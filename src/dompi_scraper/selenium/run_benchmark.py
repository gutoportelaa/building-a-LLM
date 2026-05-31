#!/usr/bin/env python3
"""
Main benchmark orchestrator for DOM-PI scraper comparison.

Runs all approaches (requests_sync, aiohttp_async, selenium_hybrid, selenium_full)
across multiple concurrency levels and generates a comprehensive performance report.

Usage:
    python run_benchmark.py --dataset dados/scraping_results/scraping_carnaubais_2025_deduplicados.json
    python run_benchmark.py --dataset <file> --approaches requests_sync aiohttp_async
    python run_benchmark.py --dataset <file> --limit 50 --runs 2
"""

import argparse
import cProfile
import json
import io
import logging
import pstats
import sys
import time
from pathlib import Path
from typing import Dict, List, Callable, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parent))
    from common_downloader import (  # type: ignore
        setup_logging, load_urls_from_json, load_records_from_json,
        sample_records_by_city_entity, records_to_urls,
        CSVLogger, BenchmarkResult,
    )
else:
    from .common_downloader import (
        setup_logging, load_urls_from_json, load_records_from_json,
        sample_records_by_city_entity, records_to_urls,
        CSVLogger, BenchmarkResult,
    )

try:
    if __package__ in (None, ""):
        import requests_bench  # type: ignore
        import aiohttp_bench  # type: ignore
        import selenium_bench  # type: ignore
    else:
        from . import requests_bench, aiohttp_bench, selenium_bench
except ImportError:
    requests_bench = None  # type: ignore[assignment]
    aiohttp_bench = None  # type: ignore[assignment]
    selenium_bench = None  # type: ignore[assignment]


class BenchmarkComparator:
    """Orchestrate and compare multiple benchmark approaches."""
    
    def __init__(
        self,
        output_dir: str = "bench_results",
        log_level: int = logging.INFO,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logging(
            str(self.output_dir / f"benchmark_{int(time.time())}.log"),
            level=log_level
        )
        
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def get_approach_factories(self) -> Dict[str, Callable]:
        """Return mapping of approach names to benchmark functions."""
        if requests_bench is None or aiohttp_bench is None or selenium_bench is None:
            raise RuntimeError(
                "Benchmark modules could not be imported. Install the project dependencies before running the comparison."
            )
        return {
            'requests_sync': requests_bench.benchmark_func,
            'aiohttp_async': aiohttp_bench.benchmark_func,
            'selenium_hybrid': selenium_bench.benchmark_func_hybrid,
            'selenium_full': selenium_bench.benchmark_func_full,
        }
    
    def run_scenario(
        self,
        scenario_name: str,
        urls: List[str],
        approach: str,
        concurrency: int,
        timeout: int,
        num_runs: int,
        num_warmups: int,
        benchmark_func: Callable,
        csv_logger: CSVLogger,
        profile_enabled: bool = False,
        profile_dir: Optional[Path] = None,
    ) -> List[BenchmarkResult]:
        """Execute a single benchmark scenario."""
        self.logger.info(
            f"\n{'='*70}\n"
            f"Scenario: {scenario_name}\n"
            f"Approach: {approach} | Concurrency: {concurrency} | URLs: {len(urls)}\n"
            f"Warmups: {num_warmups} | Runs: {num_runs}\n"
            f"{'='*70}"
        )
        
        all_results = []
        
        # Warmup runs
        if num_warmups > 0:
            self.logger.info(f"Executing {num_warmups} warmup run(s)...")
            for warmup_idx in range(num_warmups):
                try:
                    self.logger.info(f"  Warmup {warmup_idx + 1}/{num_warmups}...")
                    warmup_results = benchmark_func(
                        urls=urls,
                        concurrency=concurrency,
                        timeout=timeout,
                    )
                    successful = sum(1 for r in warmup_results if r.success)
                    self.logger.info(f"  Warmup {warmup_idx + 1} completed: {successful}/{len(warmup_results)} success")
                except Exception as e:
                    self.logger.error(f"  Warmup failed: {e}", exc_info=True)
        
        # Measurement runs
        self.logger.info(f"Executing {num_runs} measurement run(s)...")
        for run_idx in range(num_runs):
            self.logger.info(f"  Measurement run {run_idx + 1}/{num_runs}...")
            try:
                profiler = cProfile.Profile() if profile_enabled else None
                if profiler is not None:
                    profiler.enable()

                run_results = benchmark_func(
                    urls=urls,
                    concurrency=concurrency,
                    timeout=timeout,
                )

                if profiler is not None:
                    profiler.disable()
                    if profile_dir is not None:
                        profile_dir.mkdir(parents=True, exist_ok=True)
                        profile_base = profile_dir / f"{scenario_name}_run{run_idx + 1}"
                        profiler.dump_stats(str(profile_base.with_suffix('.prof')))
                        stream = io.StringIO()
                        pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(40)
                        profile_base.with_suffix('.txt').write_text(stream.getvalue(), encoding='utf-8')

                all_results.extend(run_results)
                
                successful = sum(1 for r in run_results if r.success)
                total_bytes = sum(r.bytes_downloaded for r in run_results)
                avg_time = sum(r.wall_time_s for r in run_results) / len(run_results) if run_results else 0
                
                self.logger.info(
                    f"  Run {run_idx + 1} completed: "
                    f"{successful}/{len(run_results)} success, "
                    f"{total_bytes / (1024**2):.2f} MB, "
                    f"avg time: {avg_time:.3f}s"
                )
                
                # Append to CSV
                for result in run_results:
                    csv_logger.append(result)
                
            except Exception as e:
                self.logger.error(f"  Measurement run {run_idx + 1} failed: {e}", exc_info=True)
        
        return all_results
    
    def run_comparison(
        self,
        urls: List[str],
        approaches: List[str],
        concurrency_levels: List[int],
        timeout: int = 30,
        num_runs: int = 2,
        num_warmups: int = 1,
        profile_enabled: bool = False,
        profile_dir: Optional[str] = None,
    ) -> Dict[str, List[BenchmarkResult]]:
        """Execute full comparison across approaches and concurrency levels."""
        
        # Create CSV logger
        timestamp = int(time.time())
        csv_file = self.output_dir / f"results_{timestamp}.csv"
        csv_logger = CSVLogger(str(csv_file))
        self.logger.info(f"Results CSV: {csv_file}")
        profile_path = Path(profile_dir) if profile_dir else self.output_dir / 'profiles'
        
        # Get approach factories
        factories = self.get_approach_factories()
        
        self.logger.info(
            f"\n\n{'#'*70}\n"
            f"# BENCHMARK COMPARISON STARTED\n"
            f"# Approaches: {', '.join(approaches)}\n"
            f"# Concurrency levels: {concurrency_levels}\n"
            f"# URLs: {len(urls)}\n"
            f"# Runs: {num_runs} (warmups: {num_warmups})\n"
            f"{'#'*70}\n"
        )
        
        start_time = time.time()
        all_scenario_results = {}
        
        for approach in approaches:
            if approach not in factories:
                self.logger.error(f"Unknown approach: {approach}")
                continue
            
            benchmark_func = factories[approach]
            
            for concurrency in concurrency_levels:
                scenario_name = f"{approach}_conc{concurrency}"
                
                try:
                    results = self.run_scenario(
                        scenario_name=scenario_name,
                        urls=urls,
                        approach=approach,
                        concurrency=concurrency,
                        timeout=timeout,
                        num_runs=num_runs,
                        num_warmups=num_warmups,
                        benchmark_func=benchmark_func,
                        csv_logger=csv_logger,
                        profile_enabled=profile_enabled,
                        profile_dir=profile_path,
                    )
                    
                    all_scenario_results[scenario_name] = results
                
                except Exception as e:
                    self.logger.error(f"Scenario {scenario_name} failed: {e}", exc_info=True)
        
        elapsed = time.time() - start_time
        
        self.logger.info(
            f"\n{'#'*70}\n"
            f"# BENCHMARK COMPARISON COMPLETED\n"
            f"# Total elapsed time: {elapsed:.2f} seconds\n"
            f"# Results saved to: {csv_file}\n"
            f"{'#'*70}\n"
        )
        
        return all_scenario_results
    
    def print_summary(self, results: Dict[str, List[BenchmarkResult]]) -> None:
        """Print summary statistics."""
        self.logger.info("\n" + "="*70)
        self.logger.info("SUMMARY STATISTICS")
        self.logger.info("="*70)
        
        for scenario_name, scenario_results in sorted(results.items()):
            if not scenario_results:
                continue
            
            successful = [r for r in scenario_results if r.success]
            success_rate = len(successful) / len(scenario_results) if scenario_results else 0
            
            if successful:
                total_time = sum(r.wall_time_s for r in successful)
                total_bytes = sum(r.bytes_downloaded for r in successful)
                avg_time = total_time / len(successful)
                throughput = len(successful) / total_time if total_time > 0 else 0
                avg_cpu = sum(r.cpu_user_s + r.cpu_sys_s for r in successful) / len(successful)
                avg_mem = sum(r.memory_rss_mb for r in successful) / len(successful)
                
                self.logger.info(
                    f"\n{scenario_name}:\n"
                    f"  Success rate: {success_rate*100:.1f}% ({len(successful)}/{len(scenario_results)})\n"
                    f"  Avg time/doc: {avg_time:.3f}s\n"
                    f"  Throughput: {throughput:.2f} docs/sec\n"
                    f"  Total data: {total_bytes/(1024**2):.2f} MB\n"
                    f"  Avg CPU (user+sys): {avg_cpu:.3f}s\n"
                    f"  Avg memory RSS: {avg_mem:.2f} MB"
                )
            else:
                self.logger.warning(f"{scenario_name}: All runs failed!")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark and compare DOM-PI scraper approaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 50 URLs
  python run_benchmark.py --dataset dados/scraping_results/scraping_carnaubais_2025_deduplicados.json --limit 50
  
  # Full comparison (all approaches, multiple concurrency levels)
  python run_benchmark.py --dataset dados/scraping_results/scraping_carnaubais_2025_deduplicados.json --concurrency 1 5 10
  
  # Compare only requests and aiohttp
  python run_benchmark.py --dataset <file> --approaches requests_sync aiohttp_async --runs 3
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='JSON file with URLs to download (e.g., dados/scraping_results/scraping_carnaubais_2025_deduplicados.json)'
    )
    parser.add_argument(
        '--approaches',
        type=str,
        nargs='+',
        default=['requests_sync', 'aiohttp_async', 'selenium_hybrid'],
        choices=['requests_sync', 'aiohttp_async', 'selenium_hybrid', 'selenium_full'],
        help='Approaches to benchmark'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        nargs='+',
        default=[1, 5, 10],
        help='Concurrency levels to test'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='HTTP request timeout in seconds'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=2,
        help='Number of measurement runs per scenario'
    )
    parser.add_argument(
        '--warmups',
        type=int,
        default=1,
        help='Number of warmup runs per scenario'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of URLs to test (useful for quick validation)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='bench_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable cProfile collection per scenario'
    )
    parser.add_argument(
        '--profile-dir',
        type=str,
        default=None,
        help='Directory to store cProfile outputs (default: <output>/profiles)'
    )
    parser.add_argument(
        '--sample-cities',
        type=int,
        default=None,
        help='Sample this many municipalities from the raw dataset before benchmarking'
    )
    parser.add_argument(
        '--sample-per-city',
        type=int,
        default=20,
        help='Sample this many records per municipality when sampling the dataset'
    )
    parser.add_argument(
        '--sample-entities',
        nargs='*',
        default=['Prefeitura', 'Camara'],
        help='Entities to keep when sampling the raw dataset'
    )
    parser.add_argument(
        '--sample-output',
        type=str,
        default=None,
        help='Write the sampled dataset to this JSON file'
    )
    
    args = parser.parse_args()
    
    # Validate dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {args.dataset}")
        sys.exit(1)
    
    # Load dataset records, optionally sample, then convert to URLs
    records = load_records_from_json(str(dataset_path))
    if args.sample_cities is not None:
        records = sample_records_by_city_entity(
            records,
            cities=args.sample_cities,
            per_city=args.sample_per_city,
            entities=args.sample_entities,
        )
        if args.sample_output:
            sample_path = Path(args.sample_output)
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"Sample dataset written to {sample_path}")

    urls = records_to_urls(records) if records else load_urls_from_json(str(dataset_path))
    
    if args.limit:
        urls = urls[:args.limit]
    
    if not urls:
        print(f"Error: No URLs loaded from {args.dataset}")
        sys.exit(1)
    
    print(f"Loaded {len(urls)} URLs from {dataset_path}")
    
    # Create comparator and run
    comparator = BenchmarkComparator(output_dir=args.output)
    
    results = comparator.run_comparison(
        urls=urls,
        approaches=args.approaches,
        concurrency_levels=args.concurrency,
        timeout=args.timeout,
        num_runs=args.runs,
        num_warmups=args.warmups,
        profile_enabled=args.profile,
        profile_dir=args.profile_dir,
    )
    
    comparator.print_summary(results)
    
    print(f"\nBenchmark complete! Results in: {args.output}")


if __name__ == '__main__':
    main()
