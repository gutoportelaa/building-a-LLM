"""
Benchmark runner orchestrator.

Coordinates execution of different scraping/download approaches with
parametric runs, logging, and progress tracking.
"""

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parent))
    from common_downloader import (  # type: ignore
        CSVLogger, setup_logging, load_urls_from_json, get_timestamp_iso,
        BenchmarkResult, DownloadStatus,
    )
else:
    from .common_downloader import (
        CSVLogger, setup_logging, load_urls_from_json, get_timestamp_iso,
        BenchmarkResult, DownloadStatus,
    )


class BenchmarkRunner:
    """Orchestrate benchmark runs across approaches and parameters."""
    
    def __init__(
        self,
        output_dir: str = "bench_results",
        log_level: int = logging.INFO,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = setup_logging(str(log_file), level=log_level)
        
        # Initialize CSV logger
        csv_file = self.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_logger = CSVLogger(str(csv_file))
        self.csv_file = csv_file
        
        self.logger.info(f"Benchmark output directory: {self.output_dir}")
        self.logger.info(f"Results CSV: {csv_file}")
    
    def run_scenario(
        self,
        scenario_name: str,
        urls: List[str],
        approach: str,
        concurrency: int,
        timeout: int = 30,
        num_runs: int = 1,
        num_warmups: int = 1,
        benchmark_func: Optional[Callable] = None,
    ) -> List[BenchmarkResult]:
        """
        Execute a single benchmark scenario.
        
        Args:
            scenario_name: Descriptive name for this scenario
            urls: List of URLs to download
            approach: Approach name (e.g., 'requests_sync', 'aiohttp_async')
            concurrency: Number of concurrent connections
            timeout: HTTP request timeout in seconds
            num_runs: Number of measurement runs to perform
            num_warmups: Number of warmup runs to discard
            benchmark_func: Function(urls, concurrency, timeout) -> List[BenchmarkResult]
        
        Returns:
            List of BenchmarkResult objects from measurement runs (excluding warmups)
        """
        if benchmark_func is None:
            self.logger.error(f"No benchmark function provided for {approach}")
            return []
        
        self.logger.info(
            f"\n{'='*70}\n"
            f"Scenario: {scenario_name}\n"
            f"Approach: {approach} | Concurrency: {concurrency} | URLs: {len(urls)}\n"
            f"Warmups: {num_warmups} | Measurement runs: {num_runs}\n"
            f"{'='*70}"
        )
        
        all_results = []
        
        # Warmup runs
        if num_warmups > 0:
            self.logger.info(f"Executing {num_warmups} warmup run(s)...")
            for warmup_idx in range(num_warmups):
                self.logger.info(f"  Warmup {warmup_idx + 1}/{num_warmups}...")
                try:
                    warmup_results = benchmark_func(
                        urls=urls,
                        concurrency=concurrency,
                        timeout=timeout,
                    )
                    self.logger.info(f"  Warmup {warmup_idx + 1} completed: {len(warmup_results)} items")
                except Exception as e:
                    self.logger.error(f"  Warmup failed: {e}", exc_info=True)
        
        # Measurement runs
        self.logger.info(f"Executing {num_runs} measurement run(s)...")
        for run_idx in range(num_runs):
            self.logger.info(f"  Measurement run {run_idx + 1}/{num_runs}...")
            try:
                run_results = benchmark_func(
                    urls=urls,
                    concurrency=concurrency,
                    timeout=timeout,
                )
                all_results.extend(run_results)
                
                # Log aggregate stats for this run
                successful = sum(1 for r in run_results if r.success)
                total_bytes = sum(r.bytes_downloaded for r in run_results)
                avg_time = sum(r.wall_time_s for r in run_results) / len(run_results) if run_results else 0
                
                self.logger.info(
                    f"  Run {run_idx + 1} completed: "
                    f"{successful}/{len(run_results)} success, "
                    f"{total_bytes / (1024**2):.2f} MB, "
                    f"avg time: {avg_time:.2f}s"
                )
                
                # Append to CSV immediately
                for result in run_results:
                    self.csv_logger.append(result)
                
            except Exception as e:
                self.logger.error(f"  Measurement run failed: {e}", exc_info=True)
        
        return all_results
    
    def run_comparison(
        self,
        urls: List[str],
        approaches: Dict[str, Callable],
        concurrency_levels: List[int] = [1, 5, 10, 20],
        timeout: int = 30,
        num_runs: int = 3,
        num_warmups: int = 1,
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Execute a full comparison across multiple approaches and concurrency levels.
        
        Args:
            urls: URLs to benchmark
            approaches: Dict mapping approach name -> benchmark function
            concurrency_levels: List of concurrency values to test
            timeout: HTTP request timeout
            num_runs: Number of measurement runs per scenario
            num_warmups: Number of warmup runs per scenario
        
        Returns:
            Dict mapping scenario -> list of results
        """
        all_scenario_results = {}
        
        self.logger.info(
            f"\n\n{'#'*70}\n"
            f"# BENCHMARK COMPARISON STARTED\n"
            f"# Approaches: {', '.join(approaches.keys())}\n"
            f"# URLs: {len(urls)}\n"
            f"# Concurrency levels: {concurrency_levels}\n"
            f"{'#'*70}\n"
        )
        
        start_time = time.time()
        
        for approach_name, benchmark_func in approaches.items():
            for concurrency in concurrency_levels:
                scenario_name = f"{approach_name}_conc{concurrency}"
                
                try:
                    results = self.run_scenario(
                        scenario_name=scenario_name,
                        urls=urls,
                        approach=approach_name,
                        concurrency=concurrency,
                        timeout=timeout,
                        num_runs=num_runs,
                        num_warmups=num_warmups,
                        benchmark_func=benchmark_func,
                    )
                    
                    all_scenario_results[scenario_name] = results
                    
                except Exception as e:
                    self.logger.error(f"Scenario {scenario_name} failed: {e}", exc_info=True)
        
        elapsed = time.time() - start_time
        self.logger.info(
            f"\n{'#'*70}\n"
            f"# BENCHMARK COMPARISON COMPLETED\n"
            f"# Total elapsed time: {elapsed:.2f} seconds\n"
            f"# Results saved to: {self.csv_file}\n"
            f"{'#'*70}\n"
        )
        
        return all_scenario_results
    
    def summary(self, results: Dict[str, List[BenchmarkResult]]) -> None:
        """Print summary statistics for all scenarios."""
        self.logger.info("\n" + "="*70)
        self.logger.info("SUMMARY STATISTICS")
        self.logger.info("="*70)
        
        for scenario_name, scenario_results in results.items():
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
                    f"  Avg CPU time: {avg_cpu:.3f}s\n"
                    f"  Avg memory: {avg_mem:.2f} MB"
                )
            else:
                self.logger.warning(f"{scenario_name}: All runs failed!")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark runner for DOM-PI scraper/downloader approaches"
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
        help='Approaches to benchmark (space-separated)'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        nargs='+',
        default=[1, 5, 10, 20],
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
        default=3,
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
    
    args = parser.parse_args()
    
    # Load URLs
    if not Path(args.dataset).exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    urls = load_urls_from_json(args.dataset)
    
    if args.limit:
        urls = urls[:args.limit]
    
    print(f"Loaded {len(urls)} URLs from {args.dataset}")
    
    # Create runner
    runner = BenchmarkRunner(output_dir=args.output)
    
    # Import approach modules dynamically
    # For now, we'll just inform the user that they need to implement these
    runner.logger.warning(
        "Benchmark runner initialized. "
        "You need to implement approach modules: requests_bench.py, aiohttp_bench.py, selenium_bench.py"
    )
    
    print(f"\nBenchmark runner ready!")
    print(f"Output directory: {args.output}")
    print(f"CSV results file: {runner.csv_file}")


if __name__ == '__main__':
    main()
