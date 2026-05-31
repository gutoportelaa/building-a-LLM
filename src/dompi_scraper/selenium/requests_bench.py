"""
Benchmark for requests + BeautifulSoup approach (synchronous and threaded).

Implements both single-threaded and thread-pool based downloading with
requests library. Used to extract PDF URLs from the DOM-PI website and download them.
"""

import argparse
import logging
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Add parent directory to path to import common modules
sys.path.insert(0, str(Path(__file__).parent))

from common_downloader import (
    BenchmarkResult, CSVLogger, ResourceMonitor, DownloadStatus,
    compute_sha256, get_timestamp_iso, create_benchmark_result,
    setup_logging
)


class RequestsSyncBenchmark:
    """Benchmark downloading with requests library (sync + optional threading)."""
    
    def __init__(
        self,
        concurrency: int = 1,
        timeout: int = 30,
        use_threading: bool = True,
        chunk_size: int = 8192,
    ):
        self.concurrency = concurrency
        self.timeout = timeout
        self.use_threading = use_threading or concurrency > 1
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
        
        # Session for connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=min(concurrency, 10),
            pool_maxsize=min(concurrency, 10),
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def download_single(
        self,
        url: str,
        output_dir: str,
        run_id: str,
    ) -> BenchmarkResult:
        """
        Download a single file and record metrics.
        
        Args:
            url: URL to download
            output_dir: Directory to save file
            run_id: Benchmark run identifier
        
        Returns:
            BenchmarkResult with metrics
        """
        timestamp_start = get_timestamp_iso()
        monitor = ResourceMonitor()
        monitor.start()
        
        try:
            # Construct output filename from URL
            filename = url.split('/')[-1]
            if not filename or len(filename) < 3:
                filename = f"document_{hash(url) % 100000}.pdf"
            
            output_path = Path(output_dir) / filename
            
            # Perform download
            response = self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=True,
                stream=True,
            )
            response.raise_for_status()
            
            # Write file
            bytes_written = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)
            
            # Compute SHA256
            sha256 = compute_sha256(str(output_path))
            
            # Compute metrics
            wall_time, cpu_user, cpu_sys, peak_rss = monitor.end()
            timestamp_end = get_timestamp_iso()
            
            return create_benchmark_result(
                run_id=run_id,
                approach='requests_sync',
                concurrency=self.concurrency,
                url=url,
                status=DownloadStatus.SUCCESS,
                success=True,
                wall_time_s=wall_time,
                bytes_downloaded=bytes_written,
                sha256=sha256,
                cpu_user_s=cpu_user,
                cpu_sys_s=cpu_sys,
                memory_rss_mb=peak_rss,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                http_status_code=response.status_code,
            )
        
        except requests.Timeout:
            wall_time, cpu_user, cpu_sys, peak_rss = monitor.end()
            timestamp_end = get_timestamp_iso()
            return create_benchmark_result(
                run_id=run_id,
                approach='requests_sync',
                concurrency=self.concurrency,
                url=url,
                status=DownloadStatus.TIMEOUT,
                success=False,
                wall_time_s=wall_time,
                bytes_downloaded=0,
                sha256=None,
                cpu_user_s=cpu_user,
                cpu_sys_s=cpu_sys,
                memory_rss_mb=peak_rss,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                error_msg='Request timeout',
            )
        
        except requests.RequestException as e:
            wall_time, cpu_user, cpu_sys, peak_rss = monitor.end()
            timestamp_end = get_timestamp_iso()
            return create_benchmark_result(
                run_id=run_id,
                approach='requests_sync',
                concurrency=self.concurrency,
                url=url,
                status=DownloadStatus.HTTP_ERROR,
                success=False,
                wall_time_s=wall_time,
                bytes_downloaded=0,
                sha256=None,
                cpu_user_s=cpu_user,
                cpu_sys_s=cpu_sys,
                memory_rss_mb=peak_rss,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                http_status_code=getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None,
                error_msg=str(e),
            )
        
        except Exception as e:
            wall_time, cpu_user, cpu_sys, peak_rss = monitor.end()
            timestamp_end = get_timestamp_iso()
            return create_benchmark_result(
                run_id=run_id,
                approach='requests_sync',
                concurrency=self.concurrency,
                url=url,
                status=DownloadStatus.UNKNOWN_ERROR,
                success=False,
                wall_time_s=wall_time,
                bytes_downloaded=0,
                sha256=None,
                cpu_user_s=cpu_user,
                cpu_sys_s=cpu_sys,
                memory_rss_mb=peak_rss,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                error_msg=str(e),
            )
    
    def run(
        self,
        urls: List[str],
        output_dir: Optional[str] = None,
    ) -> List[BenchmarkResult]:
        """
        Run benchmark for a set of URLs.
        
        Args:
            urls: List of URLs to download
            output_dir: Output directory (default: temp dir)
        
        Returns:
            List of BenchmarkResult objects
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix='bench_requests_')
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        run_id = f"requests_{int(time.time() * 1000)}"
        
        self.logger.info(
            f"Starting requests benchmark: "
            f"{len(urls)} URLs, concurrency={self.concurrency}, "
            f"use_threading={self.use_threading}"
        )
        
        results = []
        
        if self.use_threading and self.concurrency > 1:
            # Multi-threaded download
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {
                    executor.submit(self.download_single, url, output_dir, run_id): url
                    for url in urls
                }
                
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    try:
                        result = future.result()
                        results.append(result)
                        if completed % 10 == 0 or completed == len(urls):
                            self.logger.debug(f"Completed {completed}/{len(urls)}")
                    except Exception as e:
                        self.logger.error(f"Future failed: {e}", exc_info=True)
        else:
            # Single-threaded download
            for idx, url in enumerate(urls, 1):
                result = self.download_single(url, output_dir, run_id)
                results.append(result)
                if idx % 10 == 0 or idx == len(urls):
                    self.logger.debug(f"Completed {idx}/{len(urls)}")
        
        successful = sum(1 for r in results if r.success)
        self.logger.info(
            f"Requests benchmark completed: "
            f"{successful}/{len(urls)} successful"
        )
        
        return results


def benchmark_func(
    urls: List[str],
    concurrency: int,
    timeout: int = 30,
) -> List[BenchmarkResult]:
    """
    Wrapper function for integration with runner_bench.py.
    
    Args:
        urls: URLs to download
        concurrency: Number of concurrent connections
        timeout: Request timeout in seconds
    
    Returns:
        List of BenchmarkResult objects
    """
    benchmark = RequestsSyncBenchmark(
        concurrency=concurrency,
        timeout=timeout,
        use_threading=concurrency > 1,
    )
    return benchmark.run(urls)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark requests + BeautifulSoup scraper"
    )
    parser.add_argument(
        '--urls',
        type=str,
        required=True,
        help='JSON file with URLs to download'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=5,
        help='Number of concurrent threads'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request timeout in seconds'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for downloaded files'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='results_requests.csv',
        help='CSV file for results'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of URLs to test'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level=logging.INFO)
    
    # Load URLs
    import json
    with open(args.urls, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    urls = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                url = item.get('pdf_url') or item.get('url')
                if url:
                    urls.append(url)
    
    if args.limit:
        urls = urls[:args.limit]
    
    logger.info(f"Loaded {len(urls)} URLs")
    
    # Run benchmark
    csv_logger = CSVLogger(args.csv)
    benchmark = RequestsSyncBenchmark(
        concurrency=args.concurrency,
        timeout=args.timeout,
    )
    
    results = benchmark.run(urls, output_dir=args.output)
    
    # Save results
    for result in results:
        csv_logger.append(result)
    
    # Print summary
    successful = sum(1 for r in results if r.success)
    avg_time = sum(r.wall_time_s for r in results) / len(results) if results else 0
    total_bytes = sum(r.bytes_downloaded for r in results)
    
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)
    logger.info(f"Success rate: {successful}/{len(results)}")
    logger.info(f"Average time per URL: {avg_time:.3f}s")
    logger.info(f"Total data transferred: {total_bytes/(1024**2):.2f} MB")
    logger.info(f"Results saved to: {args.csv}")


if __name__ == '__main__':
    main()
