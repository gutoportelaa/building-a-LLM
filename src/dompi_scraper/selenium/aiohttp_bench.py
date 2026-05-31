"""
Benchmark for aiohttp + asyncio approach (asynchronous).

Implements async downloading with aiohttp library. Used to extract PDF URLs
and download them concurrently with semaphore-limited connection pooling.
"""

import argparse
import asyncio
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import aiohttp

# Add parent directory to path to import common modules
sys.path.insert(0, str(Path(__file__).parent))

from common_downloader import (
    BenchmarkResult, CSVLogger, ResourceMonitor, DownloadStatus,
    compute_sha256, get_timestamp_iso, create_benchmark_result,
    setup_logging
)


class AiohttpAsyncBenchmark:
    """Benchmark downloading with aiohttp library (async)."""
    
    def __init__(
        self,
        concurrency: int = 10,
        timeout: int = 30,
        chunk_size: int = 8192,
    ):
        self.concurrency = concurrency
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
        self.semaphore = asyncio.Semaphore(concurrency)
    
    async def download_single(
        self,
        session: aiohttp.ClientSession,
        url: str,
        output_dir: str,
        run_id: str,
    ) -> BenchmarkResult:
        """
        Download a single file asynchronously and record metrics.
        
        Args:
            session: aiohttp ClientSession
            url: URL to download
            output_dir: Directory to save file
            run_id: Benchmark run identifier
        
        Returns:
            BenchmarkResult with metrics
        """
        timestamp_start = get_timestamp_iso()
        monitor = ResourceMonitor()
        monitor.start()
        
        async with self.semaphore:
            try:
                # Construct output filename from URL
                filename = url.split('/')[-1]
                if not filename or len(filename) < 3:
                    filename = f"document_{hash(url) % 100000}.pdf"
                
                output_path = Path(output_dir) / filename
                
                # Perform download
                timeout_obj = aiohttp.ClientTimeout(total=self.timeout)
                async with session.get(
                    url,
                    timeout=timeout_obj,
                    allow_redirects=True,
                ) as response:
                    # Write file
                    bytes_written = 0
                    with open(output_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(self.chunk_size):
                            if chunk:
                                f.write(chunk)
                                bytes_written += len(chunk)
                    
                    # Verify HTTP status
                    if response.status >= 400:
                        return create_benchmark_result(
                            run_id=run_id,
                            approach='aiohttp_async',
                            concurrency=self.concurrency,
                            url=url,
                            status=DownloadStatus.HTTP_ERROR,
                            success=False,
                            wall_time_s=0,
                            bytes_downloaded=0,
                            sha256=None,
                            cpu_user_s=0,
                            cpu_sys_s=0,
                            memory_rss_mb=0,
                            timestamp_start=timestamp_start,
                            timestamp_end=get_timestamp_iso(),
                            http_status_code=response.status,
                            error_msg=f"HTTP {response.status}",
                        )
                
                # Compute SHA256
                sha256 = compute_sha256(str(output_path))
                
                # Compute metrics
                wall_time, cpu_user, cpu_sys, peak_rss = monitor.end()
                timestamp_end = get_timestamp_iso()
                
                return create_benchmark_result(
                    run_id=run_id,
                    approach='aiohttp_async',
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
                    http_status_code=200,
                )
            
            except asyncio.TimeoutError:
                wall_time, cpu_user, cpu_sys, peak_rss = monitor.end()
                return create_benchmark_result(
                    run_id=run_id,
                    approach='aiohttp_async',
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
                    timestamp_end=get_timestamp_iso(),
                    error_msg='Request timeout',
                )
            
            except aiohttp.ClientError as e:
                wall_time, cpu_user, cpu_sys, peak_rss = monitor.end()
                return create_benchmark_result(
                    run_id=run_id,
                    approach='aiohttp_async',
                    concurrency=self.concurrency,
                    url=url,
                    status=DownloadStatus.NETWORK_ERROR,
                    success=False,
                    wall_time_s=wall_time,
                    bytes_downloaded=0,
                    sha256=None,
                    cpu_user_s=cpu_user,
                    cpu_sys_s=cpu_sys,
                    memory_rss_mb=peak_rss,
                    timestamp_start=timestamp_start,
                    timestamp_end=get_timestamp_iso(),
                    error_msg=str(e),
                )
            
            except Exception as e:
                wall_time, cpu_user, cpu_sys, peak_rss = monitor.end()
                return create_benchmark_result(
                    run_id=run_id,
                    approach='aiohttp_async',
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
                    timestamp_end=get_timestamp_iso(),
                    error_msg=str(e),
                )
    
    async def run_async(
        self,
        urls: List[str],
        output_dir: str,
    ) -> List[BenchmarkResult]:
        """
        Run benchmark for a set of URLs asynchronously.
        
        Args:
            urls: List of URLs to download
            output_dir: Output directory
        
        Returns:
            List of BenchmarkResult objects
        """
        run_id = f"aiohttp_{int(time.time() * 1000)}"
        
        self.logger.info(
            f"Starting aiohttp benchmark: "
            f"{len(urls)} URLs, concurrency={self.concurrency}"
        )
        
        # Create aiohttp session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.concurrency,
            limit_per_host=self.concurrency,
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        ) as session:
            tasks = [
                self.download_single(session, url, output_dir, run_id)
                for url in urls
            ]
            
            results = []
            completed = 0
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                completed += 1
                if completed % 10 == 0 or completed == len(urls):
                    self.logger.debug(f"Completed {completed}/{len(urls)}")
        
        successful = sum(1 for r in results if r.success)
        self.logger.info(
            f"Aiohttp benchmark completed: "
            f"{successful}/{len(urls)} successful"
        )
        
        return results
    
    def run(
        self,
        urls: List[str],
        output_dir: Optional[str] = None,
    ) -> List[BenchmarkResult]:
        """
        Run benchmark (sync wrapper around async).
        
        Args:
            urls: List of URLs to download
            output_dir: Output directory (default: temp dir)
        
        Returns:
            List of BenchmarkResult objects
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix='bench_aiohttp_')
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run async event loop
        return asyncio.run(self.run_async(urls, output_dir))


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
    benchmark = AiohttpAsyncBenchmark(
        concurrency=concurrency,
        timeout=timeout,
    )
    return benchmark.run(urls)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark aiohttp async scraper"
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
        default=10,
        help='Number of concurrent connections'
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
        default='results_aiohttp.csv',
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
    benchmark = AiohttpAsyncBenchmark(
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
