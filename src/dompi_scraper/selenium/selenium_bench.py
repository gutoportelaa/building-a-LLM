"""
Benchmark for Selenium approach (hybrid and full modes).

Hybrid mode: Uses Selenium only for navigation/link extraction, then downloads via requests.
Full mode: Uses Selenium to handle downloads completely (simulates browser-driven download).

Allows comparison of Selenium rendering overhead vs pure download overhead.
"""

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select

# Add parent directory to path to import common modules
sys.path.insert(0, str(Path(__file__).parent))

from common_downloader import (
    BenchmarkResult, CSVLogger, ResourceMonitor, DownloadStatus,
    compute_sha256, get_timestamp_iso, create_benchmark_result,
    setup_logging
)


# Search page used by the PoC and the real DOM-PI consultation flow.
SEARCH_BASE_URL = "https://www.diarioficialdosmunicipios.org/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php"


class SeleniumHybridBenchmark:
    """
    Benchmark Selenium for navigation + link extraction, then download via requests.
    
    This mode isolates the cost of JavaScript rendering and dynamic content loading
    from the actual file download cost.
    """
    
    def __init__(
        self,
        concurrency: int = 5,
        timeout: int = 30,
        headless: bool = True,
    ):
        self.concurrency = concurrency
        self.timeout = timeout
        self.headless = headless
        self.logger = logging.getLogger(__name__)
    
    def create_driver(self) -> webdriver.Chrome:
        """Create a Selenium WebDriver instance."""
        options = webdriver.ChromeOptions()
        
        if self.headless:
            options.add_argument('--headless')
        
        # Additional options for better performance
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        
        return webdriver.Chrome(options=options)

    def _switch_to_results_context(self, driver: webdriver.Chrome) -> None:
        """Switch into the first iframe if the results page is embedded there."""
        driver.switch_to.default_content()
        iframes = driver.find_elements(By.TAG_NAME, 'iframe')
        if iframes:
            driver.switch_to.frame(iframes[0])

    def _expand_results_table(self, driver: webdriver.Chrome) -> bool:
        """
        Try to switch the results table to the largest available page-size option.

        This is intentionally defensive: different Scriptcase skins expose the
        rows-per-page selector with different names/ids, so we scan all selects.
        """
        changed = False
        selects = driver.find_elements(By.TAG_NAME, 'select')
        for select_element in selects:
            try:
                select = Select(select_element)
                options = [option.text.strip() for option in select.options if option.text.strip()]
                numeric_options = [option for option in options if option.isdigit()]
                if not numeric_options:
                    continue
                largest = max(numeric_options, key=lambda value: int(value))
                current = select.first_selected_option.text.strip()
                if current != largest:
                    select.select_by_visible_text(largest)
                    changed = True
            except Exception:
                continue
        return changed

    def _collect_document_links(self, driver: webdriver.Chrome) -> List[str]:
        """Collect PDF/download links from the current page or iframe context."""
        links: List[str] = []
        for context in ('default', 'iframe'):
            try:
                if context == 'iframe':
                    self._switch_to_results_context(driver)
                else:
                    driver.switch_to.default_content()

                anchors = driver.find_elements(By.TAG_NAME, 'a')
                for anchor in anchors:
                    href = anchor.get_attribute('href') or ''
                    if '.pdf' in href.lower() or 'nm_gp_submit5' in href.lower():
                        links.append(href)
                if links:
                    break
            except Exception:
                continue
        # Deduplicate preserving order.
        seen = set()
        deduped = []
        for link in links:
            if link and link not in seen:
                deduped.append(link)
                seen.add(link)
        return deduped

    def collect_links_from_search(
        self,
        municipio: str,
        entidade: str,
        search_url: str = SEARCH_BASE_URL,
        expand_max_rows: bool = True,
    ) -> List[str]:
        """Run the DOM-PI search form and collect the resulting document links."""
        driver = self.create_driver()
        wait = WebDriverWait(driver, self.timeout)
        try:
            driver.get(search_url)

            select_municipio = Select(wait.until(EC.presence_of_element_located((By.NAME, 'nomemunicipio'))))
            select_municipio.select_by_visible_text(municipio)

            select_entidade = Select(wait.until(EC.presence_of_element_located((By.NAME, 'nomeentidade'))))
            select_entidade.select_by_visible_text(entidade)

            search_button = wait.until(EC.element_to_be_clickable((
                By.XPATH,
                "//a[contains(text(), 'Pesquisa Avançada') or contains(@onclick, 'pesq') or contains(@href, 'pesq')]"
            )))
            search_button.click()

            time.sleep(2)
            self._switch_to_results_context(driver)
            if expand_max_rows:
                self._expand_results_table(driver)

            time.sleep(1)
            return self._collect_document_links(driver)
        finally:
            try:
                driver.quit()
            except Exception:
                pass
    
    def download_file_requests(
        self,
        url: str,
        output_dir: str,
        timeout: int,
    ) -> tuple:
        """Download file using requests (called after Selenium extracts link)."""
        try:
            filename = url.split('/')[-1]
            if not filename or len(filename) < 3:
                filename = f"document_{hash(url) % 100000}.pdf"
            
            output_path = Path(output_dir) / filename
            
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=5, pool_maxsize=5)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            response = session.get(
                url,
                timeout=timeout,
                allow_redirects=True,
                stream=True,
            )
            response.raise_for_status()
            
            bytes_written = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)
            
            sha256 = compute_sha256(str(output_path))
            
            return DownloadStatus.SUCCESS, True, bytes_written, sha256, response.status_code, None
        
        except Exception as e:
            return DownloadStatus.HTTP_ERROR, False, 0, None, None, str(e)
    
    def download_single(
        self,
        url: str,
        output_dir: str,
        run_id: str,
    ) -> BenchmarkResult:
        """
        Download a single file using Selenium (navigation) + requests (download).
        """
        timestamp_start = get_timestamp_iso()
        monitor = ResourceMonitor()
        monitor.start()
        
        driver = None
        try:
            # Phase 1: Selenium navigation (simulated - in real scenario, this would navigate and extract)
            # For benchmarking purposes, we'll just visit the URL and measure
            driver = self.create_driver()
            driver.set_page_load_timeout(self.timeout)
            
            # Navigate to URL (this is the Selenium rendering cost)
            driver.get(url)
            
            # Phase 2: Download via requests (after Selenium navigation)
            status, success, bytes_written, sha256, http_code, error_msg = \
                self.download_file_requests(url, output_dir, self.timeout)
            
            wall_time, cpu_user, cpu_sys, peak_rss = monitor.end()
            timestamp_end = get_timestamp_iso()
            
            return create_benchmark_result(
                run_id=run_id,
                approach='selenium_hybrid',
                concurrency=self.concurrency,
                url=url,
                status=status,
                success=success,
                wall_time_s=wall_time,
                bytes_downloaded=bytes_written,
                sha256=sha256,
                cpu_user_s=cpu_user,
                cpu_sys_s=cpu_sys,
                memory_rss_mb=peak_rss,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                http_status_code=http_code,
                error_msg=error_msg,
            )
        
        except Exception as e:
            wall_time, cpu_user, cpu_sys, peak_rss = monitor.end()
            timestamp_end = get_timestamp_iso()
            
            return create_benchmark_result(
                run_id=run_id,
                approach='selenium_hybrid',
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
        
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    def run(
        self,
        urls: List[str],
        output_dir: Optional[str] = None,
    ) -> List[BenchmarkResult]:
        """
        Run benchmark for a set of URLs using Selenium + requests.
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix='bench_selenium_hybrid_')
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        run_id = f"selenium_hybrid_{int(time.time() * 1000)}"
        
        self.logger.info(
            f"Starting Selenium hybrid benchmark: "
            f"{len(urls)} URLs, concurrency={self.concurrency}, headless={self.headless}"
        )
        
        results = []
        
        # Use threading to manage concurrent Selenium drivers
        if self.concurrency > 1:
            with ThreadPoolExecutor(max_workers=min(self.concurrency, 4)) as executor:
                futures = [
                    executor.submit(self.download_single, url, output_dir, run_id)
                    for url in urls
                ]
                
                completed = 0
                for future in futures:
                    try:
                        result = future.result(timeout=self.timeout + 10)
                        results.append(result)
                        completed += 1
                        if completed % 5 == 0 or completed == len(urls):
                            self.logger.debug(f"Completed {completed}/{len(urls)}")
                    except Exception as e:
                        self.logger.error(f"Future failed: {e}", exc_info=True)
        else:
            # Single-threaded
            for idx, url in enumerate(urls, 1):
                result = self.download_single(url, output_dir, run_id)
                results.append(result)
                if idx % 5 == 0 or idx == len(urls):
                    self.logger.debug(f"Completed {idx}/{len(urls)}")
        
        successful = sum(1 for r in results if r.success)
        self.logger.info(
            f"Selenium hybrid benchmark completed: "
            f"{successful}/{len(urls)} successful"
        )
        
        return results


class SeleniumFullBenchmark:
    """
    Benchmark Selenium for complete workflow (navigation + download via browser).
    
    This mode measures the full Selenium overhead including browser download handling.
    """
    
    def __init__(
        self,
        concurrency: int = 2,  # Lower default due to resource constraints
        timeout: int = 30,
        headless: bool = True,
    ):
        self.concurrency = concurrency
        self.timeout = timeout
        self.headless = headless
        self.logger = logging.getLogger(__name__)
    
    def create_driver(self, download_dir: str) -> webdriver.Chrome:
        """Create a Selenium WebDriver with download directory configured."""
        options = webdriver.ChromeOptions()
        
        # Configure download directory
        prefs = {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "profile.default_content_settings.popups": 0,
        }
        options.add_experimental_option("prefs", prefs)
        
        if self.headless:
            options.add_argument('--headless')
        
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        
        return webdriver.Chrome(options=options)
    
    def download_single(
        self,
        url: str,
        output_dir: str,
        run_id: str,
    ) -> BenchmarkResult:
        """Download a single file using Selenium browser download."""
        timestamp_start = get_timestamp_iso()
        monitor = ResourceMonitor()
        monitor.start()
        
        driver = None
        try:
            driver = self.create_driver(output_dir)
            driver.set_page_load_timeout(self.timeout)
            
            # Navigate to URL (triggers download via browser)
            driver.get(url)
            
            # Wait briefly for download to start/complete
            time.sleep(2)
            
            # Find the downloaded file
            import glob
            downloaded_files = glob.glob(str(Path(output_dir) / '*'))
            
            bytes_written = 0
            sha256 = None
            
            if downloaded_files:
                # Use the most recently modified file
                latest_file = max(downloaded_files, key=lambda p: Path(p).stat().st_mtime)
                bytes_written = Path(latest_file).stat().st_size
                sha256 = compute_sha256(latest_file)
                success = True
                status = DownloadStatus.SUCCESS
                http_code = 200
            else:
                success = False
                status = DownloadStatus.UNKNOWN_ERROR
                http_code = None
            
            wall_time, cpu_user, cpu_sys, peak_rss = monitor.end()
            timestamp_end = get_timestamp_iso()
            
            return create_benchmark_result(
                run_id=run_id,
                approach='selenium_full',
                concurrency=self.concurrency,
                url=url,
                status=status,
                success=success,
                wall_time_s=wall_time,
                bytes_downloaded=bytes_written,
                sha256=sha256,
                cpu_user_s=cpu_user,
                cpu_sys_s=cpu_sys,
                memory_rss_mb=peak_rss,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                http_status_code=http_code,
            )
        
        except Exception as e:
            wall_time, cpu_user, cpu_sys, peak_rss = monitor.end()
            timestamp_end = get_timestamp_iso()
            
            return create_benchmark_result(
                run_id=run_id,
                approach='selenium_full',
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
        
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    def run(
        self,
        urls: List[str],
        output_dir: Optional[str] = None,
    ) -> List[BenchmarkResult]:
        """Run benchmark for a set of URLs using Selenium for everything."""
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix='bench_selenium_full_')
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        run_id = f"selenium_full_{int(time.time() * 1000)}"
        
        self.logger.info(
            f"Starting Selenium full benchmark: "
            f"{len(urls)} URLs, concurrency={self.concurrency}, headless={self.headless}"
        )
        
        results = []
        
        # Selenium is resource-heavy, so use fewer concurrent workers
        if self.concurrency > 1:
            with ThreadPoolExecutor(max_workers=min(self.concurrency, 2)) as executor:
                futures = [
                    executor.submit(self.download_single, url, output_dir, run_id)
                    for url in urls
                ]
                
                completed = 0
                for future in futures:
                    try:
                        result = future.result(timeout=self.timeout + 15)
                        results.append(result)
                        completed += 1
                        if completed % 5 == 0 or completed == len(urls):
                            self.logger.debug(f"Completed {completed}/{len(urls)}")
                    except Exception as e:
                        self.logger.error(f"Future failed: {e}", exc_info=True)
        else:
            # Single-threaded
            for idx, url in enumerate(urls, 1):
                result = self.download_single(url, output_dir, run_id)
                results.append(result)
                if idx % 5 == 0 or idx == len(urls):
                    self.logger.debug(f"Completed {idx}/{len(urls)}")
        
        successful = sum(1 for r in results if r.success)
        self.logger.info(
            f"Selenium full benchmark completed: "
            f"{successful}/{len(urls)} successful"
        )
        
        return results


def benchmark_func_hybrid(
    urls: List[str],
    concurrency: int,
    timeout: int = 30,
) -> List[BenchmarkResult]:
    """Wrapper for Selenium hybrid mode."""
    benchmark = SeleniumHybridBenchmark(
        concurrency=concurrency,
        timeout=timeout,
        headless=True,
    )
    return benchmark.run(urls)


def benchmark_func_full(
    urls: List[str],
    concurrency: int,
    timeout: int = 30,
) -> List[BenchmarkResult]:
    """Wrapper for Selenium full mode."""
    benchmark = SeleniumFullBenchmark(
        concurrency=concurrency,
        timeout=timeout,
        headless=True,
    )
    return benchmark.run(urls)


def benchmark_func_search(
    municipio: str,
    entidade: str,
    concurrency: int,
    timeout: int = 30,
) -> List[BenchmarkResult]:
    """Search DOM-PI, collect links, then download via requests."""
    benchmark = SeleniumHybridBenchmark(
        concurrency=concurrency,
        timeout=timeout,
        headless=True,
    )
    links = benchmark.collect_links_from_search(municipio=municipio, entidade=entidade)
    return benchmark.run(links)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Selenium scraper (hybrid or full mode)"
    )
    parser.add_argument(
        '--urls',
        type=str,
        required=True,
        help='JSON file with URLs to download'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['hybrid', 'full'],
        default='hybrid',
        help='Selenium mode: hybrid (nav+requests) or full (browser download)'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=3,
        help='Number of concurrent drivers'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request/driver timeout in seconds'
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
        default='results_selenium.csv',
        help='CSV file for results'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of URLs to test'
    )
    parser.add_argument(
        '--no-headless',
        action='store_true',
        help='Run in visual mode (not headless)'
    )
    parser.add_argument(
        '--search-municipio',
        type=str,
        default=None,
        help='Run the DOM-PI consultation flow for a given municipality'
    )
    parser.add_argument(
        '--search-entidade',
        type=str,
        default=None,
        help='Filter the DOM-PI consultation flow by entity (e.g. Prefeitura or Camara)'
    )
    parser.add_argument(
        '--search-url',
        type=str,
        default=SEARCH_BASE_URL,
        help='DOM-PI consultation page URL'
    )
    parser.add_argument(
        '--expand-max-rows',
        action='store_true',
        help='Try to expand the result table to the largest row count available'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level=logging.INFO)
    
    csv_logger = CSVLogger(args.csv)

    if args.search_municipio and args.search_entidade:
        logger.info(
            f"Running DOM-PI search flow for {args.search_municipio} / {args.search_entidade}"
        )
        benchmark = SeleniumHybridBenchmark(
            concurrency=args.concurrency,
            timeout=args.timeout,
            headless=not args.no_headless,
        )
        urls = benchmark.collect_links_from_search(
            municipio=args.search_municipio,
            entidade=args.search_entidade,
            search_url=args.search_url,
            expand_max_rows=args.expand_max_rows,
        )
        if args.limit:
            urls = urls[:args.limit]
        logger.info(f"Collected {len(urls)} document links from search flow")
        results = benchmark.run(urls, output_dir=args.output)
    else:
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
        if args.mode == 'hybrid':
            benchmark = SeleniumHybridBenchmark(
                concurrency=args.concurrency,
                timeout=args.timeout,
                headless=not args.no_headless,
            )
        else:
            benchmark = SeleniumFullBenchmark(
                concurrency=args.concurrency,
                timeout=args.timeout,
                headless=not args.no_headless,
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
