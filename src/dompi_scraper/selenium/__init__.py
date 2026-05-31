"""
Benchmark package for DOM-PI scraper performance comparison.

Provides implementations for comparing:
- requests + BeautifulSoup (sync + threaded)
- aiohttp + asyncio (async)
- Selenium (hybrid + full modes)

Usage:
    python run_benchmark.py --dataset <json_file> --concurrency 1 5 10 20
"""

__version__ = "0.1.0"
