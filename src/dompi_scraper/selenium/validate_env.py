#!/usr/bin/env python3
"""
Environment validation script for benchmark suite.

Checks:
- Python version
- Required packages
- Optional packages
- Dataset availability
- Chrome/Selenium availability
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Verify Python 3.8+."""
    if sys.version_info < (3, 8):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} detected. Python 3.8+ required.")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_package(package_name: str, import_name: str = None, required: bool = True) -> bool:
    """Check if a package is installed."""
    import_name = import_name or package_name
    try:
        __import__(import_name)
        status = "✓"
        req_str = ""
    except ImportError:
        status = "❌" if required else "⚠"
        req_str = " (REQUIRED)" if required else " (optional)"
    
    print(f"{status} {package_name}{req_str}")
    return status == "✓"


def check_dependencies():
    """Check all required and optional packages."""
    print("\n📦 Checking dependencies:")
    
    # Required
    required_ok = all([
        check_package("requests", required=True),
        check_package("beautifulsoup4", "bs4", required=True),
        check_package("aiohttp", required=True),
        check_package("selenium", required=True),
        check_package("psutil", required=False),
    ])
    
    # Optional
    print("\nOptional packages for analysis:")
    check_package("pandas", required=False)
    check_package("matplotlib", required=False)
    
    return required_ok


def check_dataset():
    """Check if benchmark datasets exist."""
    print("\n📁 Checking dataset files:")
    
    dataset_paths = [
        Path("../../../dados/scraping_results/scraping_carnaubais_2025_deduplicados.json"),
        Path("../../../dados/scraping_results/scraping_teste_deduplicados.json"),
    ]
    
    found_any = False
    for path in dataset_paths:
        if path.exists():
            print(f"✓ {path} ({path.stat().st_size / 1024:.1f} KB)")
            found_any = True
        else:
            print(f"⚠ {path} (not found)")
    
    return found_any


def check_chrome():
    """Check if Chrome/Chromium is available for Selenium."""
    print("\n🌐 Checking Chrome/Chromium:")
    
    commands = ['google-chrome', 'chrome', 'chromium', 'chromium-browser']
    
    for cmd in commands:
        try:
            result = subprocess.run(
                [cmd, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✓ {result.stdout.strip()}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    
    print("⚠ Chrome/Chromium not found (required for Selenium benchmarks)")
    return False


def check_chromedriver():
    """Check if ChromeDriver is available."""
    print("\n🔧 Checking ChromeDriver:")
    
    try:
        # Try Selenium's auto-discovery
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        
        # This will fail gracefully if chromedriver is not found
        try:
            driver = webdriver.Chrome()
            driver.quit()
            print("✓ ChromeDriver available and working")
            return True
        except Exception as e:
            print(f"⚠ ChromeDriver not found or not working: {e}")
            print("  Install: pip install webdriver-manager")
            return False
    
    except Exception as e:
        print(f"⚠ Could not check ChromeDriver: {e}")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("Benchmark Suite - Environment Validation")
    print("=" * 60)
    
    checks = {
        "Python version": check_python_version(),
        "Required packages": check_dependencies(),
        "Dataset files": check_dataset(),
        "Chrome/Chromium": check_chrome(),
        "ChromeDriver": check_chromedriver(),
    }
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    for check_name, passed in checks.items():
        status = "✓" if passed else "⚠"
        print(f"{status} {check_name}")
    
    python_ok = checks["Python version"]
    packages_ok = checks["Required packages"]
    
    if not (python_ok and packages_ok):
        print("\n❌ Critical issues found. Please install missing packages:")
        print("   pip install requests beautifulsoup4 aiohttp selenium")
        return 1
    
    print("\n✓ Environment ready for benchmarks!")
    print("\nNext steps:")
    print("1. python run_benchmark.py --help")
    print("2. ./quick_test.sh 50")
    print("3. python analyze_results.py bench_results_quick/results_*.csv --output report.md")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
