"""
Common downloader and instrumentation utilities for benchmark suite.

Provides reusable functions for:
- HTTP downloads with retry logic and error handling
- Resource monitoring (CPU, memory) via psutil
- Structured logging with CSV output
- URL parsing and validation
"""

import csv
import hashlib
import logging
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
from datetime import datetime
from enum import Enum

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime dependency in this environment
    psutil = None

try:
    import resource
except ImportError:  # pragma: no cover - not available on all platforms
    resource = None


class DownloadStatus(Enum):
    """Enumeration for download outcomes."""
    SUCCESS = "success"
    HTTP_ERROR = "http_error"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    WRITE_ERROR = "write_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class BenchmarkResult:
    """Structured record of a single download benchmark run."""
    run_id: str
    approach: str  # 'requests_sync', 'aiohttp_async', 'selenium_hybrid', 'selenium_full'
    concurrency: int
    url: str
    status: str  # DownloadStatus value
    success: bool
    wall_time_s: float  # Total elapsed time (seconds)
    bytes_downloaded: int
    sha256: Optional[str]
    cpu_user_s: float  # CPU user-mode time (seconds)
    cpu_sys_s: float  # CPU system-mode time (seconds)
    memory_rss_mb: float  # Resident set size in MB at end
    timestamp_start: str  # ISO 8601
    timestamp_end: str  # ISO 8601
    http_status_code: Optional[int]
    error_msg: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export."""
        return asdict(self)


class ResourceMonitor:
    """Monitor CPU and memory usage for a process."""
    
    def __init__(self):
        self.process = psutil.Process() if psutil is not None else None
        self.start_time = None
        self.start_cpu_times = None
        self.start_rss = None

    def _read_fallback_stats(self) -> Tuple[float, float, float]:
        """Read CPU and RSS information without psutil when possible."""
        if resource is None:
            return 0.0, 0.0, 0.0

        usage = resource.getrusage(resource.RUSAGE_SELF)
        cpu_user = float(usage.ru_utime)
        cpu_sys = float(usage.ru_stime)
        rss_mb = float(usage.ru_maxrss) / 1024.0
        return cpu_user, cpu_sys, rss_mb
    
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        if self.process is not None:
            self.start_cpu_times = self.process.cpu_times()
            self.start_rss = self.process.memory_info().rss / (1024 ** 2)  # MB
        else:
            self.start_cpu_times = self._read_fallback_stats()
            self.start_rss = self.start_cpu_times[2]
    
    def end(self) -> Tuple[float, float, float, float]:
        """
        Stop monitoring and return (wall_time, cpu_user, cpu_sys, peak_rss_mb).
        """
        end_time = time.time()
        if self.process is not None:
            end_cpu_times = self.process.cpu_times()
            end_rss = self.process.memory_info().rss / (1024 ** 2)  # MB
            cpu_user = end_cpu_times.user - self.start_cpu_times.user
            cpu_sys = end_cpu_times.system - self.start_cpu_times.system
        else:
            end_cpu_user, end_cpu_sys, end_rss = self._read_fallback_stats()
            cpu_user = end_cpu_user - self.start_cpu_times[0]
            cpu_sys = end_cpu_sys - self.start_cpu_times[1]
        
        wall_time = end_time - self.start_time
        peak_rss = max(self.start_rss, end_rss)
        
        return wall_time, cpu_user, cpu_sys, peak_rss


def compute_sha256(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Setup logging with optional file output."""
    logger = logging.getLogger("benchmark")
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler (if specified)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


class CSVLogger:
    """Write benchmark results to CSV file."""
    
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.results: List[BenchmarkResult] = []
        
        # Create parent directory if needed
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Write header if file doesn't exist
        if not Path(output_file).exists():
            self._write_header()
    
    def _write_header(self):
        """Write CSV header."""
        fieldnames = [field.name for field in BenchmarkResult.__dataclass_fields__.values()]
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    def append(self, result: BenchmarkResult):
        """Append a result to the CSV."""
        self.results.append(result)
        fieldnames = [field.name for field in BenchmarkResult.__dataclass_fields__.values()]
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(result.to_dict())
    
    def flush(self):
        """Flush all results to disk."""
        pass  # Already written incrementally


def load_urls_from_json(json_file: str) -> List[str]:
    """
    Load URLs from a JSON file (expected format: list of dicts with 'pdf_url' or 'url' key).
    """
    import json
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    urls = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # Try multiple common keys
                url = item.get('pdf_url') or item.get('url') or item.get('link')
                if url:
                    urls.append(url)
            elif isinstance(item, str):
                urls.append(item)
    
    return urls


def load_records_from_json(json_file: str) -> List[Dict]:
    """Load raw records from a JSON list file."""
    import json

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        records = data.get('records')
        if isinstance(records, list):
            return [item for item in records if isinstance(item, dict)]

    return []


def get_record_url(record: Dict) -> Optional[str]:
    """Return the best URL field from a dataset record."""
    return record.get('pdf_url') or record.get('url') or record.get('link')


def sample_records_by_city_entity(
    records: List[Dict],
    cities: int = 5,
    per_city: int = 20,
    entities: Optional[List[str]] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    Deterministically sample records by municipality and entity.

    The sampler tries to balance the sample across municipalities and, when
    provided, across entities such as "Prefeitura" and "Camara".
    """
    rng = random.Random(seed)
    entities_normalized = [entity.lower() for entity in entities] if entities else None

    filtered = []
    for record in records:
        municipality = str(record.get('municipio', '')).strip()
        entity = str(record.get('entidade', '')).strip().lower()
        url = get_record_url(record)
        if not municipality or not url:
            continue
        if entities_normalized and entity not in entities_normalized:
            continue
        filtered.append(record)

    by_city: Dict[str, List[Dict]] = {}
    for record in filtered:
        city = str(record.get('municipio', '')).strip()
        by_city.setdefault(city, []).append(record)

    selected_cities = sorted(by_city.keys())
    if len(selected_cities) > cities:
        selected_cities = rng.sample(selected_cities, cities)

    sampled: List[Dict] = []
    for city in selected_cities:
        city_records = by_city[city]
        city_records = sorted(
            city_records,
            key=lambda item: (
                str(item.get('entidade', '')).lower(),
                str(item.get('identificador_oficial') or item.get('id_publicacao') or get_record_url(item) or ''),
            ),
        )
        if len(city_records) > per_city:
            city_records = rng.sample(city_records, per_city)
        sampled.extend(city_records)

    return sampled


def records_to_urls(records: Iterable[Dict]) -> List[str]:
    """Extract URLs from dataset records."""
    urls: List[str] = []
    for record in records:
        url = get_record_url(record)
        if url:
            urls.append(url)
    return urls


def validate_url(url: str) -> bool:
    """Basic URL validation."""
    return url.startswith(('http://', 'https://'))


def get_timestamp_iso() -> str:
    """Get current timestamp in ISO 8601 format."""
    return datetime.utcnow().isoformat() + 'Z'


def format_bytes(num_bytes: int) -> str:
    """Format bytes to human-readable string (KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def parse_content_length(headers: Dict) -> int:
    """Extract Content-Length from HTTP headers."""
    for key in headers:
        if key.lower() == 'content-length':
            try:
                return int(headers[key])
            except (ValueError, TypeError):
                pass
    return 0


def create_benchmark_result(
    run_id: str,
    approach: str,
    concurrency: int,
    url: str,
    status: DownloadStatus,
    success: bool,
    wall_time_s: float,
    bytes_downloaded: int,
    sha256: Optional[str],
    cpu_user_s: float,
    cpu_sys_s: float,
    memory_rss_mb: float,
    timestamp_start: str,
    timestamp_end: str,
    http_status_code: Optional[int] = None,
    error_msg: Optional[str] = None,
) -> BenchmarkResult:
    """Factory function to create a BenchmarkResult."""
    return BenchmarkResult(
        run_id=run_id,
        approach=approach,
        concurrency=concurrency,
        url=url,
        status=status.value,
        success=success,
        wall_time_s=wall_time_s,
        bytes_downloaded=bytes_downloaded,
        sha256=sha256,
        cpu_user_s=cpu_user_s,
        cpu_sys_s=cpu_sys_s,
        memory_rss_mb=memory_rss_mb,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        http_status_code=http_status_code,
        error_msg=error_msg,
    )
