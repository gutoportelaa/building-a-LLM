#!/bin/bash
# Quick benchmark test script
# Usage: ./quick_test.sh [num_urls] [output_dir]

set -e

NUM_URLS=${1:-50}
OUTPUT_DIR=${2:-bench_results_quick}
DATASET="../../../dados/scraping_results/scraping_carnaubais_2025_deduplicados.json"

echo "============================================"
echo "Quick Benchmark Test"
echo "============================================"
echo "Dataset: $DATASET"
echo "Limit: $NUM_URLS URLs"
echo "Output: $OUTPUT_DIR"
echo "Approaches: requests_sync, aiohttp_async, selenium_hybrid"
echo "Concurrency levels: 1, 5, 10"
echo "============================================"
echo ""

if [ ! -f "$DATASET" ]; then
    echo "Error: Dataset not found at $DATASET"
    exit 1
fi

echo "[1/2] Running benchmarks..."
python run_benchmark.py \
    --dataset "$DATASET" \
    --approaches requests_sync aiohttp_async selenium_hybrid \
    --concurrency 1 5 10 \
    --limit "$NUM_URLS" \
    --runs 2 \
    --warmups 1 \
    --output "$OUTPUT_DIR"

echo ""
echo "[2/2] Analyzing results..."
python analyze_results.py "$OUTPUT_DIR"/results_*.csv \
    --output "$OUTPUT_DIR/report.md" \
    --plots-dir "$OUTPUT_DIR/plots"

echo ""
echo "============================================"
echo "Done!"
echo "Results: $OUTPUT_DIR/results_*.csv"
echo "Report:  $OUTPUT_DIR/report.md"
echo "Plots:   $OUTPUT_DIR/plots/"
echo "Log:     $OUTPUT_DIR/benchmark_*.log"
echo "============================================"
