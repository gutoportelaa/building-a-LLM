#!/usr/bin/env python3
"""
Create a benchmark sample from a raw scraping dataset.

The default sampling target is 5 municipalities with 20 documents each,
filtered to the provided entities (default: Prefeitura and Camara).

Usage:
    python sample_benchmark_input.py \
      --input dados/scraping_results/scraping_carnaubais_2025_deduplicados.json \
      --output dados/scraping_results/scraping_benchmark_sample.json \
      --cities 5 --per-city 20 --entities Prefeitura Camara
"""

import argparse
import json
from pathlib import Path

from common_downloader import load_records_from_json, sample_records_by_city_entity


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a benchmark sample JSON")
    parser.add_argument("--input", required=True, help="Raw scraping JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--cities", type=int, default=5, help="Number of municipalities to sample")
    parser.add_argument("--per-city", type=int, default=20, help="Documents per municipality")
    parser.add_argument("--entities", nargs="*", default=["Prefeitura", "Camara"], help="Entities to include")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sample seed")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    records = load_records_from_json(str(input_path))
    sampled = sample_records_by_city_entity(
        records,
        cities=args.cities,
        per_city=args.per_city,
        entities=args.entities,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    cities = sorted({str(record.get('municipio', '')).strip() for record in sampled if record.get('municipio')})
    entities = sorted({str(record.get('entidade', '')).strip() for record in sampled if record.get('entidade')})

    print(f"Sample saved to: {output_path}")
    print(f"Records: {len(sampled)}")
    print(f"Cities: {len(cities)} -> {', '.join(cities)}")
    print(f"Entities: {', '.join(entities)}")


if __name__ == "__main__":
    main()
