#!/usr/bin/env python3
"""CSV schema definitions and migration helpers for DOMPI artifacts."""

from __future__ import annotations

import csv
import os

try:
    from .shared_utils import normalize_spaces  # type: ignore[reportMissingImports]
except ImportError:
    from shared_utils import normalize_spaces  # type: ignore

PUBLICACOES_FIELDS = [
    "edicao",
    "ano",
    "data",
    "municipio",
    "entidade",
    "categoria",
    "documento",
    "identificador",
    "pdf_url",
    "pdf_url_edicao",
    "pdf_url_arquivo",
    "pdf_source",
    "pdf_path",
    "md_path",
    "download_status",
    "conversion_status",
]

MANIFEST_FIELDS = [
    "timestamp",
    "municipio",
    "entidade",
    "identificador",
    "pdf_source",
    "pdf_url",
    "pdf_url_edicao",
    "pdf_url_arquivo",
    "pdf_path",
    "download_status",
    "md_path",
    "conversion_status",
    "error",
]


def _csv_has_content(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def read_csv_fieldnames(path: str) -> list[str] | None:
    if not _csv_has_content(path):
        return None

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            fieldnames = [normalize_spaces(col) for col in row if normalize_spaces(col)]
            if fieldnames:
                return fieldnames
    return None


def ensure_csv_schema(path: str, required_fieldnames: list[str]) -> list[str]:
    """Ensure CSV header contains required fields first, preserving unknown extras."""
    if not _csv_has_content(path):
        return list(required_fieldnames)

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        raw_fieldnames = reader.fieldnames or []
        current = [normalize_spaces(name) for name in raw_fieldnames if normalize_spaces(name)]

        if not current:
            return list(required_fieldnames)

        final_fieldnames = list(required_fieldnames)
        for field in current:
            if field not in final_fieldnames:
                final_fieldnames.append(field)

        if current == final_fieldnames:
            return final_fieldnames

        rows = []
        for row in reader:
            normalized_row: dict[str, str] = {}
            for key, value in row.items():
                normalized_key = normalize_spaces(str(key)) if key is not None else ""
                if normalized_key:
                    normalized_row[normalized_key] = value or ""
            rows.append(normalized_row)

    tmp_path = f"{path}.schema_tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in final_fieldnames})

    os.replace(tmp_path, path)
    return final_fieldnames
