#!/usr/bin/env python3
"""Shared utility helpers used across DOMPI pipeline scripts."""

from __future__ import annotations

import csv
import os
import re


def normalize_spaces(value: str) -> str:
    """Normalize whitespace to a single space and trim ends."""
    return re.sub(r"\s+", " ", value or "").strip()


def slugify(value: str, fallback: str = "sem_nome", trim_chars: str = "_") -> str:
    """Build filesystem-safe slugs with configurable fallback behavior."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    cleaned = cleaned.strip(trim_chars)
    return cleaned or fallback


def read_csv_rows(path: str) -> list[dict[str, str]]:
    """Load all CSV rows as dictionaries, returning an empty list when missing."""
    if not os.path.exists(path):
        return []

    with open(path, newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
