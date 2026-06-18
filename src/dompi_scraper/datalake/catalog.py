"""
catalog.py — catálogo/linhagem do data lake DOM-PI.

Consolida num só lugar o que antes vivia espalhado em 4 arquivos
(`registro_dedup_*.json` + DLA `.txt`):

- `_catalog/dedup_global.parquet` — content_hash canônico (id_limpo) cross-território.
  É a fonte da deduplicação L3 (pós-limpeza). Mantém a 1ª ocorrência de cada hash.
- `_catalog/manifest.parquet` — uma linha por documento com a linhagem (hash original
  → hash limpo, território, extrator, flags de revisão, contagens).

Tudo em Polars + Parquet; escritas atômicas (tmp + os.replace).
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from . import zone_dir

_DEDUP_GLOBAL_COLUMNS = [
    "id_limpo", "id_publicacao", "territorio", "municipio",
    "tipo_ato", "ano", "fonte_ingest",
]


def dedup_global_path(root=None) -> Path:
    return zone_dir("_catalog", root) / "dedup_global.parquet"


def manifest_path(root=None) -> Path:
    return zone_dir("_catalog", root) / "manifest.parquet"


def _atomic_write_parquet(df: pl.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(dest) + ".tmp"
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, str(dest))


def load_dedup_global(root=None) -> set[str]:
    """Conjunto de id_limpo já canonizados (vazio se ainda não existe)."""
    p = dedup_global_path(root)
    if not p.exists():
        return set()
    return set(pl.read_parquet(p, columns=["id_limpo"])["id_limpo"].to_list())


def update_dedup_global(new_rows: pl.DataFrame, root=None) -> int:
    """
    Anexa novas entradas canônicas ao dedup_global, ignorando id_limpo já presentes.
    Retorna o nº de hashes efetivamente adicionados.
    """
    if new_rows.is_empty():
        return 0
    new_rows = (
        new_rows.select(_DEDUP_GLOBAL_COLUMNS)
        .unique(subset=["id_limpo"], keep="first")
    )
    p = dedup_global_path(root)
    if p.exists():
        existing = pl.read_parquet(p)
        known = set(existing["id_limpo"].to_list())
        new_rows = new_rows.filter(~pl.col("id_limpo").is_in(known))
        if new_rows.is_empty():
            return 0
        combined = pl.concat([existing, new_rows], how="vertical_relaxed")
    else:
        combined = new_rows
    _atomic_write_parquet(combined, p)
    return new_rows.height


def upsert_manifest(rows: pl.DataFrame, root=None) -> int:
    """
    Insere/atualiza linhas de linhagem no manifest (chave = id_publicacao).
    Retorna o total de linhas no manifest após a operação.
    """
    if rows.is_empty():
        return 0
    p = manifest_path(root)
    if p.exists():
        existing = pl.read_parquet(p)
        novos_ids = set(rows["id_publicacao"].to_list())
        existing = existing.filter(~pl.col("id_publicacao").is_in(novos_ids))
        combined = pl.concat([existing, rows], how="vertical_relaxed")
    else:
        combined = rows
    _atomic_write_parquet(combined, p)
    return combined.height
