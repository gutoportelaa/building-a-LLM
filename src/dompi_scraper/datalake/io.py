"""
io.py — escrita/leitura Parquet particionada (Hive) sobre o data lake.

Escreve com DuckDB (`COPY ... PARTITION_BY`). Idempotência por território: antes de
gravar, removemos as partições de 1º nível (`<col>=<valor>`) presentes no DataFrame,
para que reprocessar um território não duplique arquivos nem apague os demais.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import polars as pl


def write_partitioned_parquet(
    df: pl.DataFrame, dest: Path | str, partition_by: list[str]
) -> int:
    """Grava `df` como dataset Parquet Hive em `dest`. Retorna nº de linhas escritas."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if df.is_empty():
        return 0

    # Limpa só as partições de 1º nível que vamos reescrever (ex.: territorio=<slug>).
    first = partition_by[0]
    for val in df[first].unique().to_list():
        sub = dest / f"{first}={val}"
        if sub.exists():
            shutil.rmtree(sub)

    con = duckdb.connect()
    try:
        con.register("_df", df.to_arrow())
        cols = ", ".join(partition_by)
        con.execute(
            f"COPY (SELECT * FROM _df) TO '{dest.as_posix()}' "
            f"(FORMAT parquet, PARTITION_BY ({cols}), OVERWRITE_OR_IGNORE, COMPRESSION zstd)"
        )
    finally:
        con.close()
    return df.height


def zone_glob(zone_path: Path | str) -> str:
    """Glob recursivo para `read_parquet(..., hive_partitioning=true)`."""
    return (Path(zone_path) / "**" / "*.parquet").as_posix()


def read_zone(zone_path: Path | str, territorio: str | None = None) -> pl.DataFrame:
    """Lê uma zona inteira (ou um território) como Polars DataFrame. Vazio se não existir."""
    zone_path = Path(zone_path)
    if not zone_path.exists() or not any(zone_path.rglob("*.parquet")):
        return pl.DataFrame()
    con = duckdb.connect()
    try:
        # union_by_name=true reconcilia schemas que evoluíram entre partições
        # (ex.: corpus legado sem colunas de proveniência → preenchidas com null).
        q = (f"SELECT * FROM read_parquet('{zone_glob(zone_path)}', "
             f"hive_partitioning=true, union_by_name=true)")
        if territorio:
            q += f" WHERE territorio = '{territorio}'"
        return con.execute(q).pl()
    finally:
        con.close()
