#!/usr/bin/env python3
"""
query.py — consultas DuckDB ad-hoc sobre o data lake DOM-PI.

Registra views Hive para cada zona/catálogo e roda SQL arbitrário. Sem argumento de
SQL, imprime um relatório de cobertura (docs por território/ano, % needs_review,
contagem por extrator) — útil para inspecionar o acervo sem gastar GPU.

Uso:
    python -m dompi_scraper.datalake.query
    python -m dompi_scraper.datalake.query "SELECT tipo_ato, count(*) c FROM limpo GROUP BY 1 ORDER BY c DESC"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

from . import zone_dir
from .io import zone_glob


def _conexao(root=None) -> duckdb.DuckDBPyConnection:
    """Conexão com views extraido/limpo/corpus/manifest/dedup_global (as que existirem)."""
    con = duckdb.connect()
    views = {
        "extraido": zone_dir("extraido", root),
        "limpo": zone_dir("limpo", root),
        "corpus": zone_dir("corpus", root) / "corpus_llm",
    }
    for name, path in views.items():
        if Path(path).exists() and any(Path(path).rglob("*.parquet")):
            con.execute(
                f"CREATE VIEW {name} AS "
                f"SELECT * FROM read_parquet('{zone_glob(path)}', "
                f"hive_partitioning=true, union_by_name=true)"
            )
    catalogo = zone_dir("_catalog", root)
    for name in ("manifest", "dedup_global"):
        p = catalogo / f"{name}.parquet"
        if p.exists():
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{p.as_posix()}')")
    return con


def _relatorio(con: duckdb.DuckDBPyConnection) -> None:
    tem = {r[0] for r in con.execute(
        "SELECT view_name FROM duckdb_views() WHERE NOT internal"
    ).fetchall()}
    if "limpo" in tem:
        print("\n== Cobertura por território/ano (limpo) ==")
        print(con.execute(
            "SELECT territorio, ano, count(*) docs, "
            "round(100.0*sum(needs_human_review::int)/count(*),1) pct_review "
            "FROM limpo GROUP BY 1,2 ORDER BY 1,2"
        ).pl())
        print("\n== Top tipos de ato (limpo) ==")
        print(con.execute(
            "SELECT tipo_ato, count(*) docs FROM limpo GROUP BY 1 ORDER BY docs DESC LIMIT 15"
        ).pl())
        print("\n== Extrator (limpo) ==")
        print(con.execute(
            "SELECT extrator, count(*) docs FROM limpo GROUP BY 1 ORDER BY docs DESC"
        ).pl())
    elif "extraido" in tem:
        print("\n(Só há extraido — rode build_limpo para o relatório completo.)")
        print(con.execute(
            "SELECT territorio, ano, count(*) docs FROM extraido GROUP BY 1,2 ORDER BY 1,2"
        ).pl())
    else:
        print("Data lake vazio — rode ingest_extraido primeiro.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Consulta DuckDB sobre o data lake.")
    ap.add_argument("sql", nargs="?", default=None, help="SQL a executar (views: extraido/limpo/corpus/manifest/dedup_global).")
    ap.add_argument("--root", default=None, help="Raiz do data lake (padrão: ./datalake).")
    args = ap.parse_args()

    con = _conexao(args.root)
    if args.sql:
        print(con.execute(args.sql).pl())
    else:
        _relatorio(con)
    con.close()


if __name__ == "__main__":
    sys.exit(main())
