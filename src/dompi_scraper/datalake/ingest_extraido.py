#!/usr/bin/env python3
"""
ingest_extraido.py — NDJSON de extração → zona EXTRAIDO (Parquet Hive).

Converte o corpus entregue pela extração (`corpus_<slug>.ndjson`, produzido no lab
SLURM ou local) em Parquet colunar particionado por `territorio`/`ano`. NÃO altera a
extração: apenas lê o NDJSON e normaliza ao schema extraido (`EXTRAIDO_COLUMNS`).

Cobre os dois schemas que existem no projeto:
  • novo  : id_publicacao, municipio, tipo_ato, data_publicacao, extrator, texto, n_chars
  • legado: doc_id, territorio, municipio, data_publicacao, texto_markdown,
            metadados_extraidos, engine_extracao, needs_reprocessing  (marca extrator='legado')

Uso:
    python -m dompi_scraper.datalake.ingest_extraido --territorio tabuleiros_alto_parnaiba
    python -m dompi_scraper.datalake.ingest_extraido --source corpus_tabuleiros.ndjson --territorio tabuleiros_alto_parnaiba
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

try:
    from ..shared_utils import classify_act_type
except ImportError:  # execução fora do pacote
    from dompi_scraper.shared_utils import classify_act_type

from . import EXTRAIDO_COLUMNS, ensure_zones, zone_dir
from .io import write_partitioned_parquet

log = logging.getLogger("ingest_extraido")


def _candidate_sources(territorio: str) -> list[Path]:
    """Locais onde o NDJSON do território costuma estar (em ordem de preferência)."""
    slug = territorio
    return [
        Path(f"corpus_{slug}.ndjson"),
        Path("extraidos") / slug / f"corpus_{slug}.ndjson",
        Path("extraidos") / slug / f"corpus_{slug}.jsonl",
    ]


def _resolve_source(territorio: str, source: str | None) -> Path:
    if source:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"--source não encontrado: {p}")
        return p
    for cand in _candidate_sources(territorio):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Nenhum corpus encontrado para '{territorio}'. Procurei: "
        + ", ".join(str(c) for c in _candidate_sources(territorio))
    )


def _first_col(df: pl.DataFrame, names: list[str], default: pl.Expr) -> pl.Expr:
    """Retorna a 1ª coluna existente entre `names`; senão a expressão `default`."""
    for n in names:
        if n in df.columns:
            return pl.col(n)
    return default


def normalizar_para_extraido(df: pl.DataFrame, territorio: str, fonte: str) -> pl.DataFrame:
    """Mapeia qualquer dos schemas conhecidos para `EXTRAIDO_COLUMNS`."""
    texto = _first_col(df, ["texto", "texto_markdown"], pl.lit("")).cast(pl.Utf8).fill_null("")
    extrator = _first_col(df, ["extrator", "engine_extracao"], pl.lit("legado")).cast(pl.Utf8)
    id_pub = _first_col(df, ["id_publicacao", "doc_id"], pl.lit("")).cast(pl.Utf8)
    municipio = _first_col(df, ["municipio"], pl.lit("")).cast(pl.Utf8).fill_null("")
    data_pub = _first_col(df, ["data_publicacao"], pl.lit("")).cast(pl.Utf8).fill_null("")
    # territorio: SEMPRE o slug do argumento (identidade canônica e partição segura).
    # Corpus legado traz aqui o nome de exibição ("Tabuleiros do Alto Parnaíba e…"),
    # que produziria partições com espaços/acentos e quebraria o dedup cross-território.

    out = df.with_columns(
        id_publicacao=id_pub,
        territorio=pl.lit(territorio),
        municipio=municipio,
        data_publicacao=data_pub,
        extrator=extrator.fill_null("legado"),
        texto=texto,
    )

    # ano (partição): aceita "AAAA..." (ano direto) ou "DD/MM/AAAA" (DOM-Teresina).
    out = out.with_columns(
        ano=pl.when(pl.col("data_publicacao").str.contains(r"^\d{4}"))
        .then(pl.col("data_publicacao").str.slice(0, 4))
        .when(pl.col("data_publicacao").str.contains(r"^\d{2}/\d{2}/\d{4}"))
        .then(pl.col("data_publicacao").str.slice(6, 4))
        .otherwise(pl.lit("sem_ano"))
    )

    # tipo_ato: usa a coluna se houver; senão classifica pelo texto.
    if "tipo_ato" in df.columns:
        out = out.with_columns(tipo_ato=pl.col("tipo_ato").cast(pl.Utf8).fill_null("sem_tipo"))
    else:
        out = out.with_columns(
            tipo_ato=pl.col("texto").map_elements(
                lambda t: classify_act_type(t or "", fallback_category="sem_tipo"),
                return_dtype=pl.Utf8,
            )
        )

    # n_chars: usa a coluna se houver; senão calcula.
    if "n_chars" in df.columns:
        out = out.with_columns(n_chars=pl.col("n_chars").cast(pl.Int64))
    else:
        out = out.with_columns(n_chars=pl.col("texto").str.len_chars().cast(pl.Int64))

    # Proveniência da extração: presente no schema novo; nula no legado.
    out = out.with_columns(
        extraido_em=_first_col(df, ["extraido_em"], pl.lit(None, dtype=pl.Utf8)).cast(pl.Utf8),
        extracao_segundos=_first_col(df, ["extracao_segundos"], pl.lit(None, dtype=pl.Float64)).cast(pl.Float64),
        paginas=_first_col(df, ["paginas"], pl.lit(None, dtype=pl.Int64)).cast(pl.Int64),
        host=_first_col(df, ["host"], pl.lit(None, dtype=pl.Utf8)).cast(pl.Utf8),
        job_id=_first_col(df, ["job_id"], pl.lit(None, dtype=pl.Utf8)).cast(pl.Utf8),
    )

    out = out.with_columns(fonte_ingest=pl.lit(fonte))
    return out.select(EXTRAIDO_COLUMNS)


def ingest(territorio: str, source: str | None = None, root=None) -> dict:
    src = _resolve_source(territorio, source)
    log.info("Lendo %s ...", src)
    df = pl.read_ndjson(src)
    n_in = df.height

    extraido = normalizar_para_extraido(df, territorio, src.name)
    # dedup defensivo dentro do próprio arquivo (id_publicacao repetido).
    extraido = extraido.filter(pl.col("id_publicacao") != "").unique(
        subset=["id_publicacao"], keep="first"
    )

    ensure_zones(root)
    dest = zone_dir("extraido", root)
    n_out = write_partitioned_parquet(extraido, dest, ["territorio", "ano"])
    log.info("Extraído gravado: %d linhas (de %d) → %s", n_out, n_in, dest)
    return {
        "territorio": territorio,
        "lidos": n_in,
        "gravados": n_out,
        "fonte": str(src),
        "anos": sorted(extraido["ano"].unique().to_list()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingere NDJSON de extração na zona extraido.")
    ap.add_argument("--territorio", required=True, help="Slug do território.")
    ap.add_argument("--source", default=None, help="Caminho do NDJSON (auto-descoberto se omitido).")
    ap.add_argument("--root", default=None, help="Raiz do data lake (padrão: ./datalake).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )
    res = ingest(args.territorio, args.source, args.root)
    print(
        f"\n  Território: {res['territorio']}\n"
        f"  Lidos:      {res['lidos']}\n"
        f"  Gravados:   {res['gravados']} (após dedup intra-arquivo)\n"
        f"  Anos:       {', '.join(res['anos'])}\n"
        f"  Fonte:      {res['fonte']}\n"
    )


if __name__ == "__main__":
    sys.exit(main())
