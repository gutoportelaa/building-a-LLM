#!/usr/bin/env python3
"""
build_limpo.py — EXTRAIDO → LIMPO (limpeza + re-hash + dedup L3 + flags).

Transform colunar que reusa a limpeza já validada do projeto:
  • `limpeza_textos.clean_text`        — remove ruído de OCR, preserva tabelas Markdown,
                                         e devolve as razões de revisão.
  • `shared_utils.compute_content_md5` — hash do CORPO normalizado.

Faz, por documento:
  1. texto_limpo = clean_text(texto)
  2. id_limpo    = compute_content_md5(texto_limpo)         (P-09: re-hash pós-limpeza)
  3. flags de severidade (P-08): `assinaturas_detectadas` é informativo (não marca);
     `needs_human_review` só para razões de alta severidade.
  4. dedup L3 por id_limpo: dentro do lote e contra OUTROS territórios já em limpo.
  5. métricas: n_chars_limpo, n_tokens (estimado).

Depois reconstrói `_catalog/dedup_global.parquet` a partir do limpo inteiro (idempotente)
e faz upsert do `_catalog/manifest.parquet`.

Uso:
    python -m dompi_scraper.datalake.build_limpo --territorio tabuleiros_alto_parnaiba
    python -m dompi_scraper.datalake.build_limpo --all
"""
from __future__ import annotations

import argparse
import logging
import sys

import polars as pl

try:
    from ..limpeza_textos import clean_text, _HIGH_SEVERITY_REASONS
    from ..shared_utils import compute_content_md5
except ImportError:
    from dompi_scraper.limpeza_textos import clean_text, _HIGH_SEVERITY_REASONS
    from dompi_scraper.shared_utils import compute_content_md5

from . import LIMPO_COLUMNS, ensure_zones, zone_dir
from .catalog import update_dedup_global, upsert_manifest, dedup_global_path
from .io import read_zone, write_partitioned_parquet

log = logging.getLogger("build_limpo")

_MANIFEST_COLUMNS = [
    "id_publicacao", "id_limpo", "territorio", "municipio", "tipo_ato", "ano",
    "data_publicacao", "extrator", "fonte_ingest", "n_chars", "n_chars_limpo",
    "needs_human_review", "review_reasons",
]


def _limpar_registro(rec: dict) -> dict:
    """Aplica clean_text + re-hash + flags a um registro extraido; devolve campos limpo."""
    texto = rec.get("texto") or ""
    texto_limpo, stats = clean_text(texto)
    reasons = stats.get("review_reasons", []) or []
    needs_review = any(r in _HIGH_SEVERITY_REASONS for r in reasons)
    n_chars_limpo = len(texto_limpo)
    rec = dict(rec)
    rec.update(
        id_limpo=compute_content_md5(texto_limpo),
        texto_limpo=texto_limpo,
        n_chars_limpo=n_chars_limpo,
        n_tokens=n_chars_limpo // 4,  # estimativa barata (~4 chars/token) p/ orçar corpus
        assinaturas_detectadas="assinaturas_detectadas" in reasons,
        needs_human_review=needs_review,
        review_reasons=", ".join(sorted(reasons)),
    )
    return rec


def build_limpo_territorio(territorio: str, root=None) -> dict:
    extraido = read_zone(zone_dir("extraido", root), territorio=territorio)
    if extraido.is_empty():
        raise FileNotFoundError(f"Extraído vazio para '{territorio}'. Rode ingest_extraido antes.")
    n_in = extraido.height

    # Limpeza row-level (volume modesto: ~dezenas de milhares de docs).
    limpos = pl.DataFrame([_limpar_registro(r) for r in extraido.to_dicts()])

    # Dedup L3 — dentro do lote.
    limpos = limpos.unique(subset=["id_limpo"], keep="first")
    n_self = limpos.height

    # Dedup L3 — contra OUTROS territórios já presentes em limpo (cross-território).
    limpo_outros = read_zone(zone_dir("limpo", root))
    if not limpo_outros.is_empty():
        known_outros = set(
            limpo_outros.filter(pl.col("territorio") != territorio)["id_limpo"].to_list()
        )
        if known_outros:
            limpos = limpos.filter(~pl.col("id_limpo").is_in(known_outros))
    n_out = limpos.height

    limpos = limpos.select(LIMPO_COLUMNS)

    ensure_zones(root)
    write_partitioned_parquet(limpos, zone_dir("limpo", root), ["territorio", "ano"])

    # Catálogo: manifest (upsert) + dedup_global (reconstruído do limpo inteiro).
    upsert_manifest(limpos.select(_MANIFEST_COLUMNS), root)
    _reconstruir_dedup_global(root)

    log.info(
        "Limpo %s: %d extraido → %d após dedup intra (%d) e cross-território (%d)",
        territorio, n_in, n_out, n_in - n_self, n_self - n_out,
    )
    return {
        "territorio": territorio,
        "extraido": n_in,
        "dedup_intra": n_in - n_self,
        "dedup_cross": n_self - n_out,
        "limpo": n_out,
        "needs_review": int(limpos["needs_human_review"].sum()),
        "assinaturas": int(limpos["assinaturas_detectadas"].sum()),
    }


def _reconstruir_dedup_global(root=None) -> None:
    """Recria dedup_global.parquet como o conjunto distinto de id_limpo do limpo."""
    limpo = read_zone(zone_dir("limpo", root))
    if limpo.is_empty():
        return
    canon = limpo.select(
        ["id_limpo", "id_publicacao", "territorio", "municipio", "tipo_ato", "ano"]
    ).unique(subset=["id_limpo"], keep="first").with_columns(
        fonte_ingest=pl.lit("limpo")
    )
    # Recria do zero (idempotente): apaga o antigo e regrava.
    p = dedup_global_path(root)
    if p.exists():
        p.unlink()
    update_dedup_global(canon, root)


def _territorios_em_extraido(root=None) -> list[str]:
    extraido = read_zone(zone_dir("extraido", root))
    if extraido.is_empty():
        return []
    return sorted(extraido["territorio"].unique().to_list())


def main() -> None:
    ap = argparse.ArgumentParser(description="Constrói a zona limpo a partir do extraido.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--territorio", help="Slug do território.")
    g.add_argument("--all", action="store_true", help="Processa todos os territórios em extraido.")
    ap.add_argument("--root", default=None, help="Raiz do data lake (padrão: ./datalake).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )

    alvos = _territorios_em_extraido(args.root) if args.all else [args.territorio]
    if not alvos:
        ap.error("Nenhum território em extraido. Rode ingest_extraido antes.")

    for t in alvos:
        res = build_limpo_territorio(t, args.root)
        pct = 100 * res["needs_review"] / max(1, res["limpo"])
        print(
            f"\n  {res['territorio']}\n"
            f"    extraido:        {res['extraido']}\n"
            f"    dedup intra:   {res['dedup_intra']}\n"
            f"    dedup cross:   {res['dedup_cross']}\n"
            f"    limpo:        {res['limpo']}\n"
            f"    needs_review:  {res['needs_review']} ({pct:.1f}%)\n"
            f"    assinaturas:   {res['assinaturas']} (informativo, não marca)\n"
        )


if __name__ == "__main__":
    sys.exit(main())
