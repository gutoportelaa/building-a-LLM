#!/usr/bin/env python3
"""
fatiar_megadocs.py — segmenta MEGA-documentos em atos atômicos (demanda D-2).

Medição do corpus: ~4% dos docs (>8k tokens) concentram ~40% dos tokens; a cauda
extrema (>32k tokens, ~587 docs) são edições/leis CONSOLIDADAS capturadas como 1
documento só. Política (híbrida): **fatiar só a cauda `mega` (>32k)** em atos;
classes `longo` (8k–32k) ficam apenas SINALIZADAS (sem fatiar) via `tamanho_classe`.

A segmentação usa as MESMAS fronteiras que a pipeline já conhece: cabeçalho de
entidade (PREFEITURA/CÂMARA MUNICIPAL DE …) e títulos de ato (PORTARIA/DECRETO/LEI/
EDITAL/… Nº). Cada fatia recebe `id` próprio (re-hash do conteúdo), herda
território/município/ano/data, re-classifica `tipo_ato` e re-estima `n_tokens`.
Tabelas Markdown ficam intactas (a fronteira é sempre início de ato, fora de tabela).

Uso (preview, sem gravar):
    python -m dompi_scraper.datalake.fatiar_megadocs --preview 20
"""
from __future__ import annotations

import argparse
import logging
import re
import sys

import polars as pl

try:
    from ..shared_utils import compute_content_md5, classify_act_type
except ImportError:
    from dompi_scraper.shared_utils import compute_content_md5, classify_act_type

log = logging.getLogger("fatiar_megadocs")

LIMITE_LONGO = 8192       # > normal
LIMITE_MEGA = 32768       # > longo  → alvo de fatiamento
_MIN_CHARS_FATIA = 200    # fatias menores que isto fundem com a anterior (evita lixo)

# Marcadores de início de ATO, ancorados em início de linha (re.MULTILINE).
# IMPORTANTE: NÃO incluir cabeçalhos de entidade/página (PREFEITURA/CÂMARA/ESTADO DO
# PIAUÍ) — eles se repetem como cabeçalho em CADA página de documentos fiscais
# escaneados (orçamento, RGF/RREO), e fatiar neles cortaria uma tabela contínua em
# pedaços por página. Só títulos de ato com numeração abrem fatia → compilações de
# muitos atos são fatiadas; tabelas fiscais únicas (sem títulos de ato) ficam intactas.
_BOUNDARY = re.compile(
    r"^\s*(?:"
    r"PORTARIA\s+N?[ºo°]?\s*\d"
    r"|DECRETO\s+(?:LEGISLATIVO\s+|EXECUTIVO\s+)?N?[ºo°]?\s*\d"
    r"|LEI\s+(?:COMPLEMENTAR\s+|ORDIN[ÁA]RIA\s+|MUNICIPAL\s+)?N?[ºo°]?\s*\d"
    r"|RESOLU[ÇC][ÃA]O\s+N?[ºo°]?\s*\d"
    r"|PROJETO\s+DE\s+LEI\s+N?[ºo°]?\s*\d"
    r"|EDITAL\s+(?:N?[ºo°]?\s*\d|DE\b|DO\b)"
    r"|AVISO\s+DE\s+(?:LICITA[ÇC][ÃA]O|PREG[ÃA]O|DISPENSA|INEXIGIBILIDADE|RESULTADO|HOMOLOGA[ÇC][ÃA]O)"
    r"|EXTRATO\s+(?:DE\s+|D[OA]\s+)?(?:CONTRATO|TERMO|ATA|REGISTRO)"
    r"|TERMO\s+(?:DE\s+|ADITIVO)(?:CONTRATO|ADITIVO|REFER[ÊE]NCIA|HOMOLOGA[ÇC][ÃA]O|RESCIS[ÃA]O)?"
    r"|ATA\s+(?:DE\s+)?(?:SESS[ÃA]O|REGISTRO\s+DE\s+PRE[ÇC]OS|JULGAMENTO)"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def segmentar(texto: str) -> list[str]:
    """Divide o texto em fatias começando em cada fronteira de ato.

    Funde fatias muito curtas (cabeçalhos isolados) com a fatia seguinte/anterior.
    Se nenhuma fronteira for encontrada, devolve o texto inteiro (1 fatia).
    """
    if not texto:
        return []
    starts = [m.start() for m in _BOUNDARY.finditer(texto)]
    if len(starts) <= 1:
        return [texto]
    # garante começo em 0 (preâmbulo antes do 1º marcador vira parte da 1ª fatia)
    if starts[0] != 0:
        starts = [0] + starts
    bruto = [texto[a:b] for a, b in zip(starts, starts[1:] + [len(texto)])]
    # funde fatias curtas com a próxima (ou anterior, se for a última)
    fatias: list[str] = []
    buffer = ""
    for seg in bruto:
        buffer += seg
        if len(buffer.strip()) >= _MIN_CHARS_FATIA:
            fatias.append(buffer)
            buffer = ""
    if buffer.strip():
        if fatias:
            fatias[-1] += buffer
        else:
            fatias.append(buffer)
    return [f for f in fatias if f.strip()]


def expandir_megadocs(
    df: pl.DataFrame, limite_split: int = LIMITE_LONGO, id_col: str = "id",
    text_col: str = "texto", token_col: str = "n_tokens", tipo_col: str = "tipo_ato",
) -> tuple[pl.DataFrame, dict]:
    """
    Substitui cada doc com n_tokens > `limite_split` por suas fatias de ato (re-hash id,
    metadados herdados, tipo_ato re-classificado). Docs <= limite passam intactos.
    O fatiador só abre fatia em TÍTULO DE ATO (PORTARIA/DECRETO/LEI/… Nº): compilações
    viram atos atômicos; documentos únicos (orçamento, planilha, uma lei) não têm
    fronteira interna → ficam INTACTOS (não corta tabela). Acrescenta `tamanho_classe`
    (normal ≤8k / longo 8k–32k / mega >32k) a TODAS as linhas. Devolve (df, stats).
    """
    meta_cols = [c for c in df.columns if c not in (id_col, text_col, token_col, tipo_col)]
    megas = df.filter(pl.col(token_col) > limite_split)
    resto = df.filter(pl.col(token_col) <= limite_split)

    novas: list[dict] = []
    n_fatias = 0
    n_fatiados = 0
    for row in megas.iter_rows(named=True):
        partes = segmentar(row[text_col])
        if len(partes) <= 1:
            novas.append(row)  # nada a fatiar (lei única gigante) — mantém
            continue
        n_fatiados += 1
        for p in partes:
            nc = len(p)
            nova = {c: row[c] for c in meta_cols}
            nova[id_col] = compute_content_md5(p)
            nova[text_col] = p
            nova[token_col] = nc // 4
            nova[tipo_col] = classify_act_type(p, row[tipo_col])
            novas.append(nova)
            n_fatias += 1

    partes_df = pl.DataFrame(novas, schema={c: df.schema[c] for c in df.columns}) \
        if novas else df.head(0)
    out = pl.concat([resto, partes_df], how="vertical_relaxed") if partes_df.height else resto

    out = out.with_columns(
        tamanho_classe=pl.when(pl.col(token_col) > LIMITE_MEGA).then(pl.lit("mega"))
        .when(pl.col(token_col) > LIMITE_LONGO).then(pl.lit("longo"))
        .otherwise(pl.lit("normal"))
    )
    stats = {
        "candidatos_entrada": megas.height,
        "docs_fatiados": n_fatiados,
        "docs_intactos": megas.height - n_fatiados,
        "fatias_geradas": n_fatias,
        "docs_saida": out.height,
        "docs_entrada": df.height,
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Fatia mega-docs (>32k tokens) em atos (preview).")
    ap.add_argument("--root", default=None)
    ap.add_argument("--limite-mega", type=int, default=LIMITE_MEGA)
    ap.add_argument("--preview", type=int, default=10)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    from . import zone_dir
    from .io import read_zone
    limpo = read_zone(zone_dir("limpo", args.root))
    df = limpo.unique(subset=["id_limpo"], keep="first").select(
        pl.col("id_limpo").alias("id"), "territorio", "municipio", "tipo_ato",
        "ano", "n_tokens", pl.col("texto_limpo").alias("texto"),
    )
    megas = df.filter(pl.col("n_tokens") > args.limite_mega).sort("n_tokens", descending=True)
    print(f"\n  Mega-docs (>{args.limite_mega} tok): {megas.height}")
    print(f"  === preview de fatiamento ({min(args.preview, megas.height)}) ===")
    tot_fatias = 0
    for row in megas.head(args.preview).iter_rows(named=True):
        partes = segmentar(row["texto"])
        tot_fatias += len(partes)
        print(f"  {row['municipio'][:22]:22s} {row['tipo_ato'][:10]:10s} "
              f"{row['n_tokens']:>7d} tok → {len(partes):>4d} fatias")
    print(f"\n  (amostra) {args.preview} mega-docs → {tot_fatias} fatias")


if __name__ == "__main__":
    sys.exit(main())
