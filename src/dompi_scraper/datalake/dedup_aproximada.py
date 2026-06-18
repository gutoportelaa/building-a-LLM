#!/usr/bin/env python3
"""
dedup_aproximada.py — detecção de QUASE-duplicatas (near-dups) por MinHash + LSH.

A dedup exata (L3/L4 em build_limpo, por `id_limpo`) só agrupa textos idênticos após
normalização. Variações de OCR, rodapé, paginação ou cabeçalho geram registros
*quase* idênticos que escapam do hash exato. Este passo:

  1. gera shingles de 5-gramas de palavras sobre o texto normalizado;
  2. estima similaridade de Jaccard via MinHash (num_perm permutações);
  3. indexa em LSH (limiar `threshold`) e liga pares acima do limiar;
  4. agrupa por componentes conexos (union-find) → clusters de near-dups;
  5. elege 1 CANÔNICO por cluster (maior texto → menos flags de revisão → data mais
     antiga → id estável) e marca os demais como redundantes.

Saída: coluna por documento — `cluster_id`, `is_near_dup` (cluster com >1 doc),
`is_canonical`. Não-destrutivo: o `build_corpus` consome essas flags para o split
'train' (só canônicos) vs 'raw' (tudo, com cluster_id para auditoria/dedup própria).

Uso (standalone, para calibrar limiar antes de aplicar):
    python -m dompi_scraper.datalake.dedup_aproximada --preview 40
    python -m dompi_scraper.datalake.dedup_aproximada --threshold 0.85 --num-perm 128
"""
from __future__ import annotations

import argparse
import logging
import re
import sys

import polars as pl

try:
    from ..shared_utils import normalize_text_for_dedup
except ImportError:  # execução fora do pacote
    from dompi_scraper.shared_utils import normalize_text_for_dedup

log = logging.getLogger("dedup_aproximada")

_DEFAULT_THRESHOLD = 0.85
_DEFAULT_NUM_PERM = 128
_DEFAULT_NGRAM = 5
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _shingles(texto: str, n: int) -> set[bytes]:
    """Conjunto de shingles de n-gramas de palavras (bytes, para o MinHash)."""
    toks = _WORD_RE.findall(normalize_text_for_dedup(texto))
    if len(toks) < n:
        # textos curtos: o próprio texto vira 1 shingle (ainda comparável entre si)
        return {(" ".join(toks)).encode("utf-8")} if toks else set()
    return {" ".join(toks[i:i + n]).encode("utf-8") for i in range(len(toks) - n + 1)}


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def marcar_near_dups(
    df: pl.DataFrame,
    threshold: float = _DEFAULT_THRESHOLD,
    num_perm: int = _DEFAULT_NUM_PERM,
    ngram: int = _DEFAULT_NGRAM,
    id_col: str = "id",
    text_col: str = "texto",
) -> pl.DataFrame:
    """
    Recebe um DataFrame com (id_col, text_col) e devolve o MESMO df acrescido de
    `cluster_id` (int), `is_near_dup` (bool) e `is_canonical` (bool).

    O canônico de cada cluster é escolhido por: maior nº de chars no texto →
    menos `review_reasons` (se existir) → `data_publicacao` mais antiga (se existir)
    → id lexicograficamente menor (desempate estável).
    """
    from datasketch import MinHash, MinHashLSH

    n = df.height
    if n == 0:
        return df.with_columns(
            cluster_id=pl.Series([], dtype=pl.Int64),
            is_near_dup=pl.Series([], dtype=pl.Boolean),
            is_canonical=pl.Series([], dtype=pl.Boolean),
        )

    ids = df[id_col].to_list()
    textos = df[text_col].to_list()

    # 1) MinHash por documento (update_batch usa numpy → viável p/ dezenas de milhares).
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: list[MinHash] = []
    log.info("MinHash de %d documentos (num_perm=%d, ngram=%d)…", n, num_perm, ngram)
    for i, txt in enumerate(textos):
        m = MinHash(num_perm=num_perm)
        sh = _shingles(txt, ngram)
        if sh:
            m.update_batch(list(sh))
        minhashes.append(m)
        lsh.insert(str(i), m)
        if (i + 1) % 10000 == 0:
            log.info("  …%d/%d", i + 1, n)

    # 2) Consulta LSH → arestas → union-find (componentes conexos = clusters).
    uf = _UnionFind(n)
    n_arestas = 0
    for i in range(n):
        for j in lsh.query(minhashes[i]):
            j = int(j)
            if j != i:
                uf.union(i, j)
                n_arestas += 1
    # agrupa índices por raiz
    grupos: dict[int, list[int]] = {}
    for i in range(n):
        grupos.setdefault(uf.find(i), []).append(i)

    # 3) Eleição de canônico + atribuição de cluster_id sequencial.
    has_reasons = "review_reasons" in df.columns
    has_data = "data_publicacao" in df.columns
    reasons = df["review_reasons"].to_list() if has_reasons else None
    datas = df["data_publicacao"].to_list() if has_data else None
    lens = [len(t or "") for t in textos]

    cluster_id = [0] * n
    is_canonical = [False] * n
    is_near_dup = [False] * n

    def _n_reasons(idx: int) -> int:
        if not has_reasons:
            return 0
        r = reasons[idx]
        if r is None:
            return 0
        return len(r) if isinstance(r, (list, tuple)) else (1 if r else 0)

    def _data_key(idx: int) -> str:
        if not has_data:
            return ""
        d = datas[idx] or ""
        # "DD/MM/AAAA" → "AAAAMMDD" p/ ordenar; valores sem data vão para o fim.
        m = re.match(r"(\d{2})/(\d{2})/(\d{4})", d)
        return f"{m.group(3)}{m.group(2)}{m.group(1)}" if m else "99999999"

    for cid, (_, membros) in enumerate(sorted(grupos.items())):
        canon = min(
            membros,
            key=lambda idx: (-lens[idx], _n_reasons(idx), _data_key(idx), ids[idx]),
        )
        near = len(membros) > 1
        for idx in membros:
            cluster_id[idx] = cid
            is_near_dup[idx] = near
        is_canonical[canon] = True

    n_clusters = len(grupos)
    n_dups = n - n_clusters  # redundantes removíveis (não-canônicos em clusters>1)
    log.info(
        "near-dup: %d docs → %d clusters (%d arestas); %d redundantes (%.1f%%)",
        n, n_clusters, n_arestas, n_dups, 100 * n_dups / n,
    )
    return df.with_columns(
        cluster_id=pl.Series(cluster_id, dtype=pl.Int64),
        is_near_dup=pl.Series(is_near_dup, dtype=pl.Boolean),
        is_canonical=pl.Series(is_canonical, dtype=pl.Boolean),
    )


def _carregar_limpo(root) -> pl.DataFrame:
    from . import zone_dir
    from .io import read_zone
    df = read_zone(zone_dir("limpo", root))
    if df.is_empty():
        raise FileNotFoundError("Limpo vazio. Rode build_limpo antes.")
    return df.unique(subset=["id_limpo"], keep="first").select(
        pl.col("id_limpo").alias("id"),
        "territorio", "municipio", "tipo_ato", "ano",
        pl.col("texto_limpo").alias("texto"),
        *( ["review_reasons"] if "review_reasons" in df.columns else [] ),
        *( ["data_publicacao"] if "data_publicacao" in df.columns else [] ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Detecta quase-duplicatas (near-dups) por MinHash+LSH.")
    ap.add_argument("--root", default=None)
    ap.add_argument("--threshold", type=float, default=_DEFAULT_THRESHOLD)
    ap.add_argument("--num-perm", type=int, default=_DEFAULT_NUM_PERM)
    ap.add_argument("--ngram", type=int, default=_DEFAULT_NGRAM)
    ap.add_argument("--preview", type=int, default=0,
                    help="Mostra N clusters de near-dup (sem gravar nada) p/ calibrar limiar.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )

    df = _carregar_limpo(args.root)
    marc = marcar_near_dups(df, args.threshold, args.num_perm, args.ngram)

    clusters = (
        marc.filter(pl.col("is_near_dup"))
        .group_by("cluster_id")
        .agg(pl.len().alias("k"), pl.col("id"), pl.col("municipio"), pl.col("tipo_ato"),
             pl.col("texto").str.len_chars().alias("chars"))
        .sort("k", descending=True)
    )
    print(f"\n  Docs:                {marc.height}")
    print(f"  Clusters near-dup:   {clusters.height}")
    print(f"  Docs redundantes:    {int((~marc['is_canonical']).sum())}")
    print(f"  Canônicos (= train): {int(marc['is_canonical'].sum())}")

    if args.preview:
        print(f"\n  === amostra de {min(args.preview, clusters.height)} clusters ===")
        for row in clusters.head(args.preview).iter_rows(named=True):
            chars = row["chars"]
            print(f"  cluster {row['cluster_id']}  k={row['k']}  "
                  f"{row['municipio'][0]}/{row['tipo_ato'][0]}  chars={chars}")


if __name__ == "__main__":
    sys.exit(main())
