#!/usr/bin/env python3
"""
corrigir_municipios.py — corrige a coluna `municipio` (camada extraído).

A extração capturou o município do CONTEÚDO OCR (cabeçalho "MUNICIPAL DE <cidade>")
→ ~5 mil valores-lixo ("Municip Al De Luzilandia", "Jú Li O Borges"...). O nome
limpo existe na fonte. Recupera por, em ordem:
  1) NOME DO ARQUIVO (via loghash hash→arquivo) casado contra a lista OFICIAL de
     municípios do território (to-do_territorios.txt) — cobre todos os territórios.
  2) campo `municipio` do scraping (arquivo→município) canonizado.
  3) o próprio valor OCR atual, canonizado contra a lista do território (fallback).
  4) DESCONHECIDO.
Sempre devolve o NOME OFICIAL completo ("Bom Princípio do Piauí").

Uso: python -m dompi_scraper.datalake.corrigir_municipios \\
        --todo to-do_territorios.txt --map dados/datas_map.json \\
        --log staging_lab/loghash.tsv --scraping dados/scraping_results [--dry-run]
"""
from __future__ import annotations
import argparse, glob, json, os, re, unicodedata
from urllib.parse import urlparse, unquote
import polars as pl
from . import zone_dir
from .io import read_zone, write_partitioned_parquet

SLUG_TO_TD = {
    "planice_litoran": 1, "cocais": 2, "carnaubais": 3, "entre_rios": 4,
    "vale_do_sambito": 5, "vale_do_rio_guaribas": 6, "chapada_vale_do_rio_itaim": 7,
    "vale_do_caninde": 8, "serra_da_capivara": 9, "vale_dos_rios_piaui_e_itaueiras": 10,
    "tabuleiros_alto_parnaiba": 11, "mangabeiras": 12,
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\bdo pi\b", "do piaui", s)      # "do Pi" (site) == "do Piauí" (oficial)
    return s


def parse_todo(path):
    """Devolve {td_num: [municípios oficiais]} a partir de to-do_territorios.txt."""
    txt = open(path, encoding="utf-8").read()
    out = {}
    parts = re.split(r"TD\s*(\d+)\s*[–-]\s*[^\(]+\(\d+\s*munic", txt)
    # parts: [pre, '1', body1, '2', body2, ...]
    for i in range(1, len(parts), 2):
        n = int(parts[i]); body = parts[i + 1]
        m = re.search(r"\)\s*(.+?)(?:\n\nTD|\Z)", body, re.S)
        if not m:
            m = re.search(r"\s*(.+?)(?:\n\nTD|\Z)", body, re.S)
        seg = m.group(1).strip() if m else ""
        munis = [x.strip().rstrip(".").strip() for x in re.split(r",|\s+e\s+", seg)]
        out[n] = [x for x in munis if x and len(x) > 1]
    return out


def build_resolver(todo_path, scraping_dir, log_path):
    td = parse_todo(todo_path)
    # slug -> [(nkey, oficial)], do mais longo p/ o mais curto (evita match parcial)
    slug_munis = {}
    for slug, tdn in SLUG_TO_TD.items():
        pares = [(norm(m), m) for m in td.get(tdn, [])]
        slug_munis[slug] = sorted(pares, key=lambda p: -len(p[0]))

    def base(u): return os.path.basename(unquote(urlparse(u).path)) if u else ""
    by_file = {}
    for f in glob.glob(os.path.join(scraping_dir, "scraping_*_2025_deduplicados.json")):
        for r in json.load(open(f)):
            u, m = r.get("pdf_url", ""), r.get("municipio", "")
            if u and m: by_file[base(u)] = m
    cm = "db_treino_carnaubais/pdfs_arquivos/download_manifest.json"
    if os.path.exists(cm):
        for rec in json.load(open(cm)).values():
            u, m = rec.get("url", ""), rec.get("municipio", "")
            if u and m: by_file[base(u)] = m

    logmap = {}
    for ln in open(log_path, encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) == 3: logmap[(p[0], p[1])] = p[2]

    def canon(slug, texto):
        """devolve o município oficial cujo nkey aparece em norm(texto), ou None."""
        nt = norm(texto)
        for nk, oficial in slug_munis.get(slug, []):
            if nk and nk in nt:
                return oficial
        return None

    def resolver(slug, idp, ocr_muni):
        fn = logmap.get((slug, idp[:8]))
        if fn:
            m = canon(slug, fn)                 # 1) nome do arquivo
            if m: return m
            sm = by_file.get(fn)                # 2) campo scraping desse arquivo
            if sm:
                m = canon(slug, sm)
                if m: return m
        m = canon(slug, ocr_muni)               # 3) OCR canonizado
        if m: return m
        return "DESCONHECIDO"

    return resolver, slug_munis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--todo", default="to-do_territorios.txt")
    ap.add_argument("--scraping", default="dados/scraping_results")
    ap.add_argument("--log", default="staging_lab/loghash.tsv")
    ap.add_argument("--map", default=None)  # compat; não usado
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    resolver, slug_munis = build_resolver(args.todo, args.scraping, args.log)
    tot_oficiais = sum(len(v) for v in slug_munis.values())

    b = read_zone(zone_dir("extraido", args.root))
    antes = b["municipio"].n_unique()
    novo = [resolver(t, i, m) for t, i, m in
            zip(b["territorio"].to_list(), b["id_publicacao"].to_list(), b["municipio"].to_list())]
    import collections
    cc = collections.Counter(novo)
    desconhecido = cc.get("DESCONHECIDO", 0)
    depois = len(cc)
    print(f"\n  Docs:                 {b.height}")
    print(f"  Municípios oficiais (catálogo): {tot_oficiais}")
    print(f"  municipio distintos:  ANTES {antes}  →  DEPOIS {depois}")
    print(f"  resolvidos:           {b.height - desconhecido} ({100*(b.height-desconhecido)/b.height:.1f}%)")
    print(f"  DESCONHECIDO:         {desconhecido} ({100*desconhecido/b.height:.1f}%)")
    print(f"  top municípios:       {cc.most_common(6)}")

    if args.dry_run:
        print("  [DRY-RUN] nada gravado.")
        return
    b = b.with_columns(pl.Series("municipio", novo))
    write_partitioned_parquet(b, zone_dir("extraido", args.root), ["territorio", "ano"])
    print("  ✓ camada extraído reescrita — rebuild de limpo+corpus na sequência.")


if __name__ == "__main__":
    main()
