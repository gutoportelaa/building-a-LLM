#!/usr/bin/env python3
"""
gerar_manifesto_pdfs.py — manifesto consultável do repositório de PDFs (DOM-PI).

O repo `dom-pi-pdfs-2025` é um dump de arquivos (PDFs por território), sem colunas. Este
script gera um `manifest.parquet` — **1 linha por PDF** com `arquivo`, `territorio`,
`municipio`, `data_publicacao`, `edicao` — tornando os PDFs filtráveis por município/data
sem mexer nos arquivos. Fonte: lista de arquivos do próprio repo (caminho + nome do arquivo
DOM-PI `DM_<edicao>_<seq>_<Municipio>_..._<AAAA-MM-DD>_pag_<N>.pdf`) + `dados/datas_map.json`
(by_file/by_edicao) + canonização do município contra a lista oficial por território
(reusa `corrigir_municipios`).

Uso:
    python -m dompi_scraper.datalake.gerar_manifesto_pdfs --upload
    python -m dompi_scraper.datalake.gerar_manifesto_pdfs --out manifest.parquet   # só gera
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import sys

import polars as pl

from .corrigir_municipios import norm, parse_todo, SLUG_TO_TD

log = logging.getLogger("gerar_manifesto_pdfs")
REPO = "gutoportelaa/dom-pi-pdfs-2025"
_RE_ED = re.compile(r"DM[_-](\d{3,5})[_-]")
_RE_DATA = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_RE_HASH = re.compile(r"^[0-9a-fA-F]{32}\.pdf$", re.IGNORECASE)   # pdfs_arquivos/<md5>.pdf
_RE_DOM = re.compile(r"DOM\d+-(\d{2})(\d{2})(\d{4})")              # Teresina: DOM3919-02012025.pdf


def _load_scraping_urlhash(scraping_dir):
    """{md5(pdf_url): {municipio,data,edicao}} — resolve arquivos hash-nomeados
    (pdfs_arquivos/<md5(url)>.pdf), cujo nome não carrega metadado."""
    m = {}
    for f in glob.glob(os.path.join(scraping_dir or "", "scraping_*_2025_deduplicados.json")):
        try:
            recs = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for r in recs:
            u = r.get("pdf_url", "")
            if u:
                m[hashlib.md5(u.encode()).hexdigest()] = {
                    "municipio": r.get("municipio", "") or "",
                    "data": r.get("data_publicacao", "") or "",
                    "edicao": str(r.get("edicao") or ""),
                }
    return m


def _build_canon(todo_path):
    """slug -> [(nkey, oficial)] ordenado do mais longo (evita match parcial). + teresina."""
    td = parse_todo(todo_path)
    slug_munis = {}
    for slug, tdn in SLUG_TO_TD.items():
        pares = [(norm(m), m) for m in td.get(tdn, [])]
        slug_munis[slug] = sorted(pares, key=lambda p: -len(p[0]))
    slug_munis["teresina"] = [("teresina", "Teresina")]

    def canon(slug, texto):
        nt = norm(texto)
        for nk, oficial in slug_munis.get(slug, []):
            if nk and nk in nt:
                return oficial
        return "DESCONHECIDO"
    return canon


def _arquivos_do_repo() -> list[str]:
    from huggingface_hub import list_repo_files
    return [f for f in list_repo_files(REPO, repo_type="dataset") if f.lower().endswith(".pdf")]


def gerar(todo="to-do_territorios.txt", datas="dados/datas_map.json",
          scraping="dados/scraping_results", arquivos=None) -> pl.DataFrame:
    canon = _build_canon(todo)
    dm = json.load(open(datas, encoding="utf-8")) if datas and os.path.exists(datas) else {}
    by_file, by_edicao = dm.get("by_file", {}), dm.get("by_edicao", {})
    urlhash = _load_scraping_urlhash(scraping)

    rows = []
    for path in (arquivos if arquivos is not None else _arquivos_do_repo()):
        parts = path.split("/")
        terr = parts[0]
        base = parts[-1]
        muni, data, ed = "DESCONHECIDO", "", ""
        # 1) subpasta-município (layout <terr>/pdfs/<Municipio>/<arq>)
        if len(parts) >= 4 and parts[1] == "pdfs":
            muni = canon(terr, parts[2])

        if _RE_HASH.match(base):
            # 2a) arquivo hash-nomeado (pdfs_arquivos/<md5(url)>.pdf) → scraping por md5(url)
            rec = urlhash.get(base[:-4].lower())
            if rec:
                if muni == "DESCONHECIDO":
                    c = canon(terr, rec["municipio"])
                    muni = c if c != "DESCONHECIDO" else (rec["municipio"] or "DESCONHECIDO")
                data, ed = rec["data"], rec["edicao"]
        else:
            # 2b) nome descritivo DOM-PI: município/edição/data do próprio nome
            if muni == "DESCONHECIDO":
                muni = canon(terr, base)
            mE = _RE_ED.search(base)
            ed = mE.group(1) if mE else ""
            data = by_file.get(base) or by_edicao.get(ed) or ""
            if not data:
                mD = _RE_DATA.search(base)
                if mD:
                    data = f"{mD.group(3)}/{mD.group(2)}/{mD.group(1)}"
                else:
                    mom = _RE_DOM.search(base)        # Teresina: DOM<ed>-DDMMAAAA
                    if mom:
                        data = f"{mom.group(1)}/{mom.group(2)}/{mom.group(3)}"
        rows.append({"arquivo": path, "territorio": terr, "municipio": muni,
                     "data_publicacao": data, "edicao": ed})
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera manifest.parquet do repo de PDFs.")
    ap.add_argument("--todo", default="to-do_territorios.txt")
    ap.add_argument("--datas", default="dados/datas_map.json")
    ap.add_argument("--scraping", default="dados/scraping_results")
    ap.add_argument("--out", default="manifest.parquet")
    ap.add_argument("--upload", action="store_true", help="Sobe manifest.parquet ao repo de PDFs.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    df = gerar(args.todo, args.datas, args.scraping)
    df.write_parquet(args.out, compression="zstd")
    n = df.height
    desc = int((df["municipio"] == "DESCONHECIDO").sum())
    sem_data = int((df["data_publicacao"] == "").sum())
    print(f"\n  Manifesto: {n} PDFs")
    print(f"  municípios distintos: {df['municipio'].n_unique()}  | DESCONHECIDO: {desc} ({100*desc/max(n,1):.1f}%)")
    print(f"  sem data: {sem_data} ({100*sem_data/max(n,1):.1f}%)")
    print(f"  por território: {df.group_by('territorio').agg(pl.len()).sort('len',descending=True).to_dicts()[:5]}")
    print(f"  → {args.out}")

    if args.upload:
        from huggingface_hub import upload_file
        upload_file(path_or_fileobj=args.out, path_in_repo="manifest.parquet",
                    repo_id=REPO, repo_type="dataset",
                    commit_message=f"Adiciona manifest.parquet ({n} PDFs: arquivo→municipio/data/edicao)")
        print(f"  ✓ enviado para {REPO}/manifest.parquet")


if __name__ == "__main__":
    sys.exit(main())
