#!/usr/bin/env python3
"""
reconstruir_coleta.py — Reorganiza coletas "flat-hash" em árvore por município.

Coletas baixadas por ``download_pdfs.py`` ficam em um diretório PLANO com nomes de
arquivo = md5(url) (ex.: ``pdfs_arquivos/90f51a52....pdf``). Nesse formato a
extração PERDE metadados: o orquestrador infere o município pelo caminho
``.../pdfs/<municipio>/<arquivo>.pdf`` e a data (P-03) pelo NOME do arquivo — um
hash não tem nem município no caminho nem data no nome.

Este utilitário lê o ``download_manifest.json`` (dict ``{hash: registro}``, cada
registro com ``municipio`` e ``url`` contendo o nome descritivo original) e recria
a estrutura esperada::

    territorios/<slug>/pdfs/<municipio>/<nome_descritivo_original>.pdf

Por padrão usa HARDLINK (mesmo filesystem, instantâneo, não duplica espaço; o
rsync para o lab envia como arquivo normal). Cai para symlink/cópia se preciso.
Idempotente: reexecutar não duplica.

Uso:
    python -m dompi_scraper.reconstruir_coleta \\
        --manifest db_treino_carnaubais/pdfs_arquivos/download_manifest.json \\
        --territorio carnaubais

    # fonte dos .pdf-hash diferente da pasta do manifest, e cópia em vez de link:
    python -m dompi_scraper.reconstruir_coleta --manifest <m.json> \\
        --territorio carnaubais --pdfs-src <dir> --modo copy
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse


def _nome_descritivo(rec: dict, hash_id: str) -> str:
    """Nome de arquivo original (basename da url); fallback = <hash>.pdf."""
    url = rec.get("url") or ""
    if url:
        nome = Path(unquote(urlparse(url).path)).name
        if nome.lower().endswith(".pdf"):
            return nome
    return f"{hash_id}.pdf"


def _fonte_pdf(rec: dict, hash_id: str, pdfs_src: Path) -> Path | None:
    """Localiza o .pdf-hash de origem: <pdfs_src>/<hash>.pdf, senão rec['path']."""
    cand = pdfs_src / f"{hash_id}.pdf"
    if cand.exists():
        return cand
    p = rec.get("path")
    if p and Path(p).exists():
        return Path(p)
    return None


def _materializar(src: Path, dest: Path, modo: str) -> None:
    """Cria dest a partir de src no modo escolhido (hardlink|symlink|copy)."""
    if modo == "copy":
        shutil.copy2(src, dest)
        return
    if modo == "symlink":
        os.symlink(src.resolve(), dest)
        return
    # hardlink (padrão), com fallback automático para symlink (cross-device)
    try:
        os.link(src, dest)
    except OSError:
        os.symlink(src.resolve(), dest)


def reconstruir(
    manifest_path: Path,
    territorio: str,
    pdfs_src: Path | None,
    repo_root: Path,
    modo: str,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SystemExit(f"Manifest inesperado (esperado dict {{hash: registro}}): {manifest_path}")

    pdfs_src = pdfs_src or manifest_path.parent
    destino_raiz = repo_root / "territorios" / territorio / "pdfs"
    destino_raiz.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": len(manifest),
        "ligados": 0,
        "ja_existiam": 0,
        "status_nao_ok": 0,
        "sem_origem": 0,
        "colisoes_resolvidas": 0,
    }
    por_municipio: Counter[str] = Counter()

    for hash_id, rec in manifest.items():
        if not isinstance(rec, dict):
            continue
        if (rec.get("status") or "").upper() != "OK":
            stats["status_nao_ok"] += 1
            continue

        src = _fonte_pdf(rec, hash_id, pdfs_src)
        if src is None:
            stats["sem_origem"] += 1
            continue

        municipio = (rec.get("municipio") or "DESCONHECIDO").strip() or "DESCONHECIDO"
        pasta = destino_raiz / municipio
        pasta.mkdir(parents=True, exist_ok=True)

        nome = _nome_descritivo(rec, hash_id)
        dest = pasta / nome
        if dest.exists():
            # Mesma origem já materializada → idempotente; nome repetido com outra
            # origem → desambigua com sufixo do hash.
            if dest.stat().st_ino == src.stat().st_ino:
                stats["ja_existiam"] += 1
                por_municipio[municipio] += 1
                continue
            dest = pasta / f"{dest.stem}__{hash_id[:8]}{dest.suffix}"
            if dest.exists():
                stats["ja_existiam"] += 1
                por_municipio[municipio] += 1
                continue
            stats["colisoes_resolvidas"] += 1

        _materializar(src, dest, modo)
        stats["ligados"] += 1
        por_municipio[municipio] += 1

    stats["municipios"] = len(por_municipio)
    stats["_por_municipio"] = dict(sorted(por_municipio.items()))
    stats["_destino"] = str(destino_raiz)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="download_manifest.json (dict {hash: registro}).")
    ap.add_argument("--territorio", required=True, help="Slug do território de destino (ex.: carnaubais).")
    ap.add_argument("--pdfs-src", default=None, help="Dir dos .pdf-hash (padrão: pasta do manifest).")
    ap.add_argument("--modo", choices=["hardlink", "symlink", "copy"], default="hardlink")
    ap.add_argument("--repo-root", default=None, help="Raiz do repo (padrão: 2 níveis acima deste módulo).")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    pdfs_src = Path(args.pdfs_src).resolve() if args.pdfs_src else None

    st = reconstruir(Path(args.manifest).resolve(), args.territorio, pdfs_src, repo_root, args.modo)

    print(f"\n  Território:      {args.territorio}")
    print(f"  Destino:         {st['_destino']}")
    print(f"  Registros:       {st['total']}")
    print(f"  Ligados ({args.modo}): {st['ligados']}")
    print(f"  Já existiam:     {st['ja_existiam']}")
    print(f"  Colisões resolv: {st['colisoes_resolvidas']}")
    print(f"  Status != OK:    {st['status_nao_ok']}")
    print(f"  Sem origem:      {st['sem_origem']}")
    print(f"  Municípios:      {st['municipios']}")
    for mun, n in st["_por_municipio"].items():
        print(f"      {mun:<28} {n}")


if __name__ == "__main__":
    sys.exit(main())
