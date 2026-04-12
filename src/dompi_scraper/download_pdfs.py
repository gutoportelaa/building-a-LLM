#!/usr/bin/env python3
"""
download_pdfs.py — Download Controlado de PDFs do DOM-PI
---------------------------------------------------------
Consome o JSON deduplicado gerado pelo scraper_isolado.py e baixa os arquivos
PDF para disco local, gerando um manifesto de integridade (SHA-256) por arquivo.

Características:
- Incremental: pula PDFs que já existem em disco (verifica por hash MD5 da URL)
- Integridade: calcula SHA-256 pós-download e registra no manifesto
- Idempotente: rodar múltiplas vezes é seguro — nunca sobrescreve
- Controlável: --limite N para testes parciais, --dry-run para simulação

O manifesto gerado (download_manifest.json) é a entrada para o próximo estágio
do pipeline (processar_pdfs.py).

Uso:
    # Download dos primeiros 5 PDFs para teste
    uv run python src/dompi_scraper/download_pdfs.py \\
        --input scraping_carnaubais_2025_deduplicados.json \\
        --output-dir db_treino_carnaubais/pdfs_arquivos \\
        --limite 5

    # Download completo
    uv run python src/dompi_scraper/download_pdfs.py \\
        --input scraping_carnaubais_2025_deduplicados.json \\
        --output-dir db_treino_carnaubais/pdfs_arquivos

    # Simulação (sem download)
    uv run python src/dompi_scraper/download_pdfs.py \\
        --input scraping_carnaubais_2025_deduplicados.json \\
        --output-dir db_treino_carnaubais/pdfs_arquivos \\
        --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("download_pdfs")


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    log.setLevel(level)
    log.handlers.clear()
    log.addHandler(handler)


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def md5_string(text: str) -> str:
    """MD5 hexadecimal de uma string (usado como ID do arquivo por URL)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def compute_sha256(filepath: str) -> str:
    """Calcula SHA-256 chunked de um arquivo em disco."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def build_session() -> requests.Session:
    """Sessão HTTP com headers de browser para evitar bloqueios."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36",
    })
    return session


def download_file(session: requests.Session, url: str, dest: str, max_retries: int = 3) -> bool:
    """
    Baixa um arquivo com retry exponencial.
    Retorna True se o download foi bem-sucedido.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, stream=True, timeout=60)
            resp.raise_for_status()

            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Verifica se o arquivo tem conteúdo mínimo
            size = os.path.getsize(dest)
            if size < 100:  # PDF válido tem pelo menos alguns KB
                log.warning(f"  Arquivo muito pequeno ({size}B), pode ser inválido: {dest}")
                return False

            return True

        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                log.warning(f"  Tentativa {attempt}/{max_retries} falhou: {e}. Retry em {wait}s...")
                time.sleep(wait)
            else:
                log.error(f"  FALHA definitiva após {max_retries} tentativas: {e}")
                # Remove arquivo parcial
                if os.path.exists(dest):
                    os.remove(dest)
                return False


# ---------------------------------------------------------------------------
# Pipeline de Download
# ---------------------------------------------------------------------------

def load_dedup_json(path: str) -> list[dict]:
    """
    Carrega o JSON deduplicado e filtra registros incompletos (ex: 'Teste').
    Retorna apenas registros com pdf_url válida e data_publicacao preenchida.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid = []
    skipped = 0
    for record in data:
        url = record.get("pdf_url", "")
        if not url or not url.startswith("http"):
            skipped += 1
            continue
        if not record.get("data_publicacao"):
            skipped += 1
            continue
        valid.append(record)

    if skipped:
        log.info(f"Registros descartados (incompletos/teste): {skipped}")

    return valid


def run_download_pipeline(
    input_json: str,
    output_dir: str,
    manifest_path: str,
    limite: int,
    dry_run: bool,
) -> dict:
    """
    Executa o pipeline de download completo.

    Returns:
        Estatísticas do processo {total, novos, preexistentes, falhas}
    """
    records = load_dedup_json(input_json)
    log.info(f"Registros válidos para download: {len(records)}")

    os.makedirs(output_dir, exist_ok=True)

    # Carrega manifesto existente (incremental)
    manifest: dict[str, dict] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        log.info(f"Manifesto existente carregado: {len(manifest)} entradas")

    session = build_session()

    stats = {"total": 0, "novos": 0, "preexistentes": 0, "falhas": 0}
    processed = 0

    for i, record in enumerate(records):
        if processed >= limite:
            log.info(f"Limite de {limite} downloads atingido.")
            break

        url = record["pdf_url"]
        file_id = md5_string(url)
        filename = f"{file_id}.pdf"
        filepath = os.path.join(output_dir, filename)

        stats["total"] += 1
        processed += 1

        # Incremental: pula se já existe no manifesto com status OK
        if file_id in manifest and manifest[file_id].get("status") == "OK":
            if os.path.exists(filepath):
                stats["preexistentes"] += 1
                if processed % 500 == 0:
                    log.debug(f"  [{processed}/{min(limite, len(records))}] Já existe: {filename}")
                continue

        log.info(
            f"  [{processed}/{min(limite, len(records))}] "
            f"Baixando: {record.get('municipio', '?')} / {record.get('categoria', '?')} → {filename}"
        )

        if dry_run:
            stats["novos"] += 1
            manifest[file_id] = {
                "url": url,
                "path": filepath,
                "status": "DRY_RUN",
                "sha256": "",
                "tamanho_bytes": 0,
                "municipio": record.get("municipio", ""),
                "entidade": record.get("entidade", ""),
                "categoria": record.get("categoria", ""),
                "data_publicacao": record.get("data_publicacao", ""),
                "documento": record.get("documento", ""),
                "identificador_oficial": record.get("identificador_oficial", ""),
                "edicao": record.get("edicao", ""),
                "edicao_url_meta": record.get("edicao_url_meta", ""),
                "codigo_interno_mun_meta": record.get("codigo_interno_mun_meta", ""),
                "pagina_url_meta": record.get("pagina_url_meta", ""),
            }
            continue

        # Download real
        ok = download_file(session, url, filepath)

        if ok:
            sha256 = compute_sha256(filepath)
            size = os.path.getsize(filepath)
            stats["novos"] += 1

            manifest[file_id] = {
                "url": url,
                "path": filepath,
                "status": "OK",
                "sha256": sha256,
                "tamanho_bytes": size,
                "municipio": record.get("municipio", ""),
                "entidade": record.get("entidade", ""),
                "categoria": record.get("categoria", ""),
                "data_publicacao": record.get("data_publicacao", ""),
                "documento": record.get("documento", ""),
                "identificador_oficial": record.get("identificador_oficial", ""),
                "edicao": record.get("edicao", ""),
                "edicao_url_meta": record.get("edicao_url_meta", ""),
                "codigo_interno_mun_meta": record.get("codigo_interno_mun_meta", ""),
                "pagina_url_meta": record.get("pagina_url_meta", ""),
            }
        else:
            stats["falhas"] += 1
            manifest[file_id] = {
                "url": url,
                "path": filepath,
                "status": "FAILED",
                "sha256": "",
                "tamanho_bytes": 0,
                "municipio": record.get("municipio", ""),
                "entidade": record.get("entidade", ""),
                "categoria": record.get("categoria", ""),
                "data_publicacao": record.get("data_publicacao", ""),
                "documento": record.get("documento", ""),
                "identificador_oficial": record.get("identificador_oficial", ""),
            }

        # Escreve manifesto a cada 50 downloads (checkpoint)
        if stats["novos"] % 50 == 0 and stats["novos"] > 0:
            _save_manifest(manifest, manifest_path)
            log.info(f"  [CHECKPOINT] Manifesto salvo com {len(manifest)} entradas.")

        # Throttle para não sobrecarregar o servidor
        time.sleep(0.5)

    # Salva manifesto final
    _save_manifest(manifest, manifest_path)

    return stats


def _save_manifest(manifest: dict, path: str) -> None:
    """Salva manifesto de forma atômica (escrita temporária + rename)."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download controlado de PDFs do DOM-PI a partir do JSON deduplicado.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input", type=str, required=True,
        help="Caminho para o JSON deduplicado (saída do scraper_isolado.py)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="db_treino_carnaubais/pdfs_arquivos",
        help="Diretório para armazenar os PDFs baixados."
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Caminho do manifesto JSON. Padrão: <output-dir>/download_manifest.json"
    )
    parser.add_argument(
        "--limite", type=int, default=999999,
        help="Nº máximo de downloads (padrão: ilimitado)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simula o download sem efetivamente baixar os arquivos."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Ativa logs de debug."
    )

    args = parser.parse_args()
    _configure_logging(verbose=args.verbose)

    if not os.path.exists(args.input):
        log.error(f"Arquivo de entrada não encontrado: {args.input}")
        sys.exit(1)

    manifest_path = args.manifest or os.path.join(args.output_dir, "download_manifest.json")

    log.info("=" * 60)
    log.info("PIPELINE DE DOWNLOAD — DOM-PI PDFs")
    log.info("=" * 60)
    log.info(f"Entrada: {args.input}")
    log.info(f"Destino: {args.output_dir}")
    log.info(f"Manifesto: {manifest_path}")
    log.info(f"Limite: {args.limite}")
    if args.dry_run:
        log.info("MODO DRY-RUN — nenhum download será executado.")
    log.info("-" * 60)

    stats = run_download_pipeline(
        input_json=args.input,
        output_dir=args.output_dir,
        manifest_path=manifest_path,
        limite=args.limite,
        dry_run=args.dry_run,
    )

    # Resumo final
    print("\n" + "─" * 60)
    print(f"  ✔  Download concluído")
    print(f"  📊 Total processado:    {stats['total']}")
    print(f"  🆕 Novos downloads:     {stats['novos']}")
    print(f"  ♻️  Pré-existentes:      {stats['preexistentes']}")
    print(f"  ❌ Falhas:              {stats['falhas']}")
    print(f"  📄 Manifesto:           {manifest_path}")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
