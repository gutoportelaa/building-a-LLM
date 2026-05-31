#!/usr/bin/env python3
"""
construir_banco.py — Geração de Embeddings DOM-PI em JSONL (Resiliente e Autocontido)
-------------------------------------------------------------------------------------
Este script lê os arquivos .md gerados pelo extrator Marker, realiza o chunking 
por parágrafos de forma resiliente, gera embeddings via Ollama REST API pura
e salva em um arquivo JSONL para evitar segfaults ou travamentos do ChromaDB.

Pode ser importado posteriormente no banco com `importar_jsonl_chroma.py`.

Uso:
    uv run python src/vector_db/construir_banco.py --input-dir dados_brutos_marker
"""

import argparse
import json
import logging
import sys
import gc
import yaml
import time
import re
import requests
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger("vector_db")

# ─── CONFIGURAÇÃO PADRÃO ───────────────────────────────────────────────────────
CHUNK_MAX_CHARS = 1200     # ~300 tokens BERT (seguro para janela 2048)
CHUNK_OVERLAP_CHARS = 120  # ~10% de sobreposição
SEND_MAX_CHARS = 1400      # Limite de segurança antes do POST
FLUSH_EVERY_N = 50         # Escreve no JSONL a cada N chunks para poupar memória
OLLAMA_BASE_URL = "http://localhost:11434"


def _configure_logging(log_dir: Path, verbose: bool = False) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_dir / "ingestao_vector_db.log", mode='a', encoding='utf-8')
    fh.setFormatter(fmt)

    log.setLevel(level)
    log.handlers.clear()
    log.addHandler(ch)
    log.addHandler(fh)


def ensure_model_available(model_name: str) -> None:
    """Verifica se o modelo está pronto no Ollama via REST API."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        if model_name in models:
            log.info(f"✓ Modelo '{model_name}' está disponível no Ollama.")
        else:
            log.info(f"Modelo '{model_name}' ausente. Fazendo pull...")
            r = requests.post(f"{OLLAMA_BASE_URL}/api/pull",
                              json={"name": model_name}, timeout=300, stream=True)
            r.raise_for_status()
            log.info(f"✓ Modelo '{model_name}' baixado com sucesso.")
    except requests.RequestException as e:
        log.error(f"Erro ao conectar ao Ollama em {OLLAMA_BASE_URL}: {e}")
        log.error("Certifique-se de que o Ollama está rodando ('ollama serve').")
        sys.exit(1)


def get_embedding(model_name: str, text: str, chunk_id: str, max_retries: int = 3) -> Optional[list]:
    """Chama a API REST do Ollama diretamente."""
    if len(text) > SEND_MAX_CHARS:
        text = text[:SEND_MAX_CHARS]

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": model_name, "prompt": text},
                timeout=90
            )
            if resp.status_code == 200:
                emb = resp.json().get("embedding")
                if emb:
                    return emb
                log.warning(f"Embedding vazio recebido para {chunk_id}.")
                return None
            else:
                log.warning(f"Ollama retornou HTTP {resp.status_code} para {chunk_id}, "
                             f"tentativa {attempt+1}: {resp.text[:150]}")
        except requests.Timeout:
            log.warning(f"Timeout no Ollama ({chunk_id}), tentativa {attempt+1}/{max_retries}.")
        except requests.RequestException as e:
            log.warning(f"Erro de conexão ({chunk_id}), tentativa {attempt+1}: {e}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    log.error(f"Falha definitiva ao obter embedding para {chunk_id}.")
    return None


def parse_markdown(content: str) -> tuple[dict, str]:
    """Parse do frontmatter do Markdown."""
    metadata = {}
    body = content
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            try:
                metadata = yaml.safe_load(parts[1]) or {}
            except Exception as e:
                log.warning(f"Erro ao ler frontmatter YAML: {e}")
            body = parts[2].strip()
    return metadata, body


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS,
               overlap_chars: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Chunking resiliente em Python puro por parágrafos/frases/caracteres."""
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    if not text:
        return []

    units: list[str] = []
    for para in text.split('\n\n'):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
        else:
            for line in para.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if len(line) <= max_chars:
                    units.append(line)
                else:
                    # Linha gigante (ex: tabelas do marker com muitos pipes)
                    step = max_chars - overlap_chars
                    for i in range(0, len(line), step):
                        piece = line[i:i + max_chars].strip()
                        if piece:
                            units.append(piece)

    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        u_len = len(unit)
        if current_len + u_len + 2 > max_chars and current:
            chunks.append('\n\n'.join(current))
            last = current[-1]
            current = [last] if len(last) + u_len + 2 <= max_chars else []
            current_len = len(last) + 2 if current else 0
        current.append(unit)
        current_len += u_len + 2

    if current:
        chunks.append('\n\n'.join(current))

    return [c for c in chunks if c.strip()]


def load_checkpoint(checkpoint_file: Path) -> set:
    processed = set()
    if checkpoint_file.exists():
        for line in checkpoint_file.read_text(encoding='utf-8').splitlines():
            if line.strip():
                processed.add(line.strip())
    log.info(f"Checkpoint carregado. {len(processed)} documentos já processados anteriormente.")
    return processed


def save_checkpoint(checkpoint_file: Path, doc_ids: set) -> None:
    with open(checkpoint_file, 'a', encoding='utf-8') as f:
        for doc_id in doc_ids:
            f.write(f"{doc_id}\n")


def yield_documents(input_dir: Path) -> Iterator[Path]:
    return input_dir.rglob('*.md')


def build_embeddings_jsonl(input_dir: Path, output_jsonl: Path, log_dir: Path,
                           model_name: str) -> None:
    if not input_dir.exists():
        log.error(f"Diretório de entrada não encontrado: {input_dir}")
        sys.exit(1)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file = log_dir / "checkpoint_ids.txt"
    processed_ids = load_checkpoint(checkpoint_file)

    log.info(f"Destino dos embeddings: {output_jsonl}")
    log.info(f"Iniciando processamento em: {input_dir}")

    total_docs = 0
    total_chunks = 0
    skipped = 0

    buffer: list[str] = []
    pending_doc_ids: set = set()

    def flush_buffer() -> None:
        nonlocal total_chunks
        if not buffer:
            return
        with open(output_jsonl, 'a', encoding='utf-8') as f:
            for line in buffer:
                f.write(line + '\n')
        total_chunks += len(buffer)
        save_checkpoint(checkpoint_file, pending_doc_ids)
        processed_ids.update(pending_doc_ids)
        log.info(f"[{total_docs} docs lidos | {total_chunks} chunks salvos] Flush de {len(buffer)} linhas finalizado.")
        buffer.clear()
        pending_doc_ids.clear()
        gc.collect()

    for md_file in yield_documents(input_dir):
        total_docs += 1
        doc_id_base = md_file.stem

        if doc_id_base in processed_ids:
            skipped += 1
            continue

        try:
            content = md_file.read_text(encoding='utf-8')
            metadata, body = parse_markdown(content)
        except Exception as e:
            log.error(f"Erro ao ler arquivo {md_file.name}: {e}")
            continue

        doc_id = metadata.get("id_publicacao", doc_id_base)
        if doc_id in processed_ids:
            skipped += 1
            continue

        chunks = chunk_text(body)
        if not chunks:
            log.warning(f"Documento sem conteúdo válido após chunking: {doc_id}")
            continue

        # Normaliza metadados para garantir que são tipos escalares (compatível com ChromaDB posterior)
        base_meta: dict = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                base_meta[k] = v
            elif v is not None:
                base_meta[k] = str(v)

        doc_ok = True
        chunk_records: list[dict] = []

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{idx}" if len(chunks) > 1 else doc_id
            embedding = get_embedding(model_name, chunk, chunk_id)

            if embedding is None:
                log.warning(f"Falha de embedding no chunk {chunk_id}. Pulando documento inteiro.")
                doc_ok = False
                break

            chunk_meta = base_meta.copy()
            chunk_meta["chunk_index"] = idx
            chunk_meta["total_chunks"] = len(chunks)
            chunk_records.append({
                "id": chunk_id,
                "document": chunk,
                "embedding": embedding,
                "metadata": chunk_meta
            })

        if not doc_ok:
            continue

        for rec in chunk_records:
            buffer.append(json.dumps(rec, ensure_ascii=False))

        pending_doc_ids.add(doc_id)

        if len(buffer) >= FLUSH_EVERY_N:
            flush_buffer()

    flush_buffer()

    log.info(f"Ingestão finalizada! {total_docs} visitados | {skipped} pulados | {total_chunks} chunks exportados.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gerador de Embeddings DOM-PI em JSONL (Resiliente, Autocontido e Rápido)"
    )
    parser.add_argument("--input-dir", type=str, default="dados_limpos",
                        help="Diretório de markdowns limpos")
    parser.add_argument("--output-jsonl", type=str, default="dados/embeddings_marker.jsonl",
                        help="Destino do arquivo JSONL contendo os embeddings")
    parser.add_argument("--log-dir", type=str, default="dados/logs",
                        help="Diretório para salvar os logs e arquivos de checkpoint")
    parser.add_argument("--model", type=str, default="nomic-embed-text",
                        help="Modelo do Ollama para embedding")
    parser.add_argument("--verbose", action="store_true", default=True, help="Habilita logs detalhados")

    args = parser.parse_args()
    log_p = Path(args.log_dir)

    _configure_logging(log_p, args.verbose)

    log.info("=" * 60)
    log.info("INICIANDO MOTOR VETORIAL RESILIENTE (JSONL)")
    log.info("=" * 60)

    ensure_model_available(args.model)
    build_embeddings_jsonl(
        Path(args.input_dir),
        Path(args.output_jsonl),
        log_p,
        args.model
    )

    log.info("Processamento finalizado com total segurança.")


if __name__ == "__main__":
    main()
