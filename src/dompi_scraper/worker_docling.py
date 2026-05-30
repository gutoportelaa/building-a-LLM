#!/usr/bin/env python3
"""
worker_docling.py — Worker de Repaginação Estruturada (Docling GPU / Ollama CPU)
----------------------------------------------------------------------------------
Lê o arquivo reprocessamento_pendente.jsonl gerado pelo pipeline PaddleOCR e
reprocessa cada documento marcado com extração estruturada de alta fidelidade,
preservando tabelas no formato Markdown (| col | col |) e descrições de figuras.

Fluxo:
  reprocessamento_pendente.jsonl
      └─ Para cada doc com has_tables/has_figures:
          ├─ GPU disponível → Docling (DocumentConverter com CUDA)
          └─ Sem GPU        → llama3.2-vision via Ollama (já integrado no projeto)

  Resultado: substitui o campo texto_markdown no corpus_<slug>.jsonl original
  e atualiza engine_extracao para "Docling-GPU" ou "Ollama-VLM".

Uso:
    # Modo padrão (detecta GPU automaticamente)
    uv run python src/dompi_scraper/worker_docling.py \\
        --pendente extraidos/tabuleiros_alto_parnaiba/reprocessamento_pendente.jsonl \\
        --corpus  extraidos/tabuleiros_alto_parnaiba/corpus_tabuleiros_alto_parnaiba.jsonl \\
        --limite 10 --verbose

    # Forçar CPU (Ollama)
    uv run python src/dompi_scraper/worker_docling.py \\
        --pendente extraidos/tabuleiros_alto_parnaiba/reprocessamento_pendente.jsonl \\
        --corpus  extraidos/tabuleiros_alto_parnaiba/corpus_tabuleiros_alto_parnaiba.jsonl \\
        --force-cpu
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

try:
    import fitz  # PyMuPDF — renderização de páginas para imagem
except ImportError:
    print("Erro: 'pymupdf' necessário. Execute: uv add pymupdf", file=sys.stderr)
    sys.exit(1)

log = logging.getLogger("worker_docling")


# ---------------------------------------------------------------------------
# DETECÇÃO DE GPU
# ---------------------------------------------------------------------------

def detectar_gpu() -> bool:
    """Retorna True se CUDA disponível via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# BACKEND DOCLING (GPU)
# ---------------------------------------------------------------------------

def carregar_docling():
    """
    Inicializa Docling DocumentConverter.
    Requer: uv add docling  (ou uv sync --extra docling)
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        log.info("Docling DocumentConverter inicializado (GPU)")
        return converter
    except ImportError:
        log.error(
            "Docling não instalado. Execute: uv add --optional docling\n"
            "  ou: uv add docling"
        )
        return None


def extrair_com_docling(converter, pdf_path: str, pages: list[int] | None = None) -> str:
    """
    Extrai texto estruturado com Docling, retornando Markdown com tabelas.

    Args:
        converter: instância de DocumentConverter (já inicializada)
        pdf_path: caminho para o PDF
        pages: lista de páginas 0-based para extrair (None = todas)

    Returns:
        Texto em Markdown com tabelas | col | col | ou string vazia em caso de erro.
    """
    try:
        if pages:
            # Cria mini-PDF apenas com as páginas relevantes (tabela/figura)
            doc = fitz.open(pdf_path)
            # fitz.select usa índices 0-based
            doc.select(pages)
            fd, tmp_pdf = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            doc.save(tmp_pdf)
            doc.close()
            source = tmp_pdf
        else:
            source = pdf_path
            tmp_pdf = None

        result = converter.convert(source)
        markdown = result.document.export_to_markdown()

        if tmp_pdf:
            try:
                os.remove(tmp_pdf)
            except OSError:
                pass

        return markdown or ""

    except Exception as e:
        log.warning(f"Docling falhou em {os.path.basename(pdf_path)}: {e}")
        return ""


# ---------------------------------------------------------------------------
# BACKEND OLLAMA VLM (CPU FALLBACK)
# ---------------------------------------------------------------------------

_OLLAMA_URL = "http://localhost:11434/api/generate"
_OLLAMA_PROMPT = (
    "Transcreva exatamente o conteúdo desta imagem para o formato Markdown. "
    "Se houver tabelas, utilize a estrutura Markdown (| Coluna 1 | Coluna 2 |). "
    "Preserve números e valores monetários exatamente como aparecem. "
    "Ignore cabeçalhos e rodapés repetitivos de paginação. "
    "Responda APENAS com o Markdown transcrito, sem explicações adicionais."
)


def _verificar_ollama(model: str = "llama3.2-vision") -> bool:
    """Verifica se o Ollama está acessível e o modelo disponível."""
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            available = any(model in m for m in models)
            if not available:
                log.warning(
                    f"Modelo '{model}' não encontrado no Ollama. "
                    f"Modelos disponíveis: {models[:5]}"
                )
            return available
    except Exception as e:
        log.warning(f"Ollama não acessível em localhost:11434 — {e}")
        return False


def extrair_pagina_ollama(
    page: fitz.Page,
    model: str = "llama3.2-vision",
    dpi: int = 150,
) -> str:
    """
    Envia uma página como imagem para o modelo VLM via Ollama.

    Args:
        page: página fitz já carregada
        model: nome do modelo Ollama (deve suportar imagens)
        dpi: resolução de renderização (150 DPI — balanço qualidade/latência CPU)

    Returns:
        Texto Markdown transcrito pelo VLM ou string vazia em caso de erro.
    """
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img_bytes = pix.tobytes("png")
    b64_img = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "model": model,
        "prompt": _OLLAMA_PROMPT,
        "images": [b64_img],
        "stream": False,
    }

    try:
        req = urllib.request.Request(
            _OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result.get("response", "").strip()
    except Exception as e:
        log.warning(f"Ollama falhou (modelo={model}): {e}")
        return ""


def extrair_com_ollama(
    pdf_path: str,
    pages: list[int] | None = None,
    model: str = "llama3.2-vision",
) -> str:
    """
    Extrai texto estruturado via VLM Ollama, página por página.

    Args:
        pdf_path: caminho para o PDF
        pages: lista de páginas 0-based (None = todas)
        model: modelo Ollama com suporte a imagens

    Returns:
        Texto Markdown concatenado de todas as páginas processadas.
    """
    try:
        doc = fitz.open(pdf_path)
        paginas_alvo = pages if pages is not None else list(range(len(doc)))
        partes: list[str] = []

        for page_num in paginas_alvo:
            if page_num >= len(doc):
                continue
            page = doc.load_page(page_num)
            texto_pagina = extrair_pagina_ollama(page, model=model)
            if texto_pagina:
                partes.append(texto_pagina)

        doc.close()
        return "\n\n---\n\n".join(partes)

    except Exception as e:
        log.error(f"Erro ao processar {pdf_path} com Ollama: {e}")
        return ""


# ---------------------------------------------------------------------------
# ATUALIZAÇÃO DO CORPUS
# ---------------------------------------------------------------------------

def _carregar_corpus(corpus_path: str) -> dict[str, dict]:
    """Carrega o corpus JSONL indexado por doc_id."""
    corpus: dict[str, dict] = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                doc_id = rec.get("doc_id", "")
                if doc_id:
                    corpus[doc_id] = rec
            except json.JSONDecodeError:
                pass
    return corpus


def _salvar_corpus(corpus_path: str, corpus: dict[str, dict]) -> None:
    """Salva o corpus JSONL atomicamente."""
    tmp = corpus_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in corpus.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, corpus_path)


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def run_worker(
    pendente_path: str,
    corpus_path: str,
    limite: int,
    force_cpu: bool = False,
    ollama_model: str = "llama3.2-vision",
    verbose: bool = False,
) -> dict:
    """
    Processa entradas do reprocessamento_pendente.jsonl com Docling ou Ollama.

    Args:
        pendente_path: caminho para reprocessamento_pendente.jsonl
        corpus_path: caminho para corpus_<slug>.jsonl (será atualizado in-place)
        limite: máximo de documentos a processar nesta execução
        force_cpu: forçar uso de Ollama mesmo com GPU disponível
        ollama_model: nome do modelo Ollama para fallback CPU
        verbose: ativar logs DEBUG

    Returns:
        Dicionário de estatísticas da execução.
    """
    if not Path(pendente_path).exists():
        log.error(f"Arquivo não encontrado: {pendente_path}")
        return {}

    # Detecta motor disponível
    gpu = detectar_gpu() and not force_cpu
    engine_nome = "Docling-GPU" if gpu else f"Ollama-VLM ({ollama_model})"
    log.info(f"Motor selecionado: {engine_nome}")

    # Inicializa Docling se GPU disponível
    docling_converter = None
    if gpu:
        docling_converter = carregar_docling()
        if docling_converter is None:
            log.warning("Docling indisponível — fallback para Ollama.")
            gpu = False
            engine_nome = f"Ollama-VLM ({ollama_model})"

    # Verifica Ollama se for CPU
    if not gpu:
        if not _verificar_ollama(ollama_model):
            log.error(
                f"Ollama não acessível e Docling indisponível. "
                f"Certifique-se que o Ollama está rodando: ollama serve\n"
                f"E que o modelo está baixado: ollama pull {ollama_model}"
            )
            return {}

    # Carrega corpus existente para atualização in-place
    if not Path(corpus_path).exists():
        log.error(f"Corpus não encontrado: {corpus_path}")
        return {}

    log.info(f"Carregando corpus: {corpus_path}")
    corpus = _carregar_corpus(corpus_path)
    log.info(f"  {len(corpus)} documentos no corpus")

    # Carrega entradas pendentes
    pendentes: list[dict] = []
    with open(pendente_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pendentes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    log.info(f"Entradas pendentes: {len(pendentes)} | processando até {limite}")
    pendentes = pendentes[:limite]

    stats = {"total": 0, "atualizados": 0, "sem_melhoria": 0, "erros": 0, "nao_encontrados": 0}

    for entry in pendentes:
        stats["total"] += 1
        doc_id = entry.get("doc_id", "")
        pdf_path = entry.get("path", "")
        table_pages = entry.get("table_pages") or None
        figure_pages = entry.get("figure_pages") or None

        if not pdf_path or not os.path.exists(pdf_path):
            log.warning(f"PDF não encontrado: {pdf_path}")
            stats["nao_encontrados"] += 1
            continue

        # Páginas relevantes = união de tabelas + figuras (0-based)
        pages_alvo: list[int] | None = None
        if table_pages or figure_pages:
            combined = set(table_pages or []) | set(figure_pages or [])
            pages_alvo = sorted(combined)

        log.info(f"  [{stats['total']}/{len(pendentes)}] {os.path.basename(pdf_path)} | páginas={pages_alvo}")
        t0 = time.time()

        if gpu and docling_converter:
            novo_texto = extrair_com_docling(docling_converter, pdf_path, pages=pages_alvo)
        else:
            novo_texto = extrair_com_ollama(pdf_path, pages=pages_alvo, model=ollama_model)

        elapsed = time.time() - t0

        if not novo_texto or len(novo_texto.strip()) < 50:
            log.warning(f"    Extração falhou ou retornou texto vazio ({elapsed:.1f}s)")
            stats["erros"] += 1
            continue

        # Atualiza o registro no corpus
        if doc_id in corpus:
            corpus[doc_id]["texto_markdown"] = novo_texto
            corpus[doc_id]["engine_extracao"] = engine_nome
            corpus[doc_id].setdefault("metadados_extraidos", {})["reprocessado"] = True
            stats["atualizados"] += 1
            log.info(f"    Atualizado: {len(novo_texto)} chars em {elapsed:.1f}s")
        else:
            log.warning(f"    doc_id={doc_id[:12]}... não encontrado no corpus")
            stats["nao_encontrados"] += 1

    # Salva corpus atualizado
    if stats["atualizados"] > 0:
        log.info(f"Salvando corpus com {stats['atualizados']} atualizações...")
        _salvar_corpus(corpus_path, corpus)
        log.info("Corpus salvo.")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    log.setLevel(level)
    log.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(level)
    log.addHandler(ch)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Worker de repaginação Docling/Ollama para documentos com tabelas e gráficos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pendente", required=True, help="Caminho do reprocessamento_pendente.jsonl")
    parser.add_argument("--corpus", required=True, help="Caminho do corpus_<slug>.jsonl (atualizado in-place)")
    parser.add_argument("--limite", type=int, default=999_999, help="Máx. documentos a reprocessar (padrão: ilimitado)")
    parser.add_argument("--force-cpu", action="store_true", help="Forçar Ollama mesmo com GPU disponível")
    parser.add_argument("--ollama-model", default="llama3.2-vision", help="Modelo Ollama para fallback CPU (padrão: llama3.2-vision)")
    parser.add_argument("--verbose", action="store_true", help="Logs detalhados")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    log.info("=" * 65)
    log.info("WORKER DE REPAGINAÇÃO — Docling GPU / Ollama CPU")
    log.info("=" * 65)
    log.info(f"  Pendente:  {args.pendente}")
    log.info(f"  Corpus:    {args.corpus}")
    log.info(f"  Limite:    {args.limite}")
    log.info(f"  GPU:       {detectar_gpu()} | force_cpu={args.force_cpu}")
    log.info("-" * 65)

    t0 = time.time()
    stats = run_worker(
        pendente_path=args.pendente,
        corpus_path=args.corpus,
        limite=args.limite,
        force_cpu=args.force_cpu,
        ollama_model=args.ollama_model,
        verbose=args.verbose,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 65)
    print("REPAGINAÇÃO CONCLUÍDA")
    print("=" * 65)
    print(f"  Tempo total:      {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Processados:      {stats.get('total', 0)}")
    print(f"  Atualizados:      {stats.get('atualizados', 0)}")
    print(f"  Não encontrados:  {stats.get('nao_encontrados', 0)}")
    print(f"  Erros:            {stats.get('erros', 0)}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
