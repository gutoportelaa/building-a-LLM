#!/usr/bin/env python3
"""
extrator_docling.py — Extração de PDFs DOM-PI via Docling (GPU/CPU)
------------------------------------------------------------------
Wrapper fino sobre o Docling para avaliar como backend de extração do pipeline
DOM-PI, em alternativa ao Marker e ao Tesseract. Expõe uma sessão reutilizável
(o conversor carrega modelos de layout/OCR uma única vez) e uma função de
extração por PDF.

Características relevantes para o benchmark:
- Seleção explícita de dispositivo (CUDA vs CPU) via AcceleratorOptions.
- OCR opcional: para PDFs nativos digitais, `do_ocr=False` é ordens de grandeza
  mais rápido; para escaneados, `do_ocr=True` aciona o motor OCR (RapidOCR/EasyOCR).
- `page_batch` limita quantas páginas o Docling mantém em memória por vez —
  alavanca central para evitar estouro de memória em PDFs de 100+ páginas.

Uso programático:
    from dompi_scraper.extrator_docling import create_docling_session, extract_with_docling
    sess = create_docling_session(device="cuda", do_ocr=False)
    md = extract_with_docling(sess, "arquivo.pdf")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

log = logging.getLogger("extrator_docling")


def create_docling_session(
    device: str = "auto",
    do_ocr: bool = False,
    do_table_structure: bool = True,
    num_threads: int = 8,
    page_batch: int | None = None,
):
    """
    Cria um DocumentConverter Docling com dispositivo e OCR configuráveis.

    Args:
        device: "cuda", "cpu" ou "auto".
        do_ocr: aciona o motor OCR (necessário para PDFs escaneados).
        do_table_structure: reconstrói tabelas (custa memória/tempo).
        num_threads: threads para o caminho CPU.
        page_batch: nº de páginas processadas por lote (controle de memória).

    Returns:
        DocumentConverter pronto para `.convert()`.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        AcceleratorOptions,
        AcceleratorDevice,
    )

    dev_map = {
        "cuda": AcceleratorDevice.CUDA,
        "gpu": AcceleratorDevice.CUDA,
        "cpu": AcceleratorDevice.CPU,
        "auto": AcceleratorDevice.AUTO,
    }
    accel_device = dev_map.get(device.lower(), AcceleratorDevice.AUTO)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr
    pipeline_options.do_table_structure = do_table_structure
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=num_threads,
        device=accel_device,
    )
    # Limita quantas páginas o Docling rasteriza/infere por lote. Atenção:
    # este é um ajuste GLOBAL (docling.datamodel.settings), não um campo de
    # PdfPipelineOptions. Reduz o pico de memória das IMAGENS de página, mas
    # NÃO limita a acumulação do documento montado — para PDFs de 100+ páginas
    # o controle decisivo de memória é fatiar o PDF (ver worker_docling /
    # orquestrador: extração por subconjunto de páginas).
    if page_batch is not None:
        from docling.datamodel.settings import settings
        settings.perf.page_batch_size = page_batch

    log.info(
        "Sessão Docling: device=%s do_ocr=%s tables=%s threads=%d batch=%s",
        accel_device, do_ocr, do_table_structure, num_threads, page_batch,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return converter


def extract_with_docling(converter, pdf_path: str) -> str | None:
    """
    Extrai Markdown de um único PDF usando a sessão Docling fornecida.

    Returns:
        Markdown (str) ou None em caso de falha.
    """
    try:
        t0 = time.time()
        result = converter.convert(str(pdf_path))
        md = result.document.export_to_markdown()
        log.debug("Docling extraiu %s em %.1fs (%d chars)",
                  Path(pdf_path).name, time.time() - t0, len(md or ""))
        return md
    except Exception as exc:  # noqa: BLE001
        log.error("Docling falhou em %s: %s", pdf_path, exc)
        return None
