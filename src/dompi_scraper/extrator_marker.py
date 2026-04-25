#!/usr/bin/env python3
"""
extrator_marker.py — Extrator de PDFs via Marker com Qualidade OCR
-------------------------------------------------------------------
Lê PDFs da pasta db_treino_carnaubais/pdfs_arquivos, extrai texto estruturado
via Marker + DocTR (com quality gate OpenCV), e consolida em JSONL para
fine-tuning de LLMs.

Pipeline interno:
1. Descoberta de PDFs na pasta de origem
2. Quality gate com Laplaciano (OpenCV) para detectar desfoque
3. Extração de texto via Marker (PDF → Markdown)
4. Mapeamento geométrico via DocTR (detecta tabelas, assinaturas)
5. Deduplicação textual por hash MD5
6. Geração de JSONL limpo para fine-tuning

Uso:
    # Processar 50 PDFs de amostra
    python src/dompi_scraper/extrator_marker.py --tamanho-amostra 50

    # Processar TODOS os PDFs
    python src/dompi_scraper/extrator_marker.py --tamanho-amostra None

    # Com verbose (mostra processamento detalhado)
    python src/dompi_scraper/extrator_marker.py --tamanho-amostra 10 --verbose
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
    import torch
except ImportError:
    print("Erro: Bibliotecas necessárias não instaladas.")
    print("Instale com: pip install opencv-python numpy torch")
    sys.exit(1)

try:
    from pdf2image import convert_from_path
except ImportError:
    print("Erro: 'pdf2image' é necessário. Instale com: pip install pdf2image")
    sys.exit(1)

try:
    from .shared_utils import compute_content_md5, normalize_text_for_dedup
except ImportError:
    from shared_utils import compute_content_md5, normalize_text_for_dedup


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

# Variável de controle: número de PDFs a processar
# Se None, processa todos os PDFs disponíveis
TAMANHO_AMOSTRA = 5

# Caminho da pasta contendo os PDFs
PASTA_PDFS = Path("db_treino_carnaubais/pdfs_arquivos")

# Diretório de saída
DIR_SAIDA = "output_pipeline"

# Arquivo JSONL de saída
JSONL_OUTPUT = "corpus_marker.jsonl"

# Threshold mínimo de qualidade OCR (Laplaciano)
LIMIAR_DESFOQUE = 100.0

# Logging
log = logging.getLogger("extrator_marker")


# ---------------------------------------------------------------------------
# Configuração de Logging
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    log.setLevel(level)
    log.handlers.clear()
    log.addHandler(handler)


# ---------------------------------------------------------------------------
# Detecção de Poppler (para Windows)
# ---------------------------------------------------------------------------

def _find_poppler_path() -> str | None:
    """
    Detecta o caminho do Poppler instalado no Conda Base (Windows).
    Retorna o caminho ou None.
    """
    import platform

    if platform.system() != "Windows":
        return "/usr/bin"

    conda_base_paths = [
        Path(os.path.expandvars(r"%USERPROFILE%\miniconda3")),
        Path(os.path.expandvars(r"%USERPROFILE%\anaconda3")),
        Path(os.path.expandvars(r"%USERPROFILE%\Miniconda3")),
        Path(os.path.expandvars(r"%USERPROFILE%\Anaconda3")),
        Path("C:\\miniconda3"),
        Path("C:\\anaconda3"),
    ]

    for conda_path in conda_base_paths:
        lib_bin = conda_path / "Library" / "bin"
        if lib_bin.exists():
            if (lib_bin / "pdftoimage.exe").exists() or (lib_bin / "pdftoppm.exe").exists():
                log.debug(f"Poppler encontrado em: {lib_bin}")
                return str(lib_bin)

    log.warning("Poppler não encontrado. Usando valor padrão (None)")
    return None


# ---------------------------------------------------------------------------
# Pipeline Principal
# ---------------------------------------------------------------------------

class DocumentAIPipeline:
    """
    Pipeline de extração de PDFs com quality gate e deduplicação.
    """

    def __init__(
        self,
        output_dir: str = "output_pipeline",
        limiar_desfoque: float = 100.0,
        poppler_path: str | None = None,
    ):
        self.output_dir = output_dir
        self.limiar_desfoque = limiar_desfoque
        self.poppler_path = poppler_path
        os.makedirs(self.output_dir, exist_ok=True)
        log.info(f"🚀 Pipeline inicializado com limiar de desfoque: {limiar_desfoque}")

    def _limpar_vram(self, *modelos: Any) -> None:
        """Limpa memória VRAM para evitar Out Of Memory."""
        for modelo in modelos:
            if modelo is not None:
                del modelo
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def preparar_imagens(self, pdf_path: str) -> list[str]:
        """Converte PDF para PNG e salva no disco."""
        log.debug(f"📄 Convertendo {pdf_path} para imagens PNG (DPI=300)...")
        try:
            if self.poppler_path:
                imagens = convert_from_path(pdf_path, dpi=300, poppler_path=self.poppler_path)
            else:
                imagens = convert_from_path(pdf_path, dpi=300)
        except Exception as e:
            log.error(f"❌ Erro ao converter PDF: {e}")
            raise

        caminhos = []
        for i, img in enumerate(imagens):
            caminho = os.path.join(self.output_dir, f"page_{i}.png")
            img.save(caminho, "PNG")
            caminhos.append(caminho)

        return caminhos

    def avaliar_qualidade_opencv(self, caminhos_imagens: list[str]) -> tuple[list[str], list[str]]:
        """
        Quality gate: Calcula a Variância do Laplaciano para detectar desfoque.
        Retorna (imagens_validas, imagens_rejeitadas).
        """
        log.debug("🔍 Iniciando triagem de qualidade (CPU)...")
        paginas_validas = []
        paginas_rejeitadas = []

        for caminho in caminhos_imagens:
            try:
                imagem = cv2.imread(caminho)
                if imagem is None:
                    log.warning(f"  ⚠️  {os.path.basename(caminho)} não pôde ser lida")
                    paginas_rejeitadas.append(caminho)
                    continue

                cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
                nitidez = cv2.Laplacian(cinza, cv2.CV_64F).var()

                nome_arquivo = os.path.basename(caminho)
                if nitidez > self.limiar_desfoque:
                    log.debug(f"  ✅ {nome_arquivo} Aprovada (Nitidez: {nitidez:.2f})")
                    paginas_validas.append(caminho)
                else:
                    log.debug(f"  ❌ {nome_arquivo} Rejeitada por desfoque (Nitidez: {nitidez:.2f})")
                    paginas_rejeitadas.append(caminho)
            except Exception as e:
                log.error(f"  Erro ao processar {caminho}: {e}")
                paginas_rejeitadas.append(caminho)

        return paginas_validas, paginas_rejeitadas

    def extrair_texto_marker(self, pdf_path: str) -> str:
        """Extração de texto via Marker com GPU."""
        log.debug("🛠️  Iniciando Marker (Extração de Markdown)...")

        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
        except ImportError:
            log.error("❌ Marker não instalado. Instale com: pip install marker-pdf")
            raise

        try:
            # Marker gerencia GPU automaticamente via device_type (CUDA por padrão se disponível)
            if torch.cuda.is_available():
                log.debug("  📊 GPU detectada - Marker usará CUDA automaticamente")
            else:
                log.warning("  ⚠️  GPU não disponível - Marker usará CPU (lento!)")
            
            modelos_marker = create_model_dict()
            converter = PdfConverter(artifact_dict=modelos_marker)
            rendered = converter(pdf_path)
            full_text, _, _ = text_from_rendered(rendered)
            self._limpar_vram(modelos_marker, converter, rendered)
            return full_text
        except Exception as e:
            log.error(f"❌ Erro ao extrair texto com Marker: {e}")
            raise

    def mapear_geometria_doctr(self, caminhos_imagens: list[str]) -> dict:
        """Mapeamento geométrico via DocTR."""
        log.debug("📐 Iniciando DocTR (Mapeamento Geométrico)...")

        if not caminhos_imagens:
            log.warning("  ⚠️  Sem imagens para mapear com DocTR")
            return {}

        try:
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor
        except ImportError:
            log.error("❌ DocTR não instalado. Instale com: pip install python-doctr[torch]")
            return {}

        try:
            modelo_doctr = ocr_predictor(pretrained=True).cuda()
            documento = DocumentFile.from_images(caminhos_imagens)
            resultado = modelo_doctr(documento)
            dicionario_espacial = resultado.export()
            self._limpar_vram(modelo_doctr)
            return dicionario_espacial
        except Exception as e:
            log.error(f"❌ Erro ao mapear geometria: {e}")
            return {}

    def processar_documento(self, pdf_path: str) -> dict:
        """
        Processa um único PDF completo.

        Retorna:
        {
            "status": "processado" | "rejeitado_por_qualidade" | "erro",
            "texto": texto_extraido (se processado),
            "paginas_rejeitadas": num (se processado),
            "error": mensagem de erro (se erro),
        }
        """
        log.info(f"--- PROCESSANDO: {os.path.basename(pdf_path)} ---")

        try:
            # 1. Preparação
            todas_imagens = self.preparar_imagens(pdf_path)
            log.debug(f"  {len(todas_imagens)} páginas convertidas para PNG")

            # 2. Quality Gate
            imagens_validas, imagens_rejeitadas = self.avaliar_qualidade_opencv(todas_imagens)
            log.debug(
                f"  Quality Gate: {len(imagens_validas)} válidas, "
                f"{len(imagens_rejeitadas)} rejeitadas"
            )

            # Regra crítica: se nenhuma página for válida, aborta
            if not imagens_validas:
                log.warning("  🛑 Documento rejeitado: nenhuma página legível")
                return {
                    "status": "rejeitado_por_qualidade",
                    "erro": "Nenhuma página legível detectada",
                }

            # 3. Extração de texto
            try:
                texto_md = self.extrair_texto_marker(pdf_path)
            except Exception as e:
                return {
                    "status": "erro",
                    "error": f"Erro ao extrair com Marker: {str(e)}",
                }

            # 4. Mapeamento geométrico
            geometria = self.mapear_geometria_doctr(imagens_validas)

            log.info(
                f"  ✅ Processamento concluído ({len(imagens_validas)} páginas, "
                f"{len(texto_md)} caracteres)"
            )

            return {
                "status": "processado",
                "texto": texto_md,
                "paginas_rejeitadas": len(imagens_rejeitadas),
                "mapa_geometrico": geometria,
            }

        except Exception as e:
            log.error(f"  ❌ Erro durante processamento: {type(e).__name__}: {e}")
            return {
                "status": "erro",
                "error": str(e),
            }


# ---------------------------------------------------------------------------
# Descoberta de PDFs
# ---------------------------------------------------------------------------

def descobrir_pdfs(pasta: Path, limite: int | None = None) -> list[Path]:
    """
    Descobre todos os PDFs em uma pasta.

    Args:
        pasta: Caminho da pasta contendo os PDFs
        limite: Número máximo de PDFs a retornar. Se None, retorna todos.

    Returns:
        Lista de caminhos de PDFs
    """
    if not pasta.exists():
        log.error(f"❌ Pasta não encontrada: {pasta}")
        return []

    pdfs = sorted(pasta.glob("*.pdf"))
    log.info(f"📁 {len(pdfs)} PDFs encontrados em {pasta}")

    if limite is not None:
        pdfs = pdfs[:limite]
        log.info(f"   Limitado a {limite} PDFs")

    return pdfs


# ---------------------------------------------------------------------------
# Pipeline Principal
# ---------------------------------------------------------------------------

def run_extraction_pipeline(
    pasta_pdfs: Path,
    jsonl_output: str,
    tamanho_amostra: int | None = None,
    limiar_desfoque: float = 100.0,
    verbose: bool = False,
) -> dict:
    """
    Executa o pipeline completo de extração.

    Returns:
        Estatísticas de processamento
    """
    _configure_logging(verbose=verbose)

    # Inicializa pipeline
    poppler_path = _find_poppler_path()
    pipeline = DocumentAIPipeline(
        output_dir=DIR_SAIDA,
        limiar_desfoque=limiar_desfoque,
        poppler_path=poppler_path,
    )

    # Descobre PDFs
    pdfs = descobrir_pdfs(pasta_pdfs, limite=tamanho_amostra)
    if not pdfs:
        log.error("❌ Nenhum PDF encontrado")
        return {"total": 0, "processados": 0, "rejeitados": 0, "erros": 0}

    # Registro de deduplicação
    dedup_registry: set[str] = set()

    # Processamento
    stats = {"total": len(pdfs), "processados": 0, "rejeitados": 0, "erros": 0}
    jsonl_records: list[dict] = []

    for idx, pdf_path in enumerate(pdfs, 1):
        log.info(f"\n[{idx}/{len(pdfs)}] Processando {pdf_path.name}...")

        try:
            resultado = pipeline.processar_documento(str(pdf_path))

            if resultado["status"] == "processado":
                texto = resultado.get("texto", "")

                # Deduplicação
                content_hash = compute_content_md5(texto)
                if content_hash in dedup_registry:
                    log.info(f"  ♻️  DUPLICATA detectada (hash={content_hash[:12]}...)")
                    stats["rejeitados"] += 1
                    continue

                dedup_registry.add(content_hash)

                # Cria registro JSONL
                record = {"arquivo": pdf_path.name, "text": texto}
                jsonl_records.append(record)
                stats["processados"] += 1

            elif resultado["status"] == "rejeitado_por_qualidade":
                log.warning(f"  ⚠️  Rejeitado: {resultado.get('erro', 'qualidade insuficiente')}")
                stats["rejeitados"] += 1

            else:
                log.error(f"  ❌ Erro: {resultado.get('error', 'desconhecido')}")
                stats["erros"] += 1

        except Exception as e:
            log.error(f"  ❌ Erro não tratado: {e}")
            stats["erros"] += 1

    # Salva JSONL
    if jsonl_records:
        jsonl_path = Path(jsonl_output)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for record in jsonl_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        log.info(f"\n📄 JSONL salvo em: {jsonl_path} ({len(jsonl_records)} linhas)")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrator de PDFs via Marker → JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--pasta-pdfs",
        type=str,
        default=str(PASTA_PDFS),
        help=f"Caminho da pasta contendo os PDFs (padrão: {PASTA_PDFS})",
    )
    parser.add_argument(
        "--tamanho-amostra",
        type=int,
        default=TAMANHO_AMOSTRA,
        help="Número máximo de PDFs a processar. None = processar todos (padrão: None)",
    )
    parser.add_argument(
        "--jsonl-output",
        type=str,
        default=JSONL_OUTPUT,
        help=f"Caminho do arquivo JSONL de saída (padrão: {JSONL_OUTPUT})",
    )
    parser.add_argument(
        "--limiar-desfoque",
        type=float,
        default=LIMIAR_DESFOQUE,
        help=f"Threshold de qualidade OCR (Laplaciano) (padrão: {LIMIAR_DESFOQUE})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Ativa logs de debug",
    )

    args = parser.parse_args()

    # Converte None string em None
    tamanho_amostra: int | None = args.tamanho_amostra
    if isinstance(tamanho_amostra, str) and tamanho_amostra.lower() == "none":
        tamanho_amostra = None
    elif isinstance(tamanho_amostra, str):
        tamanho_amostra = int(tamanho_amostra)

    log_msg = f"""
╔═══════════════════════════════════════════════════════════╗
║          EXTRATOR DE PDFS VIA MARKER → JSONL             ║
╚═══════════════════════════════════════════════════════════╝
Pasta de PDFs:     {args.pasta_pdfs}
Tamanho da amostra: {tamanho_amostra if tamanho_amostra else 'TODOS'}
JSONL de saída:    {args.jsonl_output}
Limiar de desfoque: {args.limiar_desfoque}
"""

    print(log_msg)

    pasta_pdfs = Path(args.pasta_pdfs)
    if not pasta_pdfs.exists():
        print(f"❌ Pasta não encontrada: {pasta_pdfs}")
        sys.exit(1)

    stats = run_extraction_pipeline(
        pasta_pdfs=pasta_pdfs,
        jsonl_output=args.jsonl_output,
        tamanho_amostra=tamanho_amostra,
        limiar_desfoque=args.limiar_desfoque,
        verbose=args.verbose,
    )

    # Resumo final
    print("\n" + "─" * 60)
    print(f"  ✔  Extração concluída")
    print(f"  📊 Total processado:     {stats['total']}")
    print(f"  ✅ Sucessos:             {stats['processados']}")
    print(f"  ⚠️  Rejeitados:           {stats['rejeitados']}")
    print(f"  ❌ Erros:                {stats['erros']}")
    print(f"  📄 JSONL:                {args.jsonl_output}")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
