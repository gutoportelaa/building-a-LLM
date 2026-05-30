#!/usr/bin/env python3
"""
extrator_paddle.py — Motor PaddleOCR PP-Structure para o pipeline DOM-PI
-------------------------------------------------------------------------
Responsabilidades:
1. Análise de layout por página (text / table / figure) via PP-Structure
2. Extração de texto das regiões classificadas como 'text'
3. Detecção de valores monetários BR com regex tolerante a erros de OCR
4. Pós-processamento de acentos PT-BR corrompidos pelo modelo multilingual
5. Renderização de página PDF → numpy array (via fitz) para o PP-Structure

Uso como módulo:
    from dompi_scraper.extrator_paddle import criar_engine_paddle, extrair_pdf_paddle

Uso direto (diagnóstico):
    uv run python src/dompi_scraper/extrator_paddle.py --pdf caminho.pdf --verbose
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import fitz  # PyMuPDF — renderização de página para imagem
except ImportError:
    print("Erro: 'pymupdf' necessário. Execute: uv add pymupdf", file=sys.stderr)
    sys.exit(1)

try:
    from paddleocr import PPStructure
except ImportError:
    print("Erro: 'paddleocr' necessário. Execute: uv add paddleocr", file=sys.stderr)
    sys.exit(1)

log = logging.getLogger("extrator_paddle")

# ---------------------------------------------------------------------------
# REGEX DE VALORES MONETÁRIOS BR
# ---------------------------------------------------------------------------
# Captura os formatos encontrados no DOM-PI:
#   R$ 1.234,56   → com prefixo
#   1.234,56      → sem prefixo (maioria dos RREO/RGF)
#   13. 729.422,36 → espaço de OCR no separador de milhar
#   1.234,56-     → valor negativo (demonstrativos)
# Âncora: vírgula + exatamente 2 dígitos (reduz falso positivo com datas dd/mm)
_RE_VALOR_BR = re.compile(
    r"(?:R\$\s*)?(\d{1,3}(?:[.\s]\d{3})*,\d{2})(?:\s*[-])?(?!\d)",
    re.UNICODE,
)

# Tokens de zero corrompido pelo OCR: º·ºº, °·°°, 0·00, etc.
_RE_ZERO_CORROMPIDO = re.compile(
    r"[º°o0][·.\s][º°o0]{2}",
    re.UNICODE,
)

# Palavras-chave de documentos com alta probabilidade de tabelas fiscais
PALAVRAS_TABELA: frozenset[str] = frozenset({
    "balanço", "rreo", "rgf", "orçamentária", "orçamento", "lrf",
    "licitação", "anexo", "planilha", "dotação", "credito", "crédito",
    "folha de pagamento", "demonstrativo", "despesa", "receita",
    "contrato", "extrato", "rubrica", "suplementação", "empenho",
    "liquidação", "pagamento", "receitas correntes", "despesas correntes",
})

# ---------------------------------------------------------------------------
# ESTRUTURAS DE DADOS
# ---------------------------------------------------------------------------

@dataclass
class PageResult:
    """Resultado da análise de uma única página."""
    page_num: int
    texto: str = ""
    has_table: bool = False
    has_figure: bool = False
    valores: list[str] = field(default_factory=list)
    zeros_corrompidos: int = 0
    layout_types: list[str] = field(default_factory=list)


@dataclass
class DocResult:
    """Resultado agregado de todo o documento PDF."""
    pdf_path: str
    paginas: list[PageResult] = field(default_factory=list)

    @property
    def texto_completo(self) -> str:
        return "\n\n".join(p.texto for p in self.paginas if p.texto.strip())

    @property
    def has_tables(self) -> bool:
        return any(p.has_table for p in self.paginas)

    @property
    def has_figures(self) -> bool:
        return any(p.has_figure for p in self.paginas)

    @property
    def table_pages(self) -> list[int]:
        return [p.page_num for p in self.paginas if p.has_table]

    @property
    def figure_pages(self) -> list[int]:
        return [p.page_num for p in self.paginas if p.has_figure]

    @property
    def todos_valores(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for p in self.paginas:
            for v in p.valores:
                if v not in seen:
                    seen.add(v)
                    result.append(v)
        return result

    @property
    def valores_monetarios_total(self) -> int:
        return sum(len(p.valores) for p in self.paginas)

    @property
    def zeros_corrompidos_total(self) -> int:
        return sum(p.zeros_corrompidos for p in self.paginas)


# ---------------------------------------------------------------------------
# MAPA DE CORREÇÃO DE ACENTOS PT-BR
# ---------------------------------------------------------------------------
# O modelo multilingual do PaddleOCR tende a substituir caracteres acentuados
# pelos seus equivalentes ASCII mais próximos. Este mapa corrige os casos
# mais frequentes em documentos municipais brasileiros.
_ACENTO_MAP: list[tuple[re.Pattern, str]] = [
    # ã, Ã
    (re.compile(r"\ban\b(?=\s)", re.IGNORECASE), "ã"),   # contexto: "prefeitur~a~"
    # Substituições comuns por par (ascii → acentuado)
    # Ordem importa: substituições mais longas primeiro
    (re.compile(r"\bPrefeitura\b", re.IGNORECASE), "Prefeitura"),
    (re.compile(r"\bCam[ae]ra\b", re.IGNORECASE), "Câmara"),
    (re.compile(r"\bMunicipal\b", re.IGNORECASE), "Municipal"),
    (re.compile(r"\bMunicipio\b", re.IGNORECASE), "Município"),
    (re.compile(r"\bSecretaria\b", re.IGNORECASE), "Secretaria"),
    (re.compile(r"\bPiau[il]\b", re.IGNORECASE), "Piauí"),
    (re.compile(r"\bPortaria\b", re.IGNORECASE), "Portaria"),
    (re.compile(r"\bDecr[ae]to\b", re.IGNORECASE), "Decreto"),
    (re.compile(r"\bOrcam[ae]nto\b", re.IGNORECASE), "Orçamento"),
    (re.compile(r"\bLicitac[ao]o\b", re.IGNORECASE), "Licitação"),
    (re.compile(r"\bResoluc[ao]o\b", re.IGNORECASE), "Resolução"),
]

# Municípios de Tabuleiros do Alto Parnaíba (âncoras de correção)
_MUNICIPIOS_TABULEIROS: frozenset[str] = frozenset({
    "Marcos Parente", "Jerumenha", "Bertolínia", "Guadalupe",
    "Ribeiro Gonçalves", "Uruçuí", "Landri Sales", "Porto Alegre do Piauí",
    "Baixa Grande do Ribeiro", "Sebastião Leal", "Canavieira",
    "Antônio Almeida",
})


def corrigir_acentos_ptbr(text: str, municipios: set[str] | None = None) -> str:
    """
    Aplica heurísticas de restauração de acentos corrompidos pelo OCR multilingual.
    Usa o dicionário de municípios como âncora adicional de validação.
    """
    if not text:
        return text

    muns = municipios or _MUNICIPIOS_TABULEIROS
    resultado = text

    # Corrige zeros corrompidos: º·ºº → 0,00
    resultado = _RE_ZERO_CORROMPIDO.sub("0,00", resultado)

    # Aplica substituições do mapa de acentos
    for pattern, replacement in _ACENTO_MAP:
        resultado = pattern.sub(replacement, resultado)

    return resultado


# ---------------------------------------------------------------------------
# ENGINE PADDLEOCR PP-STRUCTURE
# ---------------------------------------------------------------------------

def criar_engine_paddle(use_gpu: bool = False) -> PPStructure:
    """
    Inicializa PP-Structure v3 uma única vez por processo.

    O modelo de layout do PP-Structure suporta apenas 'en' e 'ch'.
    O modelo OCR interno é multilingual e cobre caracteres latinos (PT-BR).

    Args:
        use_gpu: True para tentar usar CUDA. Paddle GPU requer PaddlePaddle-GPU
                 compilado para a versão CUDA do sistema — use False se não
                 instalou a versão GPU do paddlepaddle.

    Returns:
        Instância de PPStructure pronta para análise.
    """
    engine = PPStructure(
        table=True,          # detecta e analisa estrutura de tabelas
        ocr=True,            # executa OCR nas regiões de texto
        show_log=False,      # suprime logs verbosos do Paddle
        use_gpu=use_gpu,
        lang="en",           # único valor suportado pelo modelo de layout
        image_orientation=False,  # não rota imagens (mais rápido)
        recovery=False,           # sem recovery de layout (mais rápido)
    )
    log.debug(f"PP-Structure inicializado (gpu={use_gpu})")
    return engine


# ---------------------------------------------------------------------------
# EXTRAÇÃO DE VALORES MONETÁRIOS
# ---------------------------------------------------------------------------

def extrair_valores_br(text: str) -> tuple[list[str], int]:
    """
    Extrai valores monetários no formato BR do texto extraído pelo OCR.

    Returns:
        (lista_de_valores, contagem_zeros_corrompidos)
        - lista_de_valores: strings como "13.785.379,22", "4.253,69"
        - contagem_zeros_corrompidos: tokens tipo "º·ºº" encontrados
    """
    valores = [m.group(1) for m in _RE_VALOR_BR.finditer(text)]
    zeros = len(_RE_ZERO_CORROMPIDO.findall(text))
    return valores, zeros


def _tem_palavras_tabela(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in PALAVRAS_TABELA)


# ---------------------------------------------------------------------------
# ANÁLISE POR PÁGINA
# ---------------------------------------------------------------------------

def _page_to_numpy(page: fitz.Page, dpi: int = 200) -> np.ndarray:
    """Renderiza uma página fitz como array numpy RGB (para o PP-Structure)."""
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, 3)
    # PP-Structure espera BGR (OpenCV convention)
    return img[:, :, ::-1].copy()


def _extrair_texto_res(res: Any) -> list[str]:
    """
    Extrai linhas de texto do campo 'res' de uma região PP-Structure.

    O PP-Structure retorna res em formatos diferentes dependendo do tipo de região:
    - Regiões de texto/figure: lista de dicts {'text': str, 'confidence': float, ...}
    - Regiões de tabela: dict {'html': str, ...}
    - Fallback legado: lista de listas [[bbox, (texto, conf)], ...]
    """
    parts: list[str] = []

    if isinstance(res, dict):
        # Tabela — extrai texto do HTML
        html = res.get("html", "")
        if html:
            txt = re.sub(r"<[^>]+>", " ", html)
            txt = re.sub(r"\s+", " ", txt).strip()
            if txt:
                parts.append(txt)
        return parts

    if not isinstance(res, list):
        return parts

    for item in res:
        if isinstance(item, dict):
            # Formato moderno do PP-Structure: {'text': ..., 'confidence': ..., 'text_region': ...}
            txt = item.get("text", item.get("transcription", "")).strip()
            if txt:
                parts.append(txt)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            # Formato legado: [bbox, (texto, conf)]
            second = item[1]
            if isinstance(second, (list, tuple)) and len(second) >= 1:
                txt = str(second[0]).strip()
            else:
                txt = str(second).strip()
            if txt:
                parts.append(txt)
        elif isinstance(item, str):
            txt = item.strip()
            if txt:
                parts.append(txt)

    return parts


# Limite mínimo de items OCR para considerar uma região 'figure' como
# página escaneada (não gráfico real). PDFs do DOM-PI têm 50-150 linhas por página.
_FIGURE_AS_PAGE_MIN_OCR_ITEMS = 10


def analisar_pagina_paddle(
    engine: PPStructure,
    img_bgr: np.ndarray,
    page_num: int,
) -> PageResult:
    """
    Analisa uma única página via PP-Structure.

    O PP-Structure retorna uma lista de regiões, cada uma com:
      - 'type': 'text' | 'table' | 'figure' | 'title' | 'list'
      - 'res':  OCR da região (lista de dicts com 'text'/'confidence')

    Particularidade do DOM-PI: páginas escaneadas inteiras são classificadas
    como 'figure', mas 'res' contém os resultados OCR. O texto é sempre extraído
    de 'res' independente do tipo. A flag has_figure só é marcada para regiões
    com poucos itens OCR (gráfico real, não página escaneada).

    Args:
        engine: instância de PPStructure (criar uma vez por processo)
        img_bgr: imagem da página em formato BGR numpy array
        page_num: índice 0-based da página

    Returns:
        PageResult com texto extraído + flags has_table/has_figure + valores
    """
    result = PageResult(page_num=page_num)

    try:
        layout_regions = engine(img_bgr)
    except Exception as e:
        log.warning(f"PP-Structure falhou na página {page_num}: {e}")
        return result

    text_parts: list[str] = []

    for region in layout_regions:
        region_type = region.get("type", "").lower()
        result.layout_types.append(region_type)
        res = region.get("res", [])

        if region_type == "table":
            result.has_table = True
            # Extrai texto das células para o corpus linear
            cell_parts = _extrair_texto_res(res)
            text_parts.extend(cell_parts)

        elif region_type == "figure":
            ocr_items = res if isinstance(res, list) else []
            if len(ocr_items) < _FIGURE_AS_PAGE_MIN_OCR_ITEMS:
                # Poucos itens OCR → provavelmente gráfico/imagem real
                result.has_figure = True
            # Extrai texto OCR mesmo de regiões 'figure' (páginas escaneadas do DOM-PI
            # chegam inteiras como figure com 50-150 linhas de texto)
            text_parts.extend(_extrair_texto_res(res))

        else:
            # 'text', 'title', 'list' → extrai texto OCR normalmente
            text_parts.extend(_extrair_texto_res(res))

    result.texto = corrigir_acentos_ptbr("\n".join(text_parts))

    # Detecção adicional de tabelas por palavras-chave fiscais no texto
    if not result.has_table and _tem_palavras_tabela(result.texto):
        result.has_table = True

    result.valores, result.zeros_corrompidos = extrair_valores_br(result.texto)

    return result


# ---------------------------------------------------------------------------
# EXTRAÇÃO COMPLETA DE UM PDF
# ---------------------------------------------------------------------------

def extrair_pdf_paddle(
    engine: PPStructure,
    pdf_path: str,
    dpi: int = 200,
) -> DocResult:
    """
    Extrai texto e metadados de layout de todas as páginas de um PDF.

    Args:
        engine: instância de PPStructure (reutilizada entre chamadas)
        pdf_path: caminho absoluto para o PDF
        dpi: resolução de renderização (200 DPI — balanço qualidade/velocidade)

    Returns:
        DocResult com texto completo + flags de tabela/figura + valores
    """
    result = DocResult(pdf_path=pdf_path)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log.error(f"Erro ao abrir PDF {pdf_path}: {e}")
        return result

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        img_bgr = _page_to_numpy(page, dpi=dpi)
        page_result = analisar_pagina_paddle(engine, img_bgr, page_num)
        result.paginas.append(page_result)
        log.debug(
            f"  Pág {page_num + 1}: {len(page_result.texto)} chars | "
            f"table={page_result.has_table} | fig={page_result.has_figure} | "
            f"valores={len(page_result.valores)}"
        )

    doc.close()
    return result


# ---------------------------------------------------------------------------
# CLI DE DIAGNÓSTICO
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
        description="Diagnóstico: analisa um PDF com PaddleOCR PP-Structure."
    )
    parser.add_argument("--pdf", required=True, help="Caminho para o PDF a analisar.")
    parser.add_argument("--dpi", type=int, default=200, help="DPI de renderização (padrão: 200).")
    parser.add_argument("--gpu", action="store_true", help="Usar GPU (requer paddlepaddle-gpu).")
    parser.add_argument("--verbose", action="store_true", help="Logs detalhados.")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    if not Path(args.pdf).exists():
        print(f"Erro: arquivo não encontrado: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    log.info("Inicializando PP-Structure...")
    engine = criar_engine_paddle(use_gpu=args.gpu)

    log.info(f"Analisando: {args.pdf}")
    result = extrair_pdf_paddle(engine, args.pdf, dpi=args.dpi)

    print("\n" + "=" * 65)
    print(f"  RESULTADO — {Path(args.pdf).name}")
    print("=" * 65)
    print(f"  Páginas:               {len(result.paginas)}")
    print(f"  Texto total:           {len(result.texto_completo)} chars")
    print(f"  has_tables:            {result.has_tables}")
    print(f"  table_pages:           {result.table_pages}")
    print(f"  has_figures:           {result.has_figures}")
    print(f"  figure_pages:          {result.figure_pages}")
    print(f"  Valores monetários:    {result.valores_monetarios_total}")
    print(f"  Zeros corrompidos:     {result.zeros_corrompidos_total}")

    if result.todos_valores:
        print(f"\n  Amostra de valores detectados:")
        for v in result.todos_valores[:10]:
            print(f"    {v}")

    print("\n  --- Texto extraído (primeiros 800 chars) ---")
    print(result.texto_completo[:800])
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
