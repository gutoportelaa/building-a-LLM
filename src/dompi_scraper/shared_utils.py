#!/usr/bin/env python3
"""
shared_utils.py — Utilidades Compartilhadas do Pipeline DOM-PI
--------------------------------------------------------------
Funções puras reutilizáveis por todos os módulos do pipeline:
- Normalização de texto e espaços
- Geração de slugs seguros para filesystem
- Hash MD5 para deduplicação textual seletiva
- Classificação de tipo de ato governamental por regex
- Extração de datas no formato brasileiro de corpos de texto
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import unicodedata


# ---------------------------------------------------------------------------
# NORMALIZAÇÃO BÁSICA
# ---------------------------------------------------------------------------

def normalize_spaces(value: str) -> str:
    """Normalize whitespace to a single space and trim ends."""
    return re.sub(r"\s+", " ", value or "").strip()


def slugify(value: str, fallback: str = "sem_nome", trim_chars: str = "_") -> str:
    """Build filesystem-safe slugs with configurable fallback behavior."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    cleaned = cleaned.strip(trim_chars)
    return cleaned or fallback


def strip_accents(value: str) -> str:
    """Remove acentos de uma string para comparações normalizadas."""
    normalized = unicodedata.normalize("NFD", value or "")
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def read_csv_rows(path: str) -> list[dict[str, str]]:
    """Load all CSV rows as dictionaries, returning an empty list when missing."""
    if not os.path.exists(path):
        return []

    with open(path, newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


# ---------------------------------------------------------------------------
# DEDUPLICAÇÃO TEXTUAL SELETIVA
# ---------------------------------------------------------------------------

def normalize_text_for_dedup(raw_text: str) -> str:
    """
    Normaliza texto para cálculo de hash de deduplicação.

    - Converte para minúsculas
    - Remove espaços em branco extras e quebras de linha múltiplas
    - Remove caracteres de controle Unicode
    - NÃO remove acentos (preserva fidelidade linguística)

    O objetivo é que dois textos semanticamente idênticos (mesmo que com
    diferenças de formatação, espaçamento ou encoding) produzam o mesmo hash.
    """
    if not raw_text:
        return ""
    # Remove caracteres de controle (exceto newline e espaço)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", raw_text)
    # Colapsa espaços e newlines
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def compute_content_md5(raw_text: str) -> str:
    """
    Calcula MD5 do conteúdo normativo de um documento.

    Usa normalize_text_for_dedup() para garantir que textos idênticos
    (mas com diferenças de whitespace/formatação) produzam o mesmo hash.
    O hash é calculado APENAS sobre o corpo do texto — sem metadados,
    URLs ou datas de raspagem.
    """
    normalized = normalize_text_for_dedup(raw_text)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# CLASSIFICAÇÃO DE TIPO DE ATO GOVERNAMENTAL
# ---------------------------------------------------------------------------

# Mapeamento ordenado por especificidade (mais específicos primeiro)
_ACT_TYPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Portaria",    re.compile(r"\bPORTARIA\b", re.IGNORECASE)),
    ("Decreto",     re.compile(r"\bDECRETO\b", re.IGNORECASE)),
    ("Lei",         re.compile(r"\bLEI\s+(COMPLEMENTAR\s+|ORDINÁRIA\s+|MUNICIPAL\s+)?N[ºo°]", re.IGNORECASE)),
    ("Edital",      re.compile(r"\bEDITAL\b", re.IGNORECASE)),
    ("Licitação",   re.compile(r"\b(LICITA[CÇ][AÃ]O|PREG[AÃ]O|DISPENSA|INEXIGIBILIDADE|TOMADA\s+DE\s+PRE[CÇ]O)\b", re.IGNORECASE)),
    ("Ata",         re.compile(r"\bATA\s+(DE\s+)?(SESS[AÃ]O|REGISTRO|REUNI[AÃ]O|JULGAMENTO)\b", re.IGNORECASE)),
    ("Contrato",    re.compile(r"\b(CONTRATO|EXTRATO\s+DE\s+CONTRATO|TERMO\s+DE\s+CONTRATO)\b", re.IGNORECASE)),
    ("Resolução",   re.compile(r"\bRESOLU[CÇ][AÃ]O\b", re.IGNORECASE)),
    ("Termo",       re.compile(r"\bTERMO\s+(DE\s+)?(POSSE|COOPERA|ADES[AÃ]O|REFER[EÊ]NCIA|HOMOLOGA)\b", re.IGNORECASE)),
    ("Aviso",       re.compile(r"\bAVISO\b", re.IGNORECASE)),
    ("Certidão",    re.compile(r"\bCERTID[AÃ]O\b", re.IGNORECASE)),
    ("Ofício",      re.compile(r"\bOF[IÍ]CIO\b", re.IGNORECASE)),
    ("Extrato",     re.compile(r"\bEXTRATO\b", re.IGNORECASE)),
    ("LRF",         re.compile(r"\b(LRF|LEI\s+DE\s+RESPONSABILIDADE\s+FISCAL|RELAT[OÓ]RIO\s+(RESUMIDO|DE\s+GEST[AÃ]O))\b", re.IGNORECASE)),
    ("Projeto",     re.compile(r"\bPROJETO\s+DE\s+LEI\b", re.IGNORECASE)),
]


def classify_act_type(text: str, fallback_category: str = "") -> str:
    """
    Classifica o tipo de ato governamental com base no conteúdo textual.

    Percorre uma lista de regex ordenada por especificidade até encontrar match.
    Se nenhum padrão for detectado, retorna o fallback_category (normalmente
    vindo do campo 'categoria' do JSON de scraping).

    Args:
        text: Corpo do texto do documento (primeiros ~500 chars são suficientes).
        fallback_category: Categoria padrão vinda dos metadados do scraping.

    Returns:
        Nome do tipo de ato identificado.
    """
    # Usa apenas os primeiros 1000 caracteres para performance
    snippet = (text or "")[:1000]
    for act_name, pattern in _ACT_TYPE_PATTERNS:
        if pattern.search(snippet):
            return act_name
    return fallback_category or "Não Identificado"


# ---------------------------------------------------------------------------
# EXTRAÇÃO DE DATAS DO CORPO DO TEXTO
# ---------------------------------------------------------------------------

_MESES_PT = {
    "janeiro": "01", "fevereiro": "02", "março": "03", "marco": "03",
    "abril": "04", "maio": "05", "junho": "06",
    "julho": "07", "agosto": "08", "setembro": "09",
    "outubro": "10", "novembro": "11", "dezembro": "12",
}

_DATE_PATTERN_EXTENSO = re.compile(
    r"(\d{1,2})\s+de\s+("
    + "|".join(_MESES_PT.keys())
    + r")\s+de\s+(\d{4})",
    re.IGNORECASE,
)

_DATE_PATTERN_NUMERICO = re.compile(r"(\d{2})/(\d{2})/(\d{4})")


def extract_date_from_text(text: str) -> str | None:
    """
    Extrai a primeira data encontrada no corpo do texto.

    Reconhece formatos:
    - Extenso: "15 de março de 2025"
    - Numérico: "15/03/2025"

    Retorna no formato ISO 8601: "2025-03-15" ou None se não encontrar.
    """
    if not text:
        return None

    # Prioriza formato extenso (mais comum em atos oficiais)
    match = _DATE_PATTERN_EXTENSO.search(text)
    if match:
        dia = match.group(1).zfill(2)
        mes = _MESES_PT.get(match.group(2).lower(), "00")
        ano = match.group(3)
        return f"{ano}-{mes}-{dia}"

    # Fallback: numérico DD/MM/AAAA
    match = _DATE_PATTERN_NUMERICO.search(text)
    if match:
        dia, mes, ano = match.group(1), match.group(2), match.group(3)
        return f"{ano}-{mes}-{dia}"

    return None
