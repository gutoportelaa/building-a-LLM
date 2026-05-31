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
# PALAVRAS-CHAVE DE DOCUMENTOS COM TABELAS / DADOS FISCAIS
# ---------------------------------------------------------------------------
# Fonte única usada tanto pelo extrator_paddle (no venv do paddle) quanto pelo
# orquestrador (no venv do torch) para sinalizar documentos "complexos" que
# valem o custo do Docling. Mantida aqui (módulo puro, sem dependências de GPU)
# para ser importável em AMBOS os ambientes isolados.
PALAVRAS_TABELA: frozenset[str] = frozenset({
    "balanço", "rreo", "rgf", "orçamentária", "orçamento", "lrf",
    "licitação", "anexo", "planilha", "dotação", "credito", "crédito",
    "folha de pagamento", "demonstrativo", "despesa", "receita",
    "contrato", "extrato", "rubrica", "suplementação", "empenho",
    "liquidação", "pagamento", "receitas correntes", "despesas correntes",
})


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
        text: Corpo do texto do documento.
        fallback_category: Categoria padrão vinda dos metadados do scraping.

    Returns:
        Nome do tipo de ato identificado.
    """
    # Primeira tentativa: primeiros 2000 chars (cabeçalho e título do ato) (P-11)
    snippet = (text or "")[:2000]
    for act_name, pattern in _ACT_TYPE_PATTERNS:
        if pattern.search(snippet):
            return act_name
    # Fallback: texto completo para documentos onde o tipo aparece no corpo (P-11)
    full = text or ""
    if len(full) > 2000:
        for act_name, pattern in _ACT_TYPE_PATTERNS:
            if pattern.search(full):
                return act_name
    return fallback_category or "Não Identificado"


# ---------------------------------------------------------------------------
# EXTRAÇÃO DE DATA E EDIÇÃO A PARTIR DO NOME DO ARQUIVO (DOM-PI)
# ---------------------------------------------------------------------------
#
# Padrão de nome de arquivo do DOM-PI:
#   DM_{edicao}_{seq}_{Municipio_Parts}_{TipoAto}_{num}-{aa}_pag_{pag}.pdf
#
# Exemplos:
#   DM_5234_053_Antonio_Almeida_Portaria_001-25_pag_616.pdf
#   DM_5251_056_Antonio_Almeida_Licitacao_Inexigibilidade_004-24_Extrato_Contrato_002-25_pag_484.pdf
#
# Estratégia de data (apenas pelo filename, sem ler conteúdo):
#   1. Extrai número de edição DM_NNNN → campo edicao_dom
#   2. Coleta TODOS os sufixos "-NN_" antes do "_pag_" e pega o ÚLTIMO
#      (o processo antecedente tem sufixo mais à esquerda; o ato principal
#      que efetivamente foi publicado nesta edição está mais à direita)
#   3. Se não achar sufixo "-NN_", tenta padrão de ano 4 dígitos "_AAAA_"
#   4. Retorna "AAAA" (somente ano) + campo data_confianca indicando a fonte

_RE_EDICAO = re.compile(r"^DM_(\d+)_", re.IGNORECASE)
_RE_ANO_SUFIXO = re.compile(r"-(\d{2})(?:_|$)")        # padrão  -25_ ou -25
_RE_ANO_4DIGITS = re.compile(r"_(\d{4})(?:_|$)")  # padrão  _2025_ ou _2025


def extrair_edicao_filename(filename: str) -> str:
    """
    Extrai o número de edição do DOM-PI a partir do nome do arquivo.

    Exemplo: DM_5234_053_... → "5234"
    Retorna "" se o padrão não for encontrado.
    """
    basename = os.path.basename(filename)
    m = _RE_EDICAO.match(basename)
    return m.group(1) if m else ""


def extrair_data_filename(filename: str) -> tuple[str, str]:
    """
    Extrai o ano de publicação a partir do nome do arquivo DOM-PI.
    Não lê nenhum conteúdo — opera apenas sobre o nome do arquivo.

    Lógica:
    1. Remove o sufixo "_pag_NNN" para evitar falsos positivos no número de página.
    2. Remove o prefixo "DM_NNNN_MMM_" para evitar capturar a edição como ano.
    3. Coleta todos os padrões "-NN_" e usa o ÚLTIMO (o ato mais à direita no
       nome é o ato publicado; referências a processos antecedentes ficam antes).
    4. Fallback: busca por "_AAAA_" (ano com 4 dígitos).

    Retorna:
        (ano_iso, data_confianca)
        - ano_iso:        "2025"  ou  ""  se não encontrado
        - data_confianca: "filename_sufixo" | "filename_ano4d" | "ausente"

    Atenção: retorna apenas o ANO. O mês/dia exato requer mapeamento de
    edição→data que não está disponível localmente.
    """
    basename = os.path.basename(filename)
    stem = os.path.splitext(basename)[0]

    # Remove _pag_NNN para não confundir número de página com ano
    stem = re.sub(r"_pag_\d+$", "", stem, flags=re.IGNORECASE)
    
    # Remove o prefixo DM_NNNN_MMM_ para não confundir edição com ano
    stem = re.sub(r"^DM_\d+_\d+_", "", stem, flags=re.IGNORECASE)

    # Camada 1: sufixos "-NN_" — pega o ÚLTIMO (ato principal, mais à direita)
    matches = _RE_ANO_SUFIXO.findall(stem)
    if matches:
        year_2d = matches[-1]
        # Converte 2 dígitos → 4 dígitos (assumindo século 21, válido para 00-99)
        year_4d = f"20{year_2d}"
        # Sanidade: aceita apenas 2000–2099
        if 2000 <= int(year_4d) <= 2099:
            return year_4d, "filename_sufixo"

    # Camada 2: ano de 4 dígitos explícito no nome ("_2025_" ou terminando com "_2025")
    matches_4d = _RE_ANO_4DIGITS.findall(stem)
    valid = [y for y in matches_4d if 2000 <= int(y) <= 2099]
    if valid:
        return valid[-1], "filename_ano4d"

    return "", "ausente"


_MESES_PT = {
    "janeiro": "01", "fevereiro": "02", "março": "03", "marco": "03",
    "abril": "04", "maio": "05", "junho": "06",
    "julho": "07", "agosto": "08", "setembro": "09",
    "outubro": "10", "novembro": "11", "dezembro": "12",
}
_DATE_EXTENSO = re.compile(
    r"(\d{1,2})\s+de\s+(" + "|".join(_MESES_PT.keys()) + r")\s+de\s+(\d{4})",
    re.IGNORECASE,
)
_DATE_NUMERICO = re.compile(r"(\d{2})/(\d{2})/(\d{4})")


def _parse_extenso(match: re.Match) -> str:
    dia = match.group(1).zfill(2)
    mes = _MESES_PT.get(match.group(2).lower(), "00")
    return f"{match.group(3)}-{mes}-{dia}"


def extract_date_from_text(text: str) -> str | None:
    """
    Extrai a data de publicação do corpo do texto.
    NOTA: Use extrair_data_filename() quando disponível — é mais confiável.

    Estratégia (P-02): prefere a data na cauda do documento (últimos 1500
    chars), onde fica o bloco de assinatura "Cidade, DD de mês de AAAA".
    Recorre ao texto completo apenas se a cauda não tiver data.

    Reconhece formatos:
    - Extenso: "15 de março de 2025"
    - Numérico: "15/03/2025"

    Retorna no formato ISO 8601: "2025-03-15" ou None se não encontrar.
    """
    if not text:
        return None

    # 1. Busca na cauda (bloco de assinatura — mais próximo da data de publicação)
    tail = text[-1500:] if len(text) > 1500 else text
    match = _DATE_EXTENSO.search(tail)
    if match:
        return _parse_extenso(match)
    match = _DATE_NUMERICO.search(tail)
    if match:
        return f"{match.group(3)}-{match.group(2)}-{match.group(1)}"

    # 2. Fallback: primeiro match no texto completo
    match = _DATE_EXTENSO.search(text)
    if match:
        return _parse_extenso(match)
    match = _DATE_NUMERICO.search(text)
    if match:
        return f"{match.group(3)}-{match.group(2)}-{match.group(1)}"

    return None
