#!/usr/bin/env python3
"""
processar_pdfs.py — Processador de PDFs DOM-PI → Markdown + JSONL
------------------------------------------------------------------
Lê PDFs baixados pelo download_pdfs.py, extrai texto estruturado via PyMuPDF,
gera arquivos Markdown com frontmatter YAML (para RAG/LangChain) e consolida
em JSONL (para fine-tuning de LLMs).

Pipeline interno:
1. Lê manifesto de download para localizar PDFs em disco
2. Extrai texto rico via PyMuPDF (blocos com font, size, bold)
3. Classifica tipo de ato governamental por regex
4. Gera Markdown hierárquico com frontmatter YAML
5. Armazena em Data Lake particionado: dados_brutos/ano=YYYY/mes=MM/municipio=slug/
6. Deduplicação textual por hash MD5 do conteúdo normativo
7. Consolida JSONL limpo para fine-tuning

Uso:
    # Processar 5 PDFs de amostra com análise detalhada de blocos
    uv run python src/dompi_scraper/processar_pdfs.py \\
        --manifest db_treino_carnaubais/pdfs_arquivos/download_manifest.json \\
        --output-dir dados_brutos \\
        --jsonl-output corpus_treino_dompi.jsonl \\
        --limite 5 \\
        --verbose-blocos

    # Processamento completo
    uv run python src/dompi_scraper/processar_pdfs.py \\
        --manifest db_treino_carnaubais/pdfs_arquivos/download_manifest.json \\
        --output-dir dados_brutos \\
        --jsonl-output corpus_treino_dompi.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Erro: A biblioteca 'pymupdf' (fitz) é necessária. Instale com `uv add pymupdf`", file=sys.stderr)
    sys.exit(1)

try:
    from .shared_utils import (
        classify_act_type,
        compute_content_md5,
        extract_date_from_text,
        normalize_text_for_dedup,
        slugify,
        strip_accents,
    )
except ImportError:
    from shared_utils import (
        classify_act_type,
        compute_content_md5,
        extract_date_from_text,
        normalize_text_for_dedup,
        slugify,
        strip_accents,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("processar_pdfs")


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    log.setLevel(level)
    log.handlers.clear()
    log.addHandler(handler)


# ---------------------------------------------------------------------------
# CONSTANTES DE HIERARQUIZAÇÃO
# ---------------------------------------------------------------------------

# Thresholds visuais para mapeamento de headings (calibráveis via --verbose-blocos)
HEADING_H1_MIN_SIZE = 14.0   # Fonte >= 14pt → # Heading 1
HEADING_H2_MIN_SIZE = 11.0   # Fonte >= 11pt e negrito → ## Heading 2
BODY_DEFAULT_SIZE = 9.0      # Tamanho típico de corpo de texto no DOM-PI

# Threshold mínimo de qualidade OCR (0.0 = lixo total, 1.0 = texto perfeito)
# Blocos abaixo desse score são descartados como artefatos de OCR
MIN_OCR_QUALITY_SCORE = 0.35

# Regex para padrões de estruturação de atos oficiais
RE_RESOLVE = re.compile(r"^(RESOLVE|DECRETA|DETERMINA|CONSIDERANDO)\s*:", re.IGNORECASE)
RE_ARTIGO = re.compile(r"^(Art(?:igo)?\.?\s*\d+[ºo°]?)", re.IGNORECASE)
RE_PARAGRAFO = re.compile(r"^(§\s*\d+[ºo°]?|Parágrafo\s+[Úú]nico)", re.IGNORECASE)
RE_INCISO = re.compile(r"^([IVXLCDM]+\s*[-–])", re.IGNORECASE)
RE_PREFEITURA = re.compile(
    r"(?:PREFEITURA|C[AÂ]MARA)\s+(?:MUNICIPAL\s+)?(?:DE\s+)?([A-ZÀ-Ÿ\s]+?)(?:\s*[-–]?\s*(?:PI|PIAUÍ|PIAUI))?$",
    re.IGNORECASE | re.MULTILINE,
)
RE_ASSINATURA = re.compile(
    r"((?:Prefeito|Presidente|Secretário|Vereador|Gestor)(?:\s+Municipal)?)",
    re.IGNORECASE,
)

# Palavras comuns em documentos oficiais brasileiros (usadas para validar qualidade OCR)
_PALAVRAS_VALIDAS = {
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas", "que",
    "para", "por", "com", "seu", "sua", "seus", "suas", "uma", "este", "esta",
    "municipal", "prefeitura", "câmara", "estado", "piauí", "portaria",
    "decreto", "lei", "edital", "art", "resolve", "publicação", "janeiro",
    "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
    "setembro", "outubro", "novembro", "dezembro", "cargo", "nomear",
    "município", "prefeito", "secretário", "atribuições", "legais",
    "publicar", "registrar", "cumprir", "vereador", "oficial",
}


# ---------------------------------------------------------------------------
# FILTRO DE QUALIDADE OCR
# ---------------------------------------------------------------------------

def compute_ocr_quality_score(text: str) -> float:
    """
    Calcula um score de qualidade [0.0, 1.0] para um bloco de texto extraído.

    Heurísticas:
    1. Proporção de caracteres alfanuméricos/espaço vs total
       (Textos reais têm >60% alfanum, OCR garbage tem <30%)
    2. Tamanho médio de palavras (palavras reais: 2-15 chars; OCR: 1-2 chars)
    3. Presença de palavras reconhecíveis em PT-BR
    4. Densidade de caracteres especiais consecutivos

    Score >= MIN_OCR_QUALITY_SCORE → texto aproveitável
    Score <  MIN_OCR_QUALITY_SCORE → provável lixo de OCR, descarta
    """
    if not text or len(text) < 3:
        return 0.0

    total_chars = len(text)

    # 1. Proporção alfanumérica (letras + dígitos + espaços acentuados)
    alnum_count = sum(1 for c in text if c.isalnum() or c in " .,:;()")
    alnum_ratio = alnum_count / total_chars

    # 2. Tamanho médio de palavras
    words = text.split()
    if not words:
        return 0.0
    avg_word_len = sum(len(w) for w in words) / len(words)
    word_len_score = min(avg_word_len / 5.0, 1.0)  # Ideal ~5+ chars

    # 3. Presença de palavras válidas em PT-BR
    words_lower = {w.lower().strip(".,;:!?()\"'") for w in words}
    known_hits = len(words_lower & _PALAVRAS_VALIDAS)
    vocab_score = min(known_hits / max(len(words) * 0.15, 1), 1.0)

    # 4. Penalidade para sequências de caracteres especiais
    special_sequences = len(re.findall(r"[~€#&@!\[\]{}|<>]{2,}", text))
    special_penalty = min(special_sequences * 0.1, 0.5)

    # Score composto (pesos calibrados para documentos oficiais brasileiros)
    score = (
        alnum_ratio * 0.45 +
        word_len_score * 0.20 +
        vocab_score * 0.25 -
        special_penalty * 0.10
    )

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# EXTRAÇÃO RICH TEXT (PYMUPDF)
# ---------------------------------------------------------------------------

def extract_rich_blocks(page: fitz.Page) -> list[dict[str, Any]]:
    """
    Extrai blocos de texto ricos de uma página PyMuPDF.

    Para cada bloco retorna:
    - texto: conteúdo textual limpo
    - tamanho: tamanho médio de fonte (pt)
    - negrito: True se algum span no bloco usa flag de bold
    - font_names: lista de famílias de fonte detectadas
    - bbox: bounding box [x0, y0, x1, y1]
    - flags_raw: lista de flag values (para debug/calibração)
    - ocr_quality: score de qualidade [0.0, 1.0]
    """
    page_dict = page.get_text("dict")
    blocks = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # Só texto (ignora imagens)
            continue

        text_parts = []
        sizes = []
        flags_all = []
        fonts = set()

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                text_parts.append(span_text)
                sizes.append(span.get("size", 0))
                flags_all.append(span.get("flags", 0))
                font_name = span.get("font", "")
                if font_name:
                    fonts.add(font_name)

        full_text = " ".join(text_parts).strip()
        if not full_text:
            continue

        avg_size = round(sum(sizes) / len(sizes), 2) if sizes else 0
        max_flags = max(flags_all) if flags_all else 0

        # Detecção de negrito: flag bit 4 (valor 16) OU nome de fonte contém "Bold"
        is_bold = bool(max_flags & (1 << 4)) or any("bold" in f.lower() for f in fonts)

        # Score de qualidade OCR
        quality = compute_ocr_quality_score(full_text)

        blocks.append({
            "texto": full_text,
            "tamanho": avg_size,
            "negrito": is_bold,
            "font_names": sorted(fonts),
            "bbox": block.get("bbox", []),
            "flags_raw": flags_all,
            "ocr_quality": quality,
        })

    return blocks


# ---------------------------------------------------------------------------
# VERBOSE-BLOCOS: Dump Analítico para Calibração
# ---------------------------------------------------------------------------

def dump_verbose_blocks(pdf_path: str, max_pages: int = 3) -> None:
    """
    Despeja análise detalhada dos blocos de texto de um PDF.
    Útil para calibrar thresholds de heading (tamanho, negrito, etc).
    """
    doc = fitz.open(pdf_path)
    total = min(len(doc), max_pages)

    print("\n" + "=" * 80)
    print(f"  ANÁLISE DETALHADA DE BLOCOS — {os.path.basename(pdf_path)}")
    print(f"  Total de páginas: {len(doc)} (mostrando primeiras {total})")
    print("=" * 80)

    # Coleta estatísticas globais
    all_sizes = []
    all_bold_count = 0
    all_block_count = 0

    for page_num in range(total):
        page = doc.load_page(page_num)
        blocks = extract_rich_blocks(page)

        print(f"\n{'─' * 70}")
        print(f"  PÁGINA {page_num + 1}")
        print(f"{'─' * 70}")

        for i, blk in enumerate(blocks):
            all_block_count += 1
            all_sizes.append(blk["tamanho"])
            if blk["negrito"]:
                all_bold_count += 1

            # Trunca texto longo para visualização
            txt_preview = blk["texto"][:120]
            if len(blk["texto"]) > 120:
                txt_preview += "..."

            # Indica heading candidato
            heading_marker = ""
            if blk["tamanho"] >= HEADING_H1_MIN_SIZE:
                heading_marker = " ← [H1 CANDIDATO]"
            elif blk["negrito"] and blk["tamanho"] >= HEADING_H2_MIN_SIZE:
                heading_marker = " ← [H2 CANDIDATO]"

            is_bold_label = "✓BOLD" if blk["negrito"] else "     "
            quality = blk.get("ocr_quality", 0)
            quality_label = f"Q={quality:.2f}"
            if quality < MIN_OCR_QUALITY_SCORE:
                quality_label += " ✗LIXO"
            else:
                quality_label += " ✓OK  "

            print(
                f"  [{i:02d}] {is_bold_label} | "
                f"Size={blk['tamanho']:5.1f}pt | "
                f"{quality_label} | "
                f"Fonts={','.join(blk['font_names'][:2])} | "
                f"Flags={blk['flags_raw'][:3]}"
                f"{heading_marker}"
            )
            print(f"       └─ \"{txt_preview}\"")

    # Estatísticas globais
    if all_sizes:
        print(f"\n{'=' * 80}")
        print(f"  ESTATÍSTICAS GLOBAIS ({all_block_count} blocos em {total} páginas):")
        print(f"    Tamanho MIN:  {min(all_sizes):.1f}pt")
        print(f"    Tamanho MAX:  {max(all_sizes):.1f}pt")
        print(f"    Tamanho MÉD:  {sum(all_sizes)/len(all_sizes):.1f}pt")
        print(f"    Blocos BOLD:  {all_bold_count}/{all_block_count} ({100*all_bold_count/all_block_count:.0f}%)")

        # Distribuição de tamanhos
        size_dist: dict[float, int] = {}
        for s in all_sizes:
            bucket = round(s, 0)
            size_dist[bucket] = size_dist.get(bucket, 0) + 1

        print(f"\n    Distribuição de tamanhos:")
        for size, count in sorted(size_dist.items()):
            bar = "█" * min(count, 40)
            print(f"      {size:5.0f}pt │ {count:3d} │ {bar}")

    print("=" * 80 + "\n")
    doc.close()


# ---------------------------------------------------------------------------
# HIERARQUIZAÇÃO: Blocos → Markdown
# ---------------------------------------------------------------------------

def blocks_to_markdown(blocks: list[dict], metadata: dict | None = None) -> str:
    """
    Converte blocos ricos de PyMuPDF em Markdown hierárquico.

    Regras de conversão:
    - Blocos com font >= H1_THRESHOLD → # Heading 1
    - Blocos com bold + font >= H2_THRESHOLD → ## Heading 2
    - Padrões RESOLVE:/DECRETA: → **negrito**
    - Art. Xº → parágrafo com **negrito** no número
    - Texto normal → parágrafo simples
    - Assinaturas → separadas com ---
    """
    lines: list[str] = []
    last_was_heading = False

    for blk in blocks:
        txt = blk["texto"].strip()
        if not txt:
            continue

        size = blk["tamanho"]
        bold = blk["negrito"]

        # --- Detecção de Heading 1 (nome da entidade / cabeçalho principal) ---
        if size >= HEADING_H1_MIN_SIZE or (bold and RE_PREFEITURA.search(txt)):
            if lines and not last_was_heading:
                lines.append("")  # Espaço antes do heading
            lines.append(f"# {txt}")
            lines.append("")
            last_was_heading = True
            continue

        # --- Detecção de Heading 2 (título do ato) ---
        if bold and size >= HEADING_H2_MIN_SIZE and len(txt) < 150:
            if lines and not last_was_heading:
                lines.append("")
            lines.append(f"## {txt}")
            lines.append("")
            last_was_heading = True
            continue

        last_was_heading = False

        # --- Padrões estruturais de atos ---
        if RE_RESOLVE.match(txt):
            lines.append(f"**{txt}**")
            lines.append("")
            continue

        if RE_ARTIGO.match(txt):
            match = RE_ARTIGO.match(txt)
            art_prefix = match.group(1)
            rest = txt[match.end():]
            lines.append(f"**{art_prefix}**{rest}")
            lines.append("")
            continue

        if RE_PARAGRAFO.match(txt):
            lines.append(f"*{txt}*")
            lines.append("")
            continue

        if RE_INCISO.match(txt):
            lines.append(f"  - {txt}")
            continue

        # --- Assinaturas (final do documento) ---
        if RE_ASSINATURA.search(txt) and len(txt) < 200:
            lines.append("")
            lines.append("---")
            lines.append(f"*{txt}*")
            lines.append("")
            continue

        # --- Texto normal com bold → enfatizado ---
        if bold and len(txt) < 300:
            lines.append(f"**{txt}**")
            lines.append("")
            continue

        # --- Corpo de texto padrão ---
        lines.append(txt)
        lines.append("")

    return "\n".join(lines)


def generate_frontmatter(
    content_hash: str,
    municipio: str,
    entidade: str,
    tipo_ato: str,
    data_publicacao: str,
    url_origem: str,
    edicao: str = "",
    paginas: list[int] | None = None,
    sha256_pdf: str = "",
) -> str:
    """
    Gera bloco YAML frontmatter para ingestão por LangChain/LlamaIndex.

    O frontmatter é lido automaticamente por bibliotecas de ingestão e
    transformado em metadados dos vetores, permitindo buscas híbridas:
    Ex: "portarias de licitação de Campo Maior depois de 01/2025"
    """
    lines = ["---"]
    lines.append(f'id_publicacao: "{content_hash}"')
    lines.append(f'municipio: "{municipio}"')
    lines.append(f'entidade: "{entidade}"')
    lines.append(f'tipo_ato: "{tipo_ato}"')
    lines.append(f'data_publicacao: "{data_publicacao}"')

    if edicao:
        lines.append(f'edicao: "{edicao}"')
    if paginas:
        lines.append(f'paginas: {paginas}')
    if sha256_pdf:
        lines.append(f'sha256_pdf: "{sha256_pdf}"')

    lines.append(f'url_origem: "{url_origem}"')
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CONVERSÃO DE DATA DD/MM/AAAA → ISO 8601
# ---------------------------------------------------------------------------

def parse_date_br(date_str: str) -> tuple[str, str, str]:
    """
    Converte data brasileira DD/MM/AAAA para (ano, mes, dia) em ISO.
    Retorna ('', '', '') se não puder parsear.
    """
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})", date_str or "")
    if m:
        return m.group(3), m.group(2), m.group(1)  # ano, mes, dia
    return "", "", ""


# ---------------------------------------------------------------------------
# DATA LAKE PARTICIONADO
# ---------------------------------------------------------------------------

def build_datalake_path(
    base_dir: str,
    ano: str,
    mes: str,
    municipio: str,
    filename: str,
) -> str:
    """
    Constrói caminho particionado estilo Data Lake:
    dados_brutos/ano=2025/mes=03/municipio=assuncao_do_pi/hash.md
    """
    mun_slug = slugify(strip_accents(municipio).lower(), fallback="desconhecido")
    path = os.path.join(
        base_dir,
        f"ano={ano}",
        f"mes={mes}",
        f"municipio={mun_slug}",
        filename,
    )
    return path


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def process_single_pdf(
    file_id: str,
    manifest_entry: dict,
    output_dir: str,
    dedup_registry: dict,
    verbose_blocks: bool = False,
) -> list[dict]:
    """
    Processa um único PDF e gera arquivos .md.

    Retorna lista de registros JSONL gerados (um por arquivo .md criado).
    """
    pdf_path = manifest_entry.get("path", "")

    if not os.path.exists(pdf_path):
        log.warning(f"  PDF não encontrado em disco: {pdf_path}")
        return []

    # Metadados do manifesto
    municipio = manifest_entry.get("municipio", "")
    entidade = manifest_entry.get("entidade", "")
    categoria = manifest_entry.get("categoria", "")
    data_pub = manifest_entry.get("data_publicacao", "")
    url_origem = manifest_entry.get("url", "")
    sha256_pdf = manifest_entry.get("sha256", "")
    edicao = manifest_entry.get("edicao_url_meta", "") or manifest_entry.get("edicao", "")
    documento = manifest_entry.get("documento", "")

    ano, mes, dia = parse_date_br(data_pub)
    data_iso = f"{ano}-{mes}-{dia}" if ano else ""

    # --- Verbose blocks dump ---
    if verbose_blocks:
        dump_verbose_blocks(pdf_path, max_pages=5)

    # --- Extração ---
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log.error(f"  Erro ao abrir PDF {pdf_path}: {e}")
        return []

    all_blocks_raw: list[dict] = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = extract_rich_blocks(page)
        all_blocks_raw.extend(blocks)
    doc.close()

    if not all_blocks_raw:
        log.warning(f"  PDF sem blocos de texto extraíveis: {pdf_path}")
        return []

    # --- Filtragem de qualidade OCR ---
    # Descarta blocos com score abaixo do limiar (lixo de OCR)
    all_blocks = [
        b for b in all_blocks_raw
        if b.get("ocr_quality", 0) >= MIN_OCR_QUALITY_SCORE
    ]
    ocr_filtered = len(all_blocks_raw) - len(all_blocks)
    if ocr_filtered > 0:
        log.debug(
            f"  Filtro OCR: {ocr_filtered}/{len(all_blocks_raw)} blocos descartados "
            f"(score < {MIN_OCR_QUALITY_SCORE})"
        )

    if not all_blocks:
        log.warning(f"  PDF sem blocos aproveitáveis após filtro OCR: {pdf_path}")
        return []

    # --- Texto bruto para deduplicação (apenas blocos aprovados) ---
    raw_text = "\n".join(b["texto"] for b in all_blocks)
    content_hash = compute_content_md5(raw_text)

    # --- Deduplicação textual ---
    if content_hash in dedup_registry:
        existing = dedup_registry[content_hash]
        log.debug(
            f"  DUPLICATA detectada: hash={content_hash[:12]}... "
            f"já registrado para '{existing.get('municipio', '?')}'"
        )
        return []

    # --- Classificação do tipo de ato ---
    tipo_ato = classify_act_type(raw_text, fallback_category=categoria)

    # --- Extração de data do corpo (fallback se não veio do scraping) ---
    if not data_iso:
        data_iso = extract_date_from_text(raw_text) or ""
        if data_iso:
            parts = data_iso.split("-")
            ano, mes, dia = parts[0], parts[1], parts[2]

    # --- Geração de Markdown hierárquico ---
    markdown_body = blocks_to_markdown(all_blocks)
    frontmatter = generate_frontmatter(
        content_hash=content_hash,
        municipio=municipio,
        entidade=entidade,
        tipo_ato=tipo_ato,
        data_publicacao=data_iso,
        url_origem=url_origem,
        edicao=edicao,
        sha256_pdf=sha256_pdf,
    )
    full_markdown = frontmatter + markdown_body

    # --- Armazenamento em Data Lake ---
    if not ano:
        ano, mes = "sem_ano", "sem_mes"
    if not mes:
        mes = "sem_mes"

    md_filename = f"{content_hash}.md"
    md_path = build_datalake_path(output_dir, ano, mes, municipio, md_filename)

    os.makedirs(os.path.dirname(md_path), exist_ok=True)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)

    # --- Registro de deduplicação ---
    dedup_registry[content_hash] = {
        "path": md_path,
        "municipio": municipio,
        "entidade": entidade,
        "tipo_ato": tipo_ato,
        "data_publicacao": data_iso,
        "url_origem": url_origem,
        "documento": documento,
    }

    # --- Registro JSONL ---
    # Remove frontmatter para o JSONL (o modelo treina em linguagem natural pura)
    jsonl_record = {
        "text": raw_text,
        "metadata": {
            "id_publicacao": content_hash,
            "municipio": municipio,
            "entidade": entidade,
            "tipo_ato": tipo_ato,
            "data_publicacao": data_iso,
            "edicao": edicao,
        }
    }

    log.info(
        f"  ✓ {municipio} / {tipo_ato} → {md_filename[:16]}... "
        f"({len(all_blocks)}/{len(all_blocks_raw)} blocos aprovados, {len(raw_text)} chars)"
    )

    return [jsonl_record]


def run_processing_pipeline(
    manifest_path: str,
    output_dir: str,
    jsonl_output: str,
    dedup_path: str,
    limite: int,
    verbose_blocks: bool,
    verbose_blocks_only: int | None,
) -> dict:
    """
    Executa o pipeline completo de processamento.
    """
    # Carrega manifesto
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Filtra apenas downloads com status OK
    ok_entries = {
        fid: entry for fid, entry in manifest.items()
        if entry.get("status") == "OK"
    }
    log.info(f"PDFs disponíveis para processamento: {len(ok_entries)}")

    # Carrega registro de deduplicação existente
    dedup_registry: dict = {}
    if os.path.exists(dedup_path):
        with open(dedup_path, "r", encoding="utf-8") as f:
            dedup_registry = json.load(f)
        log.info(f"Registro de deduplicação carregado: {len(dedup_registry)} hashes")

    # --- Modo verbose-blocos-only (apenas análise, sem processamento) ---
    if verbose_blocks_only is not None:
        count = 0
        for fid, entry in ok_entries.items():
            if count >= verbose_blocks_only:
                break
            pdf_path = entry.get("path", "")
            if os.path.exists(pdf_path):
                dump_verbose_blocks(pdf_path, max_pages=5)
                count += 1
        return {"total": count, "gerados": 0, "duplicatas": 0}

    # --- Processamento real ---
    os.makedirs(output_dir, exist_ok=True)

    stats = {"total": 0, "gerados": 0, "duplicatas": 0, "falhas": 0}
    all_jsonl_records: list[dict] = []
    processed = 0

    for fid, entry in ok_entries.items():
        if processed >= limite:
            log.info(f"Limite de {limite} processamentos atingido.")
            break

        stats["total"] += 1
        processed += 1

        if processed % 100 == 0:
            log.info(f"  [{processed}/{min(limite, len(ok_entries))}] processando...")

        records = process_single_pdf(
            file_id=fid,
            manifest_entry=entry,
            output_dir=output_dir,
            dedup_registry=dedup_registry,
            verbose_blocks=verbose_blocks,
        )

        if records:
            all_jsonl_records.extend(records)
            stats["gerados"] += len(records)
        else:
            # Pode ser duplicata ou falha
            if compute_content_md5("") != fid:  # Não é PDF vazio
                stats["duplicatas"] += 1

    # --- Salva JSONL ---
    if all_jsonl_records:
        jsonl_path = Path(jsonl_output)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        # JSONL: uma linha JSON por documento (sem metadados no campo text)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for record in all_jsonl_records:
                # Versão para fine-tuning: apenas texto puro
                clean_record = {"text": record["text"]}
                f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

        log.info(f"JSONL salvo em: {jsonl_path} ({len(all_jsonl_records)} linhas)")

        # JSONL com metadados (versão para RAG indexação)
        jsonl_meta_path = jsonl_path.with_suffix(".meta.jsonl")
        with open(jsonl_meta_path, "w", encoding="utf-8") as f:
            for record in all_jsonl_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        log.info(f"JSONL+meta salvo em: {jsonl_meta_path}")

    # --- Salva registro de deduplicação ---
    tmp_dedup = f"{dedup_path}.tmp"
    with open(tmp_dedup, "w", encoding="utf-8") as f:
        json.dump(dedup_registry, f, ensure_ascii=False, indent=2)
    os.replace(tmp_dedup, dedup_path)
    log.info(f"Registro de deduplicação atualizado: {len(dedup_registry)} hashes")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Processador de PDFs DOM-PI → Markdown + JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Caminho para o download_manifest.json gerado pelo download_pdfs.py."
    )
    parser.add_argument(
        "--output-dir", type=str, default="dados_brutos",
        help="Diretório base do Data Lake para os .md (padrão: dados_brutos)."
    )
    parser.add_argument(
        "--jsonl-output", type=str, default="corpus_treino_dompi.jsonl",
        help="Caminho do arquivo JSONL consolidado."
    )
    parser.add_argument(
        "--dedup-registry", type=str, default=None,
        help="Caminho do registro de deduplicação. Padrão: <output-dir>/registro_dedup.json"
    )
    parser.add_argument(
        "--limite", type=int, default=999999,
        help="Nº máximo de PDFs a processar."
    )
    parser.add_argument(
        "--verbose-blocos", action="store_true",
        help="Despeja análise detalhada de blocos visuais de CADA PDF processado."
    )
    parser.add_argument(
        "--verbose-blocos-only", type=int, default=None, metavar="N",
        help="Apenas despeja análise de N PDFs SEM processar. Para calibração de thresholds."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Ativa logs de debug."
    )

    args = parser.parse_args()
    _configure_logging(verbose=args.verbose)

    if not os.path.exists(args.manifest):
        log.error(f"Manifesto não encontrado: {args.manifest}")
        sys.exit(1)

    dedup_path = args.dedup_registry or os.path.join(args.output_dir, "registro_dedup.json")

    log.info("=" * 60)
    log.info("PIPELINE DE PROCESSAMENTO — PDF → Markdown + JSONL")
    log.info("=" * 60)
    log.info(f"Manifesto:      {args.manifest}")
    log.info(f"Data Lake:      {args.output_dir}")
    log.info(f"JSONL:          {args.jsonl_output}")
    log.info(f"Dedup Registry: {dedup_path}")
    log.info(f"Limite:         {args.limite}")
    if args.verbose_blocos:
        log.info("Modo VERBOSE-BLOCOS ativado (dump de análise visual)")
    if args.verbose_blocos_only is not None:
        log.info(f"Modo VERBOSE-BLOCOS-ONLY: analisando {args.verbose_blocos_only} PDFs (sem processamento)")
    log.info("-" * 60)

    stats = run_processing_pipeline(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        jsonl_output=args.jsonl_output,
        dedup_path=dedup_path,
        limite=args.limite,
        verbose_blocks=args.verbose_blocos,
        verbose_blocks_only=args.verbose_blocos_only,
    )

    # Resumo final
    print("\n" + "─" * 60)
    print(f"  ✔  Processamento concluído")
    print(f"  📊 Total processado:      {stats['total']}")
    print(f"  📝 Markdown gerados:      {stats['gerados']}")
    print(f"  ♻️  Duplicatas detectadas:  {stats['duplicatas']}")
    if stats.get('falhas'):
        print(f"  ❌ Falhas:                {stats['falhas']}")
    print(f"  📄 JSONL:                 {args.jsonl_output}")
    print(f"  🗂️  Data Lake:             {args.output_dir}/")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
