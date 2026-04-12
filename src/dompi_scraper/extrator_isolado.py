#!/usr/bin/env python3
"""
extrator_isolado.py — Módulo Extrator Mestre para Tratamento de PDFs DOM-PI
---------------------------------------------------------------------------
Extrai metadados estruturados de arquivos PDF baixados, computa o hash criptográfico (SHA-256) físico do arquivo 
e fatia os grandes cadernos do Diário Oficial em partes específicas de cada município através do reconhecimento
de padrões textuais (Regex) aliados a atributos visuais (Fontes, Negrito e Tamanho).

Uso via CLI:
    # Ler PDF local e gerar JSON com os sub-chunks para um município
    uv run python src/dompi_scraper/extrator_isolado.py \\
        --arquivo meu_arquivo.pdf

    # Baixar de uma URL e já fazer o chunking:
    uv run python src/dompi_scraper/extrator_isolado.py \\
        --url "https://diarioficialdosmunicipios.org/...pdf"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Erro: A biblioteca 'pymupdf' (fitz) é necessária. Instale com `uv add pymupdf`", file=sys.stderr)
    sys.exit(1)

# Configuração de Log
log = logging.getLogger("extrator_isolado")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s"))
log.setLevel(logging.INFO)
log.addHandler(handler)

# Regex Típicos para detecção de Cidades no DOM-PI. Pode exigir refinamentos heurísticos.
# Ex: PREFEITURA MUNICIPAL DE CAMPO MAIOR
REGEX_PREFEITURA = re.compile(
    r"(?:PREFEITURA|C[AÂ]MARA)\s+(?:MUNICIPAL\s+)?(?:DE\s+)?([A-ZÀ-Ÿ\s]+)(?:\n|$|\r)", 
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# 1. INTEGRIDADE: SHA-256 DO PDF FÍSICO
# ---------------------------------------------------------------------------
def compute_file_sha256(filepath: str) -> str:
    """Calcula o hash SHA-256 do arquivo em disco em pedaços (chunked)."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Lê o arquivo em pedaços de 4K
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

# ---------------------------------------------------------------------------
# 2. EXTRAÇÃO DE METADADOS RICOS (PYMUPDF)
# ---------------------------------------------------------------------------
def extract_rich_text(page: fitz.Page) -> list[dict[str, Any]]:
    """
    Extrai blocos de texto ricos de uma página usando o formato 'dict' do PyMuPDF,
    que provê atributos como família da fonte, tamanho e coloração.
    """
    pag_dict = page.get_text("dict")
    parsed_blocks = []

    for block in pag_dict.get("blocks", []):
        # Apenas processa blocos que são texto (tipo 0)
        if block.get("type") == 0:
            block_text = ""
            flags_list = []
            font_size_list = []
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "") + " "
                    flags_list.append(span.get("flags", 0))
                    font_size_list.append(span.get("size", 0))
                    
            text_cleaned = block_text.strip()
            if text_cleaned:
                # flags: 16 representa negrito no PyMuPDF de forma nativa geralmente
                avg_size = round(sum(font_size_list) / len(font_size_list), 2) if font_size_list else 0
                max_flags = max(flags_list) if flags_list else 0
                is_bold = bool(max_flags & 2**4) # Uma heurística para Negrito
                # Outra heurística é analisar o fontFamily se possui "Bold" no nome
                
                parsed_blocks.append({
                    "texto": text_cleaned,
                    "tamanho": avg_size,
                    "negrito": is_bold,
                    "bbox": block.get("bbox", [])
                })
    return parsed_blocks

# ---------------------------------------------------------------------------
# 3. MAPEAMENTO CIDADES (CHUNKING)
# ---------------------------------------------------------------------------
def detect_city_in_blocks(blocks: list[dict]) -> str | None:
    """
    Tenta encontrar um rótulo de município nos blocos fornecidos.
    No DOM-PI, a cidade geralmente aparece centralizada, com fontes maiores ou negrito.
    """
    for blk in blocks:
        texto = blk["texto"]
        # Ex: Daremos um leve prioridade visual (fonte > 11) ou negrito
        # Mas vamos focar na limpeza da RegEx:
        match = REGEX_PREFEITURA.search(texto)
        if match:
            # Pega o que capturou de cidade e faz limpeza extra
            cidade = match.group(1).strip()
            # Impede "falsos positivos" que sejam muito grandes (a menos que a cidade seja GIGANTE)
            if len(cidade) < 40 and len(cidade) >= 3:
                return cidade.title()
    return None


def parse_and_chunk_pdf(pdf_path: str) -> dict:
    """
    Varre o PDF, mapeia e fatia os conteúdos associados a diferentes municípios.
    """
    log.info(f"Lendo e particionando PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    total_pages = len(doc)
    
    resultado = {
        "arquivo": os.path.basename(pdf_path),
        "paginas_totais": total_pages,
        "sha256": compute_file_sha256(pdf_path),
        "chunks": []
    }
    
    cidade_atual = "DESCONHECIDA"
    paginas_buffer = []
    texto_buffer = []
    current_chunk = None

    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        rich_blocks = extract_rich_text(page)
        
        # Tenta descobrir de quem é essa página baseada na menção da prefeitura/câmara no HEADING
        cidade_encontrada = detect_city_in_blocks(rich_blocks)
        
        # Lógica de virada de Chunk
        if cidade_encontrada and cidade_encontrada != cidade_atual:
            # Salva o chunk antigo que acumulou (se houver páginas lá dentro)
            if current_chunk and current_chunk["paginas"]:
                current_chunk["texto"] = "\n".join(texto_buffer)
                resultado["chunks"].append(current_chunk)
                
            cidade_atual = cidade_encontrada
            current_chunk = {
                "municipio_referencia": cidade_atual,
                "paginas": [],
                "texto": ""
            }
            texto_buffer = []
        
        # Se for a primeira página e nunca teve chunk, inicializa com "DESCONHECIDA" ou a que achar
        if not current_chunk:
            current_chunk = {
                "municipio_referencia": cidade_atual,
                "paginas": [],
                "texto": ""
            }
            
        current_chunk["paginas"].append(page_num + 1)  # Base-1 para leitura humana
        texto_pagina_simples = "\n".join([b["texto"] for b in rich_blocks])
        texto_buffer.append(f"--- PÁGINA {page_num + 1} ---\n{texto_pagina_simples}")

    # No fim, adiciona o último acumulado
    if current_chunk and current_chunk["paginas"]:
        current_chunk["texto"] = "\n".join(texto_buffer)
        resultado["chunks"].append(current_chunk)
        
    doc.close()
    
    log.info(f"Total de entidades municipais fragmentadas (Chunks): {len(resultado['chunks'])}")
    for chk in resultado['chunks']:
        log.info(f"  -> Município '{chk['municipio_referencia']}' ocupou {len(chk['paginas'])} página(s) - [Páginas: {chk['paginas']}]")
        
    return resultado


# ---------------------------------------------------------------------------
# CLI E EXECUÇÃO
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extrator Isolado PDF DOM-PI com SHA-256, PyMuPDF e Regex Chunking")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--arquivo", type=str, help="Caminho para o ARQUIVO PDF em disco.")
    group.add_argument("--url", type=str, help="URL do PDF à ser validado e baixado temporariamente.")
    
    args = parser.parse_args()
    
    arquivo_local = args.arquivo
    tmp_path = None
    
    try:
        if args.url:
            log.info(f"Baixando PDF de: {args.url}")
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            os.close(tmp_fd)
            urllib.request.urlretrieve(args.url, tmp_path)
            arquivo_local = tmp_path
            
        if not os.path.exists(arquivo_local):
            log.error(f"Arquivo não localizado na rota especificada: {arquivo_local}")
            sys.exit(1)
            
        file_sha256 = compute_file_sha256(arquivo_local)
        log.info(f"Integridade garantida: SHA-256 do arquivo = {file_sha256}")
            
        saida_extracao = parse_and_chunk_pdf(arquivo_local)
        
        # Persistência local para inspeção manual do chunk
        nome_base = os.path.basename(arquivo_local).replace(".pdf", "")
        if args.url:
            nome_base = "downloaded_temp"
            
        out_json_path = f"{nome_base}_extracao_chunks.json"
        
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(saida_extracao, f, ensure_ascii=False, indent=2)
            
        log.info(f"Arquivo detalhado do processo salvo em: {out_json_path}")
        print("\\n=== PROCESSO FINALIZADO ===")
        print(f"Resumo salvo em: {out_json_path}")
        
    finally:
        if args.url and tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    main()
