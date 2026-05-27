#!/usr/bin/env python3
"""
limpeza_textos.py — Limpeza Avançada de Dados DOM-PI
------------------------------------------------------------------
Lê arquivos Markdown particionados de `dados_brutos/`, extrai e preserva o
frontmatter YAML, aplica rotinas de limpeza via regex no corpo do texto (removendo
lixo de OCR, espaços em excesso e padronizando quebras de linha) e salva o
resultado em `dados_limpos/`.

Uso:
    uv run python src/dompi_scraper/limpeza_textos.py --input-dir dados_brutos --output-dir dados_limpos --verbose
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

# Configuração de Logging com nível de detalhe dinâmico
log = logging.getLogger("limpeza_textos")

def _configure_logging(verbose: bool = False, debug: bool = False) -> None:
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    log.setLevel(level)
    log.handlers.clear()
    log.addHandler(handler)

# Expressões Regulares de Limpeza
RE_MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
RE_MULTIPLE_SPACES = re.compile(r' {2,}')
RE_OCR_GARBAGE = re.compile(r'[~€#&@!\[\]{}|<>\\]{2,}')
RE_ISOLATED_CHARS = re.compile(r'(?<!\w)([a-zA-Z])(?!\w)\s+(?=[a-zA-Z](?!\w))') # Letras isoladas: A B C D

# Novas expressões para remoção de ruído
RE_REPEATING_CHARS = re.compile(r'([=\-_.>~])\1{3,}') # =======, --------, etc
RE_HEADER_FOOTER = re.compile(r'(?i)(Diário Oficial dos Municípios|A prova documental dos atos municipais|Ano [IVXLCDM]+ «.*?Edição [IVXLCDM]+|Id: [A-F0-9]{16})')

# Heurísticas para flag de revisão humana
RE_SIGNATURE = re.compile(r'(?i)(PREFEITO(?: MUNICIPAL)?|SECRETARI[OA](?: MUNICIPAL)?|C[O]?TROLADOR(?:A)?(?: GERAL)?|\d{3}\.\d{3}\.\d{3}-\d{2})')
RE_POSSIBLE_TABLE = re.compile(r'(\d{1,3}(?:\.\d{3})*,\d{2}\s+){1,}\d{1,3}(?:\.\d{3})*,\d{2}') # 2 ou mais valores monetários na linha
RE_PIPE_TABLE = re.compile(r'\|.*\|')

def parse_markdown(content: str) -> tuple[str, str]:
    """
    Separa o YAML frontmatter do corpo do texto.
    """
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            frontmatter = f"---{parts[1]}---"
            body = parts[2]
            return frontmatter, body
    return "", content

def append_review_flags(frontmatter: str, reasons: list) -> str:
    """Adiciona flags de revisão no frontmatter YAML."""
    if not frontmatter or not reasons:
        return frontmatter
    
    parts = frontmatter.rsplit('---', 1)
    if len(parts) == 2:
        reasons_str = ", ".join(reasons)
        injection = f"needs_human_review: true\nreview_reasons: \"{reasons_str}\"\n---"
        return parts[0] + injection
    return frontmatter

def clean_text(text: str) -> tuple[str, dict]:
    """
    Aplica rotinas de higienização de texto e retorna o texto limpo
    junto com estatísticas das modificações feitas (para logs de debug).
    """
    stats = {
        "len_before": len(text),
        "removed_garbage": 0,
        "reduced_newlines": 0,
        "reduced_spaces": 0,
        "review_reasons": set()
    }
    
    # Heurísticas de detecção ANTES de remover muito do texto
    if RE_SIGNATURE.search(text):
        stats["review_reasons"].add("assinaturas_detectadas")
    if RE_POSSIBLE_TABLE.search(text) or RE_PIPE_TABLE.search(text):
        stats["review_reasons"].add("tabela_achatada_detectada")
        
    # Remover cabeçalhos e rodapés repetitivos
    text = RE_HEADER_FOOTER.sub('', text)
    
    # Remover caracteres repetidos (ex: ======, -----)
    text, n_rep = RE_REPEATING_CHARS.subn('', text)
    
    # Remover lixo de OCR
    text, n = RE_OCR_GARBAGE.subn('', text)
    stats["removed_garbage"] = n + n_rep
    
    # Juntar letras isoladas (A B C -> ABC)
    text = RE_ISOLATED_CHARS.sub(r'\1', text)
    
    # Padronizar espaços em branco (exceto quebras de linha que cuidaremos depois)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.strip():
            # Reduz múltiplos espaços para um só
            cl, n2 = RE_MULTIPLE_SPACES.subn(' ', line)
            stats["reduced_spaces"] += n2
            cleaned_lines.append(cl.strip())
        else:
            cleaned_lines.append('')
    
    text = '\n'.join(cleaned_lines)
    
    # Reduzir múltiplas quebras de linha vazias para no máximo uma (duas \n consecutivas)
    text, n3 = RE_MULTIPLE_NEWLINES.subn('\n\n', text)
    stats["reduced_newlines"] = n3
    
    if stats["removed_garbage"] > 20:
        stats["review_reasons"].add("alto_indice_ruido_ocr")
        
    text = text.strip() + '\n'
    stats["len_after"] = len(text)
    
    # Converte o set para list para facilitar manuseio posterior
    stats["review_reasons"] = list(stats["review_reasons"])
    
    return text, stats

def process_directory(input_dir: Path, output_dir: Path) -> None:
    """
    Varre recursivamente o diretório de entrada, limpa os arquivos .md e
    salva a estrutura espelhada no diretório de saída.
    """
    if not input_dir.exists():
        log.error(f"Diretório de entrada não encontrado: {input_dir}")
        sys.exit(1)
        
    log.info(f"Iniciando varredura em {input_dir}")
    md_files = list(input_dir.rglob('*.md'))
    total_files = len(md_files)
    
    log.info(f"Encontrados {total_files} arquivos Markdown para processamento.")
    
    processed_count = 0
    error_count = 0
    total_bytes_saved = 0
    
    for md_file in md_files:
        try:
            rel_path = md_file.relative_to(input_dir)
            out_file = output_dir / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            frontmatter, body = parse_markdown(content)
            
            if not body.strip():
                log.debug(f"[{rel_path}] Ignorado: Corpo de texto vazio após frontmatter.")
                continue
                
            cleaned_body, stats = clean_text(body)
            
            if stats.get("review_reasons"):
                frontmatter = append_review_flags(frontmatter, stats["review_reasons"])
                log.warning(f"[{rel_path}] Marcado para revisão humana: {', '.join(stats['review_reasons'])}")
            
            final_content = frontmatter + "\n" + cleaned_body if frontmatter else cleaned_body
            
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(final_content)
                
            bytes_saved = stats["len_before"] - stats["len_after"]
            total_bytes_saved += bytes_saved
            
            log.debug(f"[{rel_path}] Processado | OCR Lixo Removido: {stats['removed_garbage']} | Espaços Reduzidos: {stats['reduced_spaces']} | Redução: {bytes_saved} chars")
            processed_count += 1
            
            if processed_count % 500 == 0:
                log.info(f"Progresso: {processed_count}/{total_files} arquivos limpos...")
                
        except Exception as e:
            log.error(f"[{md_file}] Falha no processamento: {e}")
            error_count += 1
            
    log.info("="*60)
    log.info("RESUMO DA LIMPEZA")
    log.info("="*60)
    log.info(f"Arquivos processados: {processed_count}/{total_files}")
    log.info(f"Erros encontrados: {error_count}")
    log.info(f"Caracteres irrelevantes removidos no total: {total_bytes_saved}")
    log.info(f"Saída salva em: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Limpeza avançada de textos extraídos DOM-PI")
    parser.add_argument("--input-dir", type=str, default="dados_brutos", help="Diretório contendo os markdowns brutos")
    parser.add_argument("--output-dir", type=str, default="dados_limpos", help="Diretório onde salvar os markdowns limpos")
    parser.add_argument("--verbose", action="store_true", help="Mostra logs de informação")
    parser.add_argument("--debug", action="store_true", help="Mostra logs de debug detalhados por arquivo")
    
    args = parser.parse_args()
    _configure_logging(args.verbose, args.debug)
    
    input_p = Path(args.input_dir)
    output_p = Path(args.output_dir)
    
    process_directory(input_p, output_p)

if __name__ == "__main__":
    main()
