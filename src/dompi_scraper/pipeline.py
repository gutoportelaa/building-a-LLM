#!/usr/bin/env python3
"""
Pipeline Central do Scraper DOM-PI (Diário Oficial dos Municípios do Piauí)
---------------------------------------------------------------------------
Arquitetado nativamente em SQLite. Otimizado para alimentar Foundational Models, LLMs e RAG.

Fluxo:
1. Coleta iterativa dos dados passando por Cidades e Entidades (Limitado p/ testes).
2. Parsing e Hash: O documento gera Metadados Estendidos lendo as amarras da URL Oculta.
3. Deduplicação Física (dim_documentos_pdf): Garante 1 arquivo = 1 Download.
4. Deduplicação Textual (dim_extracoes_texto): Garante Texto Limpo para Treino sem Repetições de Hash.
5. População Estrutural: A persistência interliga PKs nas tabelas base.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from urllib.parse import urljoin

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Erro: A biblioteca 'beautifulsoup4' é necessária.")
    raise

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

# Dependências locais utilitárias
try:
    from .shared_utils import normalize_spaces, slugify
except ImportError:
    from shared_utils import normalize_spaces, slugify


BASE_URL = "https://www.diarioficialdosmunicipios.org"
SEARCH_URL = f"{BASE_URL}/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php"

CARNAUBAIS_MUNICIPIOS = [
    "Assuncao do Pi", "Boa Hora", "Boqueirao do Pi", "Buriti dos Montes",
    "Cabeceiras do Pi", "Campo Maior", "Capitao de Campos", "Castelo do Pi",
    "Cocal de Telha", "Jatoba do Pi", "Juazeiro do Pi", "Nossa Senhora de Nazare",
    "Novo Santo Antonio", "Sao Joao da Serra", "Sao Miguel do Tapuio", "Sigefredo Pacheco",
]

ENTIDADES_PADRAO = ["Prefeitura", "Camara"]

log = logging.getLogger("dompi_pipeline")

def configure_logging(log_file: str) -> None:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log.addHandler(file_handler)
    log.addHandler(stream_handler)


# ==============================================================================
# SECÃO 1: BANCO DE DADOS (SQLite)
# ==============================================================================

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Metadados das publicações governamentais.
    cur.execute('''
    CREATE TABLE IF NOT EXISTS fato_publicacoes (
        publicacao_id TEXT PRIMARY KEY,
        municipio TEXT,
        entidade TEXT,
        categoria TEXT,
        documento_resumo TEXT,
        identificador_oficial TEXT,
        data_publicacao TEXT,
        edicao TEXT,
        url_arquivo_metadata TEXT,
        pagina_codigo_metadata TEXT,
        documento_pdf_id TEXT,
        FOREIGN KEY (documento_pdf_id) REFERENCES dim_documentos_pdf(documento_pdf_id)
    )
    ''')

    # Central de Blobs e Arquivos únicos armazenados na nuvem/HD.
    cur.execute('''
    CREATE TABLE IF NOT EXISTS dim_documentos_pdf (
        documento_pdf_id TEXT PRIMARY KEY,
        url_origem TEXT,
        status_download TEXT,
        path_local TEXT,
        tamanho_bytes INTEGER
    )
    ''')

    # DataGrid crucial para o modelo RAG: Apenas textos únicos! Evita poluir o treinamento.
    cur.execute('''
    CREATE TABLE IF NOT EXISTS dim_extracoes_texto (
        texto_id TEXT PRIMARY KEY,
        documento_pdf_id TEXT,
        motor_extrator TEXT,
        conteudo_raw TEXT,
        FOREIGN KEY (documento_pdf_id) REFERENCES dim_documentos_pdf(documento_pdf_id)
    )
    ''')

    conn.commit()
    return conn

def hash_string(text: str) -> str:
    """Retorna MD5 universal em hexadecimal para strings (URLs, textos extrídos)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def normalize_text_for_hash(raw_text: str) -> str:
    """Padroniza minúsculas e espaços. Textos literais iguais rodarão para o mesmo Hashlif."""
    cl = re.sub(r"\s+", " ", raw_text).strip().lower()
    return cl

# ==============================================================================
# SECÃO 2: UTILITÁRIOS E EXTRAÇÕES MÍNIMAS
# ==============================================================================

def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value or "")
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")

def normalize_lookup_key(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", strip_accents(value or ""))
    return re.sub(r"\s+", " ", cleaned).strip().lower()

def extract_metadata_from_url(url: str) -> dict:
    """Inspeçiona URLs como /pdfs_novo/5230/DM_5230_091_Campo_Maior_Portaria_pag_469.pdf"""
    meta = {'pagina': '', 'codigo_interno_mun': '', 'edicao_url': ''}
    if not url: return meta
    
    pag_match = re.search(r'pag_{0,1}(\d+)\.pdf', url, re.IGNORECASE)
    if pag_match:
        meta['pagina'] = pag_match.group(1)
        
    cod_match = re.search(r'DM_(\d+)_(\d+)_', url, re.IGNORECASE)
    if cod_match:
        meta['edicao_url'] = cod_match.group(1)
        meta['codigo_interno_mun'] = cod_match.group(2)
        
    return meta

# ==============================================================================
# SECÃO 3: REQUESTS & BEAUTIFULSOUP (CRAWLING)
# ==============================================================================

def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36"),
        "Referer": SEARCH_URL,
    })
    return session

def resolve_select_value(requested_value: str, options: dict[str, str]) -> str:
    requested = (requested_value or "").strip()
    if requested in options.values(): return requested

    normalized = normalize_lookup_key(requested)
    if normalized in options: return options[normalized]

    return requested

def parse_select_options(select_tag: object | None) -> dict[str, str]:
    opts = {}
    if not select_tag: return opts
    for op in getattr(select_tag, "find_all", lambda *_: [])("option"):
        val = str(op.get("value", "")).strip()
        txt = op.get_text(" ", strip=True) 
        if val:
            opts[normalize_lookup_key(txt)] = val
            opts[normalize_lookup_key(val)] = val
    return opts

def load_form_context(session: requests.Session) -> dict[str, object]:
    ctx = {"hidden_fields": {}, "municipio_options": {}, "entidade_options": {}, "error": ""}
    try:
        resp = session.get(SEARCH_URL, timeout=30)
        resp.raise_for_status() 
    except Exception as exc:
        ctx["error"] = str(exc)
        return ctx

    soup = BeautifulSoup(resp.text, "html.parser")
    form = soup.find("form", {"name": "F1"})
    
    if form:
        for inp in form.find_all("input"):
            if str(inp.get("type", "")).lower() == "hidden":
                ctx["hidden_fields"][str(inp.get("name", ""))] = str(inp.get("value", ""))
        
        ctx["municipio_options"] = parse_select_options(form.find("select", {"name": "nomemunicipio"}))
        ctx["entidade_options"] = parse_select_options(form.find("select", {"name": "nomeentidade"}))
    return ctx

def split_date(d: str):
    m = re.match(r"^\s*(\d{2})/(\d{2})/(\d{4})\s*$", d or "")
    if m: return m.group(1), m.group(2), m.group(3)
    return "", "", ""

def fetch_search_results(session, ctx, mun, endt, di="01/01/2025", df="31/12/2025", p=1, last_html=""):
    hid = ctx.get("hidden_fields", {})
    e_opts = ctx.get("entidade_options", {})
    m_opts = ctx.get("municipio_options", {})

    if p > 1 and last_html:
        soup = BeautifulSoup(last_html, "html.parser")
        f4 = soup.find("form", {"name": "F4"})
        if f4:
            payload2 = {i.get("name"): i.get("value") for i in f4.find_all("input", type="hidden")}
            payload2["nmgp_opcao"] = "rec"
            payload2["rec"] = str((p - 1) * 10 + 1)
            try:
                r = session.post(SEARCH_URL, data=payload2, timeout=30)
                r.raise_for_status()
                return r.text
            except Exception as e:
                log.error(f"Erro em Paginação Mun: {mun} | Entity: {endt} | Pág: {p} | Erro: {e}")
                return None

    d1, m1, y1 = split_date(di)
    d2, m2, y2 = split_date(df)

    payload = {k: v for k, v in hid.items() if isinstance(hid, dict)}
    payload.update({
        "nomeentidade": resolve_select_value(endt, e_opts if isinstance(e_opts, dict) else {}),
        "nomemunicipio": resolve_select_value(mun, m_opts if isinstance(m_opts, dict) else {}),
        "data_dia_de": di, "data_dia_ate": df, "data_dia": d1, "data_mes": m1, "data_ano": y1,
        "data_input_2_dia": d2, "data_input_2_mes": m2, "data_input_2_ano": y2,
        "bprocessa": "pesq", "nmgp_opcao": "busca", "nmsc_pag_ConPublicacaoGeral": str(p),
    })

    try:
        r = session.post(SEARCH_URL, data=payload, timeout=30)
        r.raise_for_status()
        ifr = re.search(r"src\s*=\s*'([^']+)'", r.text)
        if ifr:
            return session.get(urljoin(SEARCH_URL, ifr.group(1)), timeout=30).text
        return r.text
    except Exception as e:
        log.error(f"Erro em Mun: {mun} | Entity: {endt} | Erro: {e}")
        return None

def extract_pdf_url_from_href(href: str) -> str:
    cleaned = (href or "").strip().strip('"').strip("'")
    if "nm_gp_submit5" in cleaned:
        m = re.search(r"nm_gp_submit5\('([^']+)'", cleaned)
        if m: 
            u = m.group(1).replace("\\/", "/").strip()
            return u.split("?#?", 1)[0] if "?#?" in u else u
    low = cleaned.lower()
    if low.startswith("javascript:"): return ""
    if ".pdf" in low: return urljoin(BASE_URL, cleaned.strip())
    return ""

def get_total_pages(html: str) -> tuple[int, int]:
    txt = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    m = re.search(r"(\d+)\s*a\s*(\d+)\s*de\s*(\d+)", txt)
    if m:
        total_docs = int(m.group(3))
        total_pags = max(1, (total_docs + 10 - 1) // 10)
        return total_pags, total_docs
    return 1, 0

def parse_results_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    regs = []
    
    for span in soup.find_all("span", id=re.compile(r"^id_sc_field_codigo_\d+$")):
        s_id = str(span.get("id"))
        m_suf = re.search(r"_(\d+)$", s_id)
        if not m_suf: continue
        suf = m_suf.group(1)
        row = span.find_parent("tr")
        if not row: continue

        def xt(fld):
            el = soup.find("span", {"id": f"id_sc_field_{fld}_{suf}"})
            return el.get_text(" ", strip=True) if el else ""

        # Extrai todas as URLs dos links da linha
        cands = [extract_pdf_url_from_href(str(a.get("href"))) for a in row.find_all("a", href=True)]
        cands = [c for c in cands if c]
        
        # Prioriza URLs que pareçam "completas" (com detalhes descritivos e número de página)
        # URLs completas têm padrão: DM_XXXX_NNN_municipio_tipo_descricao_pag_NNN.pdf
        urls_detalhadas = [u for u in cands if "pag_" in u.lower() and u.count("_") >= 4]
        url_final = urls_detalhadas[0] if urls_detalhadas else (cands[0] if cands else "")

        regs.append({
            "edicao": xt("numedicao"),
            "data": xt("data"),
            "municipio": xt("nomemunicipio"),
            "entidade": xt("nomeentidade"),
            "categoria": xt("nomecategoria"),
            "documento": xt("nomedoc"),
            "identificador": xt("codigo"),
            "pdf_url": url_final, 
        })
    return regs


def download_pdf(session: requests.Session, url: str, path: str) -> bool:
    if not url: return False
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    try:
        r = session.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            for c in r.iter_content(chunk_size=8192):
                if c: f.write(c)
        return True
    except Exception as e:
        log.warning(f"  DL Fail '{url}': {e}")
        return False

# ==============================================================================
# SECÃO 4: ORQUESTRAÇÃO RELACIONAL E ML DATABASE POPULATOR
# ==============================================================================

def pipeline_main(mun_list: list, entidades: list, d_in: str, d_fif: str, out_dir: str, limit_per_entity: int = 5):
    log.info(f"== INICIANDO O PIPELINE RAG (Lim.: {limit_per_entity} por cruzamento) ==")
    
    os.makedirs(out_dir, exist_ok=True)
    pdf_dir = os.path.join(out_dir, "pdfs_arquivos")
    os.makedirs(pdf_dir, exist_ok=True)
    
    db_path = os.path.join(out_dir, "dompi_knowledge_base.sqlite")
    conn = init_db(db_path)
    cur = conn.cursor()
    
    session = build_session()
    ctx = load_form_context(session)
    if ctx.get("error"):
         log.error(f"Fatal Loading Form: {ctx['error']}")
         return

    for mun in mun_list:
        for ent in entidades:
            log.info(f" -> Avaliando: [Cidade: {mun}] | [Entidade: {ent}]")
            html1 = fetch_search_results(session, ctx, mun, ent, d_in, d_fif, p=1)
            if not html1: continue
            
            pags, max_docs = get_total_pages(html1)
            docs_salvos = 0
            
            # Navegar nas páginas apenas até atingir o limite estipulado
            for pag in range(1, pags + 1):
                if docs_salvos >= limit_per_entity: break
                
                if pag > 1:
                    time.sleep(1.0)
                    html1 = fetch_search_results(session, ctx, mun, ent, d_in, d_fif, p=pag, last_html=html1)
                    if not html1: continue
                
                regs = parse_results_page(html1)
                for reg in regs:
                    if docs_salvos >= limit_per_entity: break
                    
                    p_url = reg["pdf_url"]
                    if not p_url: continue

                    # 1. Deduplicação do Arquivo (PDF_ID) por Hashes de URL.
                    doc_pdf_id = hash_string(p_url)
                    url_meta = extract_metadata_from_url(p_url)

                    # Verificar se o PDF já foi persistido no Banco:
                    cur.execute("SELECT path_local FROM dim_documentos_pdf WHERE documento_pdf_id = ?", (doc_pdf_id,))
                    row_pdf = cur.fetchone()

                    path_pdf = None
                    if row_pdf:
                        path_pdf = row_pdf[0]
                    else:
                        path_pdf = os.path.join(pdf_dir, f"{doc_pdf_id}.pdf")
                        dl_ok = download_pdf(session, p_url, path_pdf)
                        sz = os.path.getsize(path_pdf) if dl_ok else 0
                        st = "OK" if dl_ok else "FAILED"
                        
                        cur.execute('''INSERT OR IGNORE INTO dim_documentos_pdf 
                                       (documento_pdf_id, url_origem, status_download, path_local, tamanho_bytes)
                                       VALUES (?, ?, ?, ?, ?)''', 
                                    (doc_pdf_id, p_url, st, path_pdf, sz))

                    # 2. Textual Deduplication e Extracao (OCR/Markdown)
                    if path_pdf and os.path.exists(path_pdf) and MarkItDown:
                        # Checa se esse arquivo em específico já teve o texto extraído para não re-converter à toa.
                        cur.execute("SELECT texto_id FROM dim_extracoes_texto WHERE documento_pdf_id = ?", (doc_pdf_id,))
                        if not cur.fetchone():
                            try:
                                md_cv = MarkItDown()
                                res = md_cv.convert(path_pdf)
                                raw_txt = getattr(res, "text_content", "")
                                
                                if raw_txt:
                                    norm_txt = normalize_text_for_hash(raw_txt)
                                    txt_hash_id = hash_string(norm_txt)
                                    
                                    # Grava com Insert OR IGNORE, garantindo a Deduplicação Absoluta de Textos Iguais!
                                    cur.execute('''INSERT OR IGNORE INTO dim_extracoes_texto
                                                   (texto_id, documento_pdf_id, motor_extrator, conteudo_raw)
                                                   VALUES (?, ?, ?, ?)''',
                                                (txt_hash_id, doc_pdf_id, "MarkItDown", raw_txt))
                            except Exception as e:
                                log.warning(f"      Falha na extração (MarkItDown) -> {doc_pdf_id}: {e}")

                    # 3. Registrar a Ocorrência na Tabela Fato
                    # ID = Identificador + Entidade Original Garantida + URL do documento
                    row_id = hash_string(f"{reg['identificador']}_{ent}_{p_url}")
                    
                    cur.execute('''INSERT OR IGNORE INTO fato_publicacoes
                                   (publicacao_id, municipio, entidade, categoria, documento_resumo,
                                    identificador_oficial, data_publicacao, edicao,
                                    url_arquivo_metadata, pagina_codigo_metadata, documento_pdf_id)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                (row_id, mun, ent, reg['categoria'], reg['documento'],
                                 reg['identificador'], reg['data'], reg['edicao'],
                                 p_url, url_meta['pagina'], doc_pdf_id))
                    
                    conn.commit()
                    docs_salvos += 1
            
            log.info(f"    - Salvos {docs_salvos} registros para {mun} / {ent}.")

    conn.close()
    log.info("== PREPARAÇÃO DO DATASET LLM CONCLUÍDA ==")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--territorio-carnaubais", action="store_true", help="Alvo nos 16 Muns.")
    parser.add_argument("--municipio", type=str, default="", help="Aplicável para testes isolados.")
    parser.add_argument("--limite", type=int, default=5, help="Qtd. max docs P/ Entidade P/ Muni. Ex: 5")
    parser.add_argument("--outdir", type=str, default="./db_treino_carnaubais", help="Onde reside o SQLite")
    
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    configure_logging(os.path.join(args.outdir, "log_treino.log"))
    
    muns = CARNAUBAIS_MUNICIPIOS if args.territorio_carnaubais else [args.municipio]
    if not muns[0]: return print("Forneça --territorio-carnaubais ou --municipio")

    pipeline_main(muns, ENTIDADES_PADRAO, "01/01/2025", "31/12/2025", args.outdir, limit_per_entity=args.limite)

if __name__ == "__main__":
    main()
