#!/usr/bin/env python3
"""
pipeline.py — Scraper DOM-PI Otimizado (Etapa 1: Coleta de Metadados)
----------------------------------------------------------------------
Coleta metadados das publicações do DOM-PI via ThreadPoolExecutor (paralelo).
Sem SQLite — persistência exclusivamente em JSON/CSV.

Saídas:
  <saida>.json                 — Todos os registros coletados
  <saida>_deduplicados.json    — PDFs únicos (entrada para download_pdfs.py)
  <saida>.csv (opcional)

Uso:
    uv run python src/dompi_scraper/pipeline.py \\
        --territorio-carnaubais --ano 2025 --max-workers 15 --verbose

    uv run python src/dompi_scraper/pipeline.py \\
        --municipio "Campo Maior" --ano 2025 --max-workers 5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Erro: 'beautifulsoup4' necessário. Instale com: uv add beautifulsoup4")
    sys.exit(1)

try:
    from .shared_utils import normalize_spaces, slugify
except ImportError:
    from shared_utils import normalize_spaces, slugify

# ==============================================================================
# CONSTANTES
# ==============================================================================

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


# ==============================================================================
# LOGGING
# ==============================================================================

def configure_logging(verbose: bool = False, log_file: str | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log.setLevel(level)
    log.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(level)
    log.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(level)
        log.addHandler(fh)
        log.info(f"Log em arquivo: {log_file}")


# ==============================================================================
# UTILITÁRIOS
# ==============================================================================

def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value or "")
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _normalize_key(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", _strip_accents(value or ""))
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def extract_metadata_from_url(url: str) -> dict:
    """Extrai metadados embutidos na URL do PDF (edição, código municipal, página)."""
    meta = {"pagina": "", "codigo_interno_mun": "", "edicao_url": ""}
    if not url:
        return meta
    pag = re.search(r"pag_{0,1}(\d+)\.pdf", url, re.IGNORECASE)
    if pag:
        meta["pagina"] = pag.group(1)
    cod = re.search(r"DM_(\d+)_(\d+)_", url, re.IGNORECASE)
    if cod:
        meta["edicao_url"] = cod.group(1)
        meta["codigo_interno_mun"] = cod.group(2)
    return meta


# ==============================================================================
# SESSÃO HTTP
# ==============================================================================

def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36",
        "Referer": SEARCH_URL,
    })
    return session


# ==============================================================================
# PARSING DO FORMULÁRIO DOM-PI
# ==============================================================================

def _parse_select_options(select_tag) -> dict[str, str]:
    opts = {}
    if not select_tag:
        return opts
    for op in getattr(select_tag, "find_all", lambda *_: [])("option"):
        val = str(op.get("value", "")).strip()
        txt = op.get_text(" ", strip=True)
        if val:
            opts[_normalize_key(txt)] = val
            opts[_normalize_key(val)] = val
    return opts


def resolve_select_value(requested: str, options: dict[str, str]) -> str:
    req = (requested or "").strip()
    if req in options.values():
        return req
    nk = _normalize_key(req)
    if nk in options:
        return options[nk]
    return req


def load_form_context(session: requests.Session) -> dict:
    """Carrega campos hidden e opções de select do formulário de busca."""
    ctx: dict = {"hidden_fields": {}, "municipio_options": {}, "entidade_options": {}, "error": ""}
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
        ctx["municipio_options"] = _parse_select_options(form.find("select", {"name": "nomemunicipio"}))
        ctx["entidade_options"] = _parse_select_options(form.find("select", {"name": "nomeentidade"}))
    return ctx


# ==============================================================================
# REQUISIÇÕES DE BUSCA
# ==============================================================================

def _split_date(d: str) -> tuple[str, str, str]:
    m = re.match(r"^\s*(\d{2})/(\d{2})/(\d{4})\s*$", d or "")
    if m:
        return m.group(1), m.group(2), m.group(3)
    return "", "", ""


def fetch_search_results(
    session: requests.Session,
    ctx: dict,
    mun: str,
    endt: str,
    di: str = "01/01/2025",
    df: str = "31/12/2025",
    p: int = 1,
    last_html: str = "",
) -> str | None:
    """
    Busca resultados do portal DOM-PI para [município × entidade × página].
    Para páginas 2+, extrai payload do formulário F4 da página anterior (correto).
    """
    hid = ctx.get("hidden_fields", {})
    e_opts = ctx.get("entidade_options", {})
    m_opts = ctx.get("municipio_options", {})

    # Paginação: extrai form F4 do HTML anterior (mantém estado da sessão no servidor)
    if p > 1 and last_html:
        soup = BeautifulSoup(last_html, "html.parser")
        f4 = soup.find("form", {"name": "F4"})
        if f4:
            payload_pag = {
                i.get("name"): i.get("value")
                for i in f4.find_all("input", type="hidden")
                if i.get("name")
            }
            payload_pag["nmgp_opcao"] = "rec"
            payload_pag["rec"] = str((p - 1) * 10 + 1)
            try:
                r = session.post(SEARCH_URL, data=payload_pag, timeout=30)
                r.raise_for_status()
                return r.text
            except Exception as e:
                log.error(f"Erro paginação {mun}/{endt} pág {p}: {e}")
                return None

    d1, m1, y1 = _split_date(di)
    d2, m2, y2 = _split_date(df)
    payload = {k: v for k, v in hid.items()}
    payload.update({
        "nomeentidade": resolve_select_value(endt, e_opts),
        "nomemunicipio": resolve_select_value(mun, m_opts),
        "data_dia_de": di, "data_dia_ate": df,
        "data_dia": d1, "data_mes": m1, "data_ano": y1,
        "data_input_2_dia": d2, "data_input_2_mes": m2, "data_input_2_ano": y2,
        "bprocessa": "pesq", "nmgp_opcao": "busca", "nmsc_pag_ConPublicacaoGeral": "1",
    })
    try:
        r = session.post(SEARCH_URL, data=payload, timeout=30)
        r.raise_for_status()
        ifr = re.search(r"src\s*=\s*'([^']+)'", r.text)
        if ifr:
            return session.get(urljoin(SEARCH_URL, ifr.group(1)), timeout=30).text
        return r.text
    except Exception as e:
        log.error(f"Erro busca {mun}/{endt}: {e}")
        return None


# ==============================================================================
# PARSING DE RESULTADOS
# ==============================================================================

def get_total_pages(html: str) -> tuple[int, int]:
    txt = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    m = re.search(r"(\d+)\s*a\s*(\d+)\s*de\s*(\d+)", txt)
    if m:
        total_docs = int(m.group(3))
        return max(1, (total_docs + 9) // 10), total_docs
    return 1, 0


def extract_pdf_url_from_href(href: str) -> str:
    cleaned = (href or "").strip().strip('"').strip("'")
    if "nm_gp_submit5" in cleaned:
        m = re.search(r"nm_gp_submit5\('([^']+)'", cleaned)
        if m:
            u = m.group(1).replace("\\/", "/").strip()
            return u.split("?#?", 1)[0] if "?#?" in u else u
    low = cleaned.lower()
    if low.startswith("javascript:"):
        return ""
    if ".pdf" in low:
        return urljoin(BASE_URL, cleaned.strip())
    return ""


def parse_results_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    regs = []
    for span in soup.find_all("span", id=re.compile(r"^id_sc_field_codigo_\d+$")):
        s_id = str(span.get("id"))
        m_suf = re.search(r"_(\d+)$", s_id)
        if not m_suf:
            continue
        suf = m_suf.group(1)
        row = span.find_parent("tr")
        if not row:
            continue

        def xt(fld: str) -> str:
            el = soup.find("span", {"id": f"id_sc_field_{fld}_{suf}"})
            return el.get_text(" ", strip=True) if el else ""

        cands = [extract_pdf_url_from_href(str(a.get("href"))) for a in row.find_all("a", href=True)]
        cands = [c for c in cands if c]
        urls_det = [u for u in cands if "pag_" in u.lower() and u.count("_") >= 4]
        url_final = urls_det[0] if urls_det else (cands[0] if cands else "")
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


# ==============================================================================
# SCRAPER COM THREADPOOL
# ==============================================================================

class ThreadPoolScraper:
    """
    Orquestra scraping paralelo de [município × entidade] via ThreadPoolExecutor.
    Cada thread processa um cruzamento de forma independente (sessão + contexto
    próprios), com paginação correta via form F4 extraído do HTML anterior.
    """

    def __init__(self, max_workers: int = 15, max_retries: int = 3, backoff_base: float = 2.0):
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.stats = {
            "cruzamentos_ok": 0,
            "cruzamentos_falha": 0,
            "documentos_coletados": 0,
            "tempo_inicio": None,
            "tempo_fim": None,
        }
        log.info(f"ThreadPoolScraper: max_workers={max_workers} | max_retries={max_retries}")

    def _scrape_cruzamento(
        self, mun: str, ent: str, d_ini: str, d_fim: str, limit: int
    ) -> tuple[list[dict], dict]:
        """
        Coleta publicações de [município × entidade] com paginação correta (form F4).
        Cria sessão própria por thread — necessário pois hidden_fields são por sessão.
        """
        session = build_session()
        ctx = None
        for attempt in range(1, self.max_retries + 1):
            ctx = load_form_context(session)
            if not ctx.get("error"):
                break
            wait = self.backoff_base ** attempt
            log.warning(f"  Contexto {mun}/{ent} tentativa {attempt}/{self.max_retries}, retry em {wait:.0f}s")
            time.sleep(wait)

        if not ctx or ctx.get("error"):
            log.error(f"Falha definitiva ao carregar contexto: {mun}/{ent}")
            return [], {"municipio": mun, "entidade": ent, "status": "FALHA_CONTEXTO", "coletados": 0}

        registros: list[dict] = []
        html_anterior = ""
        ano = int(d_ini[-4:])

        html1 = fetch_search_results(session, ctx, mun, ent, d_ini, d_fim, p=1)
        if not html1:
            return [], {"municipio": mun, "entidade": ent, "status": "FALHA_CONEXAO", "coletados": 0}

        total_pags, total_docs = get_total_pages(html1)
        log.info(f"  {mun}/{ent}: {total_docs} docs em {total_pags} pág(s)")

        if total_docs == 0:
            return [], {"municipio": mun, "entidade": ent, "status": "VAZIO", "coletados": 0}

        def _processar_pagina(html: str) -> None:
            for reg in parse_results_page(html):
                if len(registros) >= limit:
                    return
                if not reg.get("pdf_url"):
                    continue
                url_meta = extract_metadata_from_url(reg["pdf_url"])
                registros.append({
                    "municipio": mun,
                    "entidade": ent,
                    "ano_consulta": ano,
                    "data_publicacao": reg.get("data", ""),
                    "edicao": reg.get("edicao", ""),
                    "categoria": reg.get("categoria", ""),
                    "documento": reg.get("documento", ""),
                    "identificador_oficial": reg.get("identificador", ""),
                    "pdf_url": reg["pdf_url"],
                    "pagina_url_meta": url_meta.get("pagina", ""),
                    "edicao_url_meta": url_meta.get("edicao_url", ""),
                    "codigo_interno_mun_meta": url_meta.get("codigo_interno_mun", ""),
                    "coletado_em": datetime.now().isoformat(timespec="seconds"),
                })

        _processar_pagina(html1)
        html_anterior = html1

        for pag in range(2, total_pags + 1):
            if len(registros) >= limit:
                break
            time.sleep(0.3)
            html_pag = fetch_search_results(session, ctx, mun, ent, d_ini, d_fim, p=pag, last_html=html_anterior)
            if not html_pag:
                log.warning(f"  Falha pág {pag}/{total_pags}: {mun}/{ent}")
                continue
            _processar_pagina(html_pag)
            html_anterior = html_pag

        log.info(f"  ✅ {mun}/{ent}: {len(registros)} registros coletados")
        return registros, {"municipio": mun, "entidade": ent, "status": "OK", "coletados": len(registros)}

    def scrape_territorio(
        self,
        mun_list: list[str],
        entidades: list[str],
        ano: int,
        limit: int,
    ) -> tuple[list[dict], list[dict]]:
        """Executa scraping paralelo de todos os cruzamentos [município × entidade]."""
        self.stats["tempo_inicio"] = datetime.now()
        d_ini = f"01/01/{ano}"
        d_fim = f"31/12/{ano}"
        tarefas = [(mun, ent) for mun in mun_list for ent in entidades]
        log.info(f"Agendando {len(tarefas)} cruzamentos | workers={self.max_workers}")

        todos_registros: list[dict] = []
        discrepancias: list[dict] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._scrape_cruzamento, mun, ent, d_ini, d_fim, limit): (mun, ent)
                for mun, ent in tarefas
            }
            for future in as_completed(futures):
                mun, ent = futures[future]
                try:
                    registros, info = future.result()
                    todos_registros.extend(registros)
                    self.stats["documentos_coletados"] += len(registros)
                    if info["status"] == "OK":
                        self.stats["cruzamentos_ok"] += 1
                    else:
                        self.stats["cruzamentos_falha"] += 1
                        discrepancias.append(info)
                except Exception as e:
                    log.error(f"Erro processando futuro {mun}/{ent}: {e}")
                    self.stats["cruzamentos_falha"] += 1
                    discrepancias.append({"municipio": mun, "entidade": ent, "status": "ERRO", "coletados": 0})

        self.stats["tempo_fim"] = datetime.now()
        elapsed = (self.stats["tempo_fim"] - self.stats["tempo_inicio"]).total_seconds()
        log.info(f"Scraping concluído: {len(todos_registros)} registros em {elapsed:.1f}s")
        return todos_registros, discrepancias


# ==============================================================================
# PERSISTÊNCIA JSON / CSV
# ==============================================================================

def _salvar_json(dados: list[dict], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    tmp.replace(caminho)
    log.info(f"JSON salvo: {caminho} ({len(dados)} registros)")


def _salvar_csv(dados: list[dict], caminho: Path) -> None:
    if not dados:
        return
    caminho.parent.mkdir(parents=True, exist_ok=True)
    campos: list[str] = []
    for row in dados:
        for k in row:
            if k not in campos:
                campos.append(k)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(dados)
    log.info(f"CSV salvo: {caminho} ({len(dados)} registros)")


def _upsert(existentes: list[dict], novos: list[dict]) -> list[dict]:
    """Combina registros antigos com novos via chave composta, priorizando novos."""
    def pk(r: dict) -> str:
        return f"{r.get('municipio','')}|{r.get('entidade','')}|{r.get('identificador_oficial','')}|{r.get('pdf_url','')}"
    mapa = {pk(r): r for r in existentes}
    for r in novos:
        mapa[pk(r)] = r
    return list(mapa.values())


def _deduplicar_por_url(registros: list[dict]) -> list[dict]:
    """Agrupa registros pelo PDF URL único, acumulando municípios referenciados."""
    unicos: dict[str, dict] = {}
    for r in registros:
        url = r.get("pdf_url", "")
        if not url:
            continue
        if url not in unicos:
            unicos[url] = r.copy()
            unicos[url]["municipios_referenciados"] = [r["municipio"]]
            unicos[url]["entidades_referenciadas"] = [r["entidade"]]
        else:
            if r["municipio"] not in unicos[url]["municipios_referenciados"]:
                unicos[url]["municipios_referenciados"].append(r["municipio"])
            if r["entidade"] not in unicos[url]["entidades_referenciadas"]:
                unicos[url]["entidades_referenciadas"].append(r["entidade"])
    return list(unicos.values())


# ==============================================================================
# CLI
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DOM-PI Scraper — Coleta paralela de metadados (Etapa 1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--territorio-carnaubais", action="store_true",
                       help="Todos os 16 municípios do Território Carnaubais.")
    grupo.add_argument("--municipio", type=str, metavar="NOME",
                       help='Nome do município. Ex: "Campo Maior"')

    parser.add_argument("--entidade", type=str, default=None,
                        help="Filtrar por entidade. Ex: Prefeitura ou Camara.")
    parser.add_argument("--ano", type=int, default=2025,
                        help="Ano das publicações (padrão: 2025).")
    parser.add_argument("--limite", type=int, default=100000,
                        help="Máx publicações por cruzamento (padrão: ilimitado).")
    parser.add_argument("--max-workers", type=int, default=15,
                        help="Workers paralelos (padrão: 15).")
    parser.add_argument("--saida", type=str, default=None,
                        help="Caminho base de saída (sem extensão).")
    parser.add_argument("--so-json", action="store_true",
                        help="Omite geração de CSV.")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Caminho para arquivo de log.")
    parser.add_argument("--verbose", action="store_true",
                        help="Ativa logs DEBUG.")

    args = parser.parse_args()
    configure_logging(verbose=args.verbose, log_file=args.log_file)

    muns = CARNAUBAIS_MUNICIPIOS if args.territorio_carnaubais else [args.municipio.strip()]
    entidades = [args.entidade.strip()] if args.entidade else ENTIDADES_PADRAO

    log.info(f"Alvo: {len(muns)} município(s) × {len(entidades)} entidade(s) | ano={args.ano}")

    scraper = ThreadPoolScraper(max_workers=args.max_workers)
    resultados, discrepancias = scraper.scrape_territorio(muns, entidades, args.ano, args.limite)

    if not resultados:
        log.error("Nenhum resultado coletado. Verifique conectividade e parâmetros.")
        sys.exit(1)

    base = args.saida or f"scraping_carnaubais_{args.ano}"
    path_json = Path(base).with_suffix(".json")

    # Upsert com base existente
    if path_json.exists():
        try:
            with open(path_json, encoding="utf-8") as f:
                antigos = json.load(f)
            resultados = _upsert(antigos, resultados)
            log.info(f"[UPSERT] Base consolidada: {len(resultados)} registros")
        except Exception as e:
            log.error(f"Erro ao carregar base antiga: {e}")
            sys.exit(1)

    _salvar_json(resultados, path_json)
    if not args.so_json:
        _salvar_csv(resultados, Path(base).with_suffix(".csv"))

    # Deduplicação pré-download
    unicos = _deduplicar_por_url(resultados)
    path_dedup = Path(base).with_name(f"{Path(base).name}_deduplicados.json")
    _salvar_json(unicos, path_dedup)
    if not args.so_json:
        csv_dedup = []
        for d in unicos:
            dc = d.copy()
            dc["municipios_referenciados"] = "; ".join(dc.get("municipios_referenciados", []))
            dc["entidades_referenciadas"] = "; ".join(dc.get("entidades_referenciadas", []))
            csv_dedup.append(dc)
        _salvar_csv(csv_dedup, Path(base).with_name(f"{Path(base).name}_deduplicados.csv"))

    # Resumo
    elapsed = (scraper.stats["tempo_fim"] - scraper.stats["tempo_inicio"]).total_seconds()
    print("\n" + "=" * 65)
    print("✅  SCRAPING CONCLUÍDO")
    print("=" * 65)
    print(f"  ⏰ Tempo total:          {elapsed:.1f}s")
    print(f"  ✅ Cruzamentos OK:       {scraper.stats['cruzamentos_ok']}")
    print(f"  ❌ Cruzamentos falha:    {scraper.stats['cruzamentos_falha']}")
    print(f"  📄 Publicações:          {len(resultados)}")
    print(f"  📦 PDFs únicos:          {len(unicos)}")
    print(f"  💾 Downloads evitados:   {len(resultados) - len(unicos)}")
    print(f"\n  Saídas:")
    print(f"    JSON bruto:    {path_json}")
    print(f"    JSON dedup:    {path_dedup}")
    if discrepancias:
        print(f"\n  ⚠️  {len(discrepancias)} cruzamentos com problemas:")
        for d in discrepancias:
            print(f"    - {d['municipio']}/{d['entidade']}: {d['status']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
