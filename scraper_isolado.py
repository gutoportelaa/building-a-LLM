#!/usr/bin/env python3
"""
scraper_isolado.py — Validador de Scraping DOM-PI (Território Carnaubais / 2025)
----------------------------------------------------------------------------------
Etapa 1 do pipeline isolada: coleta APENAS metadados das publicações do Diário
Oficial dos Municípios do Piauí (DOM-PI) via requests + BeautifulSoup.

NÃO baixa PDFs. NÃO acessa SQLite. NÃO faz OCR.
Persiste o resultado em JSON e/ou CSV para inspeção e validação do scraping.

Uso rápido:
    # 1 município, 1 entidade
    uv run python src/dompi_scraper/scraper_isolado.py \\
        --municipio "Assuncao do Pi" --entidade Prefeitura --ano 2025 --limite 10

    # Território Carnaubais completo (16 municípios × 2 entidades)
    uv run python src/dompi_scraper/scraper_isolado.py \\
        --territorio-carnaubais --ano 2025 --limite 5

    # Saída customizada
    uv run python src/dompi_scraper/scraper_isolado.py \\
        --territorio-carnaubais --ano 2025 --limite 3 \\
        --saida meus_resultados.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import http.client as http_client
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Importa o núcleo de scraping do pipeline principal (sem efeitos colaterais)
# ---------------------------------------------------------------------------
try:
    from .pipeline import (
        CARNAUBAIS_MUNICIPIOS,
        ENTIDADES_PADRAO,
        SEARCH_URL,
        build_session,
        extract_metadata_from_url,
        fetch_search_results,
        get_total_pages,
        load_form_context,
        parse_results_page,
    )
except ImportError:
    from pipeline import (
        CARNAUBAIS_MUNICIPIOS,
        ENTIDADES_PADRAO,
        SEARCH_URL,
        build_session,
        extract_metadata_from_url,
        fetch_search_results,
        get_total_pages,
        load_form_context,
        parse_results_page,
    )

# Registry slug→municípios dos demais territórios (gerado/validado contra o
# formulário ao vivo — ver territorios_pi.py). Opcional: scraper funciona sem ele.
try:
    try:
        from .territorios_pi import TERRITORIOS_MUNICIPIOS
    except ImportError:
        from territorios_pi import TERRITORIOS_MUNICIPIOS
except ImportError:
    TERRITORIOS_MUNICIPIOS = {}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("dompi_scraper_isolado")


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    log.setLevel(level)
    log.handlers.clear()
    log.addHandler(handler)
    
    if verbose:
        http_client.HTTPConnection.debuglevel = 1
        requests_log = logging.getLogger("urllib3")
        requests_log.setLevel(logging.DEBUG)
        requests_log.propagate = True


# ---------------------------------------------------------------------------
# Scraping Isolado
# ---------------------------------------------------------------------------

def scrape_territorio(
    mun_list: list[str],
    entidades: list[str],
    ano: int,
    limit_per_cruzamento: int,
) -> tuple[list[dict], list[dict]]:
    """
    Itera [município × entidade] e coleta metadados das publicações.
    Retorna lista de dicionários com os campos de cada publicação encontrada.
    NÃO baixa PDFs. NÃO acessa banco de dados.
    """
    d_ini = f"01/01/{ano}"
    d_fim = f"31/12/{ano}"

    session = build_session()
    log.info("Carregando contexto do formulário DOM-PI...")
    ctx = load_form_context(session)

    if ctx.get("error"):
        log.error(f"Falha ao carregar formulário: {ctx['error']}")
        return [], []

    municipios_ok = len(ctx.get("municipio_options", {}))
    entidades_ok = len(ctx.get("entidade_options", {}))
    log.info(f"Formulário carregado: {municipios_ok} municípios | {entidades_ok} entidades mapeadas.")

    resultados: list[dict] = []
    discrepancias: list[dict] = []
    total_cruzamentos = len(mun_list) * len(entidades)
    cruzamento_atual = 0

    for mun in mun_list:
        for ent in entidades:
            cruzamento_atual += 1
            log.info(
                f"[{cruzamento_atual}/{total_cruzamentos}] "
                f"Avaliando: [Cidade: {mun}] | [Entidade: {ent}]"
            )

            html1 = fetch_search_results(session, ctx, mun, ent, d_ini, d_fim, p=1)
            if not html1:
                log.warning(f"  Sem resposta para {mun} / {ent}. Pulando.")
                discrepancias.append({
                    "municipio": mun, "entidade": ent, 
                    "esperado": 0, "coletado": 0, "status": "FALHA_CONEXAO"
                })
                continue

            total_pags, total_docs = get_total_pages(html1)
            log.info(f"  -> Fonte acusou {total_docs} documentos disponíveis em {total_pags} página(s).")
            coletados = 0

            for pag in range(1, total_pags + 1):
                if coletados >= limit_per_cruzamento:
                    break

                if pag > 1:
                    time.sleep(1.2)  # Respeita o servidor
                    html1 = fetch_search_results(session, ctx, mun, ent, d_ini, d_fim, p=pag, last_html=html1)
                    if not html1:
                        continue

                regs = parse_results_page(html1)
                if not regs:
                    log.debug(f"  Pág {pag}/{total_pags} — sem registros parseados.")

                for reg in regs:
                    if coletados >= limit_per_cruzamento:
                        break

                    pdf_url = reg.get("pdf_url", "")
                    if not pdf_url:
                        log.debug(f"  Registro sem pdf_url: {reg.get('documento', '?')}")
                        continue

                    # Extrai metadados ocultos da URL (regex)
                    url_meta = extract_metadata_from_url(pdf_url)

                    entrada = {
                        "municipio": mun,
                        "entidade": ent,
                        "ano_consulta": ano,
                        "data_publicacao": reg.get("data", ""),
                        "edicao": reg.get("edicao", ""),
                        "categoria": reg.get("categoria", ""),
                        "documento": reg.get("documento", ""),
                        "identificador_oficial": reg.get("identificador", ""),
                        "pdf_url": pdf_url,
                        "pagina_url_meta": url_meta.get("pagina", ""),
                        "edicao_url_meta": url_meta.get("edicao_url", ""),
                        "codigo_interno_mun_meta": url_meta.get("codigo_interno_mun", ""),
                        "coletado_em": datetime.now().isoformat(timespec="seconds"),
                    }
                    resultados.append(entrada)
                    coletados += 1

            log.info(f"  → {coletados} publicação(ões) coletada(s) para {mun} / {ent}.")
            
            # Checa validade vs expectativa do portal
            if coletados < total_docs and coletados < limit_per_cruzamento:
                discrepancias.append({
                    "municipio": mun, "entidade": ent, 
                    "esperado": total_docs, "coletado": coletados, 
                    "status": "INCOMPLETO"
                })
            elif total_docs == 0:
                discrepancias.append({
                    "municipio": mun, "entidade": ent, 
                    "esperado": 0, "coletado": 0, 
                    "status": "VAZIO"
                })

    return resultados, discrepancias


# ---------------------------------------------------------------------------
# Persistência — JSON e CSV
# ---------------------------------------------------------------------------

def salvar_json(dados: list[dict], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    log.info(f"JSON salvo em: {caminho} ({len(dados)} registros)")


def salvar_csv(dados: list[dict], caminho: Path) -> None:
    if not dados:
        return
    caminho.parent.mkdir(parents=True, exist_ok=True)
    
    campos = []
    for row in dados:
        for key in row.keys():
            if key not in campos:
                campos.append(key)
                
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(dados)
    log.info(f"CSV salvo em: {caminho} ({len(dados)} registros)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DOM-PI Scraper Isolado — Coleta de metadados sem downloads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    grupo_alvo = parser.add_mutually_exclusive_group(required=True)
    grupo_alvo.add_argument(
        "--territorio-carnaubais",
        action="store_true",
        help="Todos os 16 municípios do Território Carnaubais.",
    )
    grupo_alvo.add_argument(
        "--municipio",
        type=str,
        metavar="NOME",
        help='Nome do município. Ex: "Assuncao do Pi"',
    )
    grupo_alvo.add_argument(
        "--territorio",
        type=str,
        metavar="SLUG",
        help="Slug de um território do registry territorios_pi.py. "
             f"Disponíveis: {', '.join(sorted(TERRITORIOS_MUNICIPIOS)) or '(registry ausente)'}.",
    )

    parser.add_argument(
        "--entidade",
        type=str,
        default=None,
        metavar="ENTIDADE",
        help="Filtrar por entidade específica. Ex: Prefeitura ou Camara. "
             "Se omitido, usa Prefeitura e Câmara.",
    )
    parser.add_argument(
        "--ano",
        type=int,
        default=2025,
        help="Ano das publicações a consultar (padrão: 2025).",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default= 100000,
        help="Nº máximo de publicações por cruzamento [município × entidade] (padrão: 10).",
    )
    parser.add_argument(
        "--saida",
        type=str,
        default=None,
        metavar="ARQUIVO",
        help="Caminho base para os arquivos de saída (sem extensão). "
             "Gera <saida>.json e <saida>.csv. "
             "Padrão: scraping_carnaubais_<ano>",
    )
    parser.add_argument(
        "--so-json",
        action="store_true",
        help="Salva apenas o JSON (omite o CSV).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Ativa logs de DEBUG.",
    )

    args = parser.parse_args()
    _configure_logging(verbose=args.verbose)

    # Resolve lista de municípios
    slug_alvo = None
    if args.territorio_carnaubais:
        muns = CARNAUBAIS_MUNICIPIOS
        slug_alvo = "carnaubais"
        log.info(f"Alvo: Território Carnaubais ({len(muns)} municípios)")
    elif args.territorio:
        slug_alvo = args.territorio.strip()
        if slug_alvo not in TERRITORIOS_MUNICIPIOS:
            log.error(f"Slug '{slug_alvo}' não está no registry. Disponíveis: "
                      f"{', '.join(sorted(TERRITORIOS_MUNICIPIOS)) or '(nenhum)'}")
            sys.exit(2)
        muns = TERRITORIOS_MUNICIPIOS[slug_alvo]
        log.info(f"Alvo: Território {slug_alvo} ({len(muns)} municípios)")
    else:
        muns = [args.municipio.strip()]
        log.info(f"Alvo: município único: {muns[0]}")

    # Resolve lista de entidades
    if args.entidade:
        entidades = [args.entidade.strip()]
    else:
        entidades = ENTIDADES_PADRAO
    log.info(f"Entidades: {entidades}")
    log.info(f"Ano: {args.ano} | Limite por cruzamento: {args.limite}")

    # Executa o scraping
    resultados, discrepancias = scrape_territorio(muns, entidades, args.ano, args.limite)

    if not resultados:
        log.warning("Nenhum resultado coletado. Verifique os parâmetros e a conectividade.")
        sys.exit(1)

    log.info(f"Total coletado: {len(resultados)} publicações.")

    # Resolve nomes de saída
    base_saida = args.saida or f"dados/scraping_results/scraping_{slug_alvo or 'municipio'}_{args.ano}"
    Path(base_saida).parent.mkdir(parents=True, exist_ok=True)
    path_json = Path(base_saida).with_suffix(".json")
    path_csv = Path(base_saida).with_suffix(".csv")

    # Lógica de Upsert (Mistura com base antiga se existir)
    novos_registros_qtd = len(resultados)
    registros_antigos_qtd = 0
    
    if path_json.exists():
        try:
            with open(path_json, "r", encoding="utf-8") as f:
                dados_antigos = json.load(f)
            
            def get_pk(r: dict) -> str:
                return f"{r.get('municipio', '')}_{r.get('entidade', '')}_{r.get('identificador_oficial', '')}_{r.get('pdf_url', '')}"
            
            # Indexa base antiga
            mapa_registros = {get_pk(r): r for r in dados_antigos}
            registros_antigos_qtd = len(mapa_registros)
            
            # Upsert dos novos
            for r in resultados:
                mapa_registros[get_pk(r)] = r
                
            resultados = list(mapa_registros.values())
            print(f"\n  [INCREMENTO] Carregados {registros_antigos_qtd} registros antigos de {path_json.name}.")
        except Exception as e:
            log.error(f"CRÍTICO: Erro ao tentar carregar base antiga {path_json}: {e}")
            log.error("Abortando execução de salvamento para proteger a base de dados original contra sobrescrita!")
            sys.exit(1)

    salvar_json(resultados, path_json)
    if not args.so_json:
        salvar_csv(resultados, path_csv)

    # Resumo final no terminal
    print("\n" + "─" * 60)
    print(f"  ✔  {len(resultados)} publicações coletadas")
    print(f"  📄 JSON: {path_json}")
    if not args.so_json:
        print(f"  📊 CSV:  {path_csv}")

    # Agrupa por município/entidade para sumário
    contagem: dict[str, int] = {}
    for r in resultados:
        chave = f"{r['municipio']} / {r['entidade']}"
        contagem[chave] = contagem.get(chave, 0) + 1

    print("\n  Distribuição por cruzamento:")
    for chave, qtd in sorted(contagem.items()):
        print(f"    {chave}: {qtd} doc(s)")
        
    if discrepancias:
        print("\n  [ATENÇÃO] Alertas de Inconsistência:")
        for disc in discrepancias:
            print(f"    - [{disc['municipio']} / {disc['entidade']}]: Status = {disc['status']} (Aguardava {disc['esperado']} | Coletou {disc['coletado']})")
    
    # Deduplicação Pré-Download (Agrupa por PDF URL)
    doc_unicos = {}
    for r in resultados:
        pdf = r.get("pdf_url")
        if not pdf: continue
        if pdf not in doc_unicos:
            doc_unicos[pdf] = r.copy()
            doc_unicos[pdf]["municipios_referenciados"] = [r["municipio"]]
            doc_unicos[pdf]["entidades_referenciadas"] = [r["entidade"]]
        else:
            if r["municipio"] not in doc_unicos[pdf]["municipios_referenciados"]:
                doc_unicos[pdf]["municipios_referenciados"].append(r["municipio"])
            if r["entidade"] not in doc_unicos[pdf]["entidades_referenciadas"]:
                doc_unicos[pdf]["entidades_referenciadas"].append(r["entidade"])
                
    lista_unicos = list(doc_unicos.values())
    
    path_json_unico = Path(base_saida).with_name(f"{Path(base_saida).name}_deduplicados.json")
    path_csv_unico = Path(base_saida).with_name(f"{Path(base_saida).name}_deduplicados.csv")
    
    lista_unicos_csv = []
    for d in lista_unicos:
        d_copy = d.copy()
        d_copy["municipios_referenciados"] = "; ".join(d_copy["municipios_referenciados"])
        d_copy["entidades_referenciadas"] = "; ".join(d_copy["entidades_referenciadas"])
        lista_unicos_csv.append(d_copy)

    salvar_json(lista_unicos, path_json_unico)
    if not args.so_json:
        salvar_csv(lista_unicos_csv, path_csv_unico)

    print("\n  📈 Extrato de Deduplicação:")
    print(f"    - Publicações processadas (linhas brutas): {len(resultados)}")
    print(f"    - Documentos PDF físicos reais (deduplicados): {len(lista_unicos)}")
    print(f"    > {len(resultados) - len(lista_unicos)} requisições de download serão economizadas!")

    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
