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


# ---------------------------------------------------------------------------
# Scraping Isolado
# ---------------------------------------------------------------------------

def scrape_territorio(
    mun_list: list[str],
    entidades: list[str],
    ano: int,
    limit_per_cruzamento: int,
) -> list[dict]:
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
        return []

    municipios_ok = len(ctx.get("municipio_options", {}))
    entidades_ok = len(ctx.get("entidade_options", {}))
    log.info(f"Formulário carregado: {municipios_ok} municípios | {entidades_ok} entidades mapeadas.")

    resultados: list[dict] = []
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
                continue

            total_pags = get_total_pages(html1)
            coletados = 0

            for pag in range(1, total_pags + 1):
                if coletados >= limit_per_cruzamento:
                    break

                if pag > 1:
                    time.sleep(1.2)  # Respeita o servidor
                    html1 = fetch_search_results(session, ctx, mun, ent, d_ini, d_fim, p=pag)
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

    return resultados


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
    campos = list(dados[0].keys())
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
    if args.territorio_carnaubais:
        muns = CARNAUBAIS_MUNICIPIOS
        log.info(f"Alvo: Território Carnaubais ({len(muns)} municípios)")
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
    resultados = scrape_territorio(muns, entidades, args.ano, args.limite)

    if not resultados:
        log.warning("Nenhum resultado coletado. Verifique os parâmetros e a conectividade.")
        sys.exit(1)

    log.info(f"Total coletado: {len(resultados)} publicações.")

    # Resolve nomes de saída
    base_saida = args.saida or f"scraping_carnaubais_{args.ano}"
    path_json = Path(base_saida).with_suffix(".json")
    path_csv = Path(base_saida).with_suffix(".csv")

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
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
