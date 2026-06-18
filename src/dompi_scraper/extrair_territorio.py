#!/usr/bin/env python3
"""
extrair_territorio.py — Script único de extração por território DOM-PI
-----------------------------------------------------------------------
Lê os PDFs da pasta territorios/<slug>/pdfs/, executa o pipeline
de extração e grava os resultados em extraidos/<slug>/.

A extração roda no ORQUESTRADOR HÍBRIDO (orquestrador_extracao.py), com motores
pesados isolados em subprocessos (engine_worker.py) e venvs separados:
  .venv         → torch + docling (engine Docling)  + orquestrador
  .venv-paddle  → paddlepaddle + paddleocr (engine PaddleOCR)
Monte os dois com: bash setup_venvs.sh   (NUNCA use 'uv run' para extrair).

Stack adaptada ao hardware (detecção automática de GPU via torch):
  GPU (CUDA):  PyMuPDF (nativo simples) | Docling-GPU (nativo+tabela) | PaddleOCR-GPU (escaneado)
  Sem GPU:     PyMuPDF (nativo simples) | PaddleOCR-CPU (nativo+tabela) | Tesseract (escaneado)

Modos de operação (--modo):
  paddle  Orquestrador híbrido (PADRÃO). Roteia por necessidade; anti-OOM por fatiamento.
  pymu    PyMuPDF direto — mais rápido, sem análise de layout/OCR. Bom p/ corpus 100% nativo.
  hibrido ALIAS → orquestrador (paddle).
  marker  DESCONTINUADO → redireciona para o orquestrador (paddle).

Uso (a partir da raiz do projeto):
    # Padrão — orquestrador híbrido (detecta GPU automaticamente)
    PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio tabuleiros_alto_parnaiba

    # WSL com PaddleOCR build CPU: force paddle em CPU (escaneados são raros)
    PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio carnaubais --gpu-paddle cpu

    # PyMuPDF apenas (mais rápido, sem análise de tabelas)
    PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio carnaubais --modo pymu

    # Teste com 5 PDFs e log DEBUG (auditoria)
    PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio parnaiba --limite 5 --verbose

Territórios válidos:
    planice_litoran, cocais, carnaubais, entre_rios, vale_do_sambito,
    vale_do_rio_guaribas, chapada_vale_do_rio_itaim, vale_do_caninde,
    serra_da_capivara, vale_dos_rios_piaui_e_itaueiras,
    tabuleiros_alto_parnaiba, teresina, parnaiba
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # Detectado mais tarde com mensagem de erro clara

log = logging.getLogger("extrair_territorio")

# ==============================================================================
# MAPEAMENTO SLUG → NOME CANÔNICO
# ==============================================================================

TERRITORIOS: dict[str, str] = {
    "planice_litoran":               "Planície Litorânea",
    "cocais":                        "Cocais",
    "carnaubais":                    "Carnaubais",
    "entre_rios":                    "Entre Rios",
    "vale_do_sambito":               "Vale do Sambito",
    "vale_do_rio_guaribas":          "Vale do Rio Guaribas",
    "chapada_vale_do_rio_itaim":     "Chapada Vale do Rio Itaim",
    "vale_do_caninde":               "Vale do Canindé",
    "serra_da_capivara":             "Serra da Capivara",
    "vale_dos_rios_piaui_e_itaueiras": "Vale dos Rios Piauí e Itaueiras",
    "tabuleiros_alto_parnaiba":      "Tabuleiros do Alto Parnaíba e Chapada das Mangabeiras",
    "teresina":                      "Teresina",
    "parnaiba":                      "Parnaíba",
}


def _configure_logging(verbose: bool = False, log_file: str | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
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
        # O arquivo de log SEMPRE grava DEBUG (auditoria), mesmo sem --verbose no console.
        fh.setLevel(logging.DEBUG)
        log.addHandler(fh)
        log.setLevel(logging.DEBUG)  # nível do logger = mais permissivo dos handlers
        ch.setLevel(level)           # console respeita --verbose


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest_from_pdfs(
    pdfs_dir: Path,
    territorio_nome: str,
    limite: int = 0,
) -> dict:
    """
    limite=0 significa sem limite (escaneia todos os PDFs).
    Passe limite=N para escanear apenas os primeiros N PDFs (útil para testes).
    """
    manifest = {}
    # Use rglob to find PDFs recursively in subdirectories (like city/entity)
    pdfs = sorted(pdfs_dir.rglob("*.pdf"))
    if limite and limite < len(pdfs):
        pdfs = pdfs[:limite]
        log.info(f"  Escaneando {len(pdfs)} PDFs (limitado a {limite}) em {pdfs_dir} (recursivo)")
    else:
        log.info(f"  Escaneando {len(pdfs)} PDFs em {pdfs_dir} (recursivo)")

    for pdf_path in pdfs:
        log.debug(f"    Calculando SHA-256: {pdf_path.name}")
        sha = sha256_file(str(pdf_path))
        # SHA-256 como chave evita colisão quando dois municípios têm PDFs com
        # o mesmo nome de arquivo (ex: edições distintas do DOM-PI). (P-10)
        fid = sha

        # Tenta inferir município e entidade pela estrutura de pastas:
        # Ex: pdfs/marcos_parente/Prefeitura/file.pdf -> municipio="marcos_parente", entidade="Prefeitura"
        municipio_inferido = territorio_nome
        entidade_inferida = ""

        # O caminho relativo ao diretório base de pdfs
        rel_path = pdf_path.relative_to(pdfs_dir)
        parts = rel_path.parts

        if len(parts) >= 2:
            # Pelo menos um subdiretório (cidade)
            # Substitui '_' por espaço e capitaliza (ex: marcos_parente -> Marcos Parente)
            cidade_dir = parts[0].replace("_", " ").title()
            # Ajuste simples de preposições para nomes mais limpos (opcional mas útil)
            for prep in [" Do ", " Da ", " De ", " Dos ", " Das "]:
                cidade_dir = cidade_dir.replace(prep, prep.lower())
            municipio_inferido = cidade_dir

            if len(parts) >= 3:
                # Tem subdiretório de entidade também (cidade/entidade)
                entidade_inferida = parts[1].replace("_", " ").title()

        manifest[fid] = {
            "path": str(pdf_path.resolve()),
            "sha256": sha,
            "status": "OK",
            "municipio": municipio_inferido,  # Inferido da pasta ou fallback pro território
            "entidade": entidade_inferida,
            "data_publicacao": "",
            "edicao": "",
            "url": "",
            "documento": pdf_path.name,
        }

    log.debug(f"  Manifesto construído com {len(manifest)} entradas (chave=SHA-256)")
    return manifest


# ==============================================================================
# MODO RÁPIDO — PyMuPDF puro com detecção de tabelas
# ==============================================================================

# Regex simples para detectar padrões de tabela em texto corrido
_RE_TABELA_LINHA = re.compile(
    r"R\$\s*[\d.,]+|\d+\s*[|│]\s*\d+|\bTotal\b.*R\$|\bQtd\.?\b|\bUnidade\b|\bItem\b.*\bValor\b",
    re.IGNORECASE,
)
# Palavras-chave de documentos com alta probabilidade de tabelas
_PALAVRAS_TABELA = {
    "balanço", "rreo", "rgf", "orçamentária", "orçamento", "lrf",
    "licitação", "anexo", "planilha", "dotação", "credito", "crédito",
    "folha de pagamento", "demonstrativo", "despesa", "receita",
    "contrato", "extrato", "rubrica", "suplementação",
}


def _detectar_tabelas(pdf_path: str) -> bool:
    """
    Detecta se o PDF provavelmente tem tabelas, usando:
    1. PyMuPDF find_tables() (detecção estrutural de grade)
    2. Heurística de palavras-chave financeiras no texto
    Retorna True se tabelas forem detectadas.
    """
    if fitz is None:
        return False
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            # Método 1: Detecção estrutural de tabelas
            tabs = page.find_tables()
            if tabs and len(tabs.tables) > 0:
                doc.close()
                return True
            # Método 2: Heurística por palavras-chave
            text_lower = page.get_text().lower()
            if any(kw in text_lower for kw in _PALAVRAS_TABELA):
                doc.close()
                return True
            # Método 3: Padrões de linhas com valores monetários/tabelas
            if len(_RE_TABELA_LINHA.findall(page.get_text())) >= 3:
                doc.close()
                return True
        doc.close()
    except Exception:
        pass
    return False


def run_modo_pymu(
    manifest: dict,
    output_dir: Path,
    slug: str,
    territorio_nome: str,
    limite: int,
    verbose: bool,
) -> dict:
    """
    Extração rápida via PyMuPDF puro (sem GPU).
    Detecta documentos com tabelas e os adiciona à fila de reprocessamento.
    Data de publicação extraída do nome do arquivo (não do texto).
    """
    from dompi_scraper.processar_pdfs import process_single_pdf
    try:
        from dompi_scraper.shared_utils import extrair_data_filename, extrair_edicao_filename
    except ImportError:
        from shared_utils import extrair_data_filename, extrair_edicao_filename

    datalake_dir = str(output_dir / "datalake")
    dedup_path = output_dir / "registro_dedup_pymu.json"
    jsonl_path = output_dir / f"corpus_{slug}.jsonl"
    reprocessar_path = output_dir / "reprocessamento_pendente.jsonl"

    # Carrega dedup existente (permite retomada)
    dedup_registry: dict = {}
    if dedup_path.exists():
        with open(dedup_path, "r", encoding="utf-8") as f:
            dedup_registry = json.load(f)
        log.debug(f"  Dedup PyMuPDF carregado: {len(dedup_registry)} hashes")

    os.makedirs(datalake_dir, exist_ok=True)

    stats = {"total": 0, "gerados": 0, "duplicatas": 0, "com_tabela": 0, "erros": 0, "pulados": 0}
    entradas = list(manifest.items())[:limite]
    total = len(entradas)

    log.info(f"  Iniciando extração RÁPIDA (PyMuPDF) de {total} PDFs...")
    log.info(f"  Tabelas detectadas serão flagadas para reprocessamento futuro.")

    with open(jsonl_path, "a", encoding="utf-8") as jf, \
         open(reprocessar_path, "a", encoding="utf-8") as rf:

        for idx, (fid, entry) in enumerate(entradas, 1):
            pdf_path = entry.get("path", "")
            if not os.path.exists(pdf_path):
                log.warning(f"  [{idx}/{total}] PDF não encontrado: {pdf_path}")
                stats["erros"] += 1
                continue

            stats["total"] += 1

            # Detecta tabelas ANTES de processar (rápido)
            tem_tabela = _detectar_tabelas(pdf_path)
            if tem_tabela:
                stats["com_tabela"] += 1
                log.debug(f"  [{idx}/{total}] 📊 Tabela detectada: {os.path.basename(pdf_path)}")

            # Extração PyMuPDF
            try:
                registros = process_single_pdf(
                    file_id=fid,
                    manifest_entry=entry,
                    output_dir=datalake_dir,
                    dedup_registry=dedup_registry,
                    verbose_blocks=False,
                    modo_chunking=False,
                )
            except Exception as e:
                log.error(f"  [{idx}/{total}] Erro ao processar {os.path.basename(pdf_path)}: {e}")
                stats["erros"] += 1
                continue

            if not registros:
                stats["duplicatas"] += 1
                continue

            # ── Data a partir do filename (não do texto) ───────────────────
            data_ano, data_confianca = extrair_data_filename(pdf_path)
            edicao_dom = extrair_edicao_filename(pdf_path)
            # data_publicacao = somente o ano (ex: "2025"); mês/dia
            # requer mapeamento edição→data não disponível localmente
            data_publicacao_final = data_ano  # "2025" ou ""

            for rec in registros:
                meta = rec.get("metadata", {})
                meta["engine_extracao"] = "PyMuPDF"
                meta["needs_reprocessing"] = tem_tabela
                meta["territorio"] = territorio_nome
                if not meta.get("municipio"):
                    meta["municipio"] = entry.get("municipio", territorio_nome)
                if not meta.get("entidade"):
                    meta["entidade"] = entry.get("entidade", "")

                # Schema JSONL unificado DOM-PI
                jsonl_record = {
                    "doc_id": entry.get("sha256", ""),
                    "territorio": meta["territorio"],
                    "municipio": meta.get("municipio", ""),
                    # Ano de publicação derivado do filename (mais confiável
                    # que extrair do texto, onde datas de contratos/vigências
                    # contaminam o resultado)
                    "data_publicacao": data_publicacao_final,
                    "texto_markdown": rec.get("text", ""),
                    "metadados_extraidos": {
                        "tipo_ato": meta.get("tipo_ato", ""),
                        "entidade": meta.get("entidade", ""),
                        "edicao_dom": edicao_dom,
                        "data_confianca": data_confianca,
                        "id_publicacao": meta.get("id_publicacao", ""),
                    },
                    "engine_extracao": "PyMuPDF",
                    "needs_reprocessing": tem_tabela,
                }
                jf.write(json.dumps(jsonl_record, ensure_ascii=False) + "\n")
                stats["gerados"] += 1

                if tem_tabela:
                    reprocessar_entry = {
                        "doc_id": entry.get("sha256", ""),
                        "path": pdf_path,
                        "municipio": meta.get("municipio", ""),
                        "territorio": territorio_nome,
                        "edicao_dom": edicao_dom,
                        "motivo": "tabela_detectada",
                    }
                    rf.write(json.dumps(reprocessar_entry, ensure_ascii=False) + "\n")

            # Salva dedup a cada 200 documentos para resiliência
            if stats["total"] % 200 == 0:
                with open(dedup_path, "w", encoding="utf-8") as df:
                    json.dump(dedup_registry, df, ensure_ascii=False)
                log.info(
                    f"  ⏳ [{stats['total']}/{total}] "
                    f"Gerados={stats['gerados']} | "
                    f"Com Tabela={stats['com_tabela']} | "
                    f"Erros={stats['erros']}"
                )

    # Salva dedup final
    with open(dedup_path, "w", encoding="utf-8") as df:
        json.dump(dedup_registry, df, ensure_ascii=False)

    return stats


# ==============================================================================
# MODO PRINCIPAL — Orquestrador Híbrido (workers isolados, anti-OOM)
# ==============================================================================

def _rotear_logs_orquestrador(verbose: bool) -> None:
    """
    Faz os loggers do orquestrador e do cliente de worker escreverem nos MESMOS
    handlers deste script (stdout + arquivo de log do território). Garante que a
    auditoria completa — incluindo stderr drenado dos workers — caia no .log.
    """
    for nome in ("orquestrador", "extrator_docling", "engine_worker"):
        lg = logging.getLogger(nome)
        lg.setLevel(logging.DEBUG if verbose else logging.INFO)
        lg.handlers.clear()
        for h in log.handlers:
            lg.addHandler(h)
        lg.propagate = False


def run_modo_orquestrador(
    manifest_path: Path,
    output_dir: Path,
    slug: str,
    territorio_nome: str,
    limite: int,
    verbose: bool,
    threshold: float,
    dpi: int,
    gpu_paddle: str | None,
    gpu_docling: str | None,
    python_paddle: str | None,
    python_docling: str | None,
    docling_max_paginas: int,
    dry_run: bool = False,
) -> dict:
    """
    Delega a extração ao orquestrador híbrido (orquestrador_extracao.py).

    Os motores pesados (PaddleOCR/Docling) rodam em subprocessos isolados, em
    venvs separados, com fatiamento por município/página (anti-OOM) e dedup
    pré-extração persistente (retomada idempotente). Ver docs/BENCHMARK_OCR.md.
    """
    try:
        from dompi_scraper.orquestrador_extracao import run_orquestrador_pipeline
    except ImportError as e:
        log.error(f"Não foi possível importar o orquestrador: {e}")
        raise

    # Unifica os logs do orquestrador/worker no .log deste território (auditoria).
    _rotear_logs_orquestrador(verbose)

    datalake_dir = str(output_dir / "datalake")
    corpus_output = str(output_dir / f"corpus_{slug}.jsonl")

    log.info("-" * 65)
    log.info("Delegando ao ORQUESTRADOR HÍBRIDO (workers isolados)")
    log.debug(f"  manifest     = {manifest_path}")
    log.debug(f"  datalake     = {datalake_dir}")
    log.debug(f"  registry_dir = {output_dir}")
    log.debug(f"  corpus       = {corpus_output}")
    log.debug(f"  limite       = {limite}")
    log.debug(f"  threshold    = {threshold} | dpi = {dpi}")
    log.debug(f"  gpu_paddle   = {gpu_paddle!r} | gpu_docling = {gpu_docling!r}")
    log.debug(f"  python_paddle  = {python_paddle or '(default .venv-paddle)'}")
    log.debug(f"  python_docling = {python_docling or '(default .venv)'}")
    log.debug(f"  docling_max_paginas = {docling_max_paginas} (cap anti-OOM por lote)")
    log.info("-" * 65)

    stats = run_orquestrador_pipeline(
        manifest_path=str(manifest_path),
        output_dir=datalake_dir,
        registry_dir=str(output_dir),
        limite=limite,
        threshold=threshold,
        corpus_output=corpus_output,
        territorio=territorio_nome,
        python_paddle=python_paddle,
        python_docling=python_docling,
        # Passagem direta: None = CPU (de '--gpu-paddle cpu'), "auto" = distribui,
        # "0"/"1" = índice de GPU. NÃO converter None→"auto" (sobrescreveria o 'cpu').
        gpu_paddle=gpu_paddle,
        gpu_docling=gpu_docling,
        dpi=dpi,
        docling_max_paginas=docling_max_paginas,
        dry_run=dry_run,
        verbose=verbose,
    )
    return stats


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[2]  # raiz do projeto
    slug = args.territorio
    territorio_nome = TERRITORIOS[slug]

    pdfs_dir = root / "territorios" / slug / "pdfs"
    output_dir = root / "extraidos" / slug
    log_file = root / "logs" / slug / f"extracao_{time.strftime('%Y%m%d_%H%M%S')}.log"

    _configure_logging(verbose=args.verbose, log_file=str(log_file))

    log.info("=" * 65)
    log.info(f"DOM-PI — Extração de Território: {territorio_nome}")
    log.info("=" * 65)
    log.info(f"  Slug:       {slug}")
    log.info(f"  PDFs:       {pdfs_dir}")
    log.info(f"  Saída:      {output_dir}")
    log.info(f"  Log:        {log_file} (arquivo grava DEBUG sempre)")
    log.info("-" * 65)

    if not pdfs_dir.exists():
        log.error(
            f"Pasta não encontrada: {pdfs_dir}\n"
            f"Execute primeiro: bash setup_territorios.sh"
        )
        sys.exit(1)

    pdfs = list(pdfs_dir.rglob("*.pdf"))
    if not pdfs:
        log.error(
            f"Nenhum PDF encontrado em {pdfs_dir}\n"
            f"Copie os PDFs do território '{territorio_nome}' para essa pasta e tente novamente."
        )
        sys.exit(1)

    log.info(f"  PDFs encontrados: {len(pdfs)}")

    # Normaliza o modo: hibrido/marker são aliases do orquestrador (paddle).
    modo = args.modo
    if modo in ("marker", "hibrido"):
        if modo == "marker":
            warnings.warn(
                "O modo 'marker' foi descontinuado. Use --modo paddle (padrão). "
                "Redirecionando para o orquestrador.",
                DeprecationWarning, stacklevel=2,
            )
        log.warning(f"Modo '{modo}' → redirecionando para o orquestrador (paddle).")
        modo = "paddle"

    # Constrói manifesto em memória (limita o scan para testes/lotes).
    scan_limite = args.limite if args.limite and args.limite < 999_999 else 0
    manifest = build_manifest_from_pdfs(pdfs_dir, territorio_nome, limite=scan_limite)

    # Salva manifesto (consumido pelo orquestrador).
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "download_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info(f"  Manifesto gerado: {manifest_path}")
    log.info(f"  Modo:       {modo.upper()}")

    try:
        t0 = time.time()

        if modo == "paddle":
            # ── MODO PRINCIPAL: Orquestrador Híbrido (workers isolados) ──────
            stats = run_modo_orquestrador(
                manifest_path=manifest_path,
                output_dir=output_dir,
                slug=slug,
                territorio_nome=territorio_nome,
                limite=args.limite,
                verbose=args.verbose,
                threshold=args.threshold,
                dpi=args.dpi,
                gpu_paddle=args.gpu_paddle,
                gpu_docling=args.gpu_docling,
                python_paddle=args.python_paddle,
                python_docling=args.python_docling,
                docling_max_paginas=args.docling_max_paginas,
                dry_run=args.dry_run_rota,
            )
            elapsed = time.time() - t0
            engine = "Orquestrador Híbrido (PyMuPDF + PaddleOCR/Docling/Tesseract)"

        elif modo == "pymu":
            # ── MODO RÁPIDO: PyMuPDF puro (sem GPU/OCR) ──────────────────────
            if fitz is None:
                log.error("pymupdf não encontrado. Rode: bash setup_venvs.sh")
                sys.exit(1)
            stats = run_modo_pymu(
                manifest=manifest,
                output_dir=output_dir,
                slug=slug,
                territorio_nome=territorio_nome,
                limite=args.limite,
                verbose=args.verbose,
            )
            elapsed = time.time() - t0
            engine = "PyMuPDF (rápido, sem OCR)"

        else:
            log.error(f"Modo desconhecido: {modo}")
            sys.exit(1)

    except ImportError as e:
        log.error(f"Erro de importação: {e}\nMonte os ambientes com: bash setup_venvs.sh")
        sys.exit(1)

    _imprimir_relatorio(modo, territorio_nome, engine, output_dir, stats, elapsed)
    log.info(f"Log completo salvo em: {log_file}")


def _imprimir_relatorio(modo, territorio_nome, engine, output_dir, stats, elapsed) -> None:
    """Imprime o resumo final, adaptado ao schema de stats de cada modo."""
    elapsed_min = elapsed / 60
    print("\n" + "=" * 65)
    print(f"EXTRAÇÃO CONCLUÍDA — {territorio_nome}")
    print("=" * 65)
    print(f"  Tempo total:      {elapsed:.1f}s ({elapsed_min:.1f} min)")
    print(f"  Motor:            {engine}")
    print(f"  Saída:            {output_dir}/")

    if modo == "paddle" and stats.get("dry_run") is not None:
        # Modo dry-run de rota (validação sem motores)
        print(f"  [DRY-RUN] classificados: {stats.get('dry_run', 0)}")
        print(f"  Por rota:         {stats.get('dry_por_rota', {})}")
        print(f"  Relatório:        {output_dir}/relatorio_rota.ndjson")
    elif modo == "paddle":
        # Schema do orquestrador
        salvos = stats.get("total", 0)
        print(f"  Chunks salvos:    {salvos}")
        print(f"  PyMuPDF:          {stats.get('pymupdf', 0)}")
        print(f"  Docling CUDA:     {stats.get('docling_cuda', 0)}")
        print(f"  Docling CPU:      {stats.get('docling_cpu', 0)}")
        print(f"  PaddleOCR CUDA:   {stats.get('paddle_cuda', 0)}")
        print(f"  PaddleOCR CPU:    {stats.get('paddle_cpu', 0)}")
        print(f"  Tesseract:        {stats.get('tesseract', 0)}")
        print(f"  Duplicatas:       {stats.get('duplicatas', 0)} "
              f"(puladas pré-extração: {stats.get('dup_pre', 0)})")
        print(f"  Erros:            {stats.get('erros', 0)}")
        if salvos:
            print(f"  Média:            {elapsed/max(1,salvos):.2f}s/chunk salvo")
    else:
        # Schema do PyMuPDF rápido
        print(f"  Gerados:          {stats.get('gerados', 0)}")
        print(f"  Duplicatas:       {stats.get('duplicatas', 0)}")
        print(f"  Com tabela:       {stats.get('com_tabela', 0)}")
        print(f"  Erros:            {stats.get('erros', 0)}")
        if stats.get("com_tabela"):
            print(f"  Fila reprocess.:  {output_dir}/reprocessamento_pendente.jsonl")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrator padronizado por território — DOM-PI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--territorio", required=False, default=None,
        choices=list(TERRITORIOS.keys()),
        metavar="SLUG",
        help=(
            "Slug do território. Valores aceitos:\n" +
            "\n".join(f"  {k} → {v}" for k, v in TERRITORIOS.items())
        ),
    )
    parser.add_argument(
        "--modo", default="paddle",
        choices=["paddle", "pymu", "hibrido", "marker"],
        help=(
            "Modo de extração:\n"
            "  paddle  → Orquestrador híbrido (PADRÃO). Roteia por necessidade; anti-OOM.\n"
            "  pymu    → PyMuPDF direto (rápido, sem OCR/layout). Ideal p/ corpus 100% nativo.\n"
            "  hibrido → ALIAS de paddle.\n"
            "  marker  → DESCONTINUADO. Redireciona para paddle.\n"
            "Padrão: paddle"
        ),
    )
    parser.add_argument(
        "--limite", type=int, default=999_999,
        help="Máx. PDFs a processar (padrão: ilimitado). Use valores pequenos para testes/lotes.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Ativa logs DEBUG no console (o arquivo de log já grava DEBUG sempre).",
    )
    parser.add_argument(
        "--listar", action="store_true",
        help="Lista todos os territórios disponíveis e sai.",
    )

    # ── Parâmetros do orquestrador (modo paddle) ─────────────────────────────
    g = parser.add_argument_group("Orquestrador (modo paddle)")
    g.add_argument(
        "--threshold", type=float, default=0.45,
        help="Score de texto nativo: acima → nativo (PyMuPDF/Docling); abaixo → escaneado (PaddleOCR). Padrão 0.45.",
    )
    g.add_argument(
        "--dpi", type=int, default=200,
        help="DPI de rasterização para OCR nos workers. Padrão 200.",
    )
    g.add_argument(
        "--docling-max-paginas", type=int, default=8,
        help="Cap anti-OOM: nº máx. de páginas por chamada ao Docling (fatiamento). Padrão 8.",
    )
    g.add_argument(
        "--dry-run-rota", action="store_true",
        help="Só valida a ROTA de cada doc (nome→fiscal/comum), gera relatorio_rota.ndjson e NÃO roda motores.",
    )
    g.add_argument(
        "--gpu-paddle", default="auto",
        help="GPU do worker PaddleOCR: índice (ex '0'), 'cpu' ou 'auto'. Em WSL com paddle CPU use 'cpu'.",
    )
    g.add_argument(
        "--gpu-docling", default="auto",
        help="GPU do worker Docling: índice (ex '1'), 'cpu' ou 'auto'.",
    )
    g.add_argument(
        "--python-paddle", default=None,
        help="Interpretador do worker PaddleOCR (padrão: ./.venv-paddle/bin/python).",
    )
    g.add_argument(
        "--python-docling", default=None,
        help="Interpretador do worker Docling (padrão: ./.venv/bin/python).",
    )

    # ── Legados (ignorados; mantidos por compatibilidade de scripts) ─────────
    parser.add_argument("--workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--force-ocr", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--min-variance", type=float, default=50.0, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.listar:
        print("\nTerritórios disponíveis:\n")
        for slug, nome in TERRITORIOS.items():
            print(f"  {slug:<40} → {nome}")
        print()
        sys.exit(0)

    if not args.territorio:
        parser.error("--territorio é obrigatório (ou use --listar para ver os slugs).")

    # Normaliza "cpu" → None para o cliente de worker do orquestrador.
    args.gpu_paddle = None if args.gpu_paddle == "cpu" else args.gpu_paddle
    args.gpu_docling = None if args.gpu_docling == "cpu" else args.gpu_docling

    run(args)


if __name__ == "__main__":
    main()
