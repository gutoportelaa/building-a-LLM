#!/usr/bin/env python3
"""
extrair_territorio.py — Script único de extração por território DOM-PI
-----------------------------------------------------------------------
Lê os PDFs da pasta territorios/<slug>/pdfs/, executa o pipeline
de extração e grava os resultados em extraidos/<slug>/.

Stack de extração (sem Marker):
  GPU (CUDA):
    Triagem DLA (PyMuPDF) → PaddleOCR CUDA (simples) | Docling CUDA (tabelas)
  CPU:
    PyMuPDF (digital nativo) | Tesseract (escaneado) | PaddleOCR CPU (tabelas)

Modos de operação (--modo):
  paddle  PaddleOCR PP-Structure paralelo. Detecta tabelas/figuras. (PADRÃO)
  pymu    PyMuPDF direto — mais rápido, sem análise de layout avançada.
  hibrido DESCONTINUADO → redireciona para paddle.
  marker  DESCONTINUADO → redireciona para paddle.

Uso:
    # Padrão — PaddleOCR com detecção de layout
    uv run python src/dompi_scraper/extrair_territorio.py --territorio tabuleiros_alto_parnaiba

    # PyMuPDF apenas (mais rápido, sem análise de tabelas)
    uv run python src/dompi_scraper/extrair_territorio.py --territorio carnaubais --modo pymu

    # Teste com 5 PDFs
    uv run python src/dompi_scraper/extrair_territorio.py --territorio parnaiba --limite 5 --verbose

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
import multiprocessing
import os
import psutil
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
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
        log.addHandler(fh)


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
# MODO PADDLE — PaddleOCR PP-Structure + ProcessPoolExecutor
# ==============================================================================

# Engine global por processo worker (inicializado uma vez em _init_paddle_worker)
_PADDLE_ENGINE = None


def _init_paddle_worker() -> None:
    """Inicializador de processo worker: cria o PP-Structure uma vez por processo."""
    global _PADDLE_ENGINE
    
    import os
    # Força as bibliotecas matemáticas subjacentes a usarem apenas 1 thread por worker.
    # Sem isso, 10 workers * 22 cores = 220 threads, causando explosão de memória e crashes (OOM) no WSL.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["MAX_JOBS"] = "1"         # Evita OOM caso haja compilação JIT de C++
    os.environ["DISABLE_NINJA"] = "1"    # Desativa ninja para compilações pesadas em background
    
    # Suprime logs verbosos do Paddle nos workers
    import logging as _logging
    _logging.disable(_logging.WARNING)
    try:
        from dompi_scraper.extrator_paddle import criar_engine_paddle
        _PADDLE_ENGINE = criar_engine_paddle(use_gpu=False)  # lang fixo em 'en' (restrição do PP-Structure)
    except Exception as e:
        # Worker continua sem engine — será reportado como erro no resultado
        _PADDLE_ENGINE = None
        import sys
        print(f"[worker] Falha ao inicializar PP-Structure: {e}", file=sys.stderr)


def _paddle_worker_task(task: tuple) -> dict | None:
    """
    Função picklável executada por cada processo worker.
    Processa um único PDF com o engine PP-Structure do processo.

    Retorna um dict com o resultado ou None em caso de erro/duplicata.
    """
    global _PADDLE_ENGINE

    fid, entry, territorio_nome, datalake_dir, dedup_snapshot = task

    if _PADDLE_ENGINE is None:
        return {"erro": "engine_nao_inicializado", "fid": fid}

    pdf_path = entry.get("path", "")
    if not pdf_path or not os.path.exists(pdf_path):
        return {"erro": "pdf_nao_encontrado", "fid": fid, "path": pdf_path}

    try:
        from dompi_scraper.extrator_paddle import extrair_pdf_paddle
        from dompi_scraper.processar_pdfs import (
            generate_frontmatter,
            build_datalake_path,
        )
        from dompi_scraper.shared_utils import (
            classify_act_type,
            compute_content_md5,
            extrair_data_filename,
            extrair_edicao_filename,
        )
    except ImportError as e:
        return {"erro": f"import: {e}", "fid": fid}

    try:
        doc_result = extrair_pdf_paddle(_PADDLE_ENGINE, pdf_path, dpi=200)
    except Exception as e:
        return {"erro": f"extracao: {e}", "fid": fid}

    texto = doc_result.texto_completo
    if not texto or len(texto.strip()) < 30:
        return {"erro": "texto_vazio", "fid": fid}

    content_hash = compute_content_md5(texto)
    if content_hash in dedup_snapshot:
        return {"duplicata": True, "fid": fid}

    # Metadados do manifesto e do filename
    municipio = entry.get("municipio", territorio_nome)
    entidade = entry.get("entidade", "")
    sha256_pdf = entry.get("sha256", fid)
    data_ano, data_confianca = extrair_data_filename(pdf_path)
    edicao_dom = extrair_edicao_filename(pdf_path)
    tipo_ato = classify_act_type(texto)

    # Salva .md no datalake
    ano = data_ano or "sem_ano"
    mes = "sem_mes"
    frontmatter = generate_frontmatter(
        content_hash=content_hash,
        municipio=municipio,
        entidade=entidade,
        tipo_ato=tipo_ato,
        data_publicacao=data_ano,
        url_origem=entry.get("url", ""),
        edicao=edicao_dom,
        sha256_pdf=sha256_pdf,
        has_tables=doc_result.has_tables,
        has_figures=doc_result.has_figures,
        table_pages=doc_result.table_pages or None,
        figure_pages=doc_result.figure_pages or None,
        valores_monetarios_total=doc_result.valores_monetarios_total,
    )
    md_path = build_datalake_path(datalake_dir, ano, mes, municipio, f"{content_hash}.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(frontmatter + texto)

    # Monta registro JSONL
    jsonl_record = {
        "doc_id": sha256_pdf,
        "territorio": territorio_nome,
        "municipio": municipio,
        "data_publicacao": data_ano,
        "texto_markdown": texto,
        "metadados_extraidos": {
            "tipo_ato": tipo_ato,
            "entidade": entidade,
            "edicao_dom": edicao_dom,
            "data_confianca": data_confianca,
            "id_publicacao": content_hash,
            "has_tables": doc_result.has_tables,
            "has_figures": doc_result.has_figures,
            "table_pages": doc_result.table_pages,
            "figure_pages": doc_result.figure_pages,
            "valores_monetarios_total": doc_result.valores_monetarios_total,
        },
        "engine_extracao": "PaddleOCR",
        "needs_reprocessing": doc_result.has_tables or doc_result.has_figures,
    }

    # Entrada de reprocessamento (apenas se necessário)
    reprocessar_entry = None
    if doc_result.has_tables or doc_result.has_figures:
        reprocessar_entry = {
            "doc_id": sha256_pdf,
            "path": pdf_path,
            "municipio": municipio,
            "territorio": territorio_nome,
            "edicao_dom": edicao_dom,
            "motivo": "tabela_ou_grafico_detectado",
            "table_pages": doc_result.table_pages,
            "figure_pages": doc_result.figure_pages,
            "valores_monetarios": doc_result.todos_valores,
            "valores_monetarios_total": doc_result.valores_monetarios_total,
        }

    return {
        "ok": True,
        "fid": fid,
        "content_hash": content_hash,
        "municipio": municipio,
        "tipo_ato": tipo_ato,
        "has_tables": doc_result.has_tables,
        "has_figures": doc_result.has_figures,
        "jsonl_record": jsonl_record,
        "reprocessar_entry": reprocessar_entry,
    }


def calcular_limite_seguro_workers(use_gpu: bool = False) -> int:
    """
    Calcula um limite dinâmico seguro de processos para evitar crashes (OOM),
    baseado na memória livre real da máquina e nos núcleos da CPU.
    """
    import platform
    import platform
    is_wsl = 'microsoft' in platform.uname().release.lower()
    
    limite_cpu = max(1, multiprocessing.cpu_count() - 1)
    
    try:
        ram_livre_gb = psutil.virtual_memory().available / (1024**3)
        
        # O log de SO revelou que CADA instância do PP-Structure consome incríveis ~7.5 GB de RAM
        # no boot devido ao multiprocessamento do PaddlePaddle no WSL.
        ram_por_worker = 10.0 if is_wsl else 8.0
        
        limite_ram = max(1, int(ram_livre_gb / ram_por_worker))
        # Hard cap: no CPU, nunca passar de 2 workers no WSL para evitar sobrecarga de barramento
        limite_final = min(limite_cpu, limite_ram)
        if is_wsl and not use_gpu:
            limite_final = min(limite_final, 2)
    except Exception as e:
        log.warning(f"  [Aviso] Falha ao ler RAM via psutil ({e}). Usando limite da CPU.")
        limite_final = limite_cpu

    if use_gpu:
        try:
            import torch
            vram_livre_gb = torch.cuda.mem_get_info()[0] / (1024**3)
            # Assumimos conservadoramente ~1.5 GB de VRAM por worker
            limite_vram = max(1, int(vram_livre_gb / 1.5))
            limite_final = min(limite_final, limite_vram)
        except Exception:
            pass
            
    return limite_final


def run_modo_paddle(
    manifest: dict,
    output_dir: Path,
    slug: str,
    territorio_nome: str,
    limite: int,
    verbose: bool,
    workers: int | None = None,
) -> dict:
    """
    Extração com PaddleOCR PP-Structure em paralelo (ProcessPoolExecutor).

    Cada processo worker inicializa seu próprio engine PP-Structure e processa
    um PDF por vez. O processo principal coleta resultados e grava JSONL/dedup.
    """
    if workers is None:
        workers = calcular_limite_seguro_workers(use_gpu=False)
        log.info(f"  [Auto-Scale] Limite dinâmico ajustado para {workers} workers (baseado em RAM/CPU/VRAM livres)")

    datalake_dir = str(output_dir / "datalake")
    dedup_path = output_dir / "registro_dedup_paddle.json"
    jsonl_path = output_dir / f"corpus_{slug}.jsonl"
    reprocessar_path = output_dir / "reprocessamento_pendente.jsonl"

    # Carrega dedup existente (permite retomada)
    dedup_registry: dict = {}
    if dedup_path.exists():
        with open(dedup_path, "r", encoding="utf-8") as f:
            dedup_registry = json.load(f)

    os.makedirs(datalake_dir, exist_ok=True)

    entradas = list(manifest.items())[:limite]
    total = len(entradas)
    stats = {
        "total": 0, "gerados": 0, "duplicatas": 0,
        "com_tabela": 0, "com_figura": 0, "erros": 0,
    }

    log.info(f"  Modo PADDLE — {total} PDFs | {workers} workers")
    log.info(f"  Cada worker inicializa PP-Structure (aguarde ~30s no startup)...")

    # Snapshot imutável do dedup para enviar aos workers (evita pickle de dict mutável grande)
    dedup_snapshot = set(dedup_registry.keys())

    tasks = [
        (fid, entry, territorio_nome, datalake_dir, dedup_snapshot)
        for fid, entry in entradas
    ]

    checkpoint_every = 500  # salva dedup a cada N documentos processados

    with (
        open(jsonl_path, "a", encoding="utf-8") as jf,
        open(reprocessar_path, "a", encoding="utf-8") as rf,
        ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_paddle_worker,
        ) as pool,
    ):
        futures = {pool.submit(_paddle_worker_task, task): task[0] for task in tasks}

        for i, future in enumerate(as_completed(futures), 1):
            stats["total"] += 1
            try:
                result = future.result()
            except Exception as e:
                log.error(f"  Worker exception: {e}")
                stats["erros"] += 1
                continue

            if result is None:
                stats["erros"] += 1
                continue

            if result.get("duplicata"):
                stats["duplicatas"] += 1
                continue

            if result.get("erro"):
                log.debug(f"  Erro [{result['fid'][:8]}]: {result['erro']}")
                stats["erros"] += 1
                continue

            if result.get("ok"):
                # Grava no JSONL
                jf.write(json.dumps(result["jsonl_record"], ensure_ascii=False) + "\n")
                stats["gerados"] += 1

                if result.get("has_tables"):
                    stats["com_tabela"] += 1
                if result.get("has_figures"):
                    stats["com_figura"] += 1

                # Grava em reprocessamento_pendente se necessário
                if result.get("reprocessar_entry"):
                    rf.write(json.dumps(result["reprocessar_entry"], ensure_ascii=False) + "\n")

                # Atualiza dedup em memória
                content_hash = result["content_hash"]
                dedup_registry[content_hash] = {
                    "municipio": result["municipio"],
                    "tipo_ato": result["tipo_ato"],
                }
                dedup_snapshot.add(content_hash)

                if verbose:
                    log.debug(
                        f"  [{i}/{total}] {result['municipio']} | "
                        f"tabela={result['has_tables']} | fig={result['has_figures']}"
                    )

            # Checkpoint periódico
            if stats["total"] % checkpoint_every == 0:
                _salvar_dedup(dedup_path, dedup_registry)
                log.info(
                    f"  ⏳ [{stats['total']}/{total}] "
                    f"Gerados={stats['gerados']} | Tabelas={stats['com_tabela']} | "
                    f"Erros={stats['erros']}"
                )

    _salvar_dedup(dedup_path, dedup_registry)
    return stats


def _salvar_dedup(path: Path, registry: dict) -> None:
    """Salva o registro de deduplicação atomicamente (write + rename)."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False)
    os.replace(tmp, path)


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
    log.info(f"  Log:        {log_file}")
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

    # Constrói manifesto em memória.
    # Aplica o limite no scan para modos sem GPU (evita hashing de 13k PDFs em testes).
    scan_limite = args.limite if args.modo in ("pymu", "paddle") else 0
    manifest = build_manifest_from_pdfs(pdfs_dir, territorio_nome, limite=scan_limite)

    # Salva manifesto para referência (usado pelo orquestrador legado)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "download_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info(f"  Manifesto gerado: {manifest_path}")
    log.info(f"  Modo:       {args.modo.upper()}")

    # Importa e executa o pipeline
    try:
        t0 = time.time()

        if args.modo == "paddle":
            # ── MODO PRINCIPAL: PaddleOCR PP-Structure + ProcessPoolExecutor ──
            stats = run_modo_paddle(
                manifest=manifest,
                output_dir=output_dir,
                slug=slug,
                territorio_nome=territorio_nome,
                limite=args.limite,
                verbose=args.verbose,
                workers=args.workers,
            )
            elapsed = time.time() - t0
            engine = f"PaddleOCR PP-Structure ({args.workers or 'auto'} workers)"

        elif args.modo == "pymu":
            # ── MODO LEGADO: PyMuPDF sem GPU ──────────────────────────────────
            if fitz is None:
                log.error("pymupdf não encontrado. Execute: uv add pymupdf")
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
            engine = "PyMuPDF (legado)"

        elif args.modo in ("marker", "hibrido"):
            # ── MODOS DESCONTINUADOS ──────────────────────────────────────────
            warnings.warn(
                f"O modo '{args.modo}' foi descontinuado. Use --modo paddle (padrão). "
                "Redirecionando para o modo paddle.",
                DeprecationWarning,
                stacklevel=2,
            )
            log.warning(f"Modo '{args.modo}' descontinuado → redirecionando para paddle.")
            stats = run_modo_paddle(
                manifest=manifest,
                output_dir=output_dir,
                slug=slug,
                territorio_nome=territorio_nome,
                limite=args.limite,
                verbose=args.verbose,
                workers=args.workers,
            )
            elapsed = time.time() - t0
            engine = f"PaddleOCR PP-Structure (redirecionado de {args.modo})"

        else:
            log.error(f"Modo desconhecido: {args.modo}")
            sys.exit(1)

    except ImportError as e:
        log.error(f"Erro de importação: {e}\nVerifique se executou: uv sync")
        sys.exit(1)

    # Relatório final
    elapsed_min = elapsed / 60
    needs_reprocessing = stats.get("com_tabela", 0) + stats.get("com_figura", 0)
    print("\n" + "=" * 65)
    print(f"EXTRAÇÃO CONCLUÍDA — {territorio_nome}")
    print("=" * 65)
    print(f"  Tempo total:      {elapsed:.1f}s ({elapsed_min:.1f} min)")
    print(f"  Motor:            {engine}")
    print(f"  Saída:            {output_dir}/")
    print(f"  Gerados:          {stats.get('gerados', 'N/A')}")
    print(f"  Duplicatas:       {stats.get('duplicatas', 'N/A')}")
    print(f"  Com tabela:       {stats.get('com_tabela', 'N/A')}")
    print(f"  Com figura:       {stats.get('com_figura', 'N/A')}")
    print(f"  Erros:            {stats.get('erros', 'N/A')}")
    if needs_reprocessing > 0:
        print(f"  Fila reprocess.:  {output_dir}/reprocessamento_pendente.jsonl")
    print("=" * 65 + "\n")

    log.info(f"Log completo salvo em: {log_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrator padronizado por território — DOM-PI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--territorio", required=True,
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
            "Modo de extração (stack sem Marker):\n"
            "  paddle  → PaddleOCR PP-Structure paralelo (PADRÃO)\n"
            "            Detecta tabelas/figuras com exatidão, grava flags no datalake.\n"
            "  pymu    → PyMuPDF direto, mais rápido mas sem análise de layout avançada.\n"
            "  hibrido → DESCONTINUADO. Redireciona para paddle.\n"
            "  marker  → DESCONTINUADO. Redireciona para paddle.\n"
            "Padrão: paddle"
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="[Modo paddle] Número de processos paralelos. Padrão: cpu_count()-1.",
    )
    parser.add_argument(
        "--limite", type=int, default=999_999,
        help="Máx. PDFs a processar (padrão: ilimitado). Use valores pequenos para testes.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.70,
        help="[Legado — ignorado] Score OCR para roteamento híbrido (padrão: 0.70).",
    )
    parser.add_argument(
        "--force-ocr", action="store_true",
        help="[Legado] Ignorado. Mantido por compatibilidade de scripts existentes.",
    )
    parser.add_argument(
        "--min-variance", type=float, default=50.0,
        help="[Legado — ignorado] Limiar de nitidez (era usado apenas pelo Marker).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Ativa logs detalhados (DEBUG).",
    )
    parser.add_argument(
        "--listar", action="store_true",
        help="Lista todos os territórios disponíveis e sai.",
    )

    args = parser.parse_args()

    if args.listar:
        print("\nTerritórios disponíveis:\n")
        for slug, nome in TERRITORIOS.items():
            print(f"  {slug:<40} → {nome}")
        print()
        sys.exit(0)

    # Compatibilidade: --force-ocr legado → modo marker
    if args.force_ocr and args.modo == "pymu":
        args.modo = "marker"

    run(args)


if __name__ == "__main__":
    main()
