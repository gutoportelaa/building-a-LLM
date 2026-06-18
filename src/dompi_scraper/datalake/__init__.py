"""
Pacote `datalake` — lakehouse local DOM-PI (medallion: extraido → limpo → corpus).

Estrutura (sob `DATALAKE_ROOT`, por padrão `./datalake`):

    datalake/
      extraido/  territorio=<slug>/ano=<AAAA>/*.parquet   1 linha/doc extraído + proveniência
      limpo/  territorio=<slug>/ano=<AAAA>/*.parquet   limpo, re-hash, dedup L3, flags por severidade
      corpus/
        corpus_llm/   ano=<AAAA>/*.parquet  +  shards/*.jsonl.zst
      _catalog/
        manifest.parquet      linhagem por doc (hashes por camada, rota, extrator)
        dedup_global.parquet  content_hash canônico cross-território

Motores: DuckDB (SQL sobre o lake, escrita Parquet particionada) + Polars (transforms
row-level, reusando `limpeza_textos.clean_text` e `shared_utils.compute_content_md5`).
Tudo CPU-leve — roda local no WSL sem GPU. A extração pesada (produtora do extraido)
roda no lab SLURM; aqui só ingerimos/transformamos texto.
"""
from __future__ import annotations

import os
from pathlib import Path


def datalake_root(root: str | os.PathLike | None = None) -> Path:
    """Raiz do data lake. Ordem: argumento explícito > env DOMPI_DATALAKE > ./datalake."""
    if root:
        return Path(root)
    env = os.environ.get("DOMPI_DATALAKE")
    return Path(env) if env else Path("datalake")


def zone_dir(zone: str, root: str | os.PathLike | None = None) -> Path:
    """Diretório de uma zona ('extraido', 'limpo', 'corpus', '_catalog')."""
    return datalake_root(root) / zone


def ensure_zones(root: str | os.PathLike | None = None) -> Path:
    """Cria a árvore de zonas se ainda não existir e devolve a raiz."""
    base = datalake_root(root)
    for z in ("extraido", "limpo", "corpus", "_catalog"):
        (base / z).mkdir(parents=True, exist_ok=True)
    return base


# Colunas canônicas de cada zona (contrato compartilhado entre os módulos).
# Mantém paridade com `orquestrador_extracao._CORPUS_SCHEMA`.
EXTRAIDO_COLUMNS = [
    "id_publicacao",    # md5 do conteúdo ORIGINAL (pré-limpeza)
    "territorio",
    "municipio",
    "tipo_ato",
    "data_publicacao",  # "AAAA-MM-DD" ou ""
    "ano",              # coluna de partição (derivada de data_publicacao)
    "extrator",
    # --- Demarcação da extração (proveniência) — nula em corpus legado ---
    "extraido_em",       # ISO-8601 Brasília do momento da extração
    "extracao_segundos", # custo de motor para o documento
    "paginas",           # nº de páginas extraídas
    "host",              # nó que extraiu (ex.: gpunode01)
    "job_id",            # SLURM_JOB_ID
    "texto",
    "n_chars",
    "fonte_ingest",     # arquivo NDJSON de origem (linhagem)
]

LIMPO_COLUMNS = EXTRAIDO_COLUMNS + [
    "id_limpo",                 # md5 do conteúdo LIMPO (P-09: re-hash pós-limpeza)
    "texto_limpo",
    "n_chars_limpo",
    "n_tokens",
    "assinaturas_detectadas",   # informativo (P-08): não marca revisão
    "needs_human_review",       # só razões de alta severidade (P-08)
    "review_reasons",
]
