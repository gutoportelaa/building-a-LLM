#!/usr/bin/env python3
"""
gerar_benchmark.py — Cria o benchmark de 100 perguntas da Q4 (50 DOM-PI + 50 docentesDC).

HELD-OUT por construção: usa as MESMAS funções de amostragem do dataset de treino com a mesma
seed, mas pega as fatias [n_treino : n_treino+50] — como o embaralhamento é determinístico, essas
50+50 passagens são DISJUNTAS das usadas no treino (que pegou [:n_treino]).

O professor (vLLM 14B) gera a pergunta (self-instruct) e a resposta COM RAG (braço B) — a resposta
RAG-grounded é a referência factual do benchmark.

Saída: benchmark_destilacao_100.jsonl  {id, source, question, reference}

Roda NO CLUSTER, na .venv-q4gen (vllm + sentence-transformers). Exemplo:
  python gerar_benchmark.py --n-treino-dompi 500 --n-treino-docentes 500 --n-bench 50 \
      --teacher Qwen/Qwen2.5-14B-Instruct --tp 2 --max-model-len 8192 \
      --dompi-seeds data/held_out.jsonl --index-dir rag/index \
      --rag-scripts trabalho-final/Q5-rag/scripts \
      --out trabalho-final/Q4-destilacao/dados/benchmark_destilacao_100.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# reusa as funções do gerador de dataset (mesmo diretório)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gerar_dataset_destilacao import (  # type: ignore
    carregar_seeds_dompi, carregar_seeds_docentes, montar_chat,
    QG_SYSTEM, ANSWER_SYSTEM_RAG, ANSWER_SYSTEM_GOLD, eh_abstencao,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera o benchmark held-out de 100 perguntas (Q4)")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--n-treino-dompi", type=int, default=500)
    ap.add_argument("--n-treino-docentes", type=int, default=500)
    ap.add_argument("--n-bench", type=int, default=50, help="por domínio (50 → 100 no total)")
    ap.add_argument("--dompi-seeds", default="data/held_out.jsonl")
    ap.add_argument("--index-dir", default="rag/index")
    ap.add_argument("--rag-scripts", default="trabalho-final/Q5-rag/scripts")
    ap.add_argument("--rag-k", type=int, default=5)
    ap.add_argument("--max-context-chars", type=int, default=6000)
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--min-seed-chars", type=int, default=400)
    ap.add_argument("--skip-docentes", action="store_true", help="só a fonte dompi-seeds (ex.: tema futebol)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gold-context", action="store_true",
                    help="usa a passagem-fonte como contexto da referência (zero 'não consta')")
    ap.add_argument("--filter-abstencao", dest="filter_abstencao", action="store_true", default=True)
    ap.add_argument("--no-filter-abstencao", dest="filter_abstencao", action="store_false")
    ap.add_argument("--out", default="trabalho-final/Q4-destilacao/dados/benchmark_destilacao_100.jsonl")
    args = ap.parse_args()

    # over-sample o held-out p/ poder descartar abstenções e ainda fechar n_bench por fonte
    pool = args.n_bench * 2 if args.filter_abstencao else args.n_bench
    # fatias HELD-OUT: [n_treino : n_treino + pool]  (disjuntas do treino, mesmo seed)
    dompi = carregar_seeds_dompi(Path(args.dompi_seeds), args.n_treino_dompi + pool, args.seed,
                                 min_chars=args.min_seed_chars)
    seeds = [("DOM-PI", p) for p in dompi[args.n_treino_dompi:]]
    if not args.skip_docentes:
        doc = carregar_seeds_docentes(args.n_treino_docentes + pool, args.seed)
        seeds += [("docentesDC", p) for p in doc[args.n_treino_docentes:]]
    print(f"{len(seeds)} sementes held-out "
          f"({sum(s=='DOM-PI' for s, _ in seeds)} DOM-PI + {sum(s=='docentesDC' for s, _ in seeds)} docentes)",
          flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    llm = LLM(model=args.teacher, tensor_parallel_size=args.tp, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len,
              enforce_eager=True, trust_remote_code=True, seed=args.seed)

    # 1) perguntas (self-instruct) — mantém a passagem-fonte p/ o contexto-ouro
    qg_prompts = [montar_chat(tok, QG_SYSTEM, f"Passagem:\n{p}\n\nPergunta:") for _, p in seeds]
    qg_outs = llm.generate(qg_prompts, SamplingParams(temperature=0.0, max_tokens=64, seed=args.seed))
    itens = []
    for (src, p), o in zip(seeds, qg_outs):
        q = o.outputs[0].text.strip().split("\n")[0].strip()
        if q:
            itens.append({"source": src, "question": q, "seed_passage": p})

    # 2) contexto + 3) referência (resposta do professor)
    if args.gold_context:
        ans_prompts = [
            montar_chat(tok, ANSWER_SYSTEM_GOLD,
                        f"CONTEXTO:\n{it['seed_passage'][: args.max_context_chars]}\n\nPERGUNTA: {it['question']}")
            for it in itens
        ]
    else:
        sys.path.insert(0, str(Path(args.rag_scripts).resolve()))
        from rag_core import E5Embedder, Retriever, _fmt_ctx  # type: ignore
        emb = E5Embedder(device=args.embed_device)
        retr = Retriever(index_dir=args.index_dir)
        ans_prompts = []
        for it in itens:
            ctx = _fmt_ctx(retr.search_vec(emb.encode_query(it["question"]), k=args.rag_k))[: args.max_context_chars]
            ans_prompts.append(montar_chat(tok, ANSWER_SYSTEM_RAG,
                                           f"CONTEXTO:\n{ctx}\n\nPERGUNTA: {it['question']}"))
    ans_outs = llm.generate(ans_prompts, SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, seed=args.seed))
    for it, o in zip(itens, ans_outs):
        it["reference"] = o.outputs[0].text.strip()

    # 4) filtra abstenções e fecha n_bench por fonte
    n_ab = 0
    por_fonte: dict[str, list[dict]] = {"DOM-PI": [], "docentesDC": []}
    for it in itens:
        if args.filter_abstencao and eh_abstencao(it["reference"]):
            n_ab += 1
            continue
        if len(por_fonte[it["source"]]) < args.n_bench:
            por_fonte[it["source"]].append(it)
    final = por_fonte["DOM-PI"] + por_fonte["docentesDC"]
    for i, it in enumerate(final):       # renumera e remove a passagem (não vai p/ o benchmark)
        it["id"] = f"bm{i:03d}"
        it.pop("seed_passage", None)
    final = [{"id": it["id"], "source": it["source"], "question": it["question"],
              "reference": it["reference"]} for it in final]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(it, ensure_ascii=False) for it in final) + "\n", encoding="utf-8")
    print(f"Benchmark salvo: {len(final)} perguntas em {out} "
          f"({len(por_fonte['DOM-PI'])} DOM-PI + {len(por_fonte['docentesDC'])} docentes; "
          f"{n_ab} abstenções descartadas)", flush=True)


if __name__ == "__main__":
    main()
