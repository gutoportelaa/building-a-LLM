#!/usr/bin/env python3
"""
gerar_dataset_destilacao.py — Gera o dataset sintético de destilação (Q4).

Roda NO CLUSTER (gpunode01, 2× L4). Usa vLLM para servir o PROFESSOR
(Qwen2.5-14B-Instruct, tensor-parallel=2) com alto throughput e captura,
na mesma passada, os top-k logprobs por token (cache offline estilo Gemma 2).

Produz o material para o eixo experimental A × B:
  • Braço A ("zerada")  — professor responde só da memória paramétrica.
  • Braço B (RAG)       — professor responde com contexto recuperado do índice DOM-PI (Q5).
O conjunto de PERGUNTAS é idêntico entre A e B (a única diferença é o contexto),
para que o comparativo isole o efeito do grounding.

Saídas (em --out-dir):
  questoes.jsonl            — {id, source, question, seed_passage?}  (perguntas, comuns a A e B)
  contexto_B.jsonl          — {id, context}                          (recuperado, só braço B)
  dataset_A.jsonl / _B.jsonl— {id, source, question, context, answer, answer_token_ids}
  logits_A.jsonl / _B.jsonl — {id, topk: [[ [tok_id, logprob], ... ] por token de resposta]}

Exemplo:
  python gerar_dataset_destilacao.py \
      --teacher Qwen/Qwen2.5-14B-Instruct --tp 2 \
      --n-dompi 500 --n-docentes 500 --topk 50 --bracos A B \
      --dompi-seeds ../../../data/held_out.jsonl \
      --index-dir ../../../rag/index \
      --rag-scripts ../../Q5-rag/scripts \
      --out-dir ../dados
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Prompts                                                                      #
# --------------------------------------------------------------------------- #

QG_SYSTEM = (
    "Você é um gerador de perguntas factuais. Dada uma passagem, formule UMA única pergunta objetiva "
    "cuja resposta esteja contida na passagem (nomes, números, valores, datas, definições, fatos). "
    "Responda apenas com a pergunta, sem preâmbulo."
)

ANSWER_SYSTEM = (
    "Você é um assistente factual e conciso. Responda de forma direta e objetiva."
)

ANSWER_SYSTEM_RAG = (
    ANSWER_SYSTEM
    + " Use exclusivamente o CONTEXTO fornecido. Se a resposta não estiver no contexto, diga que não consta."
)

# Contexto-ouro (braço G): a passagem-fonte de onde a pergunta foi gerada É o contexto.
# Como a resposta está garantidamente na passagem, o professor nunca abstém ("não consta").
ANSWER_SYSTEM_GOLD = (
    ANSWER_SYSTEM
    + " Use o CONTEXTO fornecido (que contém a resposta) e responda de forma direta, citando o fato."
)

_ABSTENCAO_RE = re.compile(
    r"n[ãa]o\s+(consta|h[áa]\b|foi poss|menciona|informa|aborda|cont[eé]m|especifica|apresenta|"
    r"est[áa]\s+(no|presente|dispon))|sem\s+informa|n[ãa]o\s+(é|e)\s+poss[íi]vel|"
    r"nenhuma\s+informa|n[ãa]o\s+(há|ha)\s+(informa|men)",
    re.IGNORECASE,
)


def eh_abstencao(txt: str) -> bool:
    """True se a resposta do professor é uma abstenção (não traz fato)."""
    return bool(_ABSTENCAO_RE.search(txt or ""))


# --------------------------------------------------------------------------- #
# Construção das perguntas                                                     #
# --------------------------------------------------------------------------- #

def carregar_seeds_dompi(path: Path, n: int, seed: int, min_chars: int = 400) -> list[str]:
    """Amostra n passagens do held-out DOM-PI (campo 'texto')."""
    rng = random.Random(seed)
    passages: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            txt = (obj.get("texto") or obj.get("text") or "").strip()
            if len(txt) >= min_chars:
                passages.append(txt[:1600])
    rng.shuffle(passages)
    return passages[:n]


def carregar_seeds_docentes(n: int, seed: int, min_chars: int = 400) -> list[str]:
    """Amostra n passagens-semente do dataset HF vickminari/docentesDC (campo 'text').

    NB: docentesDC NÃO é Q&A — colunas reais = ['text', 'nome_professor']. Tratamos como o DOM-PI:
    passagem-semente → o professor gera a pergunta (self-instruct) na etapa de QG.
    """
    from datasets import load_dataset

    ds = load_dataset("vickminari/docentesDC", split="train")
    idx = list(range(len(ds)))
    random.Random(seed + 1).shuffle(idx)   # seed distinto do DOM-PI
    passages: list[str] = []
    for i in idx:
        txt = (ds[i].get("text") or "").strip()
        if len(txt) >= min_chars:
            passages.append(txt[:1600])
        if len(passages) >= n:
            break
    return passages


# --------------------------------------------------------------------------- #
# vLLM helpers                                                                 #
# --------------------------------------------------------------------------- #

def montar_chat(tokenizer, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extrair_topk(comp, topk: int) -> list[list[list[float]]]:
    """De um CompletionOutput do vLLM, devolve por token a lista [[tok_id, logprob], ...]."""
    out = []
    for pos in (comp.logprobs or []):
        # pos: dict[int, Logprob] (já é o top-k pedido em SamplingParams.logprobs)
        ranked = sorted(pos.items(), key=lambda kv: kv[1].rank)[:topk]
        out.append([[int(tid), round(float(lp.logprob), 4)] for tid, lp in ranked])
    return out


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Gera dataset sintético de destilação (Q4)")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--tp", type=int, default=2, help="tensor_parallel_size (2 L4 no gpunode01)")
    ap.add_argument("--n-dompi", type=int, default=500)
    ap.add_argument("--n-docentes", type=int, default=500)
    ap.add_argument("--min-seed-chars", type=int, default=400,
                    help="comprimento mínimo da passagem-semente (baixar p/ corpora de passagens curtas, ex. futebol)")
    ap.add_argument("--topk", type=int, default=50, help="top-k logprobs por token (soft labels)")
    ap.add_argument("--bracos", nargs="+", default=["A", "B"], choices=["A", "B", "G"])
    ap.add_argument("--gold-context", action="store_true",
                    help="atalho p/ --bracos G: usa a passagem-fonte como contexto (zero abstenção)")
    ap.add_argument("--only-source", choices=["DOM-PI", "docentesDC"], default=None,
                    help="restringe a uma fonte (gera dataset especialista de 1 tópico)")
    ap.add_argument("--filter-abstencao", dest="filter_abstencao", action="store_true", default=True,
                    help="descarta respostas-abstenção do professor (default: ligado)")
    ap.add_argument("--no-filter-abstencao", dest="filter_abstencao", action="store_false")
    ap.add_argument("--dompi-seeds", default="data/held_out.jsonl")
    ap.add_argument("--index-dir", default="rag/index", help="índice Q5 (braço B)")
    ap.add_argument("--rag-scripts", default="trabalho-final/Q5-rag/scripts")
    ap.add_argument("--rag-k", type=int, default=5)
    ap.add_argument("--max-context-chars", type=int, default=6000,
                    help="cap do contexto RAG (braço B) p/ não estourar a janela do professor")
    ap.add_argument("--embed-device", default="cpu",
                    help="device do e5 na recuperação; 'cpu' evita contenção com o vLLM nas GPUs")
    ap.add_argument("--out-dir", default="trabalho-final/Q4-destilacao/dados")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--enforce-eager", action="store_true", default=True,
                    help="desliga torch.compile/CUDA-graphs do vLLM (evita dependência de 'ninja'); "
                         "não afeta logprobs, só throughput")
    ap.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.gold_context:
        args.bracos = ["G"]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"[1/4] Carregando professor {args.teacher} (TP={args.tp}) no vLLM...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    llm = LLM(
        model=args.teacher,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_logprobs=args.topk,          # vLLM default é 20; pedir top-50 exige elevar aqui
        enforce_eager=args.enforce_eager,  # evita compilação (ninja) — robusto no cluster
        trust_remote_code=True,
        seed=args.seed,
    )

    # ----- Etapa 1: perguntas ------------------------------------------------
    # Ambas as fontes são tratadas como passagens-semente → o professor gera 1 pergunta por passagem
    # (self-instruct grounded). docentesDC NÃO é Q&A (colunas: text, nome_professor).
    print("[2/4] Construindo conjunto de perguntas (self-instruct nas 2 fontes)...", flush=True)
    seeds: list[tuple[str, str]] = []
    if args.only_source != "docentesDC":
        seeds += [("DOM-PI", p) for p in carregar_seeds_dompi(
            Path(args.dompi_seeds), args.n_dompi, args.seed, min_chars=args.min_seed_chars)]
    if args.only_source != "DOM-PI":
        seeds += [("docentesDC", p) for p in carregar_seeds_docentes(args.n_docentes, args.seed)]

    questoes: list[dict] = []
    if seeds:
        qg_prompts = [montar_chat(tok, QG_SYSTEM, f"Passagem:\n{p}\n\nPergunta:") for _, p in seeds]
        qg_sp = SamplingParams(temperature=0.0, max_tokens=64, seed=args.seed)
        qg_outs = llm.generate(qg_prompts, qg_sp)
        for (src, p), o in zip(seeds, qg_outs):
            q = o.outputs[0].text.strip().split("\n")[0].strip()
            if q:
                questoes.append({"source": src, "question": q, "seed_passage": p})

    for i, item in enumerate(questoes):
        item["id"] = f"q{i:05d}"
    (out / "questoes.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in questoes) + "\n", encoding="utf-8"
    )
    print(f"      {len(questoes)} perguntas ({sum(q['source']=='DOM-PI' for q in questoes)} DOM-PI + "
          f"{sum(q['source']=='docentesDC' for q in questoes)} docentesDC)", flush=True)

    # ----- Etapa 2a: contexto-ouro (braço G) — a passagem-fonte é o contexto ---
    contexto: dict[str, str] = {}
    seed_passage_by_id = {q["id"]: q["seed_passage"] for q in questoes}
    if "G" in args.bracos:
        for q in questoes:
            contexto[q["id"]] = q["seed_passage"][: args.max_context_chars]

    # ----- Etapa 2b: contexto RAG (braço B) ---------------------------------
    if "B" in args.bracos:
        print("[3/4] Recuperando contexto (braço B) do índice Q5...", flush=True)
        sys.path.insert(0, str(Path(args.rag_scripts).resolve()))
        from rag_core import E5Embedder, Retriever, _fmt_ctx  # type: ignore

        emb = E5Embedder(device=args.embed_device)
        retr = Retriever(index_dir=args.index_dir)
        for q in questoes:
            qvec = emb.encode_query(q["question"])
            passages = retr.search_vec(qvec, k=args.rag_k)
            # cap de segurança: contexto não pode estourar a janela do professor
            contexto[q["id"]] = _fmt_ctx(passages)[: args.max_context_chars]
        with open(out / "contexto_B.jsonl", "w", encoding="utf-8") as f:
            for q in questoes:
                f.write(json.dumps({"id": q["id"], "context": contexto[q["id"]]}, ensure_ascii=False) + "\n")

    # ----- Etapa 3: respostas do professor + logits -------------------------
    print("[4/4] Gerando respostas do professor + top-k logits...", flush=True)
    ans_sp = SamplingParams(
        temperature=0.0, max_tokens=args.max_new_tokens, logprobs=args.topk, seed=args.seed
    )
    for braco in args.bracos:
        if braco == "A":
            prompts = [montar_chat(tok, ANSWER_SYSTEM, q["question"]) for q in questoes]
        elif braco == "G":   # contexto-ouro: passagem-fonte contém a resposta
            prompts = [
                montar_chat(tok, ANSWER_SYSTEM_GOLD,
                            f"CONTEXTO:\n{contexto[q['id']]}\n\nPERGUNTA: {q['question']}")
                for q in questoes
            ]
        else:                # B: contexto RAG
            prompts = [
                montar_chat(tok, ANSWER_SYSTEM_RAG,
                            f"CONTEXTO:\n{contexto[q['id']]}\n\nPERGUNTA: {q['question']}")
                for q in questoes
            ]
        outs = llm.generate(prompts, ans_sp)

        f_ds = open(out / f"dataset_{braco}.jsonl", "w", encoding="utf-8")
        f_lg = open(out / f"logits_{braco}.jsonl", "w", encoding="utf-8")
        n_ok, n_skip = 0, 0
        for q, pr, o in zip(questoes, prompts, outs):
            comp = o.outputs[0]
            ans = comp.text.strip()
            if args.filter_abstencao and eh_abstencao(ans):
                n_skip += 1           # descarta abstenção (não é fato → polui a destilação)
                continue
            rec = {
                "id": q["id"], "source": q["source"], "question": q["question"],
                "context": contexto.get(q["id"], ""),
                "prompt": pr,  # prompt renderizado (chat template) — usado tal-e-qual na destilação
                "answer": ans,
                "answer_token_ids": [int(t) for t in comp.token_ids],
            }
            f_ds.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_lg.write(json.dumps({"id": q["id"], "topk": extrair_topk(comp, args.topk)},
                                  ensure_ascii=False) + "\n")
            n_ok += 1
        f_ds.close()
        f_lg.close()
        print(f"      braço {braco}: {n_ok} respostas salvas, {n_skip} abstenções descartadas", flush=True)

    print(f"OK — dataset em {out}/", flush=True)


if __name__ == "__main__":
    main()
