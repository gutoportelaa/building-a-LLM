#!/usr/bin/env python3
"""
gerar_dataset_crossfamilia.py — Plano B: black-box sequence-KD CROSS-FAMÍLIA.

Professor de outra família (default google/gemma-2-9b-it) gera RESPOSTAS EM TEXTO (sem logits — vocabulário
diferente) para as MESMAS perguntas e contexto (braço B/RAG) da Q4. O texto é re-tokenizado com o tokenizador
do ALUNO (Qwen) para o SFT — é assim que a destilação cross-família funciona (sequence-level, off-policy).

Reusa: dados/questoes.jsonl + dados/contexto_B.jsonl (da Q4). Saída no MESMO formato que destilar.py espera
(prompt Qwen + answer + answer_token_ids em espaço-Qwen) → `destilar.py --method ce` roda sem alteração.

Roda na .venv-q4gen (vllm). Exemplo:
  python gerar_dataset_crossfamilia.py --teacher google/gemma-2-9b-it --tp 2 --max-model-len 8192 \
      --questoes trabalho-final/Q4-destilacao/dados/questoes.jsonl \
      --contexto trabalho-final/Q4-destilacao/dados/contexto_B.jsonl \
      --out trabalho-final/Q4-destilacao/dados/dataset_Bxf.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gerar_dataset_destilacao import ANSWER_SYSTEM_RAG  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset cross-família black-box (Plano B)")
    ap.add_argument("--teacher", default="google/gemma-2-9b-it")
    ap.add_argument("--student-tokenizer", default="Qwen/Qwen2.5-1.5B",
                    help="tokenizador do aluno p/ re-tokenizar a resposta (qualquer Qwen2.5 serve)")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--questoes", default="trabalho-final/Q4-destilacao/dados/questoes.jsonl")
    ap.add_argument("--contexto", default="trabalho-final/Q4-destilacao/dados/contexto_B.jsonl")
    ap.add_argument("--max-context-chars", type=int, default=6000)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default="trabalho-final/Q4-destilacao/dados/dataset_Bxf.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0=todos; >0 para smoke")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    quest = {}
    for l in Path(args.questoes).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l); quest[r["id"]] = r
    ctx = {}
    for l in Path(args.contexto).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l); ctx[r["id"]] = r["context"]
    ids = [i for i in quest if i in ctx]
    if args.limit:
        ids = ids[: args.limit]
    print(f"{len(ids)} exemplos (reusa questões+contexto B da Q4)", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    t_tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    s_tok = AutoTokenizer.from_pretrained(args.student_tokenizer, trust_remote_code=True)
    llm = LLM(model=args.teacher, tensor_parallel_size=args.tp, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len,
              enforce_eager=True, trust_remote_code=True, seed=args.seed)

    def teacher_prompt(q: str, c: str) -> str:
        # system mesclado no user — robusto p/ Gemma (que não aceita role 'system')
        user = f"{ANSWER_SYSTEM_RAG}\n\nCONTEXTO:\n{c}\n\nPERGUNTA: {q}"
        return t_tok.apply_chat_template([{"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True)

    def student_prompt(q: str, c: str) -> str:
        # prompt de treino do aluno = idêntico ao braço B da Q4 (Qwen, system separado)
        msgs = [{"role": "system", "content": ANSWER_SYSTEM_RAG},
                {"role": "user", "content": f"CONTEXTO:\n{c}\n\nPERGUNTA: {q}"}]
        return s_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    ctxs = {i: ctx[i][: args.max_context_chars] for i in ids}
    prompts = [teacher_prompt(quest[i]["question"], ctxs[i]) for i in ids]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, seed=args.seed))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, o in zip(ids, outs):
            ans = o.outputs[0].text.strip()
            rec = {
                "id": i, "source": quest[i]["source"], "question": quest[i]["question"],
                "context": ctxs[i], "prompt": student_prompt(quest[i]["question"], ctxs[i]),
                "answer": ans,
                "answer_token_ids": s_tok(ans, add_special_tokens=False).input_ids,  # re-tokeniza em espaço-Qwen
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"OK — {len(ids)} exemplos cross-família ({args.teacher}) em {out}", flush=True)


if __name__ == "__main__":
    main()
