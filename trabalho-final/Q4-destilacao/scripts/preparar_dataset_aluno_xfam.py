#!/usr/bin/env python3
"""
preparar_dataset_aluno_xfam.py — Converte o dataset_B (professor Qwen2.5-14B, braço B)
para o espaço de um ALUNO DE OUTRA FAMÍLIA (black-box: só o texto do professor).

Fecha a célula que faltava do quadro 2×2 white-box × black-box (§4.10b do relatório):
mesmo professor, mesmo dataset, mesmo sinal (CE) — muda só a família do aluno.

Não roda o professor de novo: reusa question/context/answer do dataset_B.jsonl e
  1) re-renderiza o prompt com o tokenizador do aluno (chat template; se o modelo base
     não tiver template — caso do Llama-3.2 base — usa o formato plano equivalente);
  2) re-tokeniza a resposta (texto) no vocabulário do aluno → answer_token_ids.

O resultado é consumido por destilar.py --method ce (hard label; KL é impossível entre
tokenizadores distintos — exatamente o ponto que o experimento quantifica).

Exemplo:
  python preparar_dataset_aluno_xfam.py --student meta-llama/Llama-3.2-1B \
      --dataset ../dados/dataset_B.jsonl --out ../dados/dataset_B_llama1b.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

# Mesmos textos de sistema do braço B (gerar_dataset_destilacao.py)
ANSWER_SYSTEM = "Você é um assistente factual e conciso. Responda de forma direta e objetiva."
ANSWER_SYSTEM_RAG = (
    ANSWER_SYSTEM
    + " Use exclusivamente o CONTEXTO fornecido. Se a resposta não estiver no contexto, diga que não consta."
)


def render_prompt(tok, question: str, context: str, max_context_chars: int) -> str:
    ctx = (context or "")[:max_context_chars]
    user = f"CONTEXTO:\n{ctx}\n\nPERGUNTA: {question}" if ctx else f"PERGUNTA: {question}"
    if getattr(tok, "chat_template", None):
        msgs = [{"role": "system", "content": ANSWER_SYSTEM_RAG},
                {"role": "user", "content": user}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # templates que não aceitam role system (ex.: Gemma)
            msgs = [{"role": "user", "content": f"{ANSWER_SYSTEM_RAG}\n\n{user}"}]
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # modelo base sem template (Llama-3.2 base): formato plano equivalente
    return f"{ANSWER_SYSTEM_RAG}\n\n{user}\nRESPOSTA:"


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset B re-tokenizado para aluno de outra família (Q4)")
    ap.add_argument("--student", required=True, help="ex.: meta-llama/Llama-3.2-1B")
    ap.add_argument("--dataset", default="trabalho-final/Q4-destilacao/dados/dataset_B.jsonl")
    ap.add_argument("--max-context-chars", type=int, default=6000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    n_in, n_out = 0, 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            n_in += 1
            r = json.loads(line)
            ans = (r.get("answer") or "").strip()
            if not ans:
                continue
            rec = {
                "id": r["id"],
                "source": r.get("source"),
                "question": r["question"],
                "context": r.get("context", ""),
                "prompt": render_prompt(tok, r["question"], r.get("context", ""), args.max_context_chars),
                "answer": ans,
                "answer_token_ids": tok(ans, add_special_tokens=False).input_ids,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"OK — {n_out}/{n_in} exemplos re-tokenizados p/ {args.student} em {out}", flush=True)


if __name__ == "__main__":
    main()
