#!/usr/bin/env python3
"""
avaliar_destilacao.py — Avalia a transferência de conhecimento (Q4).

Roda o benchmark de 100 perguntas (50 DOM-PI + 50 docentesDC) em vários modelos e mede,
SEM RAG (testando o que ficou nos pesos), o quanto cada aluno se aproxima da referência.

Benchmark esperado (`--benchmark`, jsonl): {id, source, question, reference}
  reference = resposta do PROFESSOR com RAG (braço B) — melhor proxy de verdade factual disponível.

Para cada modelo gera a resposta (greedy, sem contexto) e calcula:
  • ROUGE-L (F1) vs referência  — proximidade de conteúdo;
  • recall de termos-chave       — fração de entidades/números da referência presentes na resposta;
  • (opcional) os dois agregados por domínio (DOM-PI × docentesDC).

Compare aluno-base × aluno-destilado (e CE × KL × combined, A × B) para responder
"houve transferência?". O professor entra como teto.

Uso:
  python avaliar_destilacao.py --benchmark ../dados/benchmark_destilacao_100.jsonl \
      --model base05=Qwen/Qwen2.5-0.5B \
      --model d_05_B_combined=../modelos/aluno_qwen2.5-0.5b_B_combined \
      ... \
      --out ../resultados/avaliacao.json
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANSWER_SYSTEM = "Você é um assistente factual e conciso. Responda de forma direta e objetiva."


# --------------------------------------------------------------------------- #
# Métricas                                                                     #
# --------------------------------------------------------------------------- #

def _norm(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.findall(r"\w+", s)


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F1 baseado na subsequência comum mais longa (LCS) de tokens."""
    a, b = _norm(pred), _norm(ref)
    if not a or not b:
        return 0.0
    # LCS por programação dinâmica (O(len(a)*len(b)))
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        ai = a[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, len(b) + 1):
            row[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], row[j - 1])
    lcs = dp[len(a)][len(b)]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(a), lcs / len(b)
    return 2 * prec * rec / (prec + rec)


_KEY_RE = re.compile(r"[A-ZÀ-Ý][\wÀ-ÿ.]{2,}|\d[\d.,/-]*")

def key_term_recall(pred: str, ref: str) -> float:
    """Fração de termos-chave da referência (Maiúsculas/Números) presentes na predição."""
    keys = {k.lower() for k in _KEY_RE.findall(ref)}
    if not keys:
        return float("nan")
    pred_l = pred.lower()
    hit = sum(1 for k in keys if k.lower() in pred_l)
    return hit / len(keys)


# --------------------------------------------------------------------------- #
# Geração                                                                      #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def responder(model, tok, question: str, max_new_tokens: int = 200) -> str:
    msgs = [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": question}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = out[0, ids["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def avaliar_modelo(rotulo: str, path: str, bench: list[dict], max_new_tokens: int) -> dict:
    import os
    p = os.path.abspath(path) if os.path.isdir(path) else path
    tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        p, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()

    por_dom: dict[str, list] = {}
    detalhe = []
    for item in bench:
        ans = responder(model, tok, item["question"], max_new_tokens)
        r = rouge_l(ans, item["reference"])
        k = key_term_recall(ans, item["reference"])
        por_dom.setdefault(item["source"], []).append((r, k))
        detalhe.append({"id": item["id"], "source": item["source"], "rougeL": round(r, 4),
                        "key_recall": None if k != k else round(k, 4), "answer": ans})

    def agg(pares):
        rs = [r for r, _ in pares]
        ks = [k for _, k in pares if k == k]
        return {"n": len(pares),
                "rougeL": round(sum(rs) / len(rs), 4) if rs else None,
                "key_recall": round(sum(ks) / len(ks), 4) if ks else None}

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {"rotulo": rotulo, "path": path,
            "por_dominio": {d: agg(v) for d, v in por_dom.items()},
            "geral": agg([p for v in por_dom.values() for p in v]),
            "detalhe": detalhe}


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Avalia transferência de conhecimento na destilação (Q4)")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--model", action="append", required=True, metavar="ROTULO=CAMINHO",
                    help="repetível; ex.: base05=Qwen/Qwen2.5-0.5B  d_05_B_kl=../modelos/aluno_qwen2.5-0.5b_B_kl")
    ap.add_argument("--out", default="trabalho-final/Q4-destilacao/resultados/avaliacao.json")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    bench = [json.loads(l) for l in Path(args.benchmark).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Benchmark: {len(bench)} perguntas "
          f"({sum(b['source']=='DOM-PI' for b in bench)} DOM-PI + "
          f"{sum(b['source']=='docentesDC' for b in bench)} docentesDC)", flush=True)

    resultados = []
    for spec in args.model:
        rotulo, _, path = spec.partition("=")
        print(f"\n>> Avaliando {rotulo} ({path})...", flush=True)
        res = avaliar_modelo(rotulo, path, bench, args.max_new_tokens)
        g = res["geral"]
        print(f"   geral: ROUGE-L={g['rougeL']}  key_recall={g['key_recall']}", flush=True)
        for d, m in res["por_dominio"].items():
            print(f"     {d}: ROUGE-L={m['rougeL']}  key_recall={m['key_recall']}  (n={m['n']})", flush=True)
        resultados.append(res)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"benchmark": args.benchmark, "n": len(bench), "modelos": resultados},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados em {out}", flush=True)


if __name__ == "__main__":
    main()
