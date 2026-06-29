#!/usr/bin/env python3
"""
avaliar_sft.py — Avaliação ANTES×DEPOIS do SFT (Q2/Q3), no formato em que o
modelo foi treinado (ChatML do Qwen2.5).

Duas camadas numa só passada, sobre um arquivo de pares {instruction, input, output, tipo}:

  (1) INTRÍNSECA (teacher forcing) — NLL/PPL/CE e token-accuracy da RESPOSTA de
      referência condicionada ao prompt. Mede o quanto o modelo "concorda" com a
      resposta-alvo. Usado no held-out 20%.

  (2) GERAÇÃO (livre) — o modelo gera a resposta e comparamos com a referência
      por exact_match / contains / token_f1 (estilo SQuAD). Número interpretável
      pela banca ("acertou X de N"). Usado no benchmark CC/UFPI.

Mesmo SYSTEM_MSG e template de sft_docentes.py → avaliação fiel ao treino.

Saída: JSON com métricas agregadas + por-item (inclui a geração COMPLETA, p/ os
painéis do relatório no padrão da banca: mostrar a resposta inteira do modelo).

Uso:
  python avaliar_sft.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --data ../dados/pares_heldout.jsonl --modo intrinseca \
      --output ../resultados/heldout_baseline_1.5b.json

  python avaliar_sft.py --model ../modelos/sft_full_1.5b \
      --data ../benchmark/dc_bench.jsonl --modo geracao \
      --output ../resultados/bench_sft_full_1.5b.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM_MSG = ("Você é um assistente especialista em Ciência da Computação que responde "
              "exercícios da disciplina de forma correta, clara e objetiva.")


# ----------------------- normalização / métricas de texto ------------------ #

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(pred: str, ref: str) -> float:
    p, r = normalize(pred).split(), normalize(ref).split()
    if not p or not r:
        return 0.0
    common = Counter(p) & Counter(r)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(r)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, ref: str) -> bool:
    return normalize(pred) == normalize(ref)


def contains(pred: str, ref: str) -> bool:
    return normalize(ref) in normalize(pred)


# ----------------------------- prompt (ChatML) ----------------------------- #

def montar_prompt(tok, rec: dict) -> str:
    instr = (rec.get("instruction") or rec.get("pergunta") or "").strip()
    inp = (rec.get("input") or "").strip()
    user = instr if not inp else f"{instr}\n\n{inp}"
    msgs = [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def ref_de(rec: dict) -> str:
    return (rec.get("output") or rec.get("resposta_ref") or "").strip()


# ------------------------------ avaliações --------------------------------- #

@torch.no_grad()
def avaliar_intrinseca(model, tok, itens, max_len=1024):
    total_loss = total_correct = total_tok = 0
    per_item = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    for rec in itens:
        prompt = montar_prompt(tok, rec)
        ref = ref_de(rec)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        a_ids = tok(ref, add_special_tokens=False).input_ids + [tok.eos_token_id]
        if not a_ids:
            continue
        budget = max_len - len(a_ids)
        p_ids = p_ids[-max(8, budget):]
        ids = torch.tensor([p_ids + a_ids], device=DEVICE)
        logits = model(input_ids=ids).logits[0]
        pl = len(p_ids)
        ans_logits = logits[pl - 1: pl - 1 + len(a_ids)]
        ans_labels = torch.tensor(a_ids, device=DEVICE)
        loss = loss_fn(ans_logits, ans_labels)
        nll = loss.sum().item()
        corr = (ans_logits.argmax(-1) == ans_labels).sum().item()
        total_loss += nll
        total_correct += corr
        total_tok += len(a_ids)
        per_item.append({"instruction": rec.get("instruction"), "tipo": rec.get("tipo"),
                         "ref_nll": round(nll / len(a_ids), 4),
                         "ref_tok_acc": round(corr / len(a_ids), 4)})
    ce = total_loss / max(1, total_tok)
    return {
        "modo": "intrinseca", "n": len(per_item),
        "cross_entropy": round(ce, 4), "perplexity": round(math.exp(min(ce, 20)), 4),
        "token_accuracy": round(total_correct / max(1, total_tok), 4),
        "per_item": per_item,
    }


@torch.no_grad()
def gerar(model, tok, prompt, max_new_tokens):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][n_in:], skip_special_tokens=True).strip()


@torch.no_grad()
def avaliar_geracao(model, tok, itens, max_new_tokens=256, f1_thr=0.5):
    n_em = n_ct = n_f1 = 0
    f1_sum = 0.0
    per_item = []
    for rec in itens:
        prompt = montar_prompt(tok, rec)
        ref = ref_de(rec)
        gen = gerar(model, tok, prompt, max_new_tokens)
        em, ct, f1 = exact_match(gen, ref), contains(gen, ref), token_f1(gen, ref)
        n_em += em; n_ct += ct; n_f1 += int(f1 >= f1_thr); f1_sum += f1
        per_item.append({"instruction": rec.get("instruction"), "tipo": rec.get("tipo"),
                         "resposta_ref": ref, "geracao": gen,  # geração COMPLETA p/ a banca
                         "exact_match": em, "contains": ct, "token_f1": round(f1, 4)})
    n = max(1, len(itens))
    return {
        "modo": "geracao", "n": len(itens),
        "gen_exact_match": round(n_em / n, 4), "gen_contains": round(n_ct / n, 4),
        "gen_f1_mean": round(f1_sum / n, 4), "gen_f1_at_thr": round(n_f1 / n, 4),
        "f1_threshold": f1_thr, "per_item": per_item,
    }


def main():
    ap = argparse.ArgumentParser(description="Avaliação SFT antes×depois (formato ChatML)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="JSONL de pares {instruction,input,output,tipo}")
    ap.add_argument("--modo", choices=["intrinseca", "geracao"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    model_path = os.path.abspath(args.model) if os.path.isdir(args.model) else args.model
    print(f"Dispositivo: {DEVICE} | modelo: {model_path} | modo: {args.modo}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True)
    model.eval()

    itens = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{len(itens)} itens", flush=True)

    if args.modo == "intrinseca":
        res = avaliar_intrinseca(model, tok, itens, max_len=args.max_len)
        print(f"CE={res['cross_entropy']} PPL={res['perplexity']} TokAcc={res['token_accuracy']}", flush=True)
    else:
        res = avaliar_geracao(model, tok, itens, max_new_tokens=args.max_new_tokens)
        print(f"EM={res['gen_exact_match']} contains={res['gen_contains']} "
              f"F1={res['gen_f1_mean']} F1@thr={res['gen_f1_at_thr']}", flush=True)

    res["model"] = args.model
    res["data"] = args.data
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"Salvo em {out}", flush=True)


if __name__ == "__main__":
    main()
