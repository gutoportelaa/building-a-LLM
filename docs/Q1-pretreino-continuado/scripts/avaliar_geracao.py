#!/usr/bin/env python3
"""
avaliar_geracao.py — Acurácia de GERAÇÃO no benchmark DOM-PI (camada "ganho de domínio").

Complementa avaliar_modelo.py: em vez de medir só a NLL/PPL da resposta de
referência condicionada ao prompt (uma proxy de confiança), aqui o modelo
GERA a resposta livremente e comparamos com a referência. Isso produz um
número interpretável pela banca ("acertou X de N"), no espírito de como
projetos de DAPT reportam ganho de domínio (Juru, AdaptLLM, ChipNeMo).

Três métricas, da mais branda à mais estrita:
  - contains : a resposta de referência aparece (normalizada) na geração
  - token_f1 : F1 de tokens entre geração e referência (parcial conta)
  - exact_match : geração normalizada == referência normalizada

Uso:
    python scripts/avaliar_geracao.py \
        --model Qwen/Qwen2.5-1.5B \
        --benchmark dompi_qa.jsonl \
        --output resultados/geracao_baseline.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Normalização e métricas de texto
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """minúsculas, sem acento, sem pontuação, espaços colapsados."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(pred: str, ref: str) -> float:
    """F1 de tokens (estilo SQuAD): recompensa acerto parcial de spans factuais."""
    p_tokens = normalize(pred).split()
    r_tokens = normalize(ref).split()
    if not p_tokens or not r_tokens:
        return 0.0
    common = Counter(p_tokens) & Counter(r_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_tokens)
    recall = n_same / len(r_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, ref: str) -> bool:
    return normalize(pred) == normalize(ref)


def contains(pred: str, ref: str) -> bool:
    """A referência inteira (normalizada) aparece como substring da geração."""
    return normalize(ref) in normalize(pred)


# ---------------------------------------------------------------------------
# Geração
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # corta na primeira quebra dupla/nova pergunta para não vazar contexto
    for stop in ["\nPergunta", "\n\n", "\nQual", "\nQuem"]:
        idx = text.find(stop)
        if idx > 0:
            text = text[:idx]
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Acurácia de geração no benchmark DOM-PI")
    parser.add_argument("--model", required=True, help="HF id ou checkpoint local")
    parser.add_argument("--benchmark", default="dompi_qa.jsonl")
    parser.add_argument("--output", default="resultados/geracao.json")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--f1-threshold", type=float, default=0.5,
                        help="Limiar de token_f1 para contar como 'acerto aproximado'")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model) if os.path.isdir(args.model) else args.model
    log.info("Dispositivo: %s | Modelo: %s", DEVICE, model_path)

    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()

    items = [json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip()]
    log.info("%d perguntas no benchmark", len(items))

    per_item = []
    n_em = n_contains = n_f1ok = 0
    f1_sum = 0.0

    for i, item in enumerate(items):
        prompt = item["pergunta"] + " Resposta: "
        ref = item["resposta_ref"]
        gen = generate(model, tokenizer, prompt, args.max_new_tokens)

        em = exact_match(gen, ref)
        ct = contains(gen, ref)
        f1 = token_f1(gen, ref)
        n_em += int(em)
        n_contains += int(ct)
        n_f1ok += int(f1 >= args.f1_threshold)
        f1_sum += f1

        per_item.append({
            "pergunta": item["pergunta"], "resposta_ref": ref, "geracao": gen,
            "tipo": item.get("tipo"), "exact_match": em, "contains": ct,
            "token_f1": round(f1, 4),
        })
        if i < 3 or ct:
            log.info("[%d] f1=%.2f contains=%s | ref=%r | gen=%r", i, f1, ct, ref[:60], gen[:80])

    n = len(items)
    results = {
        "model": args.model,
        "benchmark": args.benchmark,
        "n": n,
        "gen_exact_match": round(n_em / n, 4),
        "gen_contains": round(n_contains / n, 4),
        "gen_f1_mean": round(f1_sum / n, 4),
        "gen_f1_at_threshold": round(n_f1ok / n, 4),
        "f1_threshold": args.f1_threshold,
        "n_exact_match": n_em,
        "n_contains": n_contains,
        "n_f1ok": n_f1ok,
        "per_item": per_item,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    log.info("=" * 60)
    log.info("ACURÁCIA DE GERAÇÃO — %s", args.model)
    log.info("  exact_match     : %d/%d = %.1f%%", n_em, n, 100 * n_em / n)
    log.info("  contains (ref)  : %d/%d = %.1f%%", n_contains, n, 100 * n_contains / n)
    log.info("  token_f1 >= %.2f : %d/%d = %.1f%%", args.f1_threshold, n_f1ok, n, 100 * n_f1ok / n)
    log.info("  token_f1 médio  : %.4f", f1_sum / n)
    log.info("  salvo em %s", out)


if __name__ == "__main__":
    main()
