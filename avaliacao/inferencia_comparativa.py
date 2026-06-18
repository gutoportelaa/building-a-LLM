#!/usr/bin/env python3
"""
inferencia_comparativa.py — Gera amostras de texto dos dois modelos para comparação qualitativa.

Uso:
    python avaliacao/inferencia_comparativa.py \
        --baseline Qwen/Qwen2.5-0.5B \
        --postreino treino/checkpoints/best \
        --output avaliacao/inferencias.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    {
        "id": "completar_portaria",
        "tipo": "completação_domínio",
        "prompt": "PORTARIA Nº 130, DE 12 DE MARÇO DE 2025.\n\nO PREFEITO MUNICIPAL DE COCAL, Estado do Piauí, no uso de suas atribuições legais,\n\nRESOLVE:\n\nArt. 1º",
    },
    {
        "id": "resposta_licitacao",
        "tipo": "q&a_factual",
        "prompt": "Pergunta: Qual é o objeto do contrato de prestação de serviços firmado pela Prefeitura Municipal de Pedro II com a empresa ANTARES?\n\nResposta:",
    },
    {
        "id": "conceito_juridico",
        "tipo": "q&a_conceitual",
        "prompt": "Pergunta: O que é um Pregão Eletrônico no contexto das licitações públicas brasileiras?\n\nResposta:",
    },
    {
        "id": "completar_contrato",
        "tipo": "completação_domínio",
        "prompt": "EXTRATO DE CONTRATO\n\nContratante: PREFEITURA MUNICIPAL DE TERESINA\nContratada: EMPRESA DE TECNOLOGIA LTDA\nObjeto: Prestação de serviços de",
    },
    {
        "id": "pergunta_diario",
        "tipo": "q&a_domínio",
        "prompt": "Pergunta: O que é o Diário Oficial dos Municípios do Piauí e qual é sua finalidade legal?\n\nResposta:",
    },
]


def load_model(model_path: str, dtype):
    path = model_path
    if os.path.isdir(path):
        path = os.path.abspath(path)
    log.info("Carregando %s...", path)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # greedy para reprodutibilidade
        temperature=1.0,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--postreino", default="treino/checkpoints/best")
    parser.add_argument("--output", default="avaliacao/inferencias.json")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    args = parser.parse_args()

    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    results = []

    for label, path in [("baseline", args.baseline), ("postreino", args.postreino)]:
        log.info("=== Modelo: %s ===", label)
        tokenizer, model = load_model(path, dtype)
        for item in PROMPTS:
            log.info("Prompt: %s", item["id"])
            saida = generate(model, tokenizer, item["prompt"], args.max_new_tokens)
            results.append({
                "modelo": label,
                "prompt_id": item["id"],
                "tipo": item["tipo"],
                "prompt": item["prompt"],
                "geracao": saida,
            })
        del model, tokenizer
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    log.info("Inferências salvas em %s", out_path)

    # Exibe tabela no stdout para o log do SLURM
    print("\n" + "=" * 80)
    print("INFERÊNCIAS COMPARATIVAS — BASELINE vs PÓS-TREINO")
    print("=" * 80)
    for item in PROMPTS:
        print(f"\n{'─'*80}")
        print(f"[{item['id']}] ({item['tipo']})")
        print(f"PROMPT:\n{item['prompt']}")
        for label in ("baseline", "postreino"):
            entry = next(r for r in results if r["modelo"] == label and r["prompt_id"] == item["id"])
            print(f"\n  [{label.upper()}]:\n  {entry['geracao'][:400]}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
