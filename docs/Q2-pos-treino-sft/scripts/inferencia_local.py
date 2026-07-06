#!/usr/bin/env python3
"""
inferencia_local.py — Roda os modelos pós-treinados localmente (apresentação ao vivo).

Carrega um modelo (base ou pós-treinado, caminho local em modelos/) e gera respostas
no MESMO formato ChatML do treino. Dois modos:

  • --bench : responde as 30 questões do benchmark dc_bench.jsonl (lado a lado com a referência)
  • --prompt "..." ou interativo : responde uma instrução qualquer

A RTX 4070 (8 GB) comporta os 0.5B e 1.5B (use --8bit se faltar VRAM no 1.5B).

Exemplos:
  # demo no benchmark com o SFT full 1.5B
  python inferencia_local.py --model ../modelos/sft_full_1.5b --bench

  # comparar base × full numa pergunta
  python inferencia_local.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt "O que é uma pilha?"
  python inferencia_local.py --model ../modelos/sft_full_1.5b --prompt "O que é uma pilha?"

  # modo interativo
  python inferencia_local.py --model ../modelos/sft_qlora_1.5b
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM_MSG = ("Você é um assistente especialista em Ciência da Computação que responde "
              "exercícios da disciplina de forma correta, clara e objetiva.")


def carregar(model_path, use_8bit):
    p = os.path.abspath(model_path) if os.path.isdir(model_path) else model_path
    tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(device_map="auto", trust_remote_code=True)
    if use_8bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kw["dtype"] = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(p, **kw)
    model.eval()
    return tok, model


@torch.no_grad()
def responder(tok, model, instr, inp="", max_new_tokens=256):
    user = instr if not inp else f"{instr}\n\n{inp}"
    msgs = [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ins = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ins, max_new_tokens=max_new_tokens, do_sample=False,
                         repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ins["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="caminho local (modelos/sft_*) ou HF id")
    ap.add_argument("--bench", action="store_true", help="responde o benchmark dc_bench.jsonl")
    ap.add_argument("--benchmark", default=str(Path(__file__).resolve().parent.parent / "benchmark" / "dc_bench.jsonl"))
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--8bit", dest="use_8bit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()

    print(f"[carregando] {a.model} em {DEVICE}{' (8-bit)' if a.use_8bit else ''}...", flush=True)
    tok, model = carregar(a.model, a.use_8bit)

    if a.bench:
        itens = [json.loads(l) for l in Path(a.benchmark).read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, it in enumerate(itens):
            r = responder(tok, model, it["instruction"], it.get("input", ""), a.max_new_tokens)
            print("\n" + "=" * 78)
            print(f"#{i+1} [{it.get('tipo')}] {it['instruction']}")
            if it.get("input"):
                print(f"   input: {it['input']}")
            print(f"   REF: {it['resposta_ref']}")
            print(f"   >>> {r}")
    elif a.prompt:
        print("\n>>> " + responder(tok, model, a.prompt, max_new_tokens=a.max_new_tokens))
    else:
        print("Modo interativo (Ctrl-C para sair).")
        while True:
            try:
                q = input("\nPergunta> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if q:
                print(">>> " + responder(tok, model, q, max_new_tokens=a.max_new_tokens))


if __name__ == "__main__":
    main()
