#!/usr/bin/env python3
"""
avaliar_juiz.py — LLM-as-judge (MT-Bench style) sobre as GERAÇÕES da Q2/Q3.

Lê o JSON produzido por `avaliar_sft.py --modo geracao` (que tem per_item com
instruction / resposta_ref / geracao) e pede ao MESMO professor (Qwen2.5-14B-Instruct
servido com vLLM, idêntico à geração dos pares) uma nota 1-5 para cada geração,
comparando-a com a resposta de referência.

Para instruções abertas (explicação, resumo, comparação), EM/F1 penalizam
parafrases corretas; o juiz captura "a resposta está correta e útil?" — é o
sinal forte do antes×depois.

Roda NO CLUSTER (gpunode01, venv .venv-q4gen com vLLM).

Uso:
  python avaliar_juiz.py --teacher Qwen/Qwen2.5-14B-Instruct --tp 2 \
      --gen-json ../resultados/bench_sft_full_1.5b.json \
      --output ../resultados/juiz_sft_full_1.5b.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

JUDGE_SYSTEM = (
    "Você é um avaliador rigoroso de respostas de Ciência da Computação. Recebe uma PERGUNTA, "
    "uma RESPOSTA DE REFERÊNCIA (correta) e uma RESPOSTA DO MODELO. Atribua uma nota inteira de 1 a 5:\n"
    "5 = correta, completa e clara;  4 = correta com pequenas imprecisões/omissões;\n"
    "3 = parcialmente correta;  2 = majoritariamente incorreta;  1 = errada, vazia ou sem sentido.\n"
    "Considere correção factual e se de fato responde à pergunta (paráfrases corretas valem nota alta). "
    "Responda APENAS com o número."
)


def montar_chat(tok, system, user):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser(description="LLM-as-judge das gerações Q2/Q3")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gen-json", required=True, help="saída de avaliar_sft.py --modo geracao")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--quantization", default=None, help="ex.: 'awq' p/ rodar 14B-AWQ em 1 L4")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = json.loads(Path(args.gen_json).read_text(encoding="utf-8"))
    itens = data["per_item"]
    print(f"{len(itens)} gerações a julgar (de {args.gen_json})", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    llm_kwargs = dict(model=args.teacher, tensor_parallel_size=args.tp,
                      gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len,
                      enforce_eager=True, trust_remote_code=True, seed=args.seed)
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    else:
        llm_kwargs["dtype"] = "bfloat16"
    llm = LLM(**llm_kwargs)

    prompts = [
        montar_chat(tok, JUDGE_SYSTEM,
                    f"PERGUNTA: {it.get('instruction','')}\n"
                    f"RESPOSTA DE REFERÊNCIA: {it.get('resposta_ref','')}\n"
                    f"RESPOSTA DO MODELO: {it.get('geracao','')}\n\nNota (1-5):")
        for it in itens
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=4, seed=args.seed)
    outs = llm.generate(prompts, sp)

    hist = Counter()
    por_tipo = {}
    total = 0
    for it, o in zip(itens, outs):
        m = re.search(r"[1-5]", o.outputs[0].text)
        nota = int(m.group(0)) if m else 1
        it["juiz"] = nota
        hist[nota] += 1
        total += nota
        t = it.get("tipo", "?")
        por_tipo.setdefault(t, []).append(nota)

    n = max(1, len(itens))
    res = {
        "model": data.get("model"), "gen_json": args.gen_json,
        "juiz_media": round(total / n, 4),
        "juiz_media_norm5": round(total / n / 5, 4),
        "distribuicao": dict(sorted(hist.items())),
        "media_por_tipo": {t: round(sum(v) / len(v), 3) for t, v in por_tipo.items()},
        "n": len(itens),
        "per_item": itens,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"Juiz média = {res['juiz_media']}/5  | dist={res['distribuicao']} | por tipo={res['media_por_tipo']}",
          flush=True)
    print(f"Salvo em {out}", flush=True)


if __name__ == "__main__":
    main()
