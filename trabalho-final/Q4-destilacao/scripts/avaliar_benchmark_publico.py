#!/usr/bin/env python3
"""
avaliar_benchmark_publico.py — Retenção ancorada da Q4 num benchmark PÚBLICO (PT-BR).

Mede aluno e professor num benchmark de múltipla escolha conhecido (ENEM, via HF Datasets),
para obter a métrica canônica de destilação:

    retenção = acurácia_aluno / acurácia_professor

— o número diretamente comparável ao "97% do BERT" do DistilBERT. Diferente do benchmark
próprio da Q4 (onde a referência é o professor, logo 100% por construção), aqui o professor
pontua < 100% e a retenção do aluno é uma fração legítima.

Pontuação SEM geração: para cada alternativa, calcula a log-verossimilhança média por token
da alternativa condicionada ao enunciado (mesmo princípio do benchmark Q&A da Q1/Q5), e escolhe
a de maior log-prob. Robusto para modelos 0.5B/1.5B que não seguem formato de resposta.

Uso (exemplo):
  python avaliar_benchmark_publico.py \
      --dataset eduagarcia/enem_challenge --split train --limit 200 \
      --model prof=Qwen/Qwen2.5-14B-Instruct \
      --model base15=Qwen/Qwen2.5-1.5B \
      --model d_15_B_comb=../modelos/aluno_qwen2.5-1.5b_B_combined \
      --teacher prof \
      --out ../resultados/avaliacao_benchmark_publico.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LETTERS = ["A", "B", "C", "D", "E"]


# --------------------------------------------------------------------------- #
# Carregamento do benchmark (ENEM por padrão; campos configuráveis)            #
# --------------------------------------------------------------------------- #

def _extrair_alternativas(choices) -> list[str]:
    """Normaliza o campo `choices` em uma lista de strings de alternativas.
    Esquema ENEM (eduagarcia/enem_challenge): dict {'text': [...], 'label': ['A'..'E']}.
    Também aceita lista simples de strings."""
    if isinstance(choices, dict):
        textos = choices.get("text") or choices.get("texts") or []
    else:
        textos = choices
    norm = []
    for c in textos:
        c = str(c).strip()
        for lt in LETTERS:               # remove prefixo "A) " se vier embutido
            for sep in (")", ".", "-", ":"):
                if c.upper().startswith(lt + sep):
                    c = c[len(lt) + 1:].strip()
                    break
        norm.append(c)
    return norm


def carregar_benchmark(dataset: str, split: str, limit: int) -> list[dict]:
    """Retorna [{id, question, choices:[...], answer_idx}] do ENEM (pula anuladas)."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    out = []
    for i, ex in enumerate(ds):
        if limit and len(out) >= limit:
            break
        if ex.get("nullified"):          # questão anulada — fora
            continue
        q = ex.get("question") or ex.get("enunciado") or ex.get("text")
        norm = _extrair_alternativas(ex.get("choices") or ex.get("alternatives") or ex.get("options"))
        gold = ex.get("answerKey") or ex.get("label") or ex.get("answer")
        if not q or not norm or gold is None:
            continue
        # índice da resposta (letra A-E ou inteiro)
        if isinstance(gold, str) and gold.strip().upper() in LETTERS:
            gi = LETTERS.index(gold.strip().upper())
        else:
            try:
                gi = int(gold)
            except (ValueError, TypeError):
                continue
        if gi >= len(norm):
            continue
        out.append({"id": ex.get("id", i), "question": q, "choices": norm, "answer_idx": gi})
    return out


# --------------------------------------------------------------------------- #
# Pontuação por log-verossimilhança                                            #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def logprob_continuacao(model, tok, prompt: str, continuation: str) -> float:
    """Log-prob média por token de `continuation` condicionada a `prompt`."""
    p_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    c_ids = tok(continuation, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    if c_ids.shape[1] == 0:
        return float("-inf")
    ids = torch.cat([p_ids, c_ids], dim=1)
    logits = model(ids).logits  # [1, T, V]
    # alvos: tokens da continuação; logits que os preveem são os das posições anteriores
    start = p_ids.shape[1]
    tgt = ids[:, start:]                       # [1, Lc]
    pred = logits[:, start - 1:-1, :]          # [1, Lc, V]
    logp = torch.log_softmax(pred.float(), dim=-1)
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [1, Lc]
    return float(tok_lp.mean().item())


def _eh_grande(path: str) -> bool:
    """Modelos 7B/14B não cabem em 1 L4 (24GB) em bf16 → carrega em 8-bit (quase lossless)."""
    p = path.lower()
    return any(t in p for t in ("14b", "7b", "13b", "32b"))


def avaliar_modelo(rotulo: str, path: str, bench: list[dict]) -> dict:
    print(f"\n=== {rotulo} ({path}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if _eh_grande(path):
        from transformers import BitsAndBytesConfig
        print("   modelo grande → carregando em 8-bit (bitsandbytes)", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    acertos, detalhe = 0, []
    for item in bench:
        prompt = f"Pergunta: {item['question']}\nResposta:"
        scores = [logprob_continuacao(model, tok, prompt, " " + c) for c in item["choices"]]
        pred = int(max(range(len(scores)), key=lambda j: scores[j]))
        ok = (pred == item["answer_idx"])
        acertos += int(ok)
        detalhe.append({"id": item["id"], "pred": pred, "gold": item["answer_idx"], "ok": ok})

    acc = acertos / len(bench) if bench else 0.0
    print(f"   acurácia: {acc:.4f}  ({acertos}/{len(bench)})", flush=True)
    del model
    torch.cuda.empty_cache()
    return {"rotulo": rotulo, "path": path, "n": len(bench), "acuracia": round(acc, 4),
            "detalhe": detalhe}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eduagarcia/enem_challenge")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--model", action="append", default=[], metavar="rotulo=caminho")
    ap.add_argument("--teacher", default="prof", help="rótulo do professor (para a retenção)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bench = carregar_benchmark(args.dataset, args.split, args.limit)
    print(f"Benchmark: {args.dataset}[{args.split}] — {len(bench)} questões de múltipla escolha", flush=True)

    resultados = []
    for spec in args.model:
        rot, path = spec.split("=", 1)
        resultados.append(avaliar_modelo(rot, path, bench))

    # retenção = acc_aluno / acc_professor
    prof = next((r for r in resultados if r["rotulo"] == args.teacher), None)
    if prof and prof["acuracia"] > 0:
        for r in resultados:
            r["retencao_pct"] = round(100 * r["acuracia"] / prof["acuracia"], 1)

    out = {"dataset": args.dataset, "split": args.split, "n": len(bench),
           "teacher": args.teacher, "modelos": resultados}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nEscrito: {args.out}", flush=True)
    print("\nRESUMO (acurácia | retenção % do professor):", flush=True)
    for r in resultados:
        ret = r.get("retencao_pct", "—")
        print(f"  {r['rotulo']:24s} acc={r['acuracia']:.4f}  retenção={ret}%", flush=True)


if __name__ == "__main__":
    main()
