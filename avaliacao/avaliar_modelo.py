#!/usr/bin/env python3
"""
avaliar_modelo.py — Avaliação intrínseca de LLM no corpus DOM-PI.

Mede três métricas no held-out e no benchmark Q&A:
  1. Entropia cruzada (NLL média por token)
  2. Perplexidade = exp(cross_entropy)
  3. Acurácia de previsão de token = fração de posições em que argmax(logits) == próximo token

Uso:
    python avaliacao/avaliar_modelo.py --model Qwen/Qwen2.5-0.5B \
        --held-out data/held_out.jsonl \
        --benchmark benchmark/dompi_qa.jsonl \
        --output avaliacao/resultados_baseline.json

    # Pós-treino (aponta para o checkpoint):
    python avaliacao/avaliar_modelo.py --model treino/checkpoints/final \
        --held-out data/held_out.jsonl \
        --benchmark benchmark/dompi_qa.jsonl \
        --output avaliacao/resultados_postreino.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MAX_SEQ_LEN = 512
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Tokeniza uma lista de strings em blocos de MAX_SEQ_LEN."""

    def __init__(self, texts: list[str], tokenizer, max_len: int = MAX_SEQ_LEN):
        self.examples: list[dict] = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True)
            # sliding-window de max_len com stride 50% para não perder contexto
            stride = max_len // 2
            for start in range(0, max(1, len(ids) - 1), stride):
                chunk = ids[start : start + max_len]
                if len(chunk) < 8:
                    continue
                self.examples.append({"input_ids": torch.tensor(chunk, dtype=torch.long)})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def collate_fn(batch: list[dict]) -> dict:
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        n = x["input_ids"].shape[0]
        input_ids[i, :n] = x["input_ids"]
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Retorna cross_entropy, perplexity e token_accuracy sobre o dataloader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)

        # shift: predict token t+1 from logits at position t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        # cross-entropy per token (soma)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        mask = shift_mask.view(-1).float()
        total_loss += (loss * mask).sum().item()

        # token accuracy
        preds = shift_logits.argmax(dim=-1)
        correct = (preds == shift_labels).float() * shift_mask.float()
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()

    if total_tokens == 0:
        return {"cross_entropy": float("nan"), "perplexity": float("nan"), "token_accuracy": float("nan")}

    ce = total_loss / total_tokens
    return {
        "cross_entropy": round(ce, 4),
        "perplexity": round(math.exp(ce), 4),
        "token_accuracy": round(total_correct / total_tokens, 4),
    }


@torch.no_grad()
def evaluate_benchmark(
    model: AutoModelForCausalLM,
    tokenizer,
    benchmark_path: str,
    device: str,
    max_prompt_len: int = 128,
    max_answer_len: int = 128,
) -> dict[str, float]:
    """
    Para cada item do benchmark calcula a NLL da resposta_ref dado o prompt.
    Retorna média de cross_entropy, perplexity e token_accuracy sobre as respostas.
    """
    model.eval()
    items = [json.loads(l) for l in Path(benchmark_path).read_text().splitlines() if l.strip()]
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for item in items:
        prompt = item["pergunta"] + " Resposta: "
        answer = item["resposta_ref"]

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)[-max_prompt_len:]
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)[:max_answer_len]

        if not answer_ids:
            continue

        input_ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)

        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0]  # (T, V)

        # avaliar apenas a parte da resposta
        p_len = len(prompt_ids)
        if p_len >= input_ids.shape[1] - 1:
            continue

        ans_logits = logits[p_len - 1 : p_len - 1 + len(answer_ids)]
        ans_labels = torch.tensor(answer_ids, dtype=torch.long, device=device)

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fn(ans_logits, ans_labels)
        total_loss += loss.sum().item()
        total_correct += (ans_logits.argmax(dim=-1) == ans_labels).sum().item()
        total_tokens += len(answer_ids)

    if total_tokens == 0:
        return {"bm_cross_entropy": float("nan"), "bm_perplexity": float("nan"), "bm_token_accuracy": float("nan")}

    ce = total_loss / total_tokens
    return {
        "bm_cross_entropy": round(ce, 4),
        "bm_perplexity": round(math.exp(ce), 4),
        "bm_token_accuracy": round(total_correct / total_tokens, 4),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_held_out(path: str) -> list[str]:
    lines = Path(path).read_text().splitlines()
    texts = []
    for l in lines:
        if not l.strip():
            continue
        try:
            obj = json.loads(l)
            texts.append(obj.get("texto") or obj.get("text") or "")
        except json.JSONDecodeError:
            texts.append(l)
    return [t for t in texts if len(t) > 20]


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação intrínseca LLM no corpus DOM-PI")
    parser.add_argument("--model", required=True, help="HF model id ou caminho local do checkpoint")
    parser.add_argument("--held-out", required=True, help="Arquivo JSONL do held-out (.jsonl)")
    parser.add_argument("--benchmark", default="benchmark/dompi_qa.jsonl", help="Benchmark Q&A")
    parser.add_argument("--output", default="avaliacao/resultados.json", help="JSON de saída com métricas")
    parser.add_argument("--max-samples", type=int, default=2000, help="Máx. de docs do held-out a usar")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--bf16", action="store_true", default=True, help="Usar bfloat16")
    args = parser.parse_args()

    log.info("Dispositivo: %s", DEVICE)
    log.info("Carregando modelo %s...", args.model)

    dtype = torch.bfloat16 if args.bf16 and DEVICE == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # --- Held-out ---
    log.info("Carregando held-out de %s...", args.held_out)
    texts = load_held_out(args.held_out)
    if args.max_samples:
        texts = texts[: args.max_samples]
    log.info("%d documentos no held-out (após truncagem)", len(texts))

    ds = TextDataset(texts, tokenizer, max_len=args.max_seq_len)
    log.info("%d exemplos tokenizados (blocos de %d tokens)", len(ds), args.max_seq_len)

    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)

    log.info("Calculando métricas no held-out...")
    held_out_metrics = compute_metrics(model, dl, DEVICE)
    log.info("Held-out → CE=%.4f  PPL=%.2f  TokenAcc=%.4f",
             held_out_metrics["cross_entropy"],
             held_out_metrics["perplexity"],
             held_out_metrics["token_accuracy"])

    # --- Benchmark ---
    log.info("Calculando métricas no benchmark %s...", args.benchmark)
    bm_metrics = evaluate_benchmark(model, tokenizer, args.benchmark, DEVICE)
    log.info("Benchmark → CE=%.4f  PPL=%.2f  TokenAcc=%.4f",
             bm_metrics["bm_cross_entropy"],
             bm_metrics["bm_perplexity"],
             bm_metrics["bm_token_accuracy"])

    # --- Salvar ---
    results = {
        "model": args.model,
        "held_out_path": args.held_out,
        "held_out_n_docs": len(texts),
        "held_out_n_examples": len(ds),
        **held_out_metrics,
        **bm_metrics,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    log.info("Resultados salvos em %s", out_path)


if __name__ == "__main__":
    main()
