#!/usr/bin/env python3
"""
pretreino_continuado.py — Pré-treino continuado (domain-adaptive) no corpus DOM-PI.

Modelo recomendado: Qwen/Qwen2.5-0.5B (full-FT, cabe no RTX 4070 8GB com bf16+grad_ckpt).
Alternativa LoRA: Qwen/Qwen2.5-1.5B (ajustar --use-lora).

Objetivo: causal LM (next-token) sobre texto empacotado em blocos de block_size tokens.

Uso (full-FT):
    python treino/pretreino_continuado.py \
        --model Qwen/Qwen2.5-0.5B \
        --train-data data/train_corpus.jsonl \
        --output-dir treino/checkpoints \
        --epochs 2 \
        --block-size 1024 \
        --batch-size 2 \
        --grad-accum 8 \
        --lr 2e-5

Uso (LoRA para 1.5B):
    python treino/pretreino_continuado.py \
        --model Qwen/Qwen2.5-1.5B \
        --train-data data/train_corpus.jsonl \
        --output-dir treino/checkpoints \
        --use-lora \
        --lora-r 16 \
        --lora-alpha 32 \
        --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Dataset — empacotamento em blocos (packed causal LM)
# ---------------------------------------------------------------------------

class PackedTextDataset(IterableDataset):
    """
    Lê textos de um JSONL, tokeniza e empacota em blocos de block_size tokens.
    Evita padding: concatena sequências e faz split a cada block_size.
    O EOS token é inserido entre documentos.
    """

    def __init__(self, jsonl_path: str, tokenizer, block_size: int = 1024, max_docs: int | None = None):
        self.path = jsonl_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_docs = max_docs
        self.eos_id = tokenizer.eos_token_id or 0

    def _iter_token_ids(self) -> Iterator[int]:
        n = 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("texto") or obj.get("text") or ""
                except json.JSONDecodeError:
                    text = line.strip()
                if len(text) < 20:
                    continue
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                yield from ids
                yield self.eos_id
                n += 1
                if self.max_docs and n >= self.max_docs:
                    break

    def __iter__(self) -> Iterator[dict]:
        buffer: list[int] = []
        for tok_id in self._iter_token_ids():
            buffer.append(tok_id)
            if len(buffer) >= self.block_size + 1:
                block = buffer[: self.block_size + 1]
                buffer = buffer[self.block_size:]
                input_ids = torch.tensor(block[:-1], dtype=torch.long)
                labels = torch.tensor(block[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}
        # bloco final parcial
        if len(buffer) >= 8:
            input_ids = torch.tensor(buffer[:-1], dtype=torch.long)
            labels = torch.tensor(buffer[1:], dtype=torch.long)
            yield {"input_ids": input_ids, "labels": labels}


def count_blocks(jsonl_path: str, tokenizer, block_size: int, max_docs: int | None = None) -> int:
    """Conta blocos reais do corpus completo para o scheduler de LR ser preciso."""
    ds = PackedTextDataset(jsonl_path, tokenizer, block_size, max_docs=max_docs)
    return sum(1 for _ in ds)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    log.info("Dispositivo: %s", DEVICE)
    if DEVICE == "cuda":
        log.info("GPU: %s | VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    log.info("Carregando tokenizer e modelo %s...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 and DEVICE == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto" if args.use_lora else None,
        trust_remote_code=True,
    )

    if not args.use_lora:
        model = model.to(DEVICE)

    # LoRA opcional
    if args.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            log.error("peft não instalado. Instale com: pip install peft")
            raise

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("Parâmetros treináveis: %d (%.1fM)", n_params, n_params / 1e6)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        log.info("Gradient checkpointing ativado.")

    # Dataset
    log.info("Carregando dataset de treino %s...", args.train_data)
    train_ds = PackedTextDataset(
        args.train_data, tokenizer, block_size=args.block_size, max_docs=args.max_docs
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0, pin_memory=False, drop_last=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    # Estimativa de steps para o scheduler
    # Usa estimativa conservadora de blocos (scan rápido)
    log.info("Estimando número de blocos (scan de amostra)...")
    estimated_blocks = count_blocks(args.train_data, tokenizer, args.block_size)
    steps_per_epoch = math.ceil(estimated_blocks / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.03 * total_steps))
    log.info("Estimativa: %d blocos/época | %d steps totais | %d warmup",
             estimated_blocks, total_steps, warmup_steps)

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Diretório de saída
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Treino
    global_step = 0
    best_loss = float("inf")
    accum_loss = 0.0
    accum_count = 0

    for epoch in range(1, args.epochs + 1):
        log.info("Época %d/%d iniciando...", epoch, args.epochs)
        model.train()

        # Re-instancia o dataset a cada época (IterableDataset não tem shuffle interno)
        train_ds = PackedTextDataset(
            args.train_data, tokenizer, block_size=args.block_size, max_docs=args.max_docs
        )
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0, drop_last=True)

        optimizer.zero_grad()

        for step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=dtype, enabled=(DEVICE == "cuda")):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / args.grad_accum

            loss.backward()
            accum_loss += loss.item() * args.grad_accum
            accum_count += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = accum_loss / accum_count
                accum_loss = 0.0
                accum_count = 0

                if global_step % args.log_every == 0:
                    ppl = math.exp(min(avg_loss, 20))
                    lr_now = scheduler.get_last_lr()[0]
                    log.info("Época %d | step %d | loss=%.4f | ppl=%.2f | lr=%.2e",
                             epoch, global_step, avg_loss, ppl, lr_now)

                # Checkpoint periódico
                if global_step % args.save_every == 0:
                    ckpt_path = out_dir / f"checkpoint-step{global_step}"
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    log.info("Checkpoint salvo em %s", ckpt_path)

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_path = out_dir / "best"
                        model.save_pretrained(best_path)
                        tokenizer.save_pretrained(best_path)
                        log.info("Melhor modelo atualizado em %s (loss=%.4f)", best_path, best_loss)

        log.info("Época %d finalizada. Salvando checkpoint...", epoch)
        epoch_path = out_dir / f"checkpoint-epoch{epoch}"
        model.save_pretrained(epoch_path)
        tokenizer.save_pretrained(epoch_path)

    # Modelo final
    final_path = out_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    log.info("Modelo final salvo em %s", final_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pré-treino continuado DOM-PI")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--train-data", default="data/train_corpus.jsonl")
    parser.add_argument("--output-dir", default="treino/checkpoints")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (batch efetivo = batch-size × grad-accum)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-docs", type=int, default=None, help="Limitar nº de docs (debug)")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--use-lora", action="store_true", default=False)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()

    log.info("Configuração: model=%s | block=%d | bs=%d | accum=%d | lr=%.2e | epochs=%d | lora=%s",
             args.model, args.block_size, args.batch_size,
             args.grad_accum, args.lr, args.epochs, args.use_lora)
    log.info("Batch efetivo: %d tokens/step", args.block_size * args.batch_size * args.grad_accum)

    train(args)


if __name__ == "__main__":
    main()
