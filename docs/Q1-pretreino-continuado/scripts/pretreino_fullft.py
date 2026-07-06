#!/usr/bin/env python3
"""
pretreino_fullft.py — Pré-treino continuado DOM-PI: Full FT com AdamW 8-bit.

Q3: Todos os parâmetros atualizados (sem PEFT). AdamW 8-bit (bitsandbytes) reduz
estados do otimizador de ~12 GB (fp32) para ~3 GB, tornando viável o 1.5B no L4.
Early stopping por CE no held-out previne o overfitting observado em LoRA/QLoRA (Q2).

Ref:
  Sabiá-2 (Maritaca 2024) — full FT 10B tokens PT, lr cosine.
  llm2govbr (AAAI 2024)  — full FT Llama-2 7B governo BR, lr=2e-6.
  ChipNeMo (NVIDIA 2023) — CPT full FT semicondutor, lr=2e-6.

Uso:
    .venv/bin/python3 treino/pretreino_fullft.py \
        --model Qwen/Qwen2.5-1.5B \
        --train-data data/teresina_hf/train_corpus.jsonl \
        --held-out data/teresina/held_out.jsonl \
        --output-dir treino/checkpoints_fullft_teresina \
        --lr 2e-6 --batch-size 2 --grad-accum 8 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PackedTextDataset(IterableDataset):
    """Mesmo protocolo de empacotamento dos jobs Q1/Q2 — EOS entre docs, blocos de block_size."""

    def __init__(self, jsonl_path: str, tokenizer, block_size: int = 512, max_docs: int | None = None):
        self.path = jsonl_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_docs = max_docs
        self.eos_id = tokenizer.eos_token_id or 0

    def _iter_ids(self) -> Iterator[int]:
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
                yield from self.tokenizer.encode(text, add_special_tokens=False)
                yield self.eos_id
                n += 1
                if self.max_docs and n >= self.max_docs:
                    break

    def __iter__(self) -> Iterator[dict]:
        buf: list[int] = []
        for tok in self._iter_ids():
            buf.append(tok)
            if len(buf) >= self.block_size + 1:
                block = buf[: self.block_size + 1]
                buf = buf[self.block_size :]
                # input_ids e labels ALINHADOS (mesmos tokens): o modelo HF faz o
                # shift interno (logits[:-1] vs labels[1:]). Pré-deslocar labels aqui
                # causaria DUPLO shift → objetivo "prever 2 à frente" → loss ~aleatória.
                ids = torch.tensor(block[:-1], dtype=torch.long)
                yield {"input_ids": ids, "labels": ids.clone()}


def count_blocks(path: str, tokenizer, block_size: int) -> int:
    return sum(1 for _ in PackedTextDataset(path, tokenizer, block_size))


def eval_held_out(model, tokenizer, path: str, block_size: int) -> float:
    """Retorna CE média no held-out sem alterar o modo de treino do modelo."""
    was_training = model.training
    model.eval()
    ds = PackedTextDataset(path, tokenizer, block_size)
    dl = DataLoader(ds, batch_size=4, num_workers=0, drop_last=False)
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in dl:
            iids = batch["input_ids"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                out = model(input_ids=iids, labels=lbls)
            total += out.loss.item()
            n += 1
    if was_training:
        model.train()
    return total / max(1, n)


def save_ckpt(model, tokenizer, path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    log.info("Checkpoint salvo em %s", path)


def train(args: argparse.Namespace) -> None:
    log.info("Dispositivo: %s", DEVICE)
    if DEVICE == "cuda":
        log.info("GPU: %s | VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    log.info("Carregando tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Carregando modelo em bf16 (todos os parâmetros treináveis)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(DEVICE)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        log.info("Gradient checkpointing ativado.")

    # Congela embeddings (recipe DAPT tokens-mínimos: preserva representações de
    # token → retém conhecimento geral; arxiv:2507.02964). Em Qwen2.5 a lm_head é
    # tied à embed_tokens, então congelar a entrada já congela a saída.
    if args.freeze_embeddings:
        emb = model.get_input_embeddings()
        for p in emb.parameters():
            p.requires_grad_(False)
        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight is not emb.weight:
            for p in out_emb.parameters():
                p.requires_grad_(False)
        # gradient checkpointing precisa que a saída da embedding exija grad para
        # propagar pelas camadas seguintes, mesmo com a embedding congelada.
        if args.gradient_checkpointing:
            model.enable_input_require_grads()
        log.info("Embeddings congeladas (entrada + lm_head tied).")

    n_total = sum(p.numel() for p in model.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    log.info("Parâmetros: %.1fM totais | %.1fM treináveis (%.1f%%)",
             n_total / 1e6, n_train / 1e6, 100 * n_train / n_total)

    # AdamW 8-bit: estados m+v quantizados para int8 → ~3 GB vs ~12 GB fp32
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            trainable, lr=args.lr * 0.01, weight_decay=0.1, betas=(0.9, 0.95)
        )
        log.info("Otimizador: AdamW 8-bit (bitsandbytes)")
    except ImportError:
        log.warning("bitsandbytes não encontrado — usando AdamW fp32 (mais memória)")
        optimizer = torch.optim.AdamW(
            trainable, lr=args.lr * 0.01, weight_decay=0.1, betas=(0.9, 0.95)
        )

    log.info("Contando blocos reais do corpus (para LR schedule preciso)...")
    n_blocks = count_blocks(args.train_data, tokenizer, args.block_size)
    steps_per_epoch = max(1, n_blocks // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_fraction))
    peak_lr = args.lr
    min_lr = peak_lr * 0.1
    log.info("Blocos: %d | Steps: %d | Warmup: %d | LR: %.2e → %.2e → %.2e",
             n_blocks, total_steps, warmup_steps, peak_lr * 0.01, peak_lr, min_lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline held-out (antes do treino)
    best_ho_ce = float("inf")
    if args.held_out and os.path.exists(args.held_out):
        log.info("CE held-out pré-treino (baseline)...")
        best_ho_ce = eval_held_out(model, tokenizer, args.held_out, args.block_size)
        log.info("CE held-out baseline: %.4f | PPL: %.2f", best_ho_ce, math.exp(min(best_ho_ce, 20)))
    else:
        log.info("held-out não especificado — sem early stopping por CE")

    global_step = 0
    bad_evals = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        log.info("Época %d/%d", epoch, args.epochs)
        model.train()

        ds = PackedTextDataset(args.train_data, tokenizer, args.block_size)
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=2, drop_last=True)

        optimizer.zero_grad()
        accum_loss, accum_n = 0.0, 0

        for step, batch in enumerate(dl):
            iids = batch["input_ids"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                out = model(input_ids=iids, labels=lbls)
                loss = out.loss / args.grad_accum

            loss.backward()
            accum_loss += out.loss.item()
            accum_n += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # LR schedule: warmup linear + cosine decay (Raschka Apêndice D)
                if global_step < warmup_steps:
                    lr_now = peak_lr * 0.01 + (peak_lr - peak_lr * 0.01) * global_step / warmup_steps
                else:
                    progress = min((global_step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
                    lr_now = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))

                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg = accum_loss / accum_n
                    elapsed = time.time() - start
                    eta = elapsed / global_step * (total_steps - global_step)
                    log.info("step %d/%d | loss=%.4f | lr=%.2e | elapsed=%.1fmin | ETA=%.1fmin",
                             global_step, total_steps, avg, lr_now, elapsed / 60, eta / 60)
                    accum_loss, accum_n = 0.0, 0

                # Eval held-out + early stopping
                if args.held_out and global_step % args.eval_every == 0:
                    ho_ce = eval_held_out(model, tokenizer, args.held_out, args.block_size)
                    ho_ppl = math.exp(min(ho_ce, 20))
                    log.info("[eval] step=%d | CE held-out=%.4f | PPL=%.2f | melhor=%.4f",
                             global_step, ho_ce, ho_ppl, best_ho_ce)

                    ckpt = str(out_dir / f"step_{global_step:05d}")
                    save_ckpt(model, tokenizer, ckpt)

                    # Exige melhora > min_delta para resetar a paciência: micro-oscilações
                    # na ~4ª casa não devem impedir o early stopping (visto no v3/job 489).
                    if ho_ce < best_ho_ce - args.min_delta:
                        best_ho_ce = ho_ce
                        save_ckpt(model, tokenizer, str(out_dir / "best"))
                        bad_evals = 0
                        log.info("[best] Novo melhor checkpoint: CE=%.4f PPL=%.2f", best_ho_ce, ho_ppl)
                    else:
                        # ainda salva como best se for o menor CE absoluto, sem resetar paciência
                        if ho_ce < best_ho_ce:
                            best_ho_ce = ho_ce
                            save_ckpt(model, tokenizer, str(out_dir / "best"))
                        bad_evals += 1
                        log.warning("[aviso] Sem melhora no held-out: %d/%d evals consecutivas",
                                    bad_evals, args.patience)
                        if bad_evals >= args.patience:
                            log.warning("[early stop] Parando treino. Melhor CE: %.4f", best_ho_ce)
                            # Copia best → final para compatibilidade com avaliação downstream
                            best_path = out_dir / "best"
                            final_path = out_dir / "final"
                            if best_path.exists() and not final_path.exists():
                                shutil.copytree(str(best_path), str(final_path))
                            return

                elif not args.held_out and global_step % args.save_every == 0:
                    save_ckpt(model, tokenizer, str(out_dir / f"step_{global_step:05d}"))

        log.info("Época %d concluída.", epoch)

    log.info("Salvando checkpoint final...")
    save_ckpt(model, tokenizer, str(out_dir / "final"))
    if not (out_dir / "best").exists():
        shutil.copytree(str(out_dir / "final"), str(out_dir / "best"))
    log.info("Treino concluído. Melhor CE held-out: %.4f | PPL: %.2f",
             best_ho_ce, math.exp(min(best_ho_ce, 20)))


def main() -> None:
    p = argparse.ArgumentParser(description="Full FT continuado DOM-PI (sem PEFT, AdamW 8-bit)")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--train-data", required=True)
    p.add_argument("--held-out", default=None, help="Caminho para held-out JSONL (early stopping por CE)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--freeze-embeddings", action="store_true", default=False,
                   help="Congela embed_tokens (e lm_head tied) — recipe DAPT tokens-mínimos")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=200, help="Usado quando --held-out não especificado")
    p.add_argument("--eval-every", type=int, default=200, help="Steps entre eval held-out")
    p.add_argument("--patience", type=int, default=3, help="Evals consecutivas sem melhora antes de parar")
    p.add_argument("--min-delta", type=float, default=1e-3,
                   help="Melhora mínima de CE para resetar a paciência (evita reset por micro-oscilação)")
    args = p.parse_args()

    log.info("Full FT: model=%s | bs=%d | accum=%d | lr=%.2e | block=%d | eval_every=%d | patience=%d",
             args.model, args.batch_size, args.grad_accum, args.lr,
             args.block_size, args.eval_every, args.patience)

    train(args)


if __name__ == "__main__":
    main()
