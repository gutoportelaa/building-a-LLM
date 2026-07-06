#!/usr/bin/env python3
"""
pretreino_fullft_unsloth.py — Pré-treino continuado DOM-PI: Full FT acelerado por Unsloth.

Variante de `pretreino_fullft.py` que troca APENAS o carregamento do modelo pela
API do Unsloth (`FastLanguageModel`), que aplica kernels Triton próprios, flash-
attention e cross-entropy fundida → ~2x mais rápido e 50-70% menos VRAM, com a
MESMA qualidade (perda/PPL equivalentes).

Decisão de design (importante p/ comparabilidade e p/ NÃO reintroduzir bugs):
  - Mantém o MESMO `PackedTextDataset` com input_ids/labels ALINHADOS (o shift fica
    só por conta do modelo) — evita o duplo-shift que afetou a campanha anterior.
  - Mantém o MESMO loop de treino (warmup+cosine manual, AdamW 8-bit, grad-accum),
    o MESMO `eval_held_out` (CE) e o MESMO early-stopping/saves (best/final).
  Assim, a única variável que muda vs. o full-FT de referência é o *backend* do
  modelo — isolando o ganho de throughput do Unsloth sem alterar o resultado.

Requisito: `pip install unsloth` (o sbatch companion instala se faltar).
IMPORTANTE: `import unsloth` PRECISA vir antes de `transformers` (monkey-patching).

Uso:
    python scripts/pretreino_fullft_unsloth.py \
        --model Qwen/Qwen2.5-1.5B \
        --train-data data/train_corpus_limpo.jsonl \
        --held-out data/held_out.jsonl \
        --output-dir treino/checkpoints_fullft_unsloth \
        --lr 2e-6 --batch-size 2 --grad-accum 8 --epochs 1
"""
from __future__ import annotations

# --- Unsloth deve ser importado ANTES de transformers/torch p/ aplicar os patches ---
from unsloth import FastLanguageModel  # noqa: E402  (import-order é intencional)

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Iterator  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import DataLoader, IterableDataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PackedTextDataset(IterableDataset):
    """Idêntico ao de pretreino_fullft.py: EOS entre docs, blocos de block_size,
    input_ids e labels ALINHADOS (shift interno do modelo — sem duplo-shift)."""

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
                buf = buf[self.block_size:]
                ids = torch.tensor(block[:-1], dtype=torch.long)
                yield {"input_ids": ids, "labels": ids.clone()}


def count_blocks(path: str, tokenizer, block_size: int) -> int:
    return sum(1 for _ in PackedTextDataset(path, tokenizer, block_size))


def eval_held_out(model, tokenizer, path: str, block_size: int) -> float:
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
    # save_pretrained_merged garante pesos full em fp16/bf16 carregáveis por
    # AutoModelForCausalLM (avaliar_modelo.py / HF Hub) sem dependência de Unsloth.
    try:
        model.save_pretrained_merged(path, tokenizer, save_method="merged_16bit")
    except (AttributeError, TypeError):
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
    log.info("Checkpoint salvo em %s", path)


def load_unsloth_model(args):
    """Carrega via Unsloth com full fine-tuning (todos os parâmetros treináveis)."""
    log.info("Carregando modelo via Unsloth (full_finetuning=True): %s", args.model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.block_size,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=True,           # full-FT, não LoRA
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        log.info("Gradient checkpointing ativado.")
    return model, tokenizer


def train(args: argparse.Namespace) -> None:
    log.info("Dispositivo: %s", DEVICE)
    if DEVICE == "cuda":
        log.info("GPU: %s | VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    model, tokenizer = load_unsloth_model(args)

    n_total = sum(p.numel() for p in model.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    log.info("Parâmetros: %.1fM totais | %.1fM treináveis (%.1f%%)",
             n_total / 1e6, n_train / 1e6, 100 * n_train / max(1, n_total))

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable, lr=args.lr * 0.01, weight_decay=0.1, betas=(0.9, 0.95))
        log.info("Otimizador: AdamW 8-bit (bitsandbytes)")
    except ImportError:
        log.warning("bitsandbytes não encontrado — usando AdamW fp32")
        optimizer = torch.optim.AdamW(trainable, lr=args.lr * 0.01, weight_decay=0.1, betas=(0.9, 0.95))

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

                if args.held_out and global_step % args.eval_every == 0:
                    ho_ce = eval_held_out(model, tokenizer, args.held_out, args.block_size)
                    ho_ppl = math.exp(min(ho_ce, 20))
                    log.info("[eval] step=%d | CE held-out=%.4f | PPL=%.2f | melhor=%.4f",
                             global_step, ho_ce, ho_ppl, best_ho_ce)
                    if ho_ce < best_ho_ce - args.min_delta:
                        best_ho_ce = ho_ce
                        save_ckpt(model, tokenizer, str(out_dir / "best"))
                        bad_evals = 0
                        log.info("[best] Novo melhor checkpoint: CE=%.4f PPL=%.2f", best_ho_ce, ho_ppl)
                    else:
                        if ho_ce < best_ho_ce:
                            best_ho_ce = ho_ce
                            save_ckpt(model, tokenizer, str(out_dir / "best"))
                        bad_evals += 1
                        log.warning("[aviso] Sem melhora no held-out: %d/%d evals", bad_evals, args.patience)
                        if bad_evals >= args.patience:
                            log.warning("[early stop] Parando. Melhor CE: %.4f", best_ho_ce)
                            best_path, final_path = out_dir / "best", out_dir / "final"
                            if best_path.exists() and not final_path.exists():
                                shutil.copytree(str(best_path), str(final_path))
                            return

        log.info("Época %d concluída.", epoch)

    log.info("Salvando checkpoint final...")
    save_ckpt(model, tokenizer, str(out_dir / "final"))
    if not (out_dir / "best").exists():
        shutil.copytree(str(out_dir / "final"), str(out_dir / "best"))
    log.info("Treino concluído. Melhor CE held-out: %.4f | PPL: %.2f",
             best_ho_ce, math.exp(min(best_ho_ce, 20)))


def main() -> None:
    p = argparse.ArgumentParser(description="Full FT continuado DOM-PI acelerado por Unsloth")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--train-data", required=True)
    p.add_argument("--held-out", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=1e-3)
    args = p.parse_args()

    log.info("Full FT (Unsloth): model=%s | bs=%d | accum=%d | lr=%.2e | block=%d | eval_every=%d",
             args.model, args.batch_size, args.grad_accum, args.lr, args.block_size, args.eval_every)
    train(args)


if __name__ == "__main__":
    main()
