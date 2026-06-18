#!/usr/bin/env python3
"""
pretreino_lora.py — Pré-treino continuado DOM-PI com LoRA ou QLoRA.

Questão 2: Fine-tuning com adaptadores de baixa dimensão (LoRA/QLoRA).
Suporta múltiplos modelos (Qwen2.5-0.5B, Qwen2.5-1.5B).

LoRA (Hu et al. 2022): injeta matrizes de baixo rank em camadas de atenção e MLP.
QLoRA (Dettmers et al. 2023): quantiza o modelo base em 4-bit (NF4) antes de aplicar LoRA.

Uso (LoRA):
    python treino/pretreino_lora.py \
        --model Qwen/Qwen2.5-0.5B \
        --method lora \
        --lora-r 16 --lora-alpha 32 \
        --train-data data/train_corpus.jsonl \
        --output-dir treino/checkpoints_lora_0.5b \
        --lr 3e-4

Uso (QLoRA):
    python treino/pretreino_lora.py \
        --model Qwen/Qwen2.5-1.5B \
        --method qlora \
        --lora-r 16 --lora-alpha 32 \
        --train-data data/train_corpus.jsonl \
        --output-dir treino/checkpoints_qlora_1.5b \
        --lr 2e-4
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
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Módulos alvo para LoRA no Qwen2.5 (todas as projeções lineares)
QWEN_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Dataset — idêntico ao Q1 (packed causal LM)
# ---------------------------------------------------------------------------

class PackedTextDataset(IterableDataset):
    def __init__(self, jsonl_path: str, tokenizer, block_size: int = 1024, max_docs=None):
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


def count_blocks(jsonl_path: str, tokenizer, block_size: int, max_docs=None) -> int:
    ds = PackedTextDataset(jsonl_path, tokenizer, block_size, max_docs=max_docs)
    return sum(1 for _ in ds)


# ---------------------------------------------------------------------------
# Carregamento do modelo com LoRA ou QLoRA
# ---------------------------------------------------------------------------

def load_model_with_adapter(args, tokenizer):
    method = args.method

    if method == "qlora":
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except ImportError:
            raise ImportError("QLoRA requer bitsandbytes: pip install bitsandbytes")

        log.info("Carregando %s em 4-bit NF4 (QLoRA)...", args.model)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    else:  # lora (full precision)
        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        log.info("Carregando %s em bf16 (LoRA)...", args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

    # Aplica LoRA (comum a ambos os métodos)
    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = args.lora_target_modules or QWEN_TARGET_MODULES
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def save_merged(base_model_name: str, adapter_path: Path, tokenizer, out_path: Path, method: str) -> None:
    """
    Carrega o modelo base + adapter do disco e faz merge → salva modelo completo.
    Feito desta forma para não destruir os adapters do modelo em memória durante o treino.
    merge_and_unload() modifica o modelo in-place removendo os adapters; se chamado
    durante o treino, o modelo passa a ser full-FT com lr de LoRA (catastrófico).
    """
    log.info("Mergeando adapter %s → %s ...", adapter_path, out_path)
    from peft import PeftModel

    if method == "qlora":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
    else:
        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, dtype=dtype, device_map="auto", trust_remote_code=True,
        )

    peft_model = PeftModel.from_pretrained(base, str(adapter_path))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    del merged, peft_model, base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    log.info("Merge concluído → %s", out_path)


def save_adapter(model, tokenizer, path: Path):
    """Salva apenas os pesos do adapter (leve, para checkpoints periódicos)."""
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


# ---------------------------------------------------------------------------
# Loop de treino
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    log.info("Dispositivo: %s", DEVICE)
    if DEVICE == "cuda":
        log.info("GPU: %s | VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model_with_adapter(args, tokenizer)

    # Conta blocos para o LR schedule
    log.info("Contando blocos do corpus...")
    estimated_blocks = count_blocks(args.train_data, tokenizer, args.block_size, max_docs=args.max_docs)
    steps_per_epoch = math.ceil(estimated_blocks / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup_fraction * total_steps))
    log.info("Blocos/época: %d | Steps totais: %d | Warmup: %d",
             estimated_blocks, total_steps, warmup_steps)

    # LR schedule manual (idêntico ao Q1 — Raschka Apêndice D)
    peak_lr = args.lr
    initial_lr = peak_lr * 0.01
    min_lr = peak_lr * 0.1
    lr_increment = (peak_lr - initial_lr) / max(1, warmup_steps)
    log.info("LR: %.2e → %.2e → %.2e | método: %s | r=%d α=%d",
             initial_lr, peak_lr, min_lr, args.method, args.lora_r, args.lora_alpha)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if args.method == "qlora":
        try:
            from bitsandbytes.optim import PagedAdamW32bit
            optimizer = PagedAdamW32bit(
                trainable_params, lr=initial_lr, weight_decay=0.1, betas=(0.9, 0.95)
            )
            log.info("Otimizador: PagedAdamW32bit (QLoRA)")
        except ImportError:
            log.warning("PagedAdamW32bit indisponível; usando AdamW padrão")
            optimizer = torch.optim.AdamW(
                trainable_params, lr=initial_lr, weight_decay=0.1, betas=(0.9, 0.95)
            )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params, lr=initial_lr, weight_decay=0.1, betas=(0.9, 0.95)
        )
        log.info("Otimizador: AdamW")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_loss = float("inf")
    best_step = 0
    accum_loss = 0.0
    accum_count = 0
    window_loss = 0.0
    window_count = 0

    for epoch in range(1, args.epochs + 1):
        log.info("Época %d/%d iniciando...", epoch, args.epochs)
        model.train()

        train_ds = PackedTextDataset(
            args.train_data, tokenizer, block_size=args.block_size, max_docs=args.max_docs
        )
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0, drop_last=True)

        optimizer.zero_grad()

        for step, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            dtype_ctx = torch.bfloat16 if DEVICE == "cuda" else torch.float32
            with torch.autocast(device_type=DEVICE, dtype=dtype_ctx, enabled=(DEVICE == "cuda")):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / args.grad_accum

            loss.backward()
            accum_loss += loss.item() * args.grad_accum
            accum_count += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

                # LR schedule — Raschka Apêndice D
                if global_step < warmup_steps:
                    lr_now = initial_lr + global_step * lr_increment
                else:
                    progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                    progress = min(progress, 1.0)
                    lr_now = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = accum_loss / accum_count
                accum_loss = 0.0
                accum_count = 0
                window_loss += avg_loss
                window_count += 1

                if global_step % args.log_every == 0:
                    ppl = math.exp(min(avg_loss, 20))
                    log.info("Época %d | step %d/%d | loss=%.4f | ppl=%.2f | lr=%.2e",
                             epoch, global_step, total_steps, avg_loss, ppl, lr_now)

                if global_step % args.save_every == 0:
                    ckpt_path = out_dir / f"adapter-step{global_step}"
                    save_adapter(model, tokenizer, ckpt_path)
                    log.info("Adapter salvo em %s", ckpt_path)

                    interval_loss = window_loss / max(1, window_count)
                    window_loss = 0.0
                    window_count = 0
                    if interval_loss < best_loss:
                        best_loss = interval_loss
                        best_step = global_step
                        log.info("Novo melhor adapter: step %d (loss_intervalo=%.4f)", best_step, best_loss)

        log.info("Época %d finalizada.", epoch)
        save_adapter(model, tokenizer, out_dir / f"adapter-epoch{epoch}")

    # Merge feito UMA VEZ ao final, a partir dos adapters salvos no disco.
    # Não chama merge_and_unload() no modelo em memória — isso destroiria os adapters
    # e transformaria o treino em full-FT com lr de LoRA a partir do próximo step.
    log.info("Fazendo merge do adapter final (step %d)...", global_step)
    save_merged(args.model, out_dir / f"adapter-epoch{args.epochs}", tokenizer, out_dir / "final", args.method)
    log.info("Modelo final salvo em %s/final", out_dir)

    if best_step > 0:
        best_adapter_path = out_dir / f"adapter-step{best_step}"
        if not best_adapter_path.exists():
            # best_step pode estar em adapter-epoch se coincidiu com fim de época
            best_adapter_path = out_dir / f"adapter-epoch{args.epochs}"
        log.info("Fazendo merge do melhor adapter (step %d, loss=%.4f)...", best_step, best_loss)
        save_merged(args.model, best_adapter_path, tokenizer, out_dir / "best", args.method)
        log.info("Melhor modelo salvo em %s/best", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pré-treino continuado DOM-PI — LoRA / QLoRA")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--method", choices=["lora", "qlora"], default="lora")
    parser.add_argument("--train-data", default="data/train_corpus.jsonl")
    parser.add_argument("--output-dir", default="treino/checkpoints_lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak LR (LoRA usa ~100× mais que full-FT; padrão: 3e-4)")
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=None,
                        help="Módulos alvo (padrão: todas as projeções lineares do Qwen2.5)")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()

    log.info(
        "Config: model=%s | method=%s | r=%d α=%d | lr=%.2e | bs=%d×%d | epochs=%d",
        args.model, args.method, args.lora_r, args.lora_alpha,
        args.lr, args.batch_size, args.grad_accum, args.epochs
    )
    log.info("Batch efetivo: %d tokens/step", args.block_size * args.batch_size * args.grad_accum)

    train(args)


if __name__ == "__main__":
    main()
