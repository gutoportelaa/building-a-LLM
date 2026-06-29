#!/usr/bin/env python3
"""
pretreino_fsdp.py — Pré-treino continuado DOM-PI, full-FT DISTRIBUÍDO (FSDP, multi-GPU).

Para o experimento cross-família (Llama-3.2-3B): o modelo de 3B em full fine-tuning
não cabe folgado numa L4, então fragmentamos parâmetros/gradientes/estados do
otimizador entre as 2 GPUs do nó (FSDP full_shard). Usa `transformers.Trainer`
(integração FSDP testada) lançado com `torchrun --nproc_per_node=2`.

Correção preservada (sem duplo-shift): empacotamos blocos de `block_size` e o
`DataCollatorForLanguageModeling(mlm=False)` define labels = input_ids; o shift
fica por conta do modelo (logits[:-1] vs labels[1:]). NÃO pré-deslocamos nada.

Métricas: eval_loss (= CE no held-out) guia o early stopping; PPL/CE/token-acc
finais são medidos por `avaliar_modelo.py` no checkpoint salvo.

Lançar (2 GPUs, 1 nó):
    torchrun --nproc_per_node=2 scripts/pretreino_fsdp.py \
        --model meta-llama/Llama-3.2-3B \
        --train-data data/train_corpus_limpo.jsonl \
        --held-out data/held_out.jsonl \
        --output-dir treino/checkpoints_llama32_3b_dompi \
        --lr 2e-6 --block-size 512 --per-device-batch 2 --grad-accum 4 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
IS_MAIN = LOCAL_RANK == 0


def pack_blocks(jsonl_path: str, tokenizer, block_size: int) -> list[list[int]]:
    """Tokeniza o corpus e empacota em blocos de block_size (EOS entre docs).
    Map-style (lista) — necessário p/ sharding correto entre ranks no FSDP/DDP."""
    eos = tokenizer.eos_token_id or 0
    buf: list[int] = []
    blocks: list[list[int]] = []
    n_docs = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                text = json.loads(line).get("texto") or json.loads(line).get("text") or ""
            except json.JSONDecodeError:
                text = line.strip()
            if len(text) < 20:
                continue
            buf.extend(tokenizer.encode(text, add_special_tokens=False))
            buf.append(eos)
            n_docs += 1
            # esvazia em blocos para não acumular memória demais
            while len(buf) >= block_size:
                blocks.append(buf[:block_size])
                buf = buf[block_size:]
    if IS_MAIN:
        log.info("Empacotado: %d docs -> %d blocos de %d tokens (~%.1fM tokens)",
                 n_docs, len(blocks), block_size, len(blocks) * block_size / 1e6)
    return blocks


def build_dataset(jsonl_path: str, tokenizer, block_size: int) -> Dataset:
    blocks = pack_blocks(jsonl_path, tokenizer, block_size)
    return Dataset.from_dict({"input_ids": blocks})


def main() -> None:
    p = argparse.ArgumentParser(description="Full-FT distribuído (FSDP) — DAPT DOM-PI")
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--train-data", required=True)
    p.add_argument("--held-out", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--per-device-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--max-eval-blocks", type=int, default=1500,
                   help="Cap de blocos do held-out p/ eval em loop (early stopping)")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--wrap-layer", default="LlamaDecoderLayer",
                   help="Classe do bloco transformer p/ auto_wrap do FSDP")
    args = p.parse_args()

    if IS_MAIN:
        log.info("FSDP full-FT: model=%s | per_dev_bs=%d | accum=%d | lr=%.2e | block=%d",
                 args.model, args.per_device_batch, args.grad_accum, args.lr, args.block_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True, use_cache=False,
    )

    # FSDP achata os parâmetros; com embeddings AMARRADAS (Llama tie_word_embeddings),
    # o peso de embed_tokens/lm_head vira 1-D no forward → "RuntimeError: 'weight'
    # must be 2-D". Desamarramos (untie) clonando a lm_head antes do FSDP envolver o
    # modelo. Custo: ~0,4 GB de parâmetros extras (aceitável p/ DAPT).
    if getattr(model.config, "tie_word_embeddings", False):
        import torch.nn as nn
        in_emb = model.get_input_embeddings()
        model.config.tie_word_embeddings = False
        model.lm_head.weight = nn.Parameter(in_emb.weight.detach().clone())
        if IS_MAIN:
            log.info("Embeddings DESAMARRADAS (untie) p/ compatibilidade com FSDP.")

    train_ds = build_dataset(args.train_data, tokenizer, args.block_size)
    eval_ds = build_dataset(args.held_out, tokenizer, args.block_size)
    # Eval em loop (early stopping) é caro: cap p/ ~max_eval_blocks blocos. O número
    # final antes×depois é medido depois por avaliar_modelo.py no held-out completo.
    if args.max_eval_blocks and len(eval_ds) > args.max_eval_blocks:
        eval_ds = eval_ds.select(range(args.max_eval_blocks))
        if IS_MAIN:
            log.info("Held-out de eval em loop limitado a %d blocos.", args.max_eval_blocks)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    fsdp_config = {
        "transformer_layer_cls_to_wrap": [args.wrap_layer],
        "backward_prefetch": "backward_pre",
        "forward_prefetch": False,
        "use_orig_params": True,
        "limit_all_gathers": True,
        # Checkpointing de ativação DENTRO do FSDP (não via TrainingArguments):
        # o gradient_checkpointing dos args introduz um AllGather redundante que,
        # com embeddings amarradas (Llama tie_word_embeddings), quebra com
        # "RuntimeError: 'weight' must be 2-D". Aqui é o caminho correto.
        "activation_checkpointing": True,
        # Salva o modelo desfragmentado (full) — carregável por AutoModelForCausalLM.
        "state_dict_type": "FULL_STATE_DICT",
    }

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_fraction,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        bf16=True,
        # NÃO usar gradient_checkpointing aqui — é feito via fsdp_config
        # (activation_checkpointing) para evitar o AllGather redundante/2-D error.
        optim="adamw_bnb_8bit",
        max_grad_norm=1.0,
        weight_decay=0.1,
        adam_beta2=0.95,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=args.eval_every,
        save_strategy="steps",
        save_steps=args.eval_every,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fsdp="full_shard auto_wrap",
        fsdp_config=fsdp_config,
        report_to="none",
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience,
                                         early_stopping_threshold=1e-3)],
    )

    if args.held_out and IS_MAIN:
        log.info("Baseline held-out (antes do treino)...")
    base = trainer.evaluate()
    if IS_MAIN:
        ppl = float(torch.exp(torch.tensor(min(base["eval_loss"], 20))))
        log.info("CE held-out baseline=%.4f | PPL=%.2f", base["eval_loss"], ppl)

    trainer.train()

    # salva o melhor (load_best_model_at_end já recarregou) em formato HF padrão
    best_dir = str(Path(args.output_dir) / "best")
    trainer.save_model(best_dir)
    if IS_MAIN:
        tokenizer.save_pretrained(best_dir)
        final = trainer.evaluate()
        ppl = float(torch.exp(torch.tensor(min(final["eval_loss"], 20))))
        log.info("CE held-out melhor=%.4f | PPL=%.2f | salvo em %s", final["eval_loss"], ppl, best_dir)


if __name__ == "__main__":
    main()
