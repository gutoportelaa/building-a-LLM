#!/usr/bin/env python3
"""
sft_docentes.py — Pós-treino SUPERVISIONADO (SFT) sobre os pares da Q2/Q3.

UM script, três métodos (--method), para garantir comparação CONTROLADA:
  • full   → fine-tuning de todos os parâmetros        (Questão 2)
  • lora   → adaptadores de baixo rank, base em bf16    (Questão 3)
  • qlora  → idem, base quantizada em 4-bit NF4          (Questão 3)

Mesmos pares, mesmo formato, mesmo loop → a única variável entre Q2 e Q3 é o
método. Os pares vêm de gerar_pares_docentes.py: {instruction, input, output, tipo}.

Formato (ChatML nativo do Qwen2.5):
  prompt   = apply_chat_template([system, user(instruction+input)], add_generation_prompt=True)
  resposta = output + EOS
  input_ids = prompt_ids + answer_ids
  labels    = [-100]*len(prompt_ids) + answer_ids        # LOSS SÓ NA RESPOSTA

Pegadinha da Q1 (duplo-shift): NÃO pré-deslocar labels. input_ids e labels
alinhados; o modelo HF faz o shift interno. Sanidade: loss inicial ~2-3 (não ~11).

Exemplo (Q2 full):
  python sft_docentes.py --method full --model Qwen/Qwen2.5-1.5B-Instruct \
      --train ../dados/pares_train.jsonl --out-dir ../modelos/sft_full_1.5b

Exemplo (Q3 qlora):
  python sft_docentes.py --method qlora --model Qwen/Qwen2.5-1.5B-Instruct \
      --train ../dados/pares_train.jsonl --out-dir ../modelos/sft_qlora_1.5b --lr 2e-4
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QWEN_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

SYSTEM_MSG = ("Você é um assistente especialista em Ciência da Computação que responde "
              "exercícios da disciplina de forma correta, clara e objetiva.")


# --------------------------------------------------------------------------- #
# Dados                                                                        #
# --------------------------------------------------------------------------- #

def carregar_pares(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def montar_exemplo(rec: dict, tok, max_len: int) -> dict | None:
    """Renderiza ChatML, monta input_ids/labels com máscara só na resposta."""
    instr = (rec.get("instruction") or "").strip()
    inp = (rec.get("input") or "").strip()
    out = (rec.get("output") or "").strip()
    if not instr or not out:
        return None

    user = instr if not inp else f"{instr}\n\n{inp}"
    msgs = [{"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    prompt_ids = tok(prompt, add_special_tokens=False).input_ids
    answer_ids = tok(out, add_special_tokens=False).input_ids + [tok.eos_token_id]

    budget = max_len - len(answer_ids)
    if budget < 8:
        answer_ids = answer_ids[: max_len - 8]
        prompt_ids = prompt_ids[-8:]
    else:
        prompt_ids = prompt_ids[-budget:]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}


# --------------------------------------------------------------------------- #
# Modelo                                                                       #
# --------------------------------------------------------------------------- #

def carregar_modelo(args, tok):
    if args.method == "qlora":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map="auto", trust_remote_code=True)
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, device_map="auto", trust_remote_code=True)
        if args.gradient_checkpointing and args.method == "full":
            model.gradient_checkpointing_enable()

    if args.method in ("lora", "qlora"):
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
                         lora_dropout=args.lora_dropout, target_modules=QWEN_TARGET_MODULES,
                         bias="none", inference_mode=False)
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    return model


def contar_params(model) -> tuple[int, int]:
    treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return treinaveis, total


def salvar(model, tok, path: Path, method: str, base_model: str):
    path.mkdir(parents=True, exist_ok=True)
    if method == "full":
        model.save_pretrained(path)
        tok.save_pretrained(path)
        return
    # LoRA/QLoRA: salva adapter e também faz merge p/ avaliação uniforme
    model.save_pretrained(path / "adapter")
    tok.save_pretrained(path / "adapter")
    from peft import PeftModel
    if method == "qlora":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb,
                                                    device_map="auto", trust_remote_code=True)
    else:
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16,
                                                    device_map="auto", trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, str(path / "adapter")).merge_and_unload()
    merged.save_pretrained(path)
    tok.save_pretrained(path)
    del merged, base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Treino                                                                       #
# --------------------------------------------------------------------------- #

def train(args):
    print(f"Dispositivo: {DEVICE} | método: {args.method} | modelo: {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = carregar_modelo(args, tok)
    treinaveis, total = contar_params(model)
    print(f"Parâmetros treináveis: {treinaveis:,} / {total:,} ({100*treinaveis/total:.3f}%)", flush=True)

    pares = carregar_pares(Path(args.train))
    exemplos = [e for e in (montar_exemplo(p, tok, args.max_len) for p in pares) if e is not None]
    print(f"{len(exemplos)} exemplos de treino (de {len(pares)} pares)", flush=True)

    steps_per_epoch = math.ceil(len(exemplos) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(args.warmup_frac * total_steps))

    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.method == "qlora":
        try:
            from bitsandbytes.optim import PagedAdamW32bit
            opt = PagedAdamW32bit(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
        except ImportError:
            opt = AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    else:
        opt = AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(args.seed)

    print(f"Steps totais: {total_steps} | warmup: {warmup} | lr_peak: {args.lr:.1e}", flush=True)
    model.train()
    t0 = time.time()
    global_step = 0
    first_loss = None
    losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(exemplos), generator=rng).tolist()
        opt.zero_grad()
        accum = 0.0
        for i, idx in enumerate(order):
            ex = exemplos[idx]
            input_ids = ex["input_ids"].unsqueeze(0).to(DEVICE)
            labels = ex["labels"].unsqueeze(0).to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                loss = model(input_ids=input_ids, labels=labels).loss / args.grad_accum
            loss.backward()
            accum += loss.item() * args.grad_accum

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(order):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                global_step += 1
                avg = accum / args.grad_accum
                losses.append(avg)
                if first_loss is None:
                    first_loss = avg
                    print(f"  >> loss inicial = {avg:.3f} (sanidade: deve ser ~2-3, NÃO ~11)", flush=True)
                accum = 0.0
                if global_step % args.log_every == 0:
                    ppl = math.exp(min(sum(losses[-args.log_every:]) / args.log_every, 20))
                    print(f"  época {epoch} | step {global_step}/{total_steps} | "
                          f"loss={avg:.4f} ppl={ppl:.2f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
        print(f"Época {epoch} concluída.", flush=True)

    dur = time.time() - t0
    print(f"Treino: {dur/60:.1f} min | loss_inicial={first_loss:.3f} | "
          f"loss_final={sum(losses[-10:])/min(10,len(losses)):.3f}", flush=True)

    print("Salvando modelo...", flush=True)
    salvar(model, tok, out_dir, args.method, args.model)

    vram = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
    meta = {"method": args.method, "model": args.model, "n_exemplos": len(exemplos),
            "epochs": args.epochs, "lr": args.lr, "params_treinaveis": treinaveis,
            "params_total": total, "frac_treinavel": round(100 * treinaveis / total, 4),
            "loss_inicial": round(first_loss, 4),
            "loss_final": round(sum(losses[-10:]) / min(10, len(losses)), 4),
            "tempo_min": round(dur / 60, 2), "vram_pico_gb": round(vram, 2)}
    (out_dir / "treino_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description="SFT (full/lora/qlora) sobre docentesDC")
    ap.add_argument("--method", choices=["full", "lora", "qlora"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 1e-5 (full) / 2e-4 (lora/qlora)")
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 1e-5 if args.method == "full" else 2e-4
    train(args)


if __name__ == "__main__":
    main()
