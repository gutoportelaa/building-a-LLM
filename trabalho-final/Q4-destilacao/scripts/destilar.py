#!/usr/bin/env python3
"""
destilar.py — Destilação white-box (logit KD) do professor Qwen2.5-14B-Instruct
para o aluno Qwen2.5-0.5B / 1.5B (base pristino). Mesma família → tokenizador
idêntico, requisito do logit KD.

Lê o cache offline produzido por `gerar_dataset_destilacao.py`:
  dados/dataset_<braco>.jsonl  — {id, prompt, answer, answer_token_ids, ...}
  dados/logits_<braco>.jsonl   — {id, topk: [ [[tok_id, logprob], ...] por token de resposta ]}

Métodos (--method):
  ce        — SFT na resposta do professor (apenas hard label).
  kl        — KL(professor ‖ aluno) sobre os TOP-K renormalizados, temperatura T.
  combined  — alpha·CE + (1-alpha)·T²·KL   (Hinton: o fator T² mantém a magnitude do gradiente da KL).

Decisões (registradas em ../relatorio.md §3.1):
  • Alvo soft = softmax(logprob_professor / T) RENORMALIZADO sobre os top-k (estilo Gemma 2, arXiv:2408.00118).
    Usar logprob em vez de logit é exato: logprob = logit − logZ e o termo constante cancela na softmax.
  • Prompt mascarado com −100; `labels = input_ids.clone()` (sem pré-shift — lição do bug duplo-shift da Q1).
  • Aluno = base pristino (baseline limpo); o modelo treinado da Q1/Q5 NÃO é usado (professor fraco transferiria erros).

Treino por exemplo com acumulação de gradiente (micro-batch=1) — evita bugs de gather por posição com padding;
os alunos são pequenos, então o custo é aceitável na L4.

Exemplo:
  python destilar.py --student Qwen/Qwen2.5-0.5B --braco B --method combined \
      --dataset ../dados/dataset_B.jsonl --logits ../dados/logits_B.jsonl \
      --out-dir ../modelos/aluno_0.5b_B_combined
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Dados                                                                        #
# --------------------------------------------------------------------------- #

def carregar(dataset_path: Path, logits_path: Path | None) -> list[dict]:
    data = {}
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            data[r["id"]] = r
    if logits_path is not None:
        for line in logits_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["id"] in data:
                    data[r["id"]]["topk"] = r["topk"]
    return list(data.values())


def montar_exemplo(rec: dict, tok, max_len: int) -> dict | None:
    """input_ids = prompt + answer; labels mascaram o prompt; guarda posições de resposta."""
    prompt_ids = tok(rec["prompt"], add_special_tokens=False).input_ids
    answer_ids = rec["answer_token_ids"]
    if not answer_ids:
        return None
    # trunca pela esquerda do prompt para caber answer inteiro
    budget = max_len - len(answer_ids)
    if budget < 8:
        answer_ids = answer_ids[: max_len - 8]
        prompt_ids = prompt_ids[-8:]
    else:
        prompt_ids = prompt_ids[-budget:]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "p_len": len(prompt_ids),
        "ans_len": len(answer_ids),
        "topk": rec.get("topk"),
    }


# --------------------------------------------------------------------------- #
# Perdas                                                                       #
# --------------------------------------------------------------------------- #

def perda_kl_topk(student_ans_logits: torch.Tensor, topk: list, T: float) -> torch.Tensor:
    """
    KL(professor ‖ aluno) sobre o suporte top-k renormalizado, com temperatura T.
    student_ans_logits: (ans_len, V) — logits do aluno nas posições que preveem cada token de resposta.
    topk: lista (len=ans_len) de [[tok_id, logprob_professor], ...].
    """
    device = student_ans_logits.device
    ans_len = student_ans_logits.size(0)
    n = min(ans_len, len(topk))
    if n == 0:
        return student_ans_logits.sum() * 0.0

    # log-prob do aluno sobre TODO o vocabulário (com T), depois colhe os ids do professor
    student_logp = F.log_softmax(student_ans_logits[:n] / T, dim=-1)  # (n, V)

    total = student_ans_logits.new_zeros(())
    for i in range(n):
        ids = torch.tensor([t[0] for t in topk[i]], dtype=torch.long, device=device)
        tlp = torch.tensor([t[1] for t in topk[i]], dtype=torch.float, device=device)
        teacher_soft = torch.softmax(tlp / T, dim=-1)            # renormaliza no top-k
        student_lp_k = student_logp[i].index_select(0, ids)      # log p_aluno nos mesmos ids
        # KL = Σ p_t (log p_t − log p_s)
        total = total + (teacher_soft * (torch.log(teacher_soft + 1e-12) - student_lp_k)).sum()
    return total / n


# --------------------------------------------------------------------------- #
# Treino                                                                       #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Destilação logit KD aluno Qwen2.5 (Q4)")
    ap.add_argument("--student", required=True, help="Qwen/Qwen2.5-0.5B ou Qwen/Qwen2.5-1.5B")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--logits", default=None, help="cache top-k (necessário p/ kl e combined)")
    ap.add_argument("--braco", choices=["A", "B"], required=True)
    ap.add_argument("--method", choices=["ce", "kl", "combined"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="peso da CE na combined")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    needs_logits = args.method in ("kl", "combined")
    if needs_logits and not args.logits:
        ap.error(f"--method {args.method} exige --logits")

    print(f"Aluno={args.student} braço={args.braco} método={args.method} "
          f"T={args.temperature} alpha={args.alpha}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE)
    model.config.use_cache = False          # exigido com gradient checkpointing
    model.gradient_checkpointing_enable()
    model.train()

    recs = carregar(Path(args.dataset), Path(args.logits) if needs_logits else None)
    exemplos = [e for r in recs if (e := montar_exemplo(r, tok, args.max_len))]
    if needs_logits:
        exemplos = [e for e in exemplos if e["topk"]]
    print(f"{len(exemplos)} exemplos de treino", flush=True)

    steps_total = (len(exemplos) * args.epochs) // args.grad_accum
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(
        opt, int(steps_total * args.warmup_ratio), steps_total
    )

    T, alpha = args.temperature, args.alpha
    g = torch.Generator().manual_seed(args.seed)
    micro = 0
    for ep in range(args.epochs):
        order = torch.randperm(len(exemplos), generator=g).tolist()
        run_loss = 0.0
        for j, idx in enumerate(order):
            ex = exemplos[idx]
            input_ids = ex["input_ids"].unsqueeze(0).to(DEVICE)
            out = model(input_ids=input_ids)
            logits = out.logits[0]  # (T, V)

            # posições que preveem os tokens de resposta: p_len-1 .. p_len-1+ans_len-1
            p, a = ex["p_len"], ex["ans_len"]
            ans_logits = logits[p - 1 : p - 1 + a]                      # (a, V)
            ans_labels = ex["labels"][p:].to(DEVICE)                    # (a,)

            loss = ans_logits.new_zeros(())
            if args.method in ("ce", "combined"):
                ce = F.cross_entropy(ans_logits.float(), ans_labels)
                loss = loss + (alpha * ce if args.method == "combined" else ce)
            if args.method in ("kl", "combined"):
                kl = perda_kl_topk(ans_logits.float(), ex["topk"], T)
                loss = loss + ((1 - alpha) * (T * T) * kl if args.method == "combined" else (T * T) * kl)

            (loss / args.grad_accum).backward()
            run_loss += loss.item()
            if (j + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                micro += 1
                if micro % 20 == 0:
                    print(f"ep{ep} step{micro}/{steps_total} loss={run_loss/(args.grad_accum*20):.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
                    run_loss = 0.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "config_destilacao.json").write_text(json.dumps({
        "student": args.student, "teacher": "Qwen/Qwen2.5-14B-Instruct",
        "braco": args.braco, "method": args.method, "temperature": T, "alpha": alpha,
        "epochs": args.epochs, "lr": args.lr, "n_exemplos": len(exemplos),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Aluno destilado salvo em {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
