#!/usr/bin/env python3
"""
consolidar_q2q3.py — Lê resultados/*.json e monta as TABELAS e FIGURAS dos
relatórios Q2 (SFT full) e Q3 (LoRA/QLoRA), incluindo o arco Q1→Q3.

Consome:
  resultados/heldout_<tag>.json   (avaliar_sft --modo intrinseca)  → PPL/CE/tok-acc
  resultados/bench_<tag>.json     (avaliar_sft --modo geracao)     → EM/contains/F1
  resultados/juiz_<tag>.json      (avaliar_juiz)                   → nota 1-5
  modelos/sft_<m>_<sz>/treino_meta.json                            → params/VRAM/tempo
  tags: baseline_<sz>, full_<sz>, lora_<sz>, qlora_<sz>  para sz in {1.5b,0.5b}

Produz:
  resultados/resumo_q2q3.md   — tabelas markdown prontas p/ o relatório
  resultados/fig_*.png        — figuras (juiz, ppl, custo×qualidade)

Roda offline (.venv), tolera arquivos ausentes (preenche "—").
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "resultados"
MOD = BASE / "modelos"
SIZES = ["1.5b", "0.5b"]
METHODS = ["full", "lora", "qlora"]
ROTULO = {"full": "SFT full (Q2)", "lora": "LoRA (Q3)", "qlora": "QLoRA (Q3)"}


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def g(d, k, default="—"):
    if d is None or k not in d or d[k] is None:
        return default
    return d[k]


def fmt(x, n=3):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x)


def tabela_antes_depois(sz: str) -> str:
    base_h = load(RES / f"heldout_baseline_{sz}.json")
    base_b = load(RES / f"bench_baseline_{sz}.json")
    base_j = load(RES / f"juiz_baseline_{sz}.json")
    linhas = ["| Modelo | PPL held-out ↓ | EM bench | contains | F1 médio | Juiz 1-5 ↑ |",
              "|---|---|---|---|---|---|"]
    linhas.append(f"| **Base {sz} (antes)** | {fmt(g(base_h,'perplexity'),2)} | "
                  f"{fmt(g(base_b,'gen_exact_match'))} | {fmt(g(base_b,'gen_contains'))} | "
                  f"{fmt(g(base_b,'gen_f1_mean'))} | {fmt(g(base_j,'juiz_media'),2)} |")
    for m in METHODS:
        h = load(RES / f"heldout_{m}_{sz}.json")
        b = load(RES / f"bench_{m}_{sz}.json")
        j = load(RES / f"juiz_{m}_{sz}.json")
        linhas.append(f"| {ROTULO[m]} {sz} | {fmt(g(h,'perplexity'),2)} | "
                      f"{fmt(g(b,'gen_exact_match'))} | {fmt(g(b,'gen_contains'))} | "
                      f"{fmt(g(b,'gen_f1_mean'))} | {fmt(g(j,'juiz_media'),2)} |")
    return "\n".join(linhas)


def tabela_custo(sz: str) -> str:
    linhas = ["| Método | Params treináveis | % treinável | VRAM pico (GB) | Tempo (min) | loss ini→fim |",
              "|---|---|---|---|---|---|"]
    for m in METHODS:
        meta = load(MOD / f"sft_{m}_{sz}" / "treino_meta.json")
        pt = g(meta, "params_treinaveis")
        pt_s = f"{pt:,}" if isinstance(pt, int) else "—"
        li, lf = g(meta, "loss_inicial"), g(meta, "loss_final")
        linhas.append(f"| {ROTULO[m]} | {pt_s} | {fmt(g(meta,'frac_treinavel'),2)}% | "
                      f"{fmt(g(meta,'vram_pico_gb'),2)} | {fmt(g(meta,'tempo_min'),1)} | "
                      f"{fmt(li,2)}→{fmt(lf,2)} |")
    return "\n".join(linhas)


def tabela_arco_q1() -> str:
    """Arco Q1→Q3: mesma maquinaria LoRA, objetivos diferentes."""
    return (
        "| Objetivo | LoRA | QLoRA | full-FT |\n"
        "|---|---|---|---|\n"
        "| **CPT bruto (Q1, DOM-PI/Teresina)** | colapsou (PEFT conservador) | — | venceu (ganho real de domínio) |\n"
        "| **SFT instruction (Q3, docentesDC)** | ver tabela acima | ver tabela acima | referência (Q2) |\n"
    )


def figuras():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib indisponível — pulando figuras")
        return
    import numpy as np

    # Fig 1: juiz baseline vs métodos, por tamanho
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, sz in zip(axes, SIZES):
        cats = ["base", "full", "lora", "qlora"]
        vals = []
        for c in cats:
            tag = f"baseline_{sz}" if c == "base" else f"{c}_{sz}"
            j = load(RES / f"juiz_{tag}.json")
            vals.append(g(j, "juiz_media", 0) if isinstance(g(j, "juiz_media", 0), (int, float)) else 0)
        bars = ax.bar(cats, vals, color=["#999", "#1f77b4", "#2ca02c", "#ff7f0e"])
        ax.set_title(f"Juiz 1-5 — Qwen2.5-{sz}")
        ax.set_ylim(0, 5)
        ax.set_ylabel("nota média")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.08, fmt(v, 2), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "fig_juiz.png", dpi=130)
    plt.close(fig)
    print("fig_juiz.png salvo")


def main():
    RES.mkdir(parents=True, exist_ok=True)
    out = ["# Resumo consolidado — Q2 (SFT full) × Q3 (LoRA/QLoRA)\n"]
    for sz in SIZES:
        out.append(f"\n## Qwen2.5-{sz}\n")
        out.append("### Antes × depois (qualidade)\n")
        out.append(tabela_antes_depois(sz))
        out.append("\n\n### Custo de treino (Q2 full × Q3 PEFT)\n")
        out.append(tabela_custo(sz))
    out.append("\n\n## Arco Q1 → Q3 (mesma maquinaria LoRA, objetivos diferentes)\n")
    out.append(tabela_arco_q1())
    (RES / "resumo_q2q3.md").write_text("\n".join(out), encoding="utf-8")
    print(f"resumo_q2q3.md salvo em {RES}")
    figuras()


if __name__ == "__main__":
    main()
