#!/usr/bin/env python3
"""
graficos_q2q3.py — Gera o conjunto de figuras do relatório de apresentação Q2/Q3
a partir de resultados/*.json, modelos/*/treino_meta.json e dados/stats.json.

Figuras (em resultados/):
  fig_dataset.png        — distribuição dos tipos + funil de qualidade (juiz) da geração
  fig_ppl.png            — PPL no held-out antes×depois (por método e tamanho)
  fig_terseness.png      — comprimento médio da resposta por método (o achado central)
  fig_juiz_tipo.png      — nota do juiz por tipo de questão (conceitual/código/contextual)
  fig_custo_qualidade.png— custo (params %, VRAM) × qualidade (juiz)
  fig_juiz.png           — nota do juiz por método e tamanho (já existente; regenerado)
"""
from __future__ import annotations
import json, statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "resultados"
MOD = BASE / "modelos"
SIZES = ["1.5b", "0.5b"]
METHODS = ["baseline", "full", "lora", "qlora"]
COR = {"baseline": "#9aa0a6", "full": "#1f77b4", "lora": "#2ca02c", "qlora": "#ff7f0e"}
ROT = {"baseline": "base", "full": "SFT full (Q2)", "lora": "LoRA (Q3)", "qlora": "QLoRA (Q3)"}


def L(p):
    try:
        return json.loads((RES / p).read_text())
    except Exception:
        return None


def meta(m, sz):
    try:
        return json.loads((MOD / f"sft_{m}_{sz}" / "treino_meta.json").read_text())
    except Exception:
        return {}


def fig_dataset():
    st = json.loads((BASE / "dados" / "stats.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    # tipos
    tipos = st["por_tipo"]
    nomes = ["explicacao", "factual", "codigo", "resumo", "comparacao"]
    vals = [tipos.get(n, 0) for n in nomes]
    ax[0].bar(nomes, vals, color="#2e5ca0")
    ax[0].set_title(f"Distribuição dos {st['n_total']} pares por tipo")
    ax[0].set_ylabel("nº de pares")
    for i, v in enumerate(vals):
        ax[0].text(i, v + 5, str(v), ha="center", fontsize=9)
    # funil de qualidade (juiz scores) + descartes
    sc = st.get("scores") or {}
    notas = [str(k) for k in sorted(sc, key=int)]
    nvals = [sc[k] for k in notas]
    cores = ["#c0392b" if int(k) < 3 else "#27ae60" for k in notas]
    ax[1].bar([f"nota {k}" for k in notas], nvals, color=cores)
    ax[1].set_title("Juiz LLM: aprovados (verde, ≥3) × reprovados (vermelho, <3)")
    ax[1].set_ylabel("nº de pares")
    for i, v in enumerate(nvals):
        ax[1].text(i, v + 5, str(v), ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(RES / "fig_dataset.png", dpi=130); plt.close(fig)
    print("fig_dataset.png")


def fig_ppl():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(METHODS)); w = 0.36
    cor_sz = {"1.5b": "#2e5ca0", "0.5b": "#7ba3d6"}
    for k, sz in enumerate(SIZES):
        vals = [L(f"heldout_{m}_{sz}.json")["perplexity"] if L(f"heldout_{m}_{sz}.json") else 0 for m in METHODS]
        bars = ax.bar(x + (k - 0.5) * w, vals, w, label=f"Qwen2.5-{sz}", color=cor_sz[sz])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([ROT[m] for m in METHODS])
    ax.set_ylabel("Perplexidade no held-out ↓"); ax.set_title("PPL no held-out — antes × depois do SFT")
    ax.legend(); ax.axhline(0, color="#ccc", lw=0.5)
    fig.tight_layout(); fig.savefig(RES / "fig_ppl.png", dpi=130); plt.close(fig)
    print("fig_ppl.png")


def _compr(tag):
    d = L(f"bench_{tag}.json")
    if not d:
        return 0
    return statistics.mean(len(x["geracao"]) for x in d["per_item"])


def fig_terseness():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(METHODS)); w = 0.36
    for k, sz in enumerate(SIZES):
        vals = [_compr(f"{m}_{sz}") for m in METHODS]
        bars = ax.bar(x + (k - 0.5) * w, vals, w, label=f"Qwen2.5-{sz}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([ROT[m] for m in METHODS])
    ax.set_ylabel("comprimento médio da resposta (chars)")
    ax.set_title("Achado central: o SFT comprime a resposta (full/LoRA) — QLoRA quase não comprime")
    ax.legend()
    fig.tight_layout(); fig.savefig(RES / "fig_terseness.png", dpi=130); plt.close(fig)
    print("fig_terseness.png")


def fig_juiz_tipo():
    tipos = ["conceitual", "codigo", "contextual"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(tipos)); w = 0.2
    for k, m in enumerate(METHODS):
        d = L(f"juiz_{m}_1.5b.json")
        vals = [d["media_por_tipo"].get(t, 0) for t in tipos] if d else [0, 0, 0]
        bars = ax.bar(x + (k - 1.5) * w, vals, w, label=ROT[m], color=COR[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(tipos); ax.set_ylim(0, 5)
    ax.set_ylabel("nota do juiz (1-5)")
    ax.set_title("Juiz por tipo de questão (Qwen2.5-1.5B): conceitual ↑, código ↓ no full FT")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "fig_juiz_tipo.png", dpi=130); plt.close(fig)
    print("fig_juiz_tipo.png")


def fig_custo_qualidade():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for sz, axi in zip(SIZES, ax):
        ms = ["full", "lora", "qlora"]
        vram = [meta(m, sz).get("vram_pico_gb", 0) for m in ms]
        juiz = [L(f"juiz_{m}_{sz}.json")["juiz_media"] if L(f"juiz_{m}_{sz}.json") else 0 for m in ms]
        frac = [meta(m, sz).get("frac_treinavel", 0) for m in ms]
        for m, vr, jz, fr in zip(ms, vram, juiz, frac):
            axi.scatter(vr, jz, s=240, color=COR[m], zorder=3, edgecolor="white")
            axi.annotate(f"{ROT[m]}\n{fr:.2f}% params", (vr, jz),
                         textcoords="offset points", xytext=(8, 6), fontsize=8)
        axi.set_xlabel("VRAM de pico (GB) →  (menor = melhor)")
        axi.set_ylabel("nota do juiz (1-5) ↑")
        axi.set_title(f"Custo × qualidade — Qwen2.5-{sz}", pad=18)
        axi.margins(x=0.18, y=0.22)
        axi.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(RES / "fig_custo_qualidade.png", dpi=130); plt.close(fig)
    print("fig_custo_qualidade.png")


def fig_juiz():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axi, sz in zip(axes, SIZES):
        vals = [L(f"juiz_{m}_{sz}.json")["juiz_media"] if L(f"juiz_{m}_{sz}.json") else 0 for m in METHODS]
        bars = axi.bar([ROT[m] for m in METHODS], vals, color=[COR[m] for m in METHODS])
        axi.set_title(f"Juiz 1-5 — Qwen2.5-{sz}"); axi.set_ylim(0, 5); axi.set_ylabel("nota média")
        for b, v in zip(bars, vals):
            axi.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", fontsize=9)
        axi.tick_params(axis="x", labelrotation=12)
    fig.tight_layout(); fig.savefig(RES / "fig_juiz.png", dpi=130); plt.close(fig)
    print("fig_juiz.png")


if __name__ == "__main__":
    fig_dataset(); fig_ppl(); fig_terseness(); fig_juiz_tipo(); fig_custo_qualidade(); fig_juiz()
    print("OK — figuras em", RES)
