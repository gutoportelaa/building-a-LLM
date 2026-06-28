#!/usr/bin/env python3
"""Gráficos de evidência da Q4 (destilação) a partir de resultados/avaliacao.json.

Complementa comparativo_por_questao.py (que gera heatmap/box/delta/abstenção).
Aqui o foco é a leitura "de banca": ganho por configuração, custo-benefício
(compressão × acerto) e antes→depois por domínio. Paleta = a dos relatórios HTML.

Gera (em resultados/figuras/):
  - barras_keyrecall_config.png : 14 modelos (2 bases + 12 destilados), base como referência
  - compressao_vs_keyrecall.png : razão de compressão (×) vs key_recall — eixo custo-benefício
  - antes_depois_dominio.png    : base_15 vs melhor aluno, por domínio (DOM-PI / docentesDC)
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "resultados")
FIG = os.path.join(RES, "figuras")
os.makedirs(FIG, exist_ok=True)

# Paleta dos relatórios (assets/estilo.css)
ACCENT = "#1e3a5f"
ACCENT2 = "#2e5ca0"
OK = "#2e802a"
BAD = "#a02020"
WARN = "#c08000"
MUTED = "#9a9a9a"
GRID = "#d2d2d2"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": "#888",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "figure.dpi": 130,
})

# Parâmetros (bilhões) — para razão de compressão vs professor 14B.
PARAMS_B = {"0.5b": 0.5, "1.5b": 1.5}
TEACHER_B = 14.0


def load(name):
    return json.load(open(os.path.join(RES, name), encoding="utf-8"))


def by_label(d):
    return {m["rotulo"]: m for m in d["modelos"]}


def color_for(rot):
    if rot.startswith("base"):
        return MUTED
    if rot.endswith("combined"):
        return OK
    if rot.endswith("kl"):
        return ACCENT2
    return WARN  # ce


# ───────────────────────── 1. Barras key_recall por config ─────────────────────────
def fig_barras(d):
    ms = d["modelos"]
    rot = [m["rotulo"] for m in ms]
    kr = [m["geral"]["key_recall"] for m in ms]
    cols = [color_for(r) for r in rot]
    base15 = by_label(d)["base_15"]["geral"]["key_recall"]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(rot))
    bars = ax.bar(x, kr, color=cols, edgecolor="#333", linewidth=0.4)
    ax.axhline(base15, color=BAD, ls="--", lw=1.2, label=f"base 1.5B (referência) = {base15:.3f}")
    # destaca o campeão
    imax = int(np.argmax(kr))
    bars[imax].set_edgecolor(OK)
    bars[imax].set_linewidth(2.0)
    ax.annotate("★", (x[imax], kr[imax] + 0.012), ha="center", fontsize=13)

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("d_", "").replace("_", "·") for r in rot],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("key_recall (acerto factual)")
    ax.set_ylim(0, 0.80)
    ax.set_title("Acerto factual por configuração — os 12 destilados superam as bases",
                 color=ACCENT, fontweight="bold", fontsize=12)
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=MUTED, label="base"), Patch(facecolor=WARN, label="ce (texto)"),
           Patch(facecolor=ACCENT2, label="kl (logits)"), Patch(facecolor=OK, label="combinado")]
    ax.legend(handles=leg + [plt.Line2D([], [], color=BAD, ls="--", label="base 1.5B")],
              loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    out = os.path.join(FIG, "barras_keyrecall_config.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("escrito:", out)


# ───────────────────────── 2. Compressão × key_recall ─────────────────────────
def fig_compressao(d):
    bl = by_label(d)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    # pontos: cada destilado + as bases. x = razão de compressão (×), y = key_recall
    def comp(rot):
        size = "0.5b" if "0.5b" in rot or "05" in rot else "1.5b"
        return TEACHER_B / PARAMS_B[size], size

    for m in d["modelos"]:
        rot = m["rotulo"]
        kr = m["geral"]["key_recall"]
        c, size = comp(rot)
        if rot.startswith("base"):
            ax.scatter(c, kr, s=120, marker="X", color=MUTED, edgecolor="#333",
                       zorder=3, label=None)
            ax.annotate(f"base {size}", (c, kr), textcoords="offset points",
                        xytext=(8, -4), fontsize=8, color="#555")
        else:
            col = color_for(rot)
            mk = "o" if size == "1.5b" else "^"
            ax.scatter(c, kr, s=90, marker=mk, color=col, edgecolor="#333",
                       linewidth=0.5, zorder=3, alpha=0.9)

    # campeão anotado
    champ = max(d["modelos"], key=lambda m: m["geral"]["key_recall"])
    cc, _ = comp(champ["rotulo"])
    ax.annotate("★ 1.5B·B·combinado", (cc, champ["geral"]["key_recall"]),
                textcoords="offset points", xytext=(10, 2), fontsize=9,
                color=OK, fontweight="bold")

    ax.set_xlabel("razão de compressão vs professor 14B  (× menor)")
    ax.set_ylabel("key_recall (acerto factual)")
    ax.set_title("Custo-benefício: compressão × acerto factual",
                 color=ACCENT, fontweight="bold", fontsize=12)
    ax.set_xlim(4, 32)  # 1.5B≈9× e 0.5B=28×; margem à direita p/ o rótulo do campeão
    ax.set_xticks([9.3, 28])
    ax.set_xticklabels(["9× (1.5B)", "28× (0.5B)"])
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], marker="o", ls="", color=OK, label="combinado"),
           Line2D([], [], marker="o", ls="", color=ACCENT2, label="kl (logits)"),
           Line2D([], [], marker="o", ls="", color=WARN, label="ce (texto)"),
           Line2D([], [], marker="X", ls="", color=MUTED, label="base (sem destilar)"),
           Line2D([], [], marker="o", ls="", color="#bbb", label="○ aluno 1.5B  /  △ aluno 0.5B")]
    ax.legend(handles=leg, fontsize=8, loc="center left")
    fig.tight_layout()
    out = os.path.join(FIG, "compressao_vs_keyrecall.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("escrito:", out)


# ───────────────────────── 3. Antes → depois por domínio ─────────────────────────
def fig_antes_depois(d):
    bl = by_label(d)
    base = bl["base_15"]["por_dominio"]
    best = bl["d_1.5b_B_combined"]["por_dominio"]
    doms = ["DOM-PI", "docentesDC"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(doms))
    w = 0.36
    kr_base = [base[dm]["key_recall"] for dm in doms]
    kr_best = [best[dm]["key_recall"] for dm in doms]
    b1 = ax.bar(x - w / 2, kr_base, w, color=MUTED, edgecolor="#333", label="base 1.5B")
    b2 = ax.bar(x + w / 2, kr_best, w, color=OK, edgecolor="#333", label="melhor aluno (1.5B·B·comb)")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2, r.get_height() + 0.008),
                        ha="center", fontsize=9)
    # setas de ganho
    for i, dm in enumerate(doms):
        gain = (kr_best[i] - kr_base[i]) / kr_base[i] * 100
        ax.annotate(f"+{gain:.0f}%", (x[i], max(kr_base[i], kr_best[i]) + 0.06),
                    ha="center", fontsize=11, color=OK, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(doms)
    ax.set_ylabel("key_recall (acerto factual)")
    ax.set_ylim(0, 0.95)
    ax.set_title("Antes → depois da destilação, por domínio",
                 color=ACCENT, fontweight="bold", fontsize=12)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    fig.tight_layout()
    out = os.path.join(FIG, "antes_depois_dominio.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("escrito:", out)


if __name__ == "__main__":
    d = load("avaliacao.json")
    fig_barras(d)
    fig_compressao(d)
    fig_antes_depois(d)
    print("OK — 3 figuras em", FIG)
