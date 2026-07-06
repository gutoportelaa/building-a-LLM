#!/usr/bin/env python3
"""
analisar_complementar.py — Análise complementar da Q2 sobre o experimento-irmão
(SFT LoRA da mesma base Qwen2.5-1.5B-Instruct com perguntas de conhecimento geral de CC).

Dados brutos em dados_brutos/ (execução do grupo de Pedro E. M. Carvalho, integrada a este
repositório para reanálise — github.com/PedroEmanuelMoreiraCarvalho/Trabalho_Final_IA_Fine_Tuning).

Produz, no padrão visual do relatório (azul-claro = antes, azul-escuro = depois):
  figuras/q2c_curva_epocas.png     — train loss × eval loss por época (evidência da hipótese H3)
  figuras/q2c_sim_compressao.png   — similaridade e comprimento antes/depois por rodada de avaliação
"""
import csv
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
FIG = BASE / "figuras"
FIG.mkdir(exist_ok=True)

AZUL_CLARO = "#86b6ef"
AZUL_ESC = "#256abf"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "text.color": INK,
    "axes.edgecolor": BASELINE, "axes.labelcolor": "#52514e",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def sem_moldura(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


# ── 1) Curva de épocas (trainer_state do LoRA 1.5B) ────────────────────────────
ts = json.load(open(BASE / "dados_brutos/trainer_state.json"))
tr = [(h["epoch"], h["loss"]) for h in ts["log_history"] if "loss" in h]
ev = [(h["epoch"], h["eval_loss"]) for h in ts["log_history"] if "eval_loss" in h]

fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot([e for e, _ in tr], [l for _, l in tr], color=AZUL_CLARO, lw=2, label="loss de treino", zorder=3)
ax.plot([e for e, _ in ev], [l for _, l in ev], color=AZUL_ESC, lw=2.4, marker="o", ms=7,
        label="loss de validação (held-out)", zorder=4)
for e, l in ev:
    ax.annotate(f"{l:.3f}".replace(".", ","), (e, l), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9, fontweight="bold", color=INK)
ax.axvspan(1.5, 3.05, color="#f5eaea", zorder=1)
ax.text(2.25, 1.85, "região de memorização:\ntreino cai, held-out não", ha="center",
        fontsize=8.5, color="#5a1818")
ax.set_xlabel("época")
ax.set_ylabel("cross-entropy (loss)")
ax.set_xlim(0, 3.1)
sem_moldura(ax)
ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
ax.set_title("Q2 · SFT LoRA (experimento-irmão): a validação satura após ~1,3 época",
             fontsize=11, fontweight="bold", loc="left", pad=30)
fig.savefig(FIG / "q2c_curva_epocas.png")
plt.close(fig)

# ── 2) Similaridade × comprimento, antes/depois, por rodada de avaliação ───────
rodadas = []
for f in sorted(glob.glob(str(BASE / "dados_brutos/avaliacao_comparativa_*.json"))):
    d = json.load(open(f))
    ponto = {"rodada": Path(f).stem.split("_")[-1]}
    for k, key in [("base", "antes"), ("fine_tuned", "depois")]:
        rs = d["modelos"][k]["resultados"]
        ponto[f"sim_{key}"] = st.mean(r["similaridade"] for r in rs)
        ponto[f"tam_{key}"] = st.mean(len(r["resposta_gerada"]) for r in rs)
        ponto["n"] = len(rs)
    rodadas.append(ponto)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
x = range(len(rodadas))
labels = [f"rodada {i+1}\n(n={r['n']})" for i, r in enumerate(rodadas)]
w = 0.34
for ax, met, tit, ylab in [
    (ax1, "sim", "Similaridade com a referência", "similaridade média (0–1)"),
    (ax2, "tam", "Comprimento da resposta", "caracteres (média)"),
]:
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], [r[f"{met}_antes"] for r in rodadas], w,
                color=AZUL_CLARO, label="antes (base)", zorder=3)
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], [r[f"{met}_depois"] for r in rodadas], w,
                color=AZUL_ESC, label="depois (LoRA)", zorder=3)
    fmt = (lambda v: f"{v:.2f}".replace(".", ",")) if met == "sim" else (lambda v: f"{v:.0f}")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02, fmt(b.get_height()),
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(tit, fontsize=10, fontweight="bold", loc="left")
    ax.grid(axis="x", visible=False)
    sem_moldura(ax)
    ax.spines["left"].set_visible(False)
ax1.set_ylim(0, 0.85)
ax1.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.08), ncol=2)
fig.suptitle("Q2 · Experimento-irmão: na rodada 3, a resposta encurta (713→284) e a similaridade cai",
             fontsize=11, fontweight="bold", x=0.005, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(FIG / "q2c_sim_compressao.png")
plt.close(fig)

# ── resumo no terminal ─────────────────────────────────────────────────────────
print("eval_loss por época:", [(round(e, 2), round(l, 3)) for e, l in ev])
for i, r in enumerate(rodadas):
    print(f"rodada {i+1}: sim {r['sim_antes']:.3f}→{r['sim_depois']:.3f} · "
          f"tam {r['tam_antes']:.0f}→{r['tam_depois']:.0f} chars (n={r['n']})")
print("figuras em", FIG)
