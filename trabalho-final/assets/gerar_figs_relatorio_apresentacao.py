#!/usr/bin/env python3
"""Figuras novas para o relatório de apresentação (Q1, Q5, Q6).

Padrão visual único do relatório:
  antes/sem intervenção = azul-claro (#86b6ef)
  depois/com intervenção = azul-escuro (#256abf)
Rampa ordinal de um matiz, validada (CVD-safe). Marcas finas, rótulos diretos,
grade recessiva, sem moldura.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path("/home/gutemberg/Documents/building-a-LLM/trabalho-final")

AZUL_CLARO = "#86b6ef"   # antes / sem
AZUL_ESC   = "#256abf"   # depois / com
INK        = "#0b0b0b"
MUTED      = "#898781"
GRID       = "#e1e0d9"
BASELINE   = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": "#52514e",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

def estilo(ax, ylabel=None):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)

def rotulo_barras(ax, bars, fmt="{:.2f}", dy=0.01, fontsize=9, color=INK):
    top = ax.get_ylim()[1]
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * dy,
                fmt.format(b.get_height()).replace(".", ","), ha="center", va="bottom",
                fontsize=fontsize, color=color)

# ────────────────────────────────────────────────────────────────────────────
# Q1a — PPL de domínio antes × depois (3 execuções de DAPT)
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 3.8))
labels = ["Qwen2.5-1.5B\ncorpus unificado", "Qwen2.5-1.5B\ncorpus Teresina", "Llama-3.2-3B\ncorpus unificado"]
antes  = [9.82, 6.91, 10.55]
depois = [8.74, 6.02, 8.09]
delta  = ["−11,1%", "−12,9%", "−23,3%"]
x = np.arange(len(labels)); w = 0.32
b1 = ax.bar(x - w/2 - 0.01, antes,  w, color=AZUL_CLARO, label="antes do DAPT", zorder=3)
b2 = ax.bar(x + w/2 + 0.01, depois, w, color=AZUL_ESC,   label="depois do DAPT", zorder=3)
ax.set_ylim(0, 12.4)
estilo(ax, "Perplexidade no held-out de domínio  (menor = melhor)")
rotulo_barras(ax, b1); rotulo_barras(ax, b2)
for xi, (a, d, t) in enumerate(zip(antes, depois, delta)):
    ax.annotate(t, (xi + w/2 + 0.01, d/2), ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, color=INK)
ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
ax.set_title("Q1 · DAPT reduz a perplexidade no domínio DOM-PI nas três execuções",
             fontsize=11, fontweight="bold", loc="left", pad=30)
fig.savefig(BASE / "Q1-pretreino-continuado/resultados/figuras/q1_ppl_antes_depois.png")
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Q1b — retenção: PPL de domínio cai, PPL geral não se move (full FT unificado)
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 3.4))
grupos = ["Domínio DOM-PI\n(o que queremos melhorar)", "Domínio geral — Wikipedia-PT\n(o que não podemos estragar)"]
antes  = [9.82, 8.80]
depois = [8.74, 8.79]
notas  = ["−11,1%", "−0,2%"]
x = np.arange(len(grupos)); w = 0.30
b1 = ax.bar(x - w/2 - 0.01, antes,  w, color=AZUL_CLARO, label="antes do DAPT", zorder=3)
b2 = ax.bar(x + w/2 + 0.01, depois, w, color=AZUL_ESC,   label="depois do DAPT", zorder=3)
ax.set_ylim(0, 11.6)
estilo(ax, "Perplexidade  (menor = melhor)")
rotulo_barras(ax, b1); rotulo_barras(ax, b2)
for xi, (d, t) in enumerate(zip(depois, notas)):
    ax.annotate(t, (xi + w/2 + 0.01, d/2), ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
ax.set_xticks(x); ax.set_xticklabels(grupos, fontsize=9, color=INK)
ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
ax.set_title("Q1 · Ganho de domínio sem esquecimento: a PPL geral não se move",
             fontsize=11, fontweight="bold", loc="left", pad=30)
fig.savefig(BASE / "Q1-pretreino-continuado/resultados/figuras/q1_retencao_geral.png")
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Q5a — acerto factual sem RAG × com RAG por gerador
# ────────────────────────────────────────────────────────────────────────────
figdir5 = BASE / "Q5-rag/resultados/figuras"; figdir5.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7.2, 3.8))
gers   = ["G1 · Qwen2.5-1.5B\nbase", "G2 · Qwen2.5-1.5B\nDAPT (Q1)", "G3 · qwen2.5:14b\n(Ollama)"]
sem    = [0.0, 0.0, 0.0]
com    = [11.1, 16.7, 33.3]
x = np.arange(len(gers)); w = 0.32
b1 = ax.bar(x - w/2 - 0.01, sem, w, color=AZUL_CLARO, label="sem RAG", zorder=3)
b2 = ax.bar(x + w/2 + 0.01, com, w, color=AZUL_ESC,   label="com RAG (standard)", zorder=3)
ax.set_ylim(0, 40)
estilo(ax, "Acerto factual no benchmark (%)")
for b in b1:  # zero bars: label "0%"
    ax.text(b.get_x() + b.get_width()/2, 0.8, "0%", ha="center", va="bottom",
            fontsize=9, color=MUTED)
rotulo_barras(ax, b2, fmt="{:.1f}%")
ax.set_xticks(x); ax.set_xticklabels(gers, fontsize=9, color=INK)
ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
ax.set_title("Q5 · Sem RAG nenhum gerador acerta; com RAG o ganho escala com a capacidade",
             fontsize=11, fontweight="bold", loc="left", pad=30)
fig.savefig(figdir5 / "q5_acerto_geradores.png")
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Q5b — funil: onde o pipeline perde (recall do retriever é o gargalo)
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 3.2))
etapas = ["Perguntas factuais avaliáveis", "Contexto certo recuperado\n(recall do retriever)",
          "Resposta correta gerada\n(G3, melhor gerador)"]
vals   = [100.0, 42.1, 33.3]
cores  = ["#9ec5f4", "#5598e7", "#256abf"]  # rampa ordinal azul
y = np.arange(len(etapas))[::-1]
bars = ax.barh(y, vals, height=0.52, color=cores, zorder=3)
for yi, v in zip(y, vals):
    ax.text(v + 1.5, yi, f"{v:.0f}%".replace(".", ","), va="center", fontsize=10,
            fontweight="bold", color=INK)
ax.set_yticks(y); ax.set_yticklabels(etapas, fontsize=9, color=INK)
ax.set_xlim(0, 112)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, linewidth=0.7)
ax.tick_params(length=0)
ax.set_xlabel("% das perguntas", fontsize=9)
ax.set_title("Q5 · O gargalo é a recuperação: quando o contexto vem, o 14B acerta ~75%",
             fontsize=11, fontweight="bold", loc="left", pad=12)
fig.savefig(figdir5 / "q5_funil_gargalo.png")
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Q5c — custo por modo de RAG (chamadas de LLM por pergunta)
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.8, 3.0))
modos  = ["Standard", "HyDE", "Self-reflective", "Agêntico (ReAct)"]
calls  = [1, 2, 3, 4]
labels = ["1 chamada", "2 chamadas", "2–3 chamadas", "3+ chamadas"]
y = np.arange(len(modos))[::-1]
bars = ax.barh(y, calls, height=0.5, color=["#9ec5f4", "#6da7ec", "#3987e5", "#256abf"], zorder=3)
for yi, c, l in zip(y, calls, labels):
    ax.text(c + 0.07, yi, l, va="center", fontsize=9.5, color=INK)
ax.set_yticks(y); ax.set_yticklabels(modos, fontsize=9.5, color=INK)
ax.set_xlim(0, 5.6)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, linewidth=0.7)
ax.tick_params(length=0)
ax.set_xlabel("Chamadas de LLM por pergunta (≈ custo/latência)", fontsize=9)
ax.set_title("Q5 · Custo por modo: cada chamada extra multiplica a latência",
             fontsize=11, fontweight="bold", loc="left", pad=12)
fig.savefig(figdir5 / "q5_custo_modos.png")
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Q6a — proteção por categoria, sem × com guardrails
# ────────────────────────────────────────────────────────────────────────────
figdir6 = BASE / "Q6-guardrails/resultados/figuras"; figdir6.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(7.2, 3.8))
cats  = ["Conteúdo nocivo\n(n=3)", "Fora de escopo\n(n=6)", "Vazamento de PII\n(n=6)", "Prompt injection\n(n=5)"]
sem   = [66.7, 66.7, 83.3, 80.0]
com   = [100.0, 100.0, 100.0, 100.0]
x = np.arange(len(cats)); w = 0.32
b1 = ax.bar(x - w/2 - 0.01, sem, w, color=AZUL_CLARO, label="sem guardrails", zorder=3)
b2 = ax.bar(x + w/2 + 0.01, com, w, color=AZUL_ESC,   label="com guardrails", zorder=3)
ax.set_ylim(0, 118)
estilo(ax, "Ataques corretamente tratados (%)")
rotulo_barras(ax, b1, fmt="{:.0f}%"); rotulo_barras(ax, b2, fmt="{:.0f}%")
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9, color=INK)
ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
ax.set_title("Q6 · Proteção por categoria de ameaça: 100% em todas com guardrails",
             fontsize=11, fontweight="bold", loc="left", pad=30)
fig.savefig(figdir6 / "q6_protecao_categoria.png")
plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────
# Q6b — trade-off Helpfulness × Harmlessness por configuração
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 3.8))
confs = ["Sem guardrails", "Guardrails\n+ groundedness", "Guardrails\n(recomendado)"]
harm  = [75, 100, 100]   # proteção
help_ = [100, 70, 100]   # helpfulness
x = np.arange(len(confs)); w = 0.32
b1 = ax.bar(x - w/2 - 0.01, harm,  w, color=AZUL_ESC,   label="Harmlessness (proteção)", zorder=3)
b2 = ax.bar(x + w/2 + 0.01, help_, w, color=AZUL_CLARO, label="Helpfulness (legítimas respondidas)", zorder=3)
ax.set_ylim(0, 118)
estilo(ax, "% no benchmark de 30 perguntas")
rotulo_barras(ax, b1, fmt="{:.0f}%"); rotulo_barras(ax, b2, fmt="{:.0f}%")
ax.set_xticks(x); ax.set_xticklabels(confs, fontsize=9, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
ax.set_title("Q6 · O trade-off Helpfulness × Harmlessness — e a configuração que resolve os dois",
             fontsize=11, fontweight="bold", loc="left", pad=30)
fig.savefig(figdir6 / "q6_tradeoff_configs.png")
plt.close(fig)

print("figuras geradas:")
for p in [BASE/"Q1-pretreino-continuado/resultados/figuras/q1_ppl_antes_depois.png",
          BASE/"Q1-pretreino-continuado/resultados/figuras/q1_retencao_geral.png",
          figdir5/"q5_acerto_geradores.png", figdir5/"q5_funil_gargalo.png", figdir5/"q5_custo_modos.png",
          figdir6/"q6_protecao_categoria.png", figdir6/"q6_tradeoff_configs.png"]:
    print(" ", p.relative_to(BASE), p.exists())
