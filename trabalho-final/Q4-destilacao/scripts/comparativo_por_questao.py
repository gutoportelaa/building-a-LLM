#!/usr/bin/env python3
"""Comparativo por questão do benchmark 100Q (DOM-PI + docentesDC).

Unifica os 5 JSONs que compartilham `benchmark_destilacao_100.jsonl`
(núcleo, cross-família/bxf, dapt unificado, dapt Teresina, ULD) — o
experimento de FUTEBOL é deixado de fora de propósito (benchmark próprio
de 41 fatos, Copa 2026, não casa por pergunta).

Gera (em resultados/figuras/):
  - heatmap_keyrecall.png : matriz pergunta × modelo (key_recall)
  - delta_base_vs_melhor.png : ganho por pergunta (base_15 → melhor destilado)
  - abstencao_vs_fato.png : key_recall médio por modelo, separando abstenção/fato
  - box_por_metodo.png : distribuição de key_recall por método
  - tabela_resumo.md : tabela markdown com agregados
"""
import json
import os
from collections import OrderedDict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "resultados")
FIG = os.path.join(RES, "figuras")
os.makedirs(FIG, exist_ok=True)
BENCH = os.path.join(BASE, "dados", "benchmark_destilacao_100.jsonl")

# JSONs sobre o benchmark 100Q (futebol fica de fora — extra).
FONTES = [
    "avaliacao.json",       # núcleo: 2 bases + 12 destilados
    "avaliacao_bxf.json",   # cross-família black-box (zephyr)
    "avaliacao_dapt.json",  # DAPT unificado -> aluno
    "avaliacao_daptT.json", # DAPT Teresina -> aluno
    "avaliacao_uld.json",   # ULD cross-tokenizer
]


def is_abstencao(ref: str) -> bool:
    t = ref.lower()
    return ("não consta" in t) or ("nao consta" in t) or ("não há" in t) or ("não foi" in t)


# ---- carrega benchmark e rótulos de abstenção/fato ----
recs = [json.loads(l) for l in open(BENCH)]
ids = [r["id"] for r in recs]
src = {r["id"]: r["source"] for r in recs}
abst = {r["id"]: is_abstencao(r["reference"]) for r in recs}
n_ab = sum(abst.values())
print(f"benchmark: {len(recs)} Q | abstenções: {n_ab} | fato real: {len(recs)-n_ab}")

# ---- unifica modelos (dedup por rótulo; base_15 repete em todos) ----
modelos = OrderedDict()  # rotulo -> {id -> (rougeL, key_recall)}
for f in FONTES:
    d = json.load(open(os.path.join(RES, f)))
    for m in d["modelos"]:
        rot = m["rotulo"]
        if rot in modelos:
            continue
        def num(v):
            return float(v) if isinstance(v, (int, float)) else np.nan
        modelos[rot] = {x["id"]: (num(x["rougeL"]), num(x["key_recall"])) for x in m["detalhe"]}
print(f"modelos únicos no benchmark 100Q: {len(modelos)}")
print("  " + ", ".join(modelos.keys()))

# ordem de exibição: bases, núcleo 0.5b, núcleo 1.5b, extras
ORDEM = [
    "base_05", "base_15",
    "d_0.5b_A_ce", "d_0.5b_A_kl", "d_0.5b_A_combined",
    "d_0.5b_B_ce", "d_0.5b_B_kl", "d_0.5b_B_combined",
    "d_1.5b_A_ce", "d_1.5b_A_kl", "d_1.5b_A_combined",
    "d_1.5b_B_ce", "d_1.5b_B_kl", "d_1.5b_B_combined",
    "bxf_0.5b_ce", "bxf_1.5b_ce", "uld_0.5b", "uld_1.5b",
    "dapt_raw", "dapt_B_combined", "dapt_Bxf_ce",
    "daptT_raw", "daptT_B_combined", "daptT_Bxf_ce",
]
rotulos = [r for r in ORDEM if r in modelos] + [r for r in modelos if r not in ORDEM]

# matrizes [modelo, pergunta]
KR = np.array([[modelos[r][i][1] for i in ids] for r in rotulos])
RG = np.array([[modelos[r][i][0] for i in ids] for r in rotulos])

# ================== FIG 1: heatmap key_recall ==================
fig, ax = plt.subplots(figsize=(16, 9))
cmap = matplotlib.colormaps["viridis"].with_extremes(bad="lightgray")
im = ax.imshow(np.ma.masked_invalid(KR), aspect="auto", cmap=cmap, vmin=0, vmax=1)
ax.set_yticks(range(len(rotulos)))
ax.set_yticklabels(rotulos, fontsize=8)
ax.set_xticks(range(0, len(ids), 5))
ax.set_xticklabels(range(0, len(ids), 5), fontsize=7)
ax.set_xlabel("pergunta (bm000…bm099)  —  ●=fato real abaixo, resto abstenção")
ax.set_title("key_recall por pergunta × modelo (benchmark 100Q, sem RAG; futebol excluído)")
# faixa indicando abstenção (cinza) vs fato (preto) no topo
for j, i in enumerate(ids):
    ax.add_patch(plt.Rectangle((j - .5, -1.5), 1, 1,
                 color="0.7" if abst[i] else "0.1", clip_on=False))
ax.set_ylim(len(rotulos) - .5, -2)
cbar = fig.colorbar(im, ax=ax, fraction=0.025)
cbar.set_label("key_recall")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "heatmap_keyrecall.png"), dpi=130)
plt.close(fig)

# ================== FIG 2: delta base_15 -> melhor destilado ==================
melhor = "d_1.5b_B_combined"
b = np.array([modelos["base_15"][i][1] for i in ids])
m = np.array([modelos[melhor][i][1] for i in ids])
delta = m - b
delta = np.nan_to_num(delta, nan=0.0)
order = np.argsort(delta)
colors = ["#777" if abst[ids[k]] else "#c0392b" for k in order]
fig, ax = plt.subplots(figsize=(15, 6))
ax.bar(range(len(ids)), delta[order], color=colors, width=1.0)
ax.axhline(0, color="k", lw=.6)
ax.set_title(f"Δ key_recall por pergunta: {melhor} − base_15  (ordenado)\n"
             f"cinza = abstenção  |  vermelho = fato real")
ax.set_ylabel("Δ key_recall")
ax.set_xlabel("perguntas (ordenadas por ganho)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "delta_base_vs_melhor.png"), dpi=130)
plt.close(fig)

# ================== FIG 3: key_recall médio abstenção vs fato ==================
ab_mask = np.array([abst[i] for i in ids])
kr_ab = np.nanmean(KR[:, ab_mask], axis=1)
kr_ft = np.nanmean(KR[:, ~ab_mask], axis=1)
y = np.arange(len(rotulos))
fig, ax = plt.subplots(figsize=(11, 10))
ax.barh(y - .2, kr_ab, height=.4, label=f"abstenção (n={int(ab_mask.sum())})", color="#7f8c8d")
ax.barh(y + .2, kr_ft, height=.4, label=f"fato real (n={int((~ab_mask).sum())})", color="#c0392b")
ax.set_yticks(y)
ax.set_yticklabels(rotulos, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("key_recall médio")
ax.set_title("Onde mora o ganho: abstenção vs fato real (sem RAG)\n"
             "o salto é grande na abstenção, quase nulo no fato real")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "abstencao_vs_fato.png"), dpi=130)
plt.close(fig)

# ================== FIG 4: box-plot por método (só núcleo) ==================
grupos = OrderedDict([
    ("base", ["base_05", "base_15"]),
    ("ce", [r for r in rotulos if r.startswith("d_") and r.endswith("_ce")]),
    ("kl", [r for r in rotulos if r.startswith("d_") and r.endswith("_kl")]),
    ("combined", [r for r in rotulos if r.startswith("d_") and r.endswith("_combined")]),
])
data, labels = [], []
for g, rs in grupos.items():
    vals = np.concatenate([KR[rotulos.index(r)] for r in rs]) if rs else np.array([])
    vals = vals[~np.isnan(vals)]
    if vals.size:
        data.append(vals)
        labels.append(f"{g}\n(n={len(rs)} mod)")
fig, ax = plt.subplots(figsize=(9, 6))
bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
for patch, c in zip(bp["boxes"], ["#bdc3c7", "#3498db", "#2ecc71", "#e74c3c"]):
    patch.set_facecolor(c)
ax.set_ylabel("key_recall (por pergunta)")
ax.set_title("Distribuição de key_recall por método de destilação (núcleo)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "box_por_metodo.png"), dpi=130)
plt.close(fig)

# ================== tabela resumo markdown ==================
lines = ["# Comparativo por questão — agregados (benchmark 100Q, sem RAG)\n",
         f"_Futebol/Copa-2026 excluído (benchmark próprio). Abstenções: {n_ab}/100._\n",
         "| modelo | KR geral | RG geral | KR abstenção | KR fato real |",
         "|---|---|---|---|---|"]
for r in rotulos:
    kr_all = np.nanmean(KR[rotulos.index(r)])
    rg_all = np.nanmean(RG[rotulos.index(r)])
    a = np.nanmean(KR[rotulos.index(r)][ab_mask])
    f = np.nanmean(KR[rotulos.index(r)][~ab_mask])
    lines.append(f"| {r} | {kr_all:.3f} | {rg_all:.3f} | {a:.3f} | {f:.3f} |")
open(os.path.join(RES, "tabela_resumo_por_questao.md"), "w").write("\n".join(lines) + "\n")

print("\nFiguras salvas em:", FIG)
for f in sorted(os.listdir(FIG)):
    print("  -", f)
print("Tabela:", os.path.join(RES, "tabela_resumo_por_questao.md"))
