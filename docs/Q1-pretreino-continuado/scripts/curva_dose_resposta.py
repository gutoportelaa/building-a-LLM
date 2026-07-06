#!/usr/bin/env python3
"""
curva_dose_resposta.py — Figura PPL no held-out × tokens de treino (escala log).

DAPT escala com log(N_tokens): cada ~10x mais tokens rende um incremento de ganho
aproximadamente constante na CE/PPL (vide regra empírica do projeto e Juru, que
mostra ganho crescente por checkpoint até ~7B tokens). Esta figura comunica isso
em uma imagem — e substitui a tabela de jobs bugados no relatório.

Lê um JSON com a lista de pontos medidos (NÃO inventa números):
    [
      {"tokens": 700000,  "ppl": 9.7,  "corpus": "unificado"},
      {"tokens": 8600000, "ppl": 9.2,  "corpus": "unificado"},
      {"tokens": 86000000,"ppl": 8.90, "corpus": "unificado"},
      {"tokens": 0,       "ppl": 10.03,"corpus": "unificado", "baseline": true}
    ]

Uso:
    python scripts/curva_dose_resposta.py \
        --points resultados/curva_pontos.json \
        --output resultados/figuras/curva_dose_resposta.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Curva dose-resposta PPL x tokens")
    parser.add_argument("--points", default="resultados/curva_pontos.json")
    parser.add_argument("--output", default="resultados/figuras/curva_dose_resposta.png")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = json.loads(Path(args.points).read_text())
    # agrupa por corpus
    by_corpus: dict[str, list[dict]] = {}
    for p in points:
        by_corpus.setdefault(p.get("corpus", "corpus"), []).append(p)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for corpus, pts in by_corpus.items():
        # baseline (tokens=0) vira linha horizontal de referência
        base = next((p for p in pts if p.get("baseline")), None)
        curve = sorted([p for p in pts if not p.get("baseline") and p["tokens"] > 0],
                       key=lambda x: x["tokens"])
        if curve:
            xs = [p["tokens"] for p in curve]
            ys = [p["ppl"] for p in curve]
            ax.plot(xs, ys, marker="o", label=f"{corpus} (DAPT)")
            for p in curve:
                ax.annotate(f"{p['ppl']:.2f}", (p["tokens"], p["ppl"]),
                            textcoords="offset points", xytext=(0, 7), fontsize=8)
        if base:
            ax.axhline(base["ppl"], linestyle="--", alpha=0.6,
                       label=f"{corpus} baseline (PPL {base['ppl']:.2f})")

    ax.set_xscale("log")
    ax.set_xlabel("Tokens de pré-treino continuado (escala log)")
    ax.set_ylabel("Perplexidade no held-out")
    ax.set_title("Dose-resposta do DAPT no DOM-PI: PPL cai com log(tokens)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figura salva em {out}")


if __name__ == "__main__":
    main()
