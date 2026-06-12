#!/usr/bin/env python3
"""
comparar_resultados.py — Gera tabela antes×depois a partir dos JSONs de avaliação.

Uso:
    python avaliacao/comparar_resultados.py \
        --baseline avaliacao/resultados_baseline.json \
        --postreino avaliacao/resultados_postreino.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def delta(before: float, after: float, invert: bool = False) -> str:
    """Seta indicando melhora (↓ para loss, ↑ para accuracy)."""
    diff = after - before
    if invert:
        return f"{'↓' if diff < 0 else '↑'} {diff:+.4f}"
    return f"{'↑' if diff > 0 else '↓'} {diff:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--postreino", required=True)
    args = parser.parse_args()

    b = json.loads(Path(args.baseline).read_text())
    p = json.loads(Path(args.postreino).read_text())

    print("\n" + "=" * 72)
    print(f"  Avaliação Antes × Depois — Pré-treino Continuado DOM-PI")
    print("=" * 72)
    print(f"  Baseline  : {b['model']}")
    print(f"  Pós-treino: {p['model']}")
    print(f"  Held-out  : {b.get('held_out_n_docs', '?')} docs  /  {b.get('held_out_n_examples', '?')} exemplos")
    print()

    rows = [
        ("Métrica", "Antes (baseline)", "Depois (pós-treino)", "Δ"),
        ("-" * 35, "-" * 18, "-" * 18, "-" * 14),
        # held-out
        ("HELD-OUT", "", "", ""),
        ("  Cross-Entropy",
         f"{b['cross_entropy']:.4f}",
         f"{p['cross_entropy']:.4f}",
         delta(b["cross_entropy"], p["cross_entropy"], invert=True)),
        ("  Perplexidade",
         f"{b['perplexity']:.2f}",
         f"{p['perplexity']:.2f}",
         delta(b["perplexity"], p["perplexity"], invert=True)),
        ("  Token Accuracy",
         f"{b['token_accuracy']:.4f}",
         f"{p['token_accuracy']:.4f}",
         delta(b["token_accuracy"], p["token_accuracy"])),
        ("", "", "", ""),
        # benchmark
        ("BENCHMARK Q&A (NLL da resposta dado o prompt)", "", "", ""),
        ("  Cross-Entropy",
         f"{b['bm_cross_entropy']:.4f}",
         f"{p['bm_cross_entropy']:.4f}",
         delta(b["bm_cross_entropy"], p["bm_cross_entropy"], invert=True)),
        ("  Perplexidade",
         f"{b['bm_perplexity']:.2f}",
         f"{p['bm_perplexity']:.2f}",
         delta(b["bm_perplexity"], p["bm_perplexity"], invert=True)),
        ("  Token Accuracy",
         f"{b['bm_token_accuracy']:.4f}",
         f"{p['bm_token_accuracy']:.4f}",
         delta(b["bm_token_accuracy"], p["bm_token_accuracy"])),
    ]

    col_widths = [36, 18, 18, 14]
    for row in rows:
        line = "  ".join(str(c).ljust(w) for c, w in zip(row, col_widths))
        print(line)

    print("=" * 72)
    print()
    print("Interpretação:")
    ppl_delta = p["perplexity"] - b["perplexity"]
    bm_ppl_delta = p["bm_perplexity"] - b["bm_perplexity"]
    if ppl_delta < 0:
        print(f"  ✔ Perplexidade no held-out caiu {abs(ppl_delta):.2f} pts → modelo aprendeu o domínio.")
    else:
        print(f"  ✗ Perplexidade no held-out subiu {ppl_delta:.2f} pts → verificar overfitting/lr.")
    if bm_ppl_delta < 0:
        print(f"  ✔ Perplexidade no benchmark caiu {abs(bm_ppl_delta):.2f} pts → fatos DOM-PI internalizados.")
    else:
        print(f"  ✗ Perplexidade no benchmark subiu {bm_ppl_delta:.2f} pts → pré-treino insuficiente para Q&A.")
    print()


if __name__ == "__main__":
    main()
