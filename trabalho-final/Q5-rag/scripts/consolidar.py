#!/usr/bin/env python3
"""
consolidar.py — Consolida os resultados das avaliações de RAG em tabelas e
exemplos para o relatório.

Lê:
  rag/resultados/retrieval.json        (recall de contexto standard×hyde)
  rag/resultados/gen_G1_base.json      (G1 base — antes do pré-treino)
  rag/resultados/gen_G2_dapt.json      (G2 DAPT — pré-treinado)
  rag/resultados/gen_G3_14b.json       (G3 qwen2.5:14b — inferência qualificada)
  rag/resultados/traces.json           (traços hyde/reflexivo/agentico)  [opcional]

Emite no stdout um resumo legível e salva rag/resultados/consolidado.json.
"""
from __future__ import annotations
import json
from pathlib import Path

R = Path("rag/resultados")


def load(name):
    p = R / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def acc(summary, mode, classe):
    try:
        c = summary[mode][classe]
        return c["acuracia"], c["acertos"], c["avaliaveis"]
    except (KeyError, TypeError):
        return None, 0, 0


def main():
    out = {}

    # 1. recall de contexto
    retr = load("retrieval.json")
    if retr:
        out["recall_contexto"] = retr["summary"]
        print("== RECALL DE CONTEXTO (perguntas rag/ambos) ==")
        for m, v in retr["summary"].items():
            print(f"  {m:9}: recall={v['recall']} ({v['hits']}/{v['avaliaveis']})")

    # 2. tabela de geração 3 geradores
    gens = [("G1 base (antes)", "gen_G1_base.json"),
            ("G2 DAPT (pré-treinado)", "gen_G2_dapt.json"),
            ("G3 qwen2.5:14b (qualif.)", "gen_G3_14b.json")]
    print("\n== ACURÁCIA DE RESPOSTA (perguntas rag, por conteúdo) ==")
    print(f"  {'gerador':28} {'no_rag':>10} {'standard':>10}   Δ")
    tabela = []
    for nome, fn in gens:
        d = load(fn)
        if not d:
            print(f"  {nome:28} {'(pendente)':>10}")
            continue
        a0, h0, n0 = acc(d["summary"], "no_rag", "rag")
        a1, h1, n1 = acc(d["summary"], "standard", "rag")
        delta = (a1 - a0) if (a0 is not None and a1 is not None) else None
        tabela.append({"gerador": nome, "no_rag": a0, "standard": a1,
                       "delta": delta, "acertos_norag": h0, "acertos_rag": h1,
                       "avaliaveis": n1})
        s0 = f"{a0:.3f}({h0}/{n0})" if a0 is not None else "—"
        s1 = f"{a1:.3f}({h1}/{n1})" if a1 is not None else "—"
        ds = f"+{delta:.3f}" if delta is not None else "—"
        print(f"  {nome:28} {s0:>10} {s1:>10}   {ds}")
    out["tabela_geracao"] = tabela

    # 3. exemplos lado a lado (mesma pergunta, 3 geradores, no_rag vs standard)
    exemplos = []
    g1, g2, g3 = load("gen_G1_base.json"), load("gen_G2_dapt.json"), load("gen_G3_14b.json")
    if g1 and g3:
        by_q = {}
        for d, key in [(g1, "G1_base"), (g2, "G2_DAPT"), (g3, "G3_14b")]:
            if not d:
                continue
            for r in d["results"]:
                by_q.setdefault(r["pergunta"], {"resposta_ref": r["resposta_ref"],
                                                "modo_q": r["modo_q"], "geradores": {}})
                by_q[r["pergunta"]]["geradores"][key] = r["saidas"]
        # seleciona alguns exemplos rag ilustrativos
        for q, info in by_q.items():
            if info["modo_q"] in ("rag", "ambos"):
                exemplos.append({"pergunta": q, **info})
        out["exemplos"] = exemplos[:8]

    # 4. traços
    tr = load("traces.json")
    if tr:
        out["traces"] = tr

    json.dump(out, open(R / "consolidado.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nsalvo:", R / "consolidado.json")


if __name__ == "__main__":
    main()
