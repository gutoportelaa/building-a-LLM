#!/usr/bin/env python3
"""
gerar_painel_inferencias.py — Converte um avaliacao_*.json (saída de avaliar_destilacao.py)
+ o benchmark correspondente em fragmentos HTML de painel de inferência (.inf-block),
para embutir nos relatórios da Q4.

Cada painel = uma pergunta held-out, com a referência (professor+RAG) e as respostas
verbatim dos modelos escolhidos, com badges de ROUGE-L / key-recall. Degenerações
(loops de token repetido) são truncadas com um marcador honesto — nada é inventado.

Uso:
  python gerar_painel_inferencias.py \
      --benchmark ../dados/benchmark_destilacao_100.jsonl \
      --avaliacao ../resultados/avaliacao.json \
      --item bm009:base_15,d_1.5b_B_combined \
      --item bm031:base_15,d_1.5b_B_combined \
      --tag base_15=base:tag-base --tag d_1.5b_B_combined=aluno (1.5B·B·comb):tag-fullft \
      --out fragmento_inferencias.html
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


def colapsa_degeneracao(txt: str, max_chars: int = 360) -> str:
    """Trunca loops degenerados (mesmo trecho repetido / rajada de 1 caractere) de forma honesta."""
    t = txt.strip()
    # rajada do mesmo caractere (ex.: '猞猞猞…') 6+ vezes → colapsa
    t = re.sub(r"(.)\1{5,}", lambda m: m.group(1) * 3 + "…", t)
    # mesmo bloco de ~10+ chars repetido 2+ vezes (inclui quebras de linha) → mantém 1 + marcador
    m = re.search(r"(.{12,}?)(?:\s*\1){1,}", t, flags=re.DOTALL)
    if m and len(m.group(0)) > len(m.group(1)) + 8:
        t = t[: m.start()] + m.group(1).strip() + "  ⟲ […repetição degenerada…]"
    t = re.sub(r"[�]+", "", t)  # remove caractere de substituição (ruído de encoding)
    if len(t) > max_chars:
        t = t[:max_chars].rstrip() + "…"
    return t.strip()


def badge(rg, kr) -> str:
    def fmt(v):
        return "—" if v is None else f"{v:.2f}".replace(".", ",")
    return (f'<span class="badge b-info">RG {fmt(rg)}</span> '
            f'<span class="badge b-info">KR {fmt(kr)}</span>')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--avaliacao", required=True, action="append",
                    help="repetível; modelos de vários arquivos são mesclados (1ª ocorrência vence)")
    ap.add_argument("--item", action="append", required=True,
                    help="ID:modelo1,modelo2  (repetível)")
    ap.add_argument("--tag", action="append", default=[],
                    help="modelo=Rótulo:classe-css  (repetível); classe ex.: tag-base/tag-fullft")
    ap.add_argument("--max-chars", type=int, default=360)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bench = {json.loads(l)["id"]: json.loads(l)
             for l in Path(args.benchmark).read_text(encoding="utf-8").splitlines() if l.strip()}
    M: dict[str, dict] = {}
    for caminho in args.avaliacao:
        av = json.loads(Path(caminho).read_text(encoding="utf-8"))
        for m in av["modelos"]:
            M.setdefault(m["rotulo"], {x["id"]: x for x in m["detalhe"]})

    tags = {}
    for spec in args.tag:
        mod, _, rest = spec.partition("=")
        rot, _, cls = rest.partition(":")
        tags[mod] = (rot or mod, cls or "tag-base")

    blocos = []
    for spec in args.item:
        bid, _, mods = spec.partition(":")
        bi = bench[bid]
        ref = html.escape(colapsa_degeneracao(bi["reference"], args.max_chars))
        linhas = [f'<div class="inf-block">',
                  f'  <div class="inf-prompt"><span class="lbl">Pergunta held-out · {bid} · {bi["source"]}</span>'
                  f'{html.escape(bi["question"])}</div>',
                  f'  <div class="inf-resp"><span class="model-tag tag-fullft">referência · professor 14B + RAG</span>'
                  f'<div class="inf-txt">{ref}</div></div>']
        for mod in mods.split(","):
            mod = mod.strip()
            rec = M[mod].get(bid)
            rot, cls = tags.get(mod, (mod, "tag-base"))
            ans = html.escape(colapsa_degeneracao(rec["answer"], args.max_chars)) if rec else "(sem registro)"
            b = badge(rec["rougeL"], rec["key_recall"]) if rec else ""
            linhas.append(f'  <div class="inf-resp"><span class="model-tag {cls}">{html.escape(rot)}</span> {b}'
                          f'<div class="inf-txt">{ans}</div></div>')
        linhas.append("</div>")
        blocos.append("\n".join(linhas))

    Path(args.out).write_text("\n\n".join(blocos) + "\n", encoding="utf-8")
    print(f"{len(blocos)} painéis → {args.out}")


if __name__ == "__main__":
    main()
