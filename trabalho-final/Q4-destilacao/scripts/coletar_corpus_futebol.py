#!/usr/bin/env python3
"""
coletar_corpus_futebol.py — Corpus factual da Copa do Mundo 2026 para a extensão temática (Plano A).

Fontes PÚBLICAS e ungated (sem scraping frágil, sem ToS de Transfermarkt):
  • openfootball/worldcup.json (domínio público) — grupos, jogos+placares+gols, estádios, seleções.
  • Wikipedia REST API — resumo narrativo dos artigos da Copa 2026.

Converte os dados estruturados em PASSAGENS de texto (PT) e salva um corpus JSONL pronto para o
`build_index.py` da Q5. O professor (com RAG sobre este corpus) é quem raciocina — nós só fornecemos os fatos.

Uso:
  python coletar_corpus_futebol.py --out trabalho-final/Q4-destilacao/futebol/corpus_futebol.jsonl
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

OF = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026"
WIKI = "https://pt.wikipedia.org/api/rest_v1/page/summary"
WIKI_PAGES = ["Copa_do_Mundo_FIFA_de_2026", "Eliminatórias_da_Copa_do_Mundo_FIFA_de_2026"]


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "dompi-corpus/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _ph(s: str) -> str:  # placar/half helper
    return f"{s[0]}–{s[1]}" if isinstance(s, list) and len(s) == 2 else "?"


def passagens_openfootball() -> list[dict]:
    out: list[dict] = []

    # grupos
    for g in get_json(f"{OF}/worldcup.groups.json").get("groups", []):
        out.append({"source": "openfootball:groups",
                    "texto": f"{g['name']} da Copa do Mundo FIFA de 2026: {', '.join(g['teams'])}."})

    # estádios
    for s in get_json(f"{OF}/worldcup.stadiums.json").get("stadiums", []):
        out.append({"source": "openfootball:stadiums",
                    "texto": f"Estádio-sede da Copa 2026: {s.get('name')} em {s.get('city')} "
                             f"({str(s.get('cc','')).upper()}), capacidade {s.get('capacity','?')}, "
                             f"fuso {s.get('timezone','?')}."})

    # jogos (jogados e agendados)
    for m in get_json(f"{OF}/worldcup.json").get("matches", []):
        base = (f"Copa 2026 — {m.get('round','')}, {m.get('group','')}, {m.get('date','')}: "
                f"{m['team1']} x {m['team2']} em {m.get('ground','?')}")
        sc = m.get("score")
        if sc and sc.get("ft"):
            g1 = ", ".join(f"{x['name']} ({x['minute']}')" for x in m.get("goals1", [])) or "—"
            g2 = ", ".join(f"{x['name']} ({x['minute']}')" for x in m.get("goals2", [])) or "—"
            txt = (f"{base}. Resultado: {m['team1']} {_ph(sc['ft'])} {m['team2']} "
                   f"(intervalo {_ph(sc.get('ht', []))}). Gols de {m['team1']}: {g1}. "
                   f"Gols de {m['team2']}: {g2}.")
        else:
            txt = f"{base} (agendado para {m.get('time','?')})."
        out.append({"source": "openfootball:matches", "texto": txt})

    # seleções (metadados, se houver)
    try:
        for t in get_json(f"{OF}/worldcup.teams.json").get("teams", []):
            nome = t.get("name"); code = t.get("code", "")
            if nome:
                out.append({"source": "openfootball:teams",
                            "texto": f"Seleção na Copa 2026: {nome} ({code})."})
    except Exception:
        pass
    return out


def passagens_wikipedia() -> list[dict]:
    out = []
    for p in WIKI_PAGES:
        try:
            d = get_json(f"{WIKI}/{p}")
            if d.get("extract"):
                out.append({"source": f"wikipedia:{p}", "texto": d["extract"]})
        except Exception:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Coleta corpus factual da Copa 2026 (Plano A)")
    ap.add_argument("--out", default="trabalho-final/Q4-destilacao/futebol/corpus_futebol.jsonl")
    args = ap.parse_args()

    passagens = passagens_openfootball() + passagens_wikipedia()
    for i, p in enumerate(passagens):
        p["id"] = f"fut{i:04d}"
        p["texto"] = p["texto"].strip()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in passagens) + "\n", encoding="utf-8")
    por_fonte: dict[str, int] = {}
    for p in passagens:
        por_fonte[p["source"]] = por_fonte.get(p["source"], 0) + 1
    print(f"{len(passagens)} passagens salvas em {out}")
    for s, n in sorted(por_fonte.items()):
        print(f"  {s}: {n}")


if __name__ == "__main__":
    main()
