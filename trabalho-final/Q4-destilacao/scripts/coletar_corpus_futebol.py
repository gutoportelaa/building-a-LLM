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


def _classificacao(matches: list[dict]) -> list[dict]:
    """Calcula a classificação parcial de cada grupo a partir dos jogos já disputados."""
    from collections import defaultdict
    tab = defaultdict(lambda: {"grupo": "", "pts": 0, "gp": 0, "gc": 0, "j": 0})
    for m in matches:
        sc = m.get("score") or {}
        if not sc.get("ft"):
            continue
        a, b = sc["ft"]
        t1, t2, g = m["team1"], m["team2"], m.get("group", "")
        for t, gf, ga in ((t1, a, b), (t2, b, a)):
            r = tab[t]; r["grupo"] = g; r["j"] += 1; r["gp"] += gf; r["gc"] += ga
            r["pts"] += 3 if gf > ga else (1 if gf == ga else 0)
    out = []
    grupos = defaultdict(list)
    for t, r in tab.items():
        grupos[r["grupo"]].append((t, r))
    for g, times in sorted(grupos.items()):
        ordenado = sorted(times, key=lambda x: (-x[1]["pts"], -(x[1]["gp"] - x[1]["gc"]), -x[1]["gp"]))
        linhas = "; ".join(
            f"{i+1}º {t} ({r['pts']}pts, {r['j']}j, saldo {r['gp']-r['gc']:+d})"
            for i, (t, r) in enumerate(ordenado))
        out.append({"source": "openfootball:standings",
                    "texto": f"Classificação parcial do {g} na Copa 2026: {linhas}."})
    return out


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

    # jogos (jogados e agendados) + classificação derivada
    matches = get_json(f"{OF}/worldcup.json").get("matches", [])
    for m in matches:
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
    out.extend(_classificacao(matches))

    # seleções (metadados) — JSON é uma lista no topo
    for t in get_json(f"{OF}/worldcup.teams.json"):
        if t.get("name"):
            out.append({"source": "openfootball:teams",
                        "texto": f"Seleção na Copa 2026: {t['name']} ({t.get('fifa_code','')}), "
                                 f"Grupo {t.get('group','?')}, confederação {t.get('confed','?')} "
                                 f"({t.get('continent','?')})."})

    # elencos — um passagem por seleção
    for sq in get_json(f"{OF}/worldcup.squads.json"):
        jog = "; ".join(
            f"{p.get('pos','?')} {p.get('name','?')} ({(p.get('club') or {}).get('name','?')})"
            for p in sq.get("players", []))
        out.append({"source": "openfootball:squads",
                    "texto": f"Elenco da seleção {sq.get('name','?')} (Grupo {sq.get('group','?')}) "
                             f"na Copa 2026: {jog}."})
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
