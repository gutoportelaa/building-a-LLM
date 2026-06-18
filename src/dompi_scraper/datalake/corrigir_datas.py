#!/usr/bin/env python3
"""
corrigir_datas.py — corrige a cronologia (P-03) no datalake.

A extração derivava a data do NOME do arquivo (frágil → anos impossíveis: 2001,
2054, 2099...). A data real de publicação já existe na coleta: é a data da EDIÇÃO
do diário (`edicao_url_meta → data_publicacao`, 1:1) e, no planície, está no
próprio nome (`DD-MM-AAAA_<edicao>_...`). Toda a coleta é de 2025.

Este script reescreve, na camada EXTRAÍDO (extraido), `data_publicacao` (DD/MM/AAAA
quando há match) e `ano=2025` para TODOS (ano de publicação é a verdade), usando:
  1) datas_map.json  — by_file (basename(url)→data) + by_edicao (edição→data)
  2) loghash.tsv     — (territorio, hash8_do_conteudo, arquivo) extraído dos logs SLURM
Para edições fora da amostra, interpola pela sequência (edições são ~diárias).

Uso: python -m dompi_scraper.datalake.corrigir_datas \\
        --map dados/datas_map.json --log staging_lab/loghash.tsv [--ano-colecao 2025]
Depois: rebuild de limpo + corpus.
"""
from __future__ import annotations
import argparse, bisect, datetime, json, re
import polars as pl
from . import zone_dir
from .io import read_zone, write_partitioned_parquet


def carregar_resolvedor(map_path: str, log_path: str):
    M = json.load(open(map_path, encoding="utf-8"))
    by_file, by_ed = M["by_file"], M["by_edicao"]
    pts = sorted((int(k.split("_")[0]),
                  datetime.datetime.strptime(v, "%d/%m/%Y").toordinal())
                 for k, v in by_ed.items() if k.split("_")[0].isdigit())
    eds = [p[0] for p in pts]; ords = [p[1] for p in pts]

    def interp(e: int) -> str:
        if e <= eds[0]: o = ords[0]
        elif e >= eds[-1]: o = ords[-1]
        else:
            i = bisect.bisect_left(eds, e)
            if eds[i] == e: o = ords[i]
            else:
                e0, e1, o0, o1 = eds[i-1], eds[i], ords[i-1], ords[i]
                o = round(o0 + (o1-o0)*(e-e0)/(e1-e0))
        return datetime.date.fromordinal(o).strftime("%d/%m/%Y")

    def data_de(fn: str | None):
        if not fn: return None
        m = re.match(r"(\d{2})-(\d{2})-(\d{4})_", fn)          # planície: data no prefixo
        if m: return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
        if fn in by_file: return by_file[fn]                   # exato por nome
        me = re.search(r"DM_([0-9]+)", fn)                     # por edição (+interp)
        if me:
            e = me.group(1)
            return by_ed.get(e) or interp(int(e))
        return None

    logmap = {}
    for ln in open(log_path, encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) == 3: logmap[(p[0], p[1])] = p[2]
    return data_de, logmap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--ano-colecao", default="2025")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    data_de, logmap = carregar_resolvedor(args.map, args.log)
    b = read_zone(zone_dir("extraido", args.root))
    if b.is_empty():
        print("Camada extraído vazia."); return

    ano0 = b["ano"].to_list()
    novas_data, novo_ano, n_match = [], [], 0
    for terr, idp in zip(b["territorio"].to_list(), b["id_publicacao"].to_list()):
        d = data_de(logmap.get((terr, idp[:8])))
        if d:
            n_match += 1
            novas_data.append(d); novo_ano.append(d.split("/")[-1])
        else:
            # sem match: a coleta é toda do ano-coleção → ano correto, data só o ano
            novas_data.append(args.ano_colecao); novo_ano.append(args.ano_colecao)

    b = b.with_columns([pl.Series("data_publicacao", novas_data),
                        pl.Series("ano", novo_ano)])

    # idempotente: reescreve toda a zona extraído com o ano corrigido
    write_partitioned_parquet(b, zone_dir("extraido", args.root), ["territorio", "ano"])

    import collections
    antes = collections.Counter("plausivel" if "2023" <= a <= "2026"
                                else ("sem_ano" if a == "sem_ano" else "lixo") for a in ano0)
    depois = collections.Counter(novo_ano)
    print(f"\n  Docs (extraído):      {b.height}")
    print(f"  Com data precisa:     {n_match} ({100*n_match/b.height:.1f}%)")
    print(f"  ANTES (ano):          {dict(antes)}")
    print(f"  DEPOIS (ano):         {dict(depois)}")
    print(f"  data_publicacao agora = DD/MM/AAAA (match) ou '{args.ano_colecao}' (sem match)")
    print("  → rebuild de limpo e corpus na sequência.")


if __name__ == "__main__":
    main()
