#!/usr/bin/env python3
"""
run_eval.py — Avaliação do sistema RAG sobre o benchmark DOM-PI marcado.

Como o fonte_id do benchmark não casa com os ids do corpus (ligação quebrada),
a avaliação é feita por CONTEÚDO:
  • Recall de contexto  — alguma passagem recuperada contém a entidade-chave da
    resposta de referência? (mede a qualidade da RECUPERAÇÃO)
  • Acerto da resposta   — a resposta gerada contém a entidade-chave esperada,
    ou recusa corretamente quando não há fonte? (mede a GERAÇÃO)

Tarefas:
  retrieval  — compara recuperação standard × hyde (recall de contexto). Rápido.
  generation — compara geradores (G1 base, G2 DAPT, G3 Ollama) com/sem RAG.
  traces     — salva traços qualitativos (hyde/reflexivo/agentico) de perguntas.

Uso:
  .venv/bin/python3 rag/run_eval.py --task retrieval --out rag/resultados/retrieval.json
  .venv/bin/python3 rag/run_eval.py --task generation --gen hf:modelos/Qwen2.5-1.5B:base \
      --modes no_rag standard --out rag/resultados/gen_base.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import rag_core as rc


# ───────────────────── métrica por conteúdo ─────────────────────
def keyterms(ans: str) -> list[str]:
    """Entidades distintivas da resposta de referência (nomes, CNPJ, CPF, valores)."""
    t = []
    t += re.findall(r'\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ]{2,}(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ&]{2,}){1,}', ans)
    t += re.findall(r'\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}', ans)   # CNPJ
    t += re.findall(r'\d{3}\.\d{3}\.\d{3}-\d{2}', ans)         # CPF
    t += re.findall(r'R\$\s?[\d\.]+,\d{2}', ans)               # valores
    return list({x.strip() for x in t if len(x.strip()) > 6})


def norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip().lower()


def contains_any(text: str, terms: list[str]) -> bool:
    nt = norm(text)
    return any(norm(term) in nt for term in terms)


REFUSAL = re.compile(r'não encontrad|nao encontrad|não consta|não há informaç', re.I)


def load_bench(path="benchmark/dompi_qa_tagged.jsonl"):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


# ───────────────────────── tarefas ─────────────────────────
def task_retrieval(args):
    """Recall de contexto: standard × hyde, sobre perguntas rag/ambos."""
    bench = [q for q in load_bench() if q["modo"] in ("rag", "ambos")]
    retr = rc.Retriever(args.index)
    emb = rc.E5Embedder()
    # hyde precisa de um gerador para o documento hipotético (usa Ollama por padrão)
    gen = rc.OllamaGenerator(args.hyde_gen) if args.hyde_gen else None

    results = []
    agg = {"standard": [0, 0], "hyde": [0, 0]}  # [hits, avaliaveis]
    for q in bench:
        kt = keyterms(q["resposta_ref"])
        row = {"pergunta": q["pergunta"], "tipo": q["tipo"], "modo": q["modo"],
               "resposta_ref": q["resposta_ref"], "keyterms": kt}
        # standard
        passages = retr.search_vec(emb.encode_query(q["pergunta"]), k=args.k)
        hay = " ".join(p["texto"] for p in passages)
        row["standard"] = {"hit": contains_any(hay, kt) if kt else None,
                           "top_doc": passages[0]["doc_id"],
                           "top_score": round(passages[0]["score"], 3)}
        if kt:
            agg["standard"][1] += 1
            agg["standard"][0] += int(row["standard"]["hit"])
        # hyde
        if gen is not None:
            hypo = gen.generate(
                [{"role": "system", "content":
                  "Escreva um trecho curto de documento oficial (Diário Oficial) que "
                  "responderia à pergunta. Invente detalhes plausíveis. Máx. 4 frases."},
                 {"role": "user", "content": f"PERGUNTA: {q['pergunta']}"}],
                max_new_tokens=160)
            hp = retr.search_vec(emb.encode_passage(hypo), k=args.k)
            hayh = " ".join(p["texto"] for p in hp)
            row["hyde"] = {"hit": contains_any(hayh, kt) if kt else None,
                           "top_doc": hp[0]["doc_id"], "top_score": round(hp[0]["score"], 3),
                           "hyde_doc": hypo[:300]}
            if kt:
                agg["hyde"][1] += 1
                agg["hyde"][0] += int(row["hyde"]["hit"])
        results.append(row)
        print(f"  [{q['modo']:5}] std={row['standard']['hit']} "
              f"hyde={row.get('hyde',{}).get('hit')} | {q['pergunta'][:55]}", flush=True)

    summary = {m: {"recall": round(h / n, 3) if n else None, "hits": h, "avaliaveis": n}
               for m, (h, n) in agg.items()}
    out = {"task": "retrieval", "k": args.k, "summary": summary, "results": results}
    save(args.out, out)
    print("\nRECALL DE CONTEXTO:", json.dumps(summary, ensure_ascii=False))


def build_gen(spec: str):
    """spec: 'ollama:<modelo>[|nome]' ou 'hf:<caminho>[|nome]'.

    Usa '|' como separador do nome para não colidir com ':' dos modelos Ollama
    (ex.: 'ollama:qwen2.5:14b|qualificado')."""
    kind, rest = spec.split(":", 1)
    if "|" in rest:
        target, name = rest.split("|", 1)
    else:
        target, name = rest, None
    if kind == "ollama":
        return rc.OllamaGenerator(target, name=name or target)
    return rc.HFGenerator(target, name=name or Path(target).name)


def task_generation(args):
    """Compara um gerador em vários modos no benchmark inteiro."""
    bench = load_bench()
    if args.limit:
        bench = bench[:args.limit]
    gen = build_gen(args.gen)
    retr = emb = None
    if any(m != "no_rag" for m in args.modes):
        retr = rc.Retriever(args.index)
        emb = rc.E5Embedder()

    results = []
    # acurácia agregada por (modo, classe de pergunta)
    agg = {}
    for q in bench:
        kt = keyterms(q["resposta_ref"])
        row = {"pergunta": q["pergunta"], "tipo": q["tipo"], "modo_q": q["modo"],
               "resposta_ref": q["resposta_ref"], "keyterms": kt, "saidas": {}}
        for mode in args.modes:
            t0 = time.time()
            res = rc.run_mode(mode, q["pergunta"], gen, retr, emb,
                              k=args.k, max_new_tokens=args.max_new_tokens)
            dt = time.time() - t0
            ans = res["answer"]
            # acerto: entidade-chave presente; ou recusa correta quando sem entidade no corpus
            if kt:
                acerto = contains_any(ans, kt)
            else:
                acerto = None  # conceitual/sem entidade — avaliar qualitativamente
            row["saidas"][mode] = {"answer": ans, "acerto": acerto,
                                   "recusou": bool(REFUSAL.search(ans)),
                                   "n_recuperados": len(res["retrieved"]),
                                   "segundos": round(dt, 1)}
            key = (mode, q["modo"])
            agg.setdefault(key, [0, 0])
            if acerto is not None:
                agg[key][1] += 1
                agg[key][0] += int(acerto)
            print(f"  {gen.name} | {mode:9} | {q['modo']:5} | acerto={acerto} "
                  f"| {dt:4.0f}s | {q['pergunta'][:45]}", flush=True)
        results.append(row)

    summary = {}
    for (mode, modo_q), (h, n) in agg.items():
        summary.setdefault(mode, {})[modo_q] = {
            "acuracia": round(h / n, 3) if n else None, "acertos": h, "avaliaveis": n}
    out = {"task": "generation", "gerador": gen.name, "modos": args.modes,
           "summary": summary, "results": results}
    save(args.out, out)
    print(f"\n{gen.name} — RESUMO:", json.dumps(summary, ensure_ascii=False))


def task_traces(args):
    """Traços qualitativos detalhados de modos avançados em perguntas escolhidas."""
    bench = load_bench()
    sel = [bench[i] for i in args.indices] if args.indices else bench[:args.limit]
    gen = build_gen(args.gen)
    retr = rc.Retriever(args.index)
    emb = rc.E5Embedder()
    out = {"task": "traces", "gerador": gen.name, "modos": args.modes, "casos": []}
    for q in sel:
        caso = {"pergunta": q["pergunta"], "modo_q": q["modo"],
                "resposta_ref": q["resposta_ref"], "execucoes": {}}
        for mode in args.modes:
            res = rc.run_mode(mode, q["pergunta"], gen, retr, emb,
                              k=args.k, max_new_tokens=args.max_new_tokens)
            caso["execucoes"][mode] = res
            print(f"  {mode:9} | {q['pergunta'][:55]}", flush=True)
        out["casos"].append(caso)
    save(args.out, out)


def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("salvo:", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["retrieval", "generation", "traces"])
    ap.add_argument("--index", default="rag/index")
    ap.add_argument("--gen", default="ollama:qwen2.5:14b")
    ap.add_argument("--hyde-gen", default="qwen2.5:14b", help="modelo Ollama p/ doc hipotético (retrieval)")
    ap.add_argument("--modes", nargs="+", default=["no_rag", "standard"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--indices", nargs="+", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    {"retrieval": task_retrieval, "generation": task_generation,
     "traces": task_traces}[args.task](args)
