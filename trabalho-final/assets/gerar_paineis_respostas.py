#!/usr/bin/env python3
"""
gerar_paineis_respostas.py — Painéis com TODAS as respostas dos modelos aos benchmarks.

Gera (no diretório de cada questão):
  Q1-pretreino-continuado/painel_respostas_q1.html   — 49 perguntas × 6 modelos
  Q4-destilacao/painel_respostas_q4.html             — v1 (100 × 14 modelos) + v2 (50/tópico × 4) + n100 (100/tópico × 4)
  Q5-rag/painel_respostas_q5.html                    — 49 perguntas × 3 geradores × (sem RAG, com RAG)
  Q6-guardrails/painel_respostas_q6.html             — 30 perguntas × (sem, com guardrails)

Q2/Q3 já têm painel próprio (painel_respostas_q2q3.html). Estilo alinhado ao
relatorio_apresentacao.html (azul-claro=antes/sem · azul-escuro=depois/com).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # trabalho-final/

CSS = """
  :root { --txt:#1c1c1c; --muted:#5a5a5a; --border:#d2d2d2; --bg-alt:#f7f7f5; --bg-code:#f2f2ef;
          --accent:#1e3a5f; --accent2:#2e5ca0; --a-light:#e6ecf5; --warn-bg:#fef9e7; --warn-fg:#5c4500;
          --ok:#1a4a1a; --bad:#5a1818; --mono:"Courier New",Courier,monospace;
          --sans:"Helvetica Neue",Arial,sans-serif; --serif:Georgia,"Times New Roman",serif; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:var(--serif); font-size:15px; line-height:1.7; color:var(--txt);
         max-width:1020px; margin:0 auto; padding:40px 32px 90px; background:#fff; }
  h1 { font-family:var(--sans); font-size:1.5rem; color:var(--accent);
       border-bottom:3px solid var(--accent); padding-bottom:10px; margin-bottom:8px; }
  h2 { font-family:var(--sans); font-size:1.15rem; color:var(--accent); margin:40px 0 12px;
       border-bottom:2px solid var(--border); padding-bottom:5px; }
  .meta { font-family:var(--sans); font-size:.8rem; color:var(--muted); background:var(--a-light);
          border-left:4px solid var(--accent); padding:9px 14px; margin-bottom:24px; }
  .meta a { color:var(--accent2); }
  details { border:1px solid var(--border); border-radius:6px; margin:10px 0; overflow:hidden; }
  summary { cursor:pointer; font-family:var(--sans); font-size:.88rem; padding:9px 14px;
            background:var(--bg-alt); }
  summary:hover { background:var(--a-light); }
  summary .qn { font-weight:800; color:var(--accent); margin-right:8px; }
  summary .cat { font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
                 background:var(--a-light); color:var(--accent2); border-radius:3px;
                 padding:1px 7px; margin-left:8px; }
  .ref { background:var(--warn-bg); color:var(--warn-fg); font-family:var(--sans);
         font-size:.8rem; padding:8px 14px; border-bottom:1px solid var(--border); }
  .ref b { text-transform:uppercase; font-size:.68rem; letter-spacing:.08em; }
  .resp { padding:9px 14px; border-bottom:1px solid var(--border); }
  .resp:last-child { border-bottom:none; }
  .tag { display:inline-block; font-family:var(--sans); font-size:.68rem; font-weight:700;
         text-transform:uppercase; letter-spacing:.05em; padding:2px 8px; border-radius:3px;
         margin-bottom:5px; }
  .t-antes { background:#dbe9fb; color:#1c5cab; } .t-depois { background:#1c5cab; color:#fff; }
  .t-neutro { background:var(--bg-alt); color:var(--muted); border:1px solid var(--border); }
  .txt { font-family:var(--mono); font-size:.76rem; line-height:1.5; background:var(--bg-code);
         border-radius:4px; padding:8px 11px; white-space:pre-wrap; word-break:break-word; }
  .m { font-family:var(--sans); font-size:.72rem; color:var(--muted); margin-left:8px; }
  .ok { color:var(--ok); font-weight:700; } .bad { color:var(--bad); font-weight:700; }
  footer { margin-top:50px; padding-top:14px; border-top:1px solid var(--border);
           font-family:var(--sans); font-size:.76rem; color:var(--muted); }
"""


def esc(t) -> str:
    return html.escape(str(t if t is not None else ""))


def page(titulo: str, meta: str, corpo: str, rel_prefix: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(titulo)}</title><style>{CSS}</style></head><body>
<h1>{esc(titulo)}</h1>
<div class="meta">{meta} · Use <b>Ctrl+F</b> para localizar uma pergunta/entidade ·
Relatório consolidado: <a href="{rel_prefix}relatorio_apresentacao.html">relatorio_apresentacao.html</a></div>
{corpo}
<footer>Painel gerado por <code>assets/gerar_paineis_respostas.py</code> a partir dos JSONs de resultados
(respostas integrais, sem edição). Trabalho Final — Construindo um LLM (DOM-PI / UFPI-CC).</footer>
</body></html>"""


def bloco_resp(tag_css: str, rotulo: str, texto: str, metricas: str = "") -> str:
    return (f'<div class="resp"><span class="tag {tag_css}">{esc(rotulo)}</span>'
            f'<span class="m">{metricas}</span><div class="txt">{esc(texto)}</div></div>')


# ═════════════════════════════ Q1 ═════════════════════════════
def painel_q1() -> None:
    d = BASE / "Q1-pretreino-continuado/resultados"
    modelos = [  # (arquivo, rótulo, classe)
        ("geracao_baseline.json", "Qwen2.5-1.5B baseline (antes)", "t-antes"),
        ("geracao_unificado.json", "Qwen2.5-1.5B DAPT unificado (canônico)", "t-depois"),
        ("geracao_teresina.json", "Qwen2.5-1.5B DAPT Teresina", "t-depois"),
        ("geracao_unif_limpo.json", "Qwen2.5-1.5B DAPT corpus limpo (negativo §1.7a)", "t-neutro"),
        ("geracao_llama32_baseline.json", "Llama-3.2-3B baseline (antes)", "t-antes"),
        ("geracao_llama32_dapt.json", "Llama-3.2-3B DAPT", "t-depois"),
    ]
    dados = []
    for f, rot, css in modelos:
        j = json.load(open(d / f))
        dados.append((rot, css, {it["pergunta"]: it for it in j["per_item"]}))
    perguntas = list(dados[0][2].keys())
    corpo = []
    for i, q in enumerate(perguntas, 1):
        it0 = dados[0][2][q]
        blocos = "".join(
            bloco_resp(css, rot, m[q]["geracao"], f"token-F1 {m[q]['token_f1']:.2f}")
            for rot, css, m in dados if q in m
        )
        corpo.append(
            f'<details><summary><span class="qn">{i:02d}</span>{esc(q)}'
            f'<span class="cat">{esc(it0.get("tipo",""))}</span></summary>'
            f'<div class="ref"><b>Referência</b> · {esc(it0["resposta_ref"])}</div>{blocos}</details>'
        )
    out = BASE / "Q1-pretreino-continuado/painel_respostas_q1.html"
    out.write_text(page(
        "Q1 · Respostas completas — benchmark dompi_qa (49 perguntas × 6 modelos)",
        "Geração gulosa, sem contexto — mede o que está <b>nos pesos</b>. "
        "Fonte: <code>resultados/geracao_*.json</code>", "\n".join(corpo), "../"), encoding="utf-8")
    print(out, f"({len(perguntas)} perguntas)")


# ═════════════════════════════ Q4 ═════════════════════════════
def _sec_q4(titulo: str, bench_path: Path, aval_paths: list[Path], destaque: set[str]) -> str:
    bench = {}
    for l in bench_path.read_text(encoding="utf-8").splitlines():
        if l.strip():
            b = json.loads(l)
            bench[b["id"]] = b
    modelos = []  # (rotulo, {id: item})
    for p in aval_paths:
        d = json.load(open(p))
        for m in d["modelos"]:
            modelos.append((m["rotulo"], {x["id"]: x for x in m["detalhe"]}))
    corpo = [f"<h2>{esc(titulo)}</h2>"]
    for i, (bid, b) in enumerate(sorted(bench.items()), 1):
        blocos = []
        for rot, det in modelos:
            if bid not in det:
                continue
            x = det[bid]
            css = ("t-antes" if rot.startswith("base") else
                   "t-depois" if rot in destaque else "t-neutro")
            kr = f"KR {x['key_recall']:.2f}" if x.get("key_recall") is not None else "KR —"
            blocos.append(bloco_resp(css, rot, x["answer"], f"{kr} · RG {x['rougeL']:.2f}"))
        corpo.append(
            f'<details><summary><span class="qn">{esc(bid)}</span>{esc(b["question"])}'
            f'<span class="cat">{esc(b["source"])}</span></summary>'
            f'<div class="ref"><b>Referência</b> · {esc(b["reference"])}</div>{"".join(blocos)}</details>'
        )
    return "\n".join(corpo)


def painel_q4() -> None:
    q4 = BASE / "Q4-destilacao"
    secoes = [
        _sec_q4("Benchmark v1 (100 perguntas) — 2 bases + 12 alunos do fatorial",
                q4 / "dados/benchmark_destilacao_100.jsonl",
                [q4 / "resultados/avaliacao.json"],
                {"d_1.5b_A_ce", "d_1.5b_B_combined"}),
        _sec_q4("Benchmark v2 · DOM-PI (50, contexto-ouro) — bases + especialistas",
                q4 / "dados_v2/benchmark_dompi.jsonl" if (q4 / "dados_v2/benchmark_dompi.jsonl").exists()
                else q4 / "dados_v2/benchmark_v2_100.jsonl",
                [q4 / "resultados/avaliacao_dompi.json"], {"esp_dompi_0.5b", "esp_dompi_1.5b"}),
        _sec_q4("Benchmark v2 · docentesDC (50, contexto-ouro) — bases + especialistas",
                q4 / "dados_v2/benchmark_docentes.jsonl" if (q4 / "dados_v2/benchmark_docentes.jsonl").exists()
                else q4 / "dados_v2/benchmark_v2_100.jsonl",
                [q4 / "resultados/avaliacao_docentes.json"], {"esp_docentes_0.5b", "esp_docentes_1.5b"}),
        _sec_q4("Benchmark ampliado · DOM-PI (99, §4.10b-a)",
                q4 / "dados_v2/benchmark_dompi_n100.jsonl",
                [q4 / "resultados/avaliacao_dompi_n100.json"], {"esp_dompi_0.5b", "esp_dompi_1.5b"}),
        _sec_q4("Benchmark ampliado · docentesDC (100, §4.10b-a)",
                q4 / "dados_v2/benchmark_docentes_n100.jsonl",
                [q4 / "resultados/avaliacao_docentes_n100.json"], {"esp_docentes_0.5b", "esp_docentes_1.5b"}),
        _sec_q4("Alunos cross-família (Llama, §4.10b-b) — benchmark v1",
                q4 / "dados/benchmark_destilacao_100.jsonl",
                [q4 / "resultados/avaliacao_alunos_xfam_parcial.json"], {"xfam_llama1b_B_ce", "xfam_llama3b_B_ce"}),
    ]
    out = q4 / "painel_respostas_q4.html"
    out.write_text(page(
        "Q4 · Respostas completas — destilação (benchmarks v1, v2 e ampliado)",
        "Geração gulosa, sem RAG na inferência — mede o que a destilação gravou <b>nos pesos</b>. "
        "Rótulos: <code>d_&lt;tam&gt;_&lt;braço&gt;_&lt;sinal&gt;</code>; braço B = professor aterrado por RAG. "
        "Fonte: <code>resultados/avaliacao*.json</code>", "\n".join(secoes), "../"), encoding="utf-8")
    print(out)


# ═════════════════════════════ Q5 ═════════════════════════════
def painel_q5() -> None:
    d = BASE / "Q5-rag/resultados"
    gers = [("gen_G1_base.json", "G1 · Qwen2.5-1.5B base"),
            ("gen_G2_dapt.json", "G2 · Qwen2.5-1.5B DAPT (Q1)"),
            ("gen_G3_14b.json", "G3 · qwen2.5:14b (Ollama)")]
    dados = [(rot, {r["pergunta"]: r for r in json.load(open(d / f))["results"]}) for f, rot in gers]
    perguntas = list(dados[0][1].keys())
    corpo = []
    for i, q in enumerate(perguntas, 1):
        r0 = dados[0][1][q]
        blocos = []
        for rot, m in dados:
            if q not in m:
                continue
            s = m[q]["saidas"]
            for modo, css, nome in [("no_rag", "t-antes", "sem RAG"), ("standard", "t-depois", "com RAG")]:
                if modo not in s:
                    continue
                x = s[modo]
                v = ('<span class="ok">✔ acerto</span>' if x.get("acerto")
                     else '<span class="bad">recusou</span>' if x.get("recusou") else "✘")
                blocos.append(bloco_resp(css, f"{rot} · {nome}", x["answer"],
                                         f"{v} · {x.get('segundos','?')}s"))
        corpo.append(
            f'<details><summary><span class="qn">{i:02d}</span>{esc(q)}'
            f'<span class="cat">{esc(r0.get("tipo",""))}</span></summary>'
            f'<div class="ref"><b>Referência</b> · {esc(r0["resposta_ref"])}</div>{"".join(blocos)}</details>'
        )
    out = BASE / "Q5-rag/painel_respostas_q5.html"
    out.write_text(page(
        "Q5 · Respostas completas — RAG (49 perguntas × 3 geradores, sem/com RAG)",
        "Modo standard (top-5, cosseno exato). Acerto = entidade-chave da referência presente. "
        "Fonte: <code>resultados/gen_G*.json</code>", "\n".join(corpo), "../"), encoding="utf-8")
    print(out, f"({len(perguntas)} perguntas)")


# ═════════════════════════════ Q6 ═════════════════════════════
def painel_q6() -> None:
    d = json.load(open(BASE / "Q6-guardrails/scripts/resultados_guardrails_noground.json"))
    corpo = []
    for r in d["registros"]:
        sg, cg = r["sem_guard"], r["com_guard"]
        blocos = (
            bloco_resp("t-antes", "sem guardrails", sg.get("answer"),
                       f"{sg.get('latencia_s','?')}s") +
            bloco_resp("t-depois", f"com guardrails · ação: {cg.get('action')}", cg.get("answer"),
                       f"{cg.get('latencia_s','?')}s")
        )
        corpo.append(
            f'<details><summary><span class="qn">{r["id"]:02d}</span>{esc(r["pergunta"])}'
            f'<span class="cat">{esc(r["categoria"])}</span></summary>'
            f'<div class="ref"><b>Ação esperada</b> · {esc(r["acao_esperada"])}</div>{blocos}</details>'
        )
    out = BASE / "Q6-guardrails/painel_respostas_q6.html"
    out.write_text(page(
        "Q6 · Respostas completas — guardrails (30 perguntas, sem → com)",
        "Configuração recomendada (sem groundedness). Categorias: 10 legítimas · 6 PII · "
        "6 fora-de-escopo · 5 injection · 3 nocivas. "
        "Fonte: <code>scripts/resultados_guardrails_noground.json</code>", "\n".join(corpo), "../"),
        encoding="utf-8")
    print(out, f"({len(d['registros'])} perguntas)")


if __name__ == "__main__":
    painel_q1()
    painel_q4()
    painel_q5()
    painel_q6()
