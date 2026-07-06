#!/usr/bin/env python3
"""
painel_respostas.py — Gera um HTML autocontido com o BENCHMARK e as RESPOSTAS
de cada modelo (antes×depois) a cada questão, para apresentar no relatório.

Para cada tamanho (1.5B, 0.5B) e cada uma das 30 questões mostra:
  - a pergunta + a resposta de referência;
  - a geração VERBATIM de base / SFT full / LoRA / QLoRA;
  - a nota do juiz (1-5) e o token-F1 de cada geração.

Lê resultados/bench_<metodo>_<sz>.json e resultados/juiz_<metodo>_<sz>.json.
Reusa o CSS compartilhado dos relatórios.

Uso:
  python painel_respostas.py --css <arquivo_style> --out ../painel_respostas_q2q3.html
"""
from __future__ import annotations
import argparse, html, json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "resultados"
SIZES = ["1.5b", "0.5b"]
METHODS = ["baseline", "full", "lora", "qlora"]
ROT = {"baseline": "Base (antes)", "full": "SFT full (Q2)", "lora": "LoRA (Q3)", "qlora": "QLoRA (Q3)"}
COR = {"baseline": "#9aa0a6", "full": "#1f77b4", "lora": "#2ca02c", "qlora": "#ff7f0e"}


def load(p):
    f = RES / p
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def juiz_map(tag):
    d = load(f"juiz_{tag}.json")
    if not d:
        return {}
    return {it.get("instruction"): it.get("juiz") for it in d["per_item"]}


def esc(s):
    return html.escape(str(s or ""), quote=False)


def painel_tamanho(sz):
    bench = {m: load(f"bench_{m}_{sz}.json") for m in METHODS}
    juiz = {m: juiz_map(f"{m}_{sz}") for m in METHODS}
    base = bench["baseline"]
    if not base:
        return f"<p><em>(sem dados para {sz})</em></p>"
    itens = base["per_item"]
    out = [f"<h2>Qwen2.5-{sz} — respostas por questão ({len(itens)} questões)</h2>"]
    for i, it in enumerate(itens):
        instr = it["instruction"]
        out.append('<div class="qcard">')
        out.append(f'<p class="qtipo">#{i+1} · <strong>{esc(it.get("tipo"))}</strong></p>')
        out.append(f"<p><strong>Pergunta:</strong> {esc(instr)}</p>")
        if it.get("input"):
            out.append(f'<p><strong>Input:</strong> <code>{esc(it["input"])}</code></p>')
        out.append(f'<p class="qref"><strong>Referência:</strong> {esc(it.get("resposta_ref"))}</p>')
        out.append('<table class="resp"><thead><tr><th>Modelo</th><th>Resposta gerada</th>'
                   '<th>Juiz</th><th>F1</th></tr></thead><tbody>')
        for m in METHODS:
            d = bench[m]
            cell = next((x for x in d["per_item"] if x["instruction"] == instr), None) if d else None
            gen = esc(cell["geracao"]) if cell else "—"
            f1 = f'{cell["token_f1"]:.2f}' if cell else "—"
            jv = juiz[m].get(instr, "—")
            out.append(
                f'<tr><td style="border-left:5px solid {COR[m]}"><strong>{ROT[m]}</strong></td>'
                f'<td class="gen">{gen}</td>'
                f'<td class="num">{jv}</td><td class="num">{f1}</td></tr>')
        out.append("</tbody></table></div>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--css", required=True)
    ap.add_argument("--out", default=str(BASE / "painel_respostas_q2q3.html"))
    a = ap.parse_args()
    css = Path(a.css).read_text(encoding="utf-8")
    if "<style>" not in css:
        css = f"<style>{css}</style>"
    extra = """<style>
    .qcard{border:1px solid var(--border);border-radius:8px;padding:14px 16px;margin:16px 0;background:#fff}
    .qtipo{color:var(--muted);font-size:.85em;margin:0 0 4px}
    .qref{background:var(--bg-alt);padding:8px 10px;border-radius:6px}
    table.resp{width:100%;border-collapse:collapse;margin-top:8px}
    table.resp td,table.resp th{border:1px solid var(--border);padding:7px 9px;vertical-align:top;font-size:.92em}
    table.resp td.gen{white-space:pre-wrap;max-width:680px}
    table.resp td.num{text-align:center;font-variant-numeric:tabular-nums;width:48px}
    </style>"""
    body = ["<h1>Benchmark CC/UFPI e respostas dos modelos — Q2/Q3</h1>",
            "<p>Benchmark de 30 questões (10 conceitual · 10 código · 10 contextual UFPI) e a "
            "geração <strong>verbatim</strong> de cada modelo, antes e depois do pós-treino, "
            "com a nota do juiz (1-5) e o token-F1. Dados brutos em "
            "<code>resultados/bench_*.json</code> e <code>resultados/juiz_*.json</code>.</p>"]
    for sz in SIZES:
        body.append(painel_tamanho(sz))
    doc = f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Benchmark e respostas Q2/Q3 | DOM-PI / UFPI-DC</title>
{css}{extra}</head><body><main class="container">
{"".join(body)}
</main></body></html>"""
    Path(a.out).write_text(doc, encoding="utf-8")
    print(f"Painel salvo em {a.out} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
