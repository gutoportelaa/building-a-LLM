#!/usr/bin/env python3
"""
gerar_relatorio_q2.py — Gera relatorio_questao2.html comparando LoRA e QLoRA.

Lê os JSONs de avaliação e inferência produzidos pelos jobs de treinamento
e gera um relatório HTML completo com tabelas de métricas e exemplos de geração.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load(path: Path) -> dict | list | None:
    if not path.exists():
        print(f"[aviso] {path} não encontrado — será omitido do relatório", file=sys.stderr)
        return None
    return json.loads(path.read_text())


def delta_html(before: float, after: float, invert: bool = False) -> str:
    diff = after - before
    if invert:
        good = diff < 0
    else:
        good = diff > 0
    color = "#1a7a1a" if good else "#c0392b"
    arrow = "↓" if diff < 0 else "↑"
    return f'<span style="color:{color};font-weight:600">{arrow} {diff:+.4f}</span>'


def fmt(v: float | None, decimals: int = 4) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def pct_params(r: int, d_model: int, n_layers: int, n_modules: int = 7) -> float:
    lora_params = 2 * r * d_model * n_layers * n_modules
    return lora_params / 1e6


# ---------------------------------------------------------------------------
# Carrega dados
# ---------------------------------------------------------------------------
base_05 = load(ROOT / "avaliacao/resultados_baseline.json")
base_15 = load(ROOT / "avaliacao/resultados_baseline_1.5b.json")
lora_05 = load(ROOT / "avaliacao/resultados_q2_lora_0.5b.json")
lora_15 = load(ROOT / "avaliacao/resultados_q2_lora_1.5b.json")
qlora_15 = load(ROOT / "avaliacao/resultados_q2_qlora_1.5b.json")
fullft_05 = load(ROOT / "avaliacao/resultados_postreino.json")  # Job 384

inf_lora05 = load(ROOT / "avaliacao/inferencias_q2_lora_0.5b.json")
inf_lora15 = load(ROOT / "avaliacao/inferencias_q2_lora_1.5b.json")
inf_qlora15 = load(ROOT / "avaliacao/inferencias_q2_qlora_1.5b.json")

PROMPT_IDS = [
    ("completar_portaria", "Completar Portaria"),
    ("resposta_licitacao", "Resposta Licitação"),
    ("conceito_juridico", "Conceito Jurídico"),
    ("completar_contrato", "Completar Contrato"),
    ("pergunta_diario", "Pergunta sobre DOM-PI"),
]


def get_gen(infs: list | None, model_label: str, prompt_id: str) -> str:
    if not infs:
        return "<em style='color:#888'>resultado pendente</em>"
    for r in infs:
        if r.get("modelo") == model_label and r.get("prompt_id") == prompt_id:
            text = r.get("geracao", "").strip()
            return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    return "<em style='color:#888'>não encontrado</em>"


def metrics_row(label: str, res: dict | None, base: dict | None, color: str) -> str:
    if res is None:
        return f"""<tr><td class="model-label" style="border-left:4px solid {color}">{label}</td>
        <td colspan="8" style="color:#888;font-style:italic">resultado pendente</td></tr>"""
    ce = res.get("cross_entropy")
    ppl = res.get("perplexity")
    ta = res.get("token_accuracy")
    bm_ce = res.get("bm_cross_entropy")
    bm_ppl = res.get("bm_perplexity")
    bm_ta = res.get("bm_token_accuracy")

    def d(b_val, a_val, inv=False):
        if b_val is None or a_val is None:
            return "—"
        return delta_html(b_val, a_val, invert=inv)

    b_ce = base.get("cross_entropy") if base else None
    b_ppl = base.get("perplexity") if base else None
    b_ta = base.get("token_accuracy") if base else None
    b_bce = base.get("bm_cross_entropy") if base else None

    return f"""<tr>
      <td class="model-label" style="border-left:4px solid {color}">{label}</td>
      <td>{fmt(ce)}</td><td>{d(b_ce, ce, inv=True)}</td>
      <td>{fmt(ppl, 2)}</td>
      <td>{fmt(ta)}</td>
      <td>{fmt(bm_ce)}</td><td>{d(b_bce, bm_ce, inv=True)}</td>
      <td>{fmt(bm_ppl, 2)}</td>
      <td>{fmt(bm_ta)}</td>
    </tr>"""


def inference_section(title: str, color: str, infs: list | None,
                      model_a_label: str, model_a_title: str,
                      model_b_label: str, model_b_title: str,
                      model_c_label: str | None = None,
                      model_c_title: str | None = None) -> str:
    cols = 2 + (1 if model_c_label else 0)
    grid = f"repeat({cols}, 1fr)"
    blocks = []
    for pid, ptitle in PROMPT_IDS:
        gen_a = get_gen(infs, model_a_label, pid)
        gen_b = get_gen(infs, model_b_label, pid)
        gen_c = get_gen(infs, model_c_label, pid) if model_c_label else None

        pane_c = f"""<div class="inference-pane pane-c">
          <div class="pane-header">{model_c_title}</div>
          <div class="inference-body">{gen_c}</div>
        </div>""" if gen_c is not None else ""

        blocks.append(f"""
        <div class="inference-block">
          <div class="prompt-label">{ptitle}</div>
          <div class="inference-grid" style="grid-template-columns:{grid}">
            <div class="inference-pane pane-a">
              <div class="pane-header">{model_a_title}</div>
              <div class="inference-body">{gen_a}</div>
            </div>
            <div class="inference-pane pane-b">
              <div class="pane-header">{model_b_title}</div>
              <div class="inference-body">{gen_b}</div>
            </div>
            {pane_c}
          </div>
        </div>""")

    return f"""
    <section class="section">
      <h2>{title}</h2>
      {''.join(blocks)}
    </section>"""


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
b05_ce = fmt(base_05.get("cross_entropy")) if base_05 else "—"
b05_ppl = fmt(base_05.get("perplexity"), 2) if base_05 else "—"
b15_ce = fmt(base_15.get("cross_entropy")) if base_15 else "—"
b15_ppl = fmt(base_15.get("perplexity"), 2) if base_15 else "—"

lora05_trainable = pct_params(r=16, d_model=896, n_layers=24)
lora15_trainable = pct_params(r=16, d_model=1536, n_layers=28)

html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Q2 — LoRA e QLoRA DOM-PI | Relatório</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f7fa; color: #1a1a2e; line-height: 1.6; }}
  .hero {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: #fff; padding: 3rem 2rem; text-align: center; }}
  .hero h1 {{ font-size: 2.2rem; font-weight: 700; margin-bottom: .5rem; }}
  .hero .subtitle {{ opacity: .75; font-size: 1.05rem; }}
  .hero .badges {{ margin-top: 1rem; display: flex; gap: .5rem; justify-content: center; flex-wrap: wrap; }}
  .badge {{ background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.25); border-radius: 20px; padding: .25rem .9rem; font-size: .85rem; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 0 1.5rem; }}
  .section {{ background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.06); margin: 2rem 0; padding: 2rem; }}
  .section h2 {{ font-size: 1.4rem; font-weight: 700; margin-bottom: 1.5rem; padding-bottom: .5rem; border-bottom: 2px solid #f0f0f0; color: #1a1a2e; }}
  .section h3 {{ font-size: 1.1rem; font-weight: 600; margin: 1.5rem 0 .75rem; color: #2c3e50; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
  th {{ background: #1a1a2e; color: #fff; padding: .6rem .8rem; text-align: left; font-weight: 600; }}
  td {{ padding: .55rem .8rem; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }}
  tr:hover td {{ background: #fafafa; }}
  .model-label {{ font-weight: 600; white-space: nowrap; }}
  .baseline-row {{ background: #f8f9ff; }}
  .callout {{ border-left: 4px solid #3498db; background: #f0f7ff; border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin: 1rem 0; font-size: .95rem; }}
  .callout.success {{ border-color: #27ae60; background: #f0fff4; }}
  .callout.warning {{ border-color: #e67e22; background: #fff8f0; }}
  .callout.info {{ border-color: #9b59b6; background: #faf0ff; }}
  .method-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.25rem; margin: 1rem 0; }}
  .method-card {{ background: #f8f9ff; border: 1px solid #e0e4ef; border-radius: 10px; padding: 1.25rem; }}
  .method-card h4 {{ font-size: 1rem; font-weight: 700; margin-bottom: .5rem; }}
  .method-card p {{ font-size: .88rem; color: #555; }}
  .method-card .tag {{ display: inline-block; background: #e8f0fe; color: #1a5ccc; border-radius: 4px; font-size: .78rem; padding: .15rem .5rem; margin-top: .5rem; font-weight: 600; }}
  .inference-block {{ margin-bottom: 2rem; border: 1px solid #e8e8e8; border-radius: 10px; overflow: hidden; }}
  .prompt-label {{ background: #1a1a2e; color: #fff; padding: .6rem 1rem; font-weight: 600; font-size: .9rem; }}
  .inference-grid {{ display: grid; gap: 0; }}
  .inference-pane {{ padding: 1rem; border-right: 1px solid #e8e8e8; }}
  .inference-pane:last-child {{ border-right: none; }}
  .pane-header {{ font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; margin-bottom: .6rem; padding: .2rem .5rem; border-radius: 4px; display: inline-block; }}
  .inference-body {{ font-size: .88rem; color: #333; line-height: 1.65; white-space: pre-wrap; word-break: break-word; }}
  .pane-a .pane-header {{ background: #e8f0fe; color: #1a5ccc; }}
  .pane-b .pane-header {{ background: #e8fff0; color: #1a7a3c; }}
  .pane-c .pane-header {{ background: #fff0e8; color: #a04010; }}
  @media (max-width: 768px) {{ .method-grid {{ grid-template-columns: 1fr; }} }}
  footer {{ text-align: center; padding: 2rem; color: #888; font-size: .85rem; }}
</style>
</head>
<body>

<div class="hero">
  <div class="container">
    <h1>Questão 2 — LoRA e QLoRA no DOM-PI</h1>
    <p class="subtitle">Adaptação de domínio eficiente via Low-Rank Adaptation</p>
    <div class="badges">
      <span class="badge">Qwen2.5-0.5B + LoRA</span>
      <span class="badge">Qwen2.5-1.5B + LoRA</span>
      <span class="badge">Qwen2.5-1.5B + QLoRA (4-bit NF4)</span>
      <span class="badge">Corpus DOM-PI 2024–2025</span>
    </div>
  </div>
</div>

<div class="container">

<!-- Metodologia -->
<section class="section">
  <h2>Metodologia</h2>

  <div class="method-grid">
    <div class="method-card">
      <h4>Full-FT (Q1 — referência)</h4>
      <p>Todos os ~494M parâmetros do modelo são atualizados. LR baixo (2e-6) necessário para evitar esquecimento catastrófico. Job 384 serviu de linha de base para esta comparação.</p>
      <span class="tag">~494M params treináveis (0.5B)</span>
    </div>
    <div class="method-card">
      <h4>LoRA (Hu et al. 2022)</h4>
      <p>Injeta matrizes de baixo rank (r=16, α=32) em todas as projeções lineares (q, k, v, o, gate, up, down). Modelo base congelado. LR 100× maior (3e-4 / 2e-4) que full-FT.</p>
      <span class="tag">~{lora05_trainable:.1f}M params (0.5B) | ~{lora15_trainable:.1f}M (1.5B)</span>
    </div>
    <div class="method-card">
      <h4>QLoRA (Dettmers et al. 2023)</h4>
      <p>Base quantizada em 4-bit NF4 com dupla quantização — reduz memória ~4×. Adapters LoRA treinados em bf16. PagedAdamW32bit para gerenciamento de estados do otimizador.</p>
      <span class="tag">~600MB base quantizada (1.5B)</span>
    </div>
  </div>

  <div class="callout info">
    <strong>Bug crítico corrigido:</strong> A versão original de <code>pretreino_lora.py</code> chamava
    <code>model.merge_and_unload()</code> durante o treino para salvar checkpoints "merged", o que
    destruía os adapters LoRA na memória. A partir do step seguinte, o treino continuava como
    <em>full-FT</em> com lr=3e-4 (~150× acima do seguro), causando esquecimento catastrófico
    (CE +4.09, PPL 831 no Job 395). O fix carrega base + adapter do disco em processo isolado;
    o modelo em memória nunca é tocado durante o treino.
  </div>
</section>

<!-- Tabela de Métricas -->
<section class="section">
  <h2>Comparativo de Métricas</h2>

  <table>
    <thead>
      <tr>
        <th>Modelo</th>
        <th>CE (held-out)</th><th>Δ CE</th>
        <th>PPL</th>
        <th>TokenAcc</th>
        <th>CE (benchmark)</th><th>Δ BM_CE</th>
        <th>BM_PPL</th>
        <th>BM_TokenAcc</th>
      </tr>
    </thead>
    <tbody>
      <!-- 0.5B -->
      <tr><td colspan="9" style="background:#f5f5f5;font-weight:700;font-size:.85rem;color:#555;padding:.4rem .8rem">QWEN2.5-0.5B</td></tr>
      <tr class="baseline-row">
        <td class="model-label" style="border-left:4px solid #95a5a6">Baseline 0.5B</td>
        <td>{b05_ce}</td><td>—</td>
        <td>{b05_ppl}</td>
        <td>{fmt(base_05.get('token_accuracy')) if base_05 else '—'}</td>
        <td>{fmt(base_05.get('bm_cross_entropy')) if base_05 else '—'}</td><td>—</td>
        <td>{fmt(base_05.get('bm_perplexity'), 2) if base_05 else '—'}</td>
        <td>{fmt(base_05.get('bm_token_accuracy')) if base_05 else '—'}</td>
      </tr>
      {metrics_row("Full-FT 0.5B (Job 384, lr=2e-6)", fullft_05, base_05, "#e67e22")}
      {metrics_row("LoRA 0.5B (r=16, lr=3e-4)", lora_05, base_05, "#27ae60")}
      <!-- 1.5B -->
      <tr><td colspan="9" style="background:#f5f5f5;font-weight:700;font-size:.85rem;color:#555;padding:.4rem .8rem">QWEN2.5-1.5B</td></tr>
      <tr class="baseline-row">
        <td class="model-label" style="border-left:4px solid #95a5a6">Baseline 1.5B</td>
        <td>{b15_ce}</td><td>—</td>
        <td>{b15_ppl}</td>
        <td>{fmt(base_15.get('token_accuracy')) if base_15 else '—'}</td>
        <td>{fmt(base_15.get('bm_cross_entropy')) if base_15 else '—'}</td><td>—</td>
        <td>{fmt(base_15.get('bm_perplexity'), 2) if base_15 else '—'}</td>
        <td>{fmt(base_15.get('bm_token_accuracy')) if base_15 else '—'}</td>
      </tr>
      {metrics_row("LoRA 1.5B (r=16, lr=2e-4)", lora_15, base_15, "#2980b9")}
      {metrics_row("QLoRA 1.5B (4-bit NF4, lr=2e-4)", qlora_15, base_15, "#8e44ad")}
    </tbody>
  </table>

  <div class="callout" style="margin-top:1.5rem">
    <strong>Leitura da tabela:</strong> Δ CE negativo (↓) indica melhora — o modelo ficou mais
    "surpreso" pelo corpus DOM-PI <em>antes</em> do treino e menos depois, evidenciando adaptação
    de domínio. PPL e BM_CE seguem a mesma lógica. TokenAcc positivo (↑) indica melhora.
  </div>
</section>

<!-- Inferências LoRA 0.5B -->
{inference_section(
    "Inferências — LoRA 0.5B vs Baseline",
    "#27ae60",
    inf_lora05,
    "baseline", "Baseline 0.5B",
    "lora_0.5b_final", "LoRA 0.5B"
)}

<!-- Inferências LoRA 1.5B -->
{inference_section(
    "Inferências — LoRA 1.5B vs Baseline 1.5B",
    "#2980b9",
    inf_lora15,
    "baseline_1.5b", "Baseline 1.5B",
    "lora_1.5b_final", "LoRA 1.5B"
)}

<!-- Inferências QLoRA 1.5B -->
{inference_section(
    "Inferências — QLoRA 1.5B (4-bit NF4) vs Baseline 1.5B",
    "#8e44ad",
    inf_qlora15,
    "baseline_1.5b", "Baseline 1.5B",
    "qlora_1.5b_final", "QLoRA 1.5B"
)}

<!-- Análise -->
<section class="section">
  <h2>Análise e Conclusões</h2>

  <h3>LoRA vs Full-FT (no 0.5B)</h3>
  <div class="callout">
    LoRA usa ~{lora05_trainable:.0f}M parâmetros treináveis vs ~494M no full-FT.
    Se a melhora de CE for comparável, LoRA é mais eficiente e seguro — não há risco de
    esquecimento catastrófico ao escalar o LR. O full-FT do Job 384 resultou em CE +1.107
    (corpus geral ruidoso); LoRA com parâmetros isolados deve ser mais robusto ao ruído OCR.
  </div>

  <h3>LoRA 0.5B vs LoRA 1.5B</h3>
  <div class="callout success">
    Escalar de 0.5B → 1.5B mantendo r=16 aumenta os parâmetros do adapter de
    ~{lora05_trainable:.0f}M → ~{lora15_trainable:.0f}M, mas a capacidade do modelo base
    cresce muito mais. Espera-se CE menor no 1.5B mesmo com o mesmo rank.
  </div>

  <h3>LoRA vs QLoRA (no 1.5B)</h3>
  <div class="callout warning">
    QLoRA quantiza o modelo base em 4-bit NF4, reduzindo de ~3GB → ~600MB.
    A perda por quantização tipicamente resulta em CE ligeiramente maior que LoRA puro.
    O trade-off: QLoRA permite treinar o 1.5B em GPUs menores (L4 24GB) com batch_size maior.
    Se Δ CE(QLoRA − LoRA) &lt; 0.05, o ganho de memória compensa.
  </div>
</section>

</div>

<footer>
  DOM-PI Pipeline · Jobs 396 (LoRA 1.5B) · 397 (QLoRA 1.5B) · LoRA 0.5B rerun<br>
  Gerado por <code>avaliacao/gerar_relatorio_q2.py</code>
</footer>

</body>
</html>
"""

out = ROOT / "relatorio_questao2.html"
out.write_text(html, encoding="utf-8")
print(f"Relatório salvo em {out}")
