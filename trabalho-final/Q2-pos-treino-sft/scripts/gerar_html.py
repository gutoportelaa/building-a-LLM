#!/usr/bin/env python3
"""
gerar_html.py — Converte um relatorio.md em HTML autocontido, reusando o CSS
compartilhado dos relatórios do trabalho e embutindo figuras como base64.

Suporta o subconjunto de Markdown usado nos relatórios: headings (#..###),
tabelas pipe, listas (- / 1.), **negrito**, `código`, > blockquote, links
[txt](url), imagens ![alt](path), parágrafos, regras ---.

Uso:
  python gerar_html.py --md ../relatorio.md --out ../relatorio_q2.html \
      --title "Questão 2 — Pós-treino SFT" --css <arquivo_com_bloco_style> \
      --fig-dir ../resultados
"""
from __future__ import annotations
import argparse, base64, html, re
from pathlib import Path

def inline(t: str) -> str:
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)   # negrito (permite *itálico* interno)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    return t

def conv_table(rows: list[str]) -> str:
    def cells(r): return [c.strip() for c in r.strip().strip("|").split("|")]
    head = cells(rows[0]); body = rows[2:]
    out = ["<table><thead><tr>"] + [f"<th>{inline(c)}</th>" for c in head] + ["</tr></thead><tbody>"]
    for r in body:
        if not r.strip(): continue
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells(r)) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)

def embed_img(path: str, fig_dir: Path) -> str:
    p = (fig_dir / Path(path).name) if not Path(path).is_absolute() else Path(path)
    if not p.exists(): return f"<p><em>[figura ausente: {html.escape(path)}]</em></p>"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<p><img alt="" style="max-width:100%;border:1px solid var(--border);border-radius:6px" src="data:image/png;base64,{b64}"></p>'

def md_to_html(md: str, fig_dir: Path) -> str:
    lines = md.splitlines()
    out, i = [], 0
    list_open = None  # 'ul'|'ol'|None
    def close_list():
        nonlocal list_open
        if list_open: out.append(f"</{list_open}>"); list_open = None
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            close_list(); i += 1; continue
        # tabela
        if ln.lstrip().startswith("|") and i+1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i+1]):
            close_list(); blk = [ln]; i += 1
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                blk.append(lines[i]); i += 1
            out.append(conv_table(blk)); continue
        # imagem isolada
        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)\s*$", ln.strip())
        if m: close_list(); out.append(embed_img(m.group(1), fig_dir)); i += 1; continue
        # heading
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m: close_list(); n=len(m.group(1)); out.append(f"<h{n}>{inline(m.group(2))}</h{n}>"); i+=1; continue
        # hr
        if re.match(r"^---+\s*$", ln): close_list(); out.append("<hr>"); i+=1; continue
        # blockquote
        if ln.lstrip().startswith(">"):
            close_list(); buf=[]
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?","",lines[i])); i+=1
            # linhas '>' consecutivas = soft-wrap (junta com espaço); linha '>' vazia = quebra.
            paras, cur = [], []
            for s in buf:
                if s.strip(): cur.append(s.strip())
                else:
                    if cur: paras.append(" ".join(cur)); cur=[]
            if cur: paras.append(" ".join(cur))
            out.append("<blockquote>"+"<br><br>".join(inline(p) for p in paras)+"</blockquote>"); continue
        # listas (com captura de continuações soft-wrap do item)
        m = re.match(r"^\s*([-*])\s+(.*)$", ln) or re.match(r"^\s*(\d+)\.\s+(.*)$", ln)
        if m:
            kind = "ul" if ln.lstrip()[0] in "-*" else "ol"
            if list_open != kind: close_list(); out.append(f"<{kind}>"); list_open = kind
            item = [m.group(2)]; i += 1
            while i < len(lines) and re.match(r"^\s+\S", lines[i]) and not re.match(
                    r"^\s*([-*]\s|\d+\.\s|[#>|]|!\[|---)", lines[i]):
                item.append(lines[i].strip()); i += 1
            out.append(f"<li>{inline(' '.join(item))}</li>"); continue
        # parágrafo — junta linhas plenas consecutivas (soft-wrap) num só <p>
        close_list(); buf = [ln]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^\s*([#>|*-]|\d+\.|!\[|---)", lines[i]):
            buf.append(lines[i]); i += 1
        out.append(f"<p>{inline(' '.join(s.strip() for s in buf))}</p>")
    close_list()
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--title", required=True); ap.add_argument("--css", required=True)
    ap.add_argument("--fig-dir", default=".")
    a = ap.parse_args()
    css = Path(a.css).read_text(encoding="utf-8")
    if "<style>" not in css: css = f"<style>{css}</style>"
    body = md_to_html(Path(a.md).read_text(encoding="utf-8"), Path(a.fig_dir))
    doc = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(a.title)} | DOM-PI / UFPI-DC</title>
{css}
</head><body><main class="container">
{body}
</main></body></html>"""
    Path(a.out).write_text(doc, encoding="utf-8")
    print(f"HTML salvo em {a.out} ({len(doc)} bytes)")

if __name__ == "__main__":
    main()
