#!/usr/bin/env python3
"""
limpar_corpus_ocr.py — Limpeza heurística de OCR do corpus DOM-PI (Eixo A).

O corpus unificado é texto de OCR ruidoso ("Ano XxIlI", "CRIADOPEL.SEIN01915195",
"Edicao VCccXLVI"...). A hipótese de DAPT é que um sinal de treino mais limpo
rende mais ganho por token (foi o que vimos no subcorpus curado de Teresina).
Este script NÃO mexe no held-out (mantém comparabilidade do PPL com o resultado
atual) — limpa apenas o lado de TREINO, descartando linhas/documentos degradados.

Heurísticas, em ordem:
  1. de-hifenização de quebra de linha ("pala-\\nvra" -> "palavra")
  2. por LINHA: descarta linha com razão alfabética baixa, curta demais, ou
     dominada por dígitos/símbolos (cabeçalho/rodapé/lixo de OCR)
  3. colapso de runs de caractere repetido (">>>>>", "----")
  4. por DOCUMENTO: descarta se sobrou pouco texto ou se a fração preservada
     ficou abaixo do limiar (documento majoritariamente ruído)

Uso:
    python scripts/limpar_corpus_ocr.py \
        --input data/train_corpus.jsonl \
        --output data/train_corpus_limpo.jsonl \
        --stats resultados/limpeza_stats.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RUN_RE = re.compile(r"(.)\1{4,}")            # 5+ repetições do mesmo char
DEHYPHEN_RE = re.compile(r"(\w)-\n(\w)")      # quebra hifenizada
MULTISPACE_RE = re.compile(r"[ \t]{2,}")


def alpha_ratio(s: str) -> float:
    chars = [c for c in s if not c.isspace()]
    if not chars:
        return 0.0
    letters = sum(c.isalpha() for c in chars)
    return letters / len(chars)


def digit_ratio(s: str) -> float:
    chars = [c for c in s if not c.isspace()]
    if not chars:
        return 0.0
    return sum(c.isdigit() for c in chars) / len(chars)


def _is_ocr_id_token(tok: str) -> bool:
    """Token longo, sem espaço, misturando letras e dígitos: lixo de OCR/código.
    Ex.: 'CRIADOPEL.SEIN01915195', 'OOPAC01915195'."""
    if len(tok) < 12:
        return False
    has_alpha = any(c.isalpha() for c in tok)
    has_digit = any(c.isdigit() for c in tok)
    return has_alpha and has_digit


def clean_line(line: str, min_alpha: float, min_len: int, max_digit: float) -> str | None:
    """Retorna a linha limpa, ou None se for para descartar."""
    line = RUN_RE.sub(lambda m: m.group(1) * 2, line)
    line = MULTISPACE_RE.sub(" ", line).strip()
    if len(line) < min_len:
        return None
    if alpha_ratio(line) < min_alpha:
        return None
    if digit_ratio(line) > max_digit:
        # linhas de tabela/numéricas puras: descarta a menos que tenham contexto textual
        return None
    toks = line.split()
    if toks and sum(_is_ocr_id_token(t) for t in toks) / len(toks) >= 0.30:
        # linha dominada por códigos/IDs alfanuméricos colados
        return None
    return line


def clean_doc(text: str, args) -> tuple[str, float]:
    """Limpa um documento; retorna (texto_limpo, fração_preservada_em_chars)."""
    text = unicodedata.normalize("NFC", text)
    text = DEHYPHEN_RE.sub(r"\1\2", text)
    original_len = max(1, len(text))
    kept_lines = []
    for ln in text.split("\n"):
        c = clean_line(ln, args.min_alpha, args.min_line_len, args.max_digit)
        if c:
            kept_lines.append(c)
    cleaned = "\n".join(kept_lines)
    return cleaned, len(cleaned) / original_len


def main() -> None:
    p = argparse.ArgumentParser(description="Limpeza heurística de OCR do corpus DOM-PI")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stats", default="resultados/limpeza_stats.json")
    p.add_argument("--min-alpha", type=float, default=0.55, help="Razão alfabética mínima por linha")
    p.add_argument("--min-line-len", type=int, default=12)
    p.add_argument("--max-digit", type=float, default=0.6, help="Razão de dígitos máxima por linha")
    p.add_argument("--min-doc-chars", type=int, default=200, help="Mínimo de chars no doc limpo")
    p.add_argument("--min-keep-frac", type=float, default=0.30, help="Fração mínima preservada do doc")
    p.add_argument("--text-field", default="texto")
    args = p.parse_args()

    in_path, out_path = Path(args.input), Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0
    chars_in = chars_out = 0
    dropped_short = dropped_frac = 0

    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get(args.text_field) or obj.get("text") or ""
            n_in += 1
            chars_in += len(text)

            cleaned, frac = clean_doc(text, args)
            if len(cleaned) < args.min_doc_chars:
                dropped_short += 1
                continue
            if frac < args.min_keep_frac:
                dropped_frac += 1
                continue

            obj[args.text_field] = cleaned
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1
            chars_out += len(cleaned)

    stats = {
        "input": str(in_path),
        "output": str(out_path),
        "docs_in": n_in,
        "docs_out": n_out,
        "docs_dropped_short": dropped_short,
        "docs_dropped_lowfrac": dropped_frac,
        "doc_keep_rate": round(n_out / max(1, n_in), 4),
        "chars_in": chars_in,
        "chars_out": chars_out,
        "char_keep_rate": round(chars_out / max(1, chars_in), 4),
        "params": {k: getattr(args, k) for k in
                   ["min_alpha", "min_line_len", "max_digit", "min_doc_chars", "min_keep_frac"]},
    }
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    log.info("=" * 60)
    log.info("LIMPEZA DE OCR concluída")
    log.info("  docs:  %d -> %d  (mantidos %.1f%%)", n_in, n_out, 100 * stats["doc_keep_rate"])
    log.info("  chars: %d -> %d  (mantidos %.1f%%)", chars_in, chars_out, 100 * stats["char_keep_rate"])
    log.info("  descartados: %d curtos, %d baixa-fração", dropped_short, dropped_frac)
    log.info("  -> %s", out_path)


if __name__ == "__main__":
    main()
