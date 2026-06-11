#!/usr/bin/env python3
"""
qualidade.py — tier de qualidade textual por documento (A / B / C).

A análise do corpus mostrou que NENHUMA métrica lexical isolada separa "prosa boa" de
"tabela fiscal achatada" (esta tem palavras reais, mas é sopa de números). O sinal que
funciona é a combinação:
  • real_word_ratio  — fração de tokens alfabéticos que são palavras PT reais (wordfreq);
  • numeric_ratio    — fração de tokens com dígito (densidade de tabela/código).

Tiers (limiares calibrados em amostra de 9k docs):
  A · prosa limpa   : real_word ≥ 0.88 e numeric < 0.15   (~52%; SFT/instruction)
  C · tabela/ruim   : numeric ≥ 0.35 ou real_word < 0.78   (~13%; excluir do treino de prosa)
  B · média         : o resto                               (~35%; pré-treino)

Tier C concentra-se nos atos fiscais (LRF/Lei/Decreto) — é caso de RE-EXTRAÇÃO
(Docling/VLM table-aware), não de limpeza.
"""
from __future__ import annotations

import re

from wordfreq import zipf_frequency

_RE_ALPHA = re.compile(r"[A-Za-zÀ-ÿ]{3,}")
_RE_TOKEN = re.compile(r"\S+")
_RE_HASDIGIT = re.compile(r"\d")

A_MIN_RW, A_MAX_NUM = 0.88, 0.15
C_MAX_RW, C_MIN_NUM = 0.78, 0.35


def metricas(texto: str, n_alpha: int = 400, n_tok: int = 500) -> tuple[float, float]:
    """(real_word_ratio, numeric_ratio) — amostra os primeiros tokens para custo baixo."""
    alpha = _RE_ALPHA.findall(texto or "")[:n_alpha]
    rw = (sum(1 for w in alpha if zipf_frequency(w.lower(), "pt") > 0) / len(alpha)
          ) if alpha else 0.0
    toks = _RE_TOKEN.findall(texto or "")[:n_tok]
    num = (sum(1 for w in toks if _RE_HASDIGIT.search(w)) / len(toks)) if toks else 1.0
    return rw, num


def quality_tier(texto: str) -> str:
    rw, num = metricas(texto)
    if rw >= A_MIN_RW and num < A_MAX_NUM:
        return "A"
    if num >= C_MIN_NUM or rw < C_MAX_RW:
        return "C"
    return "B"
