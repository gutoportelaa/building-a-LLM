# Relatório — Questão 1: Pré-treino Continuado (DAPT)

## 1. Enunciado
Fazer **pré-treinamento continuado** de um LLM considerando o dataset unificado *diariosPrefeituras* (DOM-PI) e
avaliar a qualidade **antes e depois**. Criar um benchmark com **≥25 perguntas e respostas de referência**.
Métricas exigidas: **perplexidade (PPL), entropia cruzada (CE) e acurácia de previsão de tokens**.

## 2. Modelo e justificativa
- **Base:** `Qwen/Qwen2.5-1.5B` (também `Qwen2.5-0.5B` nos experimentos iniciais de varredura de LR).
- Por quê: forte em português, licença permissiva, tamanho que cabe full-FT em GPU L4 (24 GB) com AdamW 8-bit.

## 3. Corpus
- **DOM-PI unificado** (Diário Oficial dos Municípios do Piauí): 224 municípios + capital. ~50 M tokens (tier A/B).
  Texto de OCR, com ruído.
- **Subcorpus Teresina** (curado, tier A+B): ~9,3 M tokens, texto mais limpo (capital).
- Held-out fixo (seed=42): 2.000 docs (unificado) e 444 docs (Teresina), separados antes do treino.

## 4. Benchmark
`dompi_qa.jsonl` — **49 perguntas** (supera o mínimo de 25), com fatos verificados do tier A. Métrica: CE da resposta
de referência condicionada ao prompt (NLL), PPL e token-accuracy.

## 5. Metodologia
Treino causal packed (blocos de 512 tokens), AdamW 8-bit (bitsandbytes), lr=2e-6 (full-FT) / 2e-5 (PEFT), warmup 5% +
cosine decay manual, early stopping por CE no held-out. Avaliação antes×depois com `avaliar_modelo.py` +
`comparar_resultados.py`.

## 6. A reviravolta: bug do duplo deslocamento de rótulos
Durante meses os experimentos exibiram "esquecimento catastrófico", atribuído ao corpus ruidoso. A causa real foi um
**bug**: o `PackedTextDataset` pré-deslocava os `labels` (`block[1:]`) **e** o modelo HuggingFace já desloca
internamente (`logits[:-1]` vs `labels[1:]`), resultando em **duplo shift** — o treino otimizava "prever o token **2
posições à frente**", um objetivo incorreto.
- **Sintomas:** loss de treino travada em ~11 (≈ log do vocabulário, ~152 k) e PPL da eval interna ~47.000 (vs ~10 real).
- **Presente nos três scripts:** `pretreino_fullft.py`, `pretreino_continuado.py`, `pretreino_lora.py`.
- **Correção:** `input_ids` e `labels` alinhados (`labels = input_ids.clone()`); o shift fica só por conta do modelo.
- **Sanidade:** após a correção, a CE do baseline interno bate o `avaliar_modelo.py` (PPL ~10 unificado / ~6,9 Teresina).

## 7. Resultados (held-out) — bug → corrigido
| Versão (job) | Corpus | Objetivo | PPL | Δ vs baseline |
|---|---|---|---|---|
| **Full FT unificado v2 (491)** | completo | corrigido | **8,90** | **−11,3%** ✓ |
| **Full FT Teresina v3 (489)** | Teresina | corrigido | **6,02** | **−12,9%** ✓ |
| QLoRA unificado v2 (492) | completo | corrigido | 9,88 | −1,4% |
| Full FT unificado (487) | completo | bug | 22,22 | +121% |
| QLoRA (421) | completo | bug | 10,47 | +4,4% |
| Full FT Teresina v1/v2 (429/488) | Teresina | bug | 8,76 / 12,64 | +27% / +83% |
| LoRA (396/426) | completo | bug | ~1.000 | colapso |

Benchmark Q&A (NLL): Full FT unificado v2 7,45→7,26; Teresina v3 7,45→7,24. Token-accuracy sobe em ambos.

## 8. Duas respostas (por desenho), ambas positivas
O enunciado pede o **dataset completo**. Com o objetivo corrigido:
1. **Canônica (dataset completo):** Full FT unificado v2 — **−11,3%** no held-out geral. Resposta literal ao enunciado e melhor PPL no held-out geral.
2. **Alternativa (corpus curado):** Full FT Teresina v3 — **−12,9%** no held-out de Teresina.

## 9. Conclusões
- O bug **invertia a conclusão**: sob o objetivo errado, o full-FT parecia catastrófico (+121%) e o QLoRA "vencia"
  (+4,4%). Corrigido, o **Full FT de parâmetros plenos supera o PEFT** nos dois corpora (−11,3% vs −1,4%) — como a
  teoria de DAPT prevê.
- **Lição central:** validar o pipeline (sanidade: PPL baseline ~10, não ~47.000) antes de culpar os dados.
- LoRA colapsa em CPT com corpus pequeno ("intruder dimensions"); QLoRA (NF4) é mais estável mas conservador.

## 10. Exemplos de inferência (completar portaria)
- **Bugado (full FT unificado):** "…limpeza urbana e *coletares resídeos* domiciliares… *Coletes Resídeos R$ 70,00*" — gramática/OCR degradados.
- **Corrigido (v2):** texto fluente e no domínio ("…publicado diariamente… normas legais aprovadas…").

## 11. Modelos publicados (HuggingFace, privados)
- `gutoportelaa/qwen2.5-1.5b-dompi-fullft-unificado` — canônica (v2 corrigido).
- `gutoportelaa/qwen2.5-1.5b-dompi-teresina-v3` — alternativa.
- `gutoportelaa/qwen2.5-1.5b-dompi-dapt` — variante QLoRA (gerador G2 da Q5).

## 12. Como reproduzir
```bash
# Full FT unificado (resposta canônica)
sbatch scripts/run_q1_fullft_unificado_v2.sbatch
# Full FT Teresina (alternativa)
sbatch scripts/run_q1_fullft_teresina_v3.sbatch
# QLoRA
sbatch scripts/run_q1_qlora_unificado_v2.sbatch
```
Avaliação: `avaliar_modelo.py` + `comparar_resultados.py`. Detalhes: §1 de `../relatorio_tecnico_completo.html`.
