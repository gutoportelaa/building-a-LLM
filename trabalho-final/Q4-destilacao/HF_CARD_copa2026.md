---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B
language: [pt]
tags: [knowledge-distillation, reasoning-distillation, rag, world-cup-2026, qwen2.5, distillation]
pipeline_tag: text-generation
---

# qwen2.5-1.5b-copa2026-reasoning-distill

Aluno destilado da **Questão 4 / Plano A** (Construindo um LLM, UFPI-DC): especialização temática na **Copa do Mundo
FIFA 2026** por **destilação de raciocínio** (receita estilo DeepSeek-R1).

## Como foi treinado
- **Professor:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` — modelo de raciocínio (`<think>`) sobre Qwen2.5-14B
  (mesma família do aluno → habilita **white-box logit KD**).
- **Aluno:** `Qwen/Qwen2.5-1.5B` (base), braço **B** (professor com **RAG**), método **combinado** (α·CE + (1−α)·T²·KL).
- **Corpus factual *ungated*:** openfootball (grupos, jogos com placares/gols, elencos, estádios, seleções) +
  **classificação derivada dos jogos** + Wikipedia PT (241 passagens) — **sem scraping de Transfermarkt**. Índice e5-base.
- 200 perguntas self-instruct; respostas do professor com top-50 logits; full fine-tune.

## Resultado (benchmark held-out de 41 fatos da Copa 2026, sem RAG na inferência)
| | ROUGE-L | key-recall |
|---|---|---|
| base 1.5B | 0,122 | 0,476 |
| **este modelo (braço B / RAG)** | 0,209 | **0,640** |

Transferência clara do conhecimento da Copa 2026 para os pesos. **Professor com RAG (B) ≫ "zerada" (A)** — para um tema
**posterior ao corte de conhecimento**, o professor sem recuperação alucina; só o RAG fornece os fatos.

## Uso e limitações
- **Uso:** perguntas factuais sobre a Copa do Mundo 2026 (grupos, sedes, classificação, históricos). O modelo emite
  raciocínio em `<think>…</think>` antes da resposta.
- **Limitações:** conhecimento é um *snapshot* (fatos voláteis mudam — para isso, prefira RAG); `key_recall` é a métrica
  robusta (o ROUGE-L é ruidificado pelo `<think>`); pode alucinar fora do tema.

## Reprodução
`coletar_corpus_futebol.py` → `build_index.py` → `gerar_dataset_destilacao.py` (professor R1) → `destilar.py` →
`avaliar_destilacao.py`. Ver `trabalho-final/Q4-destilacao/` (relatório §7/§4.2).
