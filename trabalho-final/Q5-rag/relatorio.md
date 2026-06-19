# Relatório — Questão 5: RAG

> Relatório detalhado em [`relatorio_q5_rag.html`](relatorio_q5_rag.html). Este `.md` é o resumo executivo.

## 1. Enunciado
Entregar um sistema de **RAG funcional** sobre o corpus, explorando técnicas como **HyDE, RAG agêntico e
auto-reflexão**. As demonstrações comparam o modelo **antes** do pré-treino, o modelo **pré-treinado (DAPT)** e o
**modelo mais performático no Ollama** (inferência qualificada).

## 2. Dataset e índice
- Corpus **DOM-PI** (densidade factual alta). Índice vetorial **`intfloat/multilingual-e5-base`** (GPU):
  **175.924 chunks** de **57.229 documentos** (janela 1600 chars, overlap 200). Retriever cosseno (numpy).
- nomic-embed via Ollama era inviável (~5 emb/s); e5-base GPU = ~882 emb/s.

## 3. Modos implementados (`rag/rag_core.py`)
- **Standard:** embeda a query → top-k → gera com contexto.
- **HyDE:** gera documento hipotético, embeda no espaço de passagens e recupera.
- **Self-reflective:** gera, critica a fundamentação e re-busca (até 2 iterações).
- **Agêntico:** loop ReAct (BUSCAR[]/RESPONDER[], até 3 passos).

## 4. Três geradores
- **G1** Qwen2.5-1.5B base · **G2** DAPT (QLoRA merged) · **G3** `qwen2.5:14b` (Ollama, inferência qualificada).

## 5. Resultado central (acerto factual, perguntas RAG, sem → com RAG)
| Gerador | Sem RAG | Com RAG (standard) |
|---|---|---|
| G1 base 1.5B | 0% | 11,1% |
| G2 DAPT 1.5B | 0% | 16,7% |
| **G3 qwen2.5:14b** | 0% | **33,3%** |

Conclusões: **RAG é necessário** (todos 0% sem ele); o ganho **escala com a capacidade do gerador**; o gargalo é o
**recall** (~42%) — o 14b acerta ~75% quando o contexto traz a resposta, enquanto modelos 1.5B colapsam/alucinam.

## 6. Descobertas não-óbvias
- **HyDE não superou o standard** (documento hipotético alucina entidades e desvia a busca) — só rende em queries vagas.
- `benchmark.fonte_id` não casa com ids do corpus → avaliação **por conteúdo** (entidade-chave).
- Custo ≈ nº de chamadas LLM × latência/chamada: standard (1) < reflexivo/HyDE (2-3) < agêntico (≈3+).
  Recomendação operacional: **Standard como padrão**; agêntico só em multi-hop/síntese.

## 7. Artefatos
- Código: `scripts/` (rag_core, build_index, run_eval, consolidar). Índice no HF (privado):
  `gutoportelaa/dompi-rag-index-e5`. Resultados em `resultados/`.

Detalhes completos: [`relatorio_q5_rag.html`](relatorio_q5_rag.html) e §5 de `../relatorio_tecnico_completo.html`.
