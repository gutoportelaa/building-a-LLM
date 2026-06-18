# Trabalho Final — Construindo um LLM (DOM-PI / UFPI-DC)

Pasta de entrega organizada por questão. Cada subpasta contém o **enunciado**, os
**relatórios** e os **scripts** usados para gerar os resultados daquela questão.

- **Dataset principal:** `diariosPrefeituras` (DOM-PI) — Diário Oficial dos Municípios do Piauí
- **Dataset secundário:** [`vickminari/docentesDC`](https://huggingface.co/datasets/vickminari/docentesDC) (Q2/Q3)
- **Modelo base:** `Qwen/Qwen2.5-1.5B` · **Inferência qualificada:** `qwen2.5:14b` (Ollama)
- **Relatório consolidado (todas as questões):** [`relatorio_tecnico_completo.html`](relatorio_tecnico_completo.html)

> ⚠️ **Nomenclatura interna vs. enunciado:** scripts com "q2"/"q3" no nome (ex.: `run_q2_lora_1.5b.sbatch`,
> `run_q3_fullft_teresina.sbatch`) referem-se a **experimentos internos da Questão 1** (variações de método),
> e **não** às Questões 2 e 3 do enunciado. A numeração de pastas aqui segue o **enunciado**.

---

## Correlação Questão ↔ Modelos ↔ Scripts ↔ Status

| Questão | Tema | Dataset | Modelos | Scripts principais | Status |
|---|---|---|---|---|---|
| **Q1** | Pré-treino continuado (DAPT) | DOM-PI | Qwen2.5-0.5B/1.5B · LoRA · QLoRA · Full FT | `pretreino_{continuado,lora,fullft}.py`, `avaliar_modelo.py`, `*.sbatch` | ✅ Concluída |
| **Q2** | Pós-treino SFT | docentesDC | Qwen2.5-1.5B-Instruct | *(a criar)* `gerar_pares_sft.py`, `sft_docentes.py` | ⏳ Pendente |
| **Q3** | Pós-treino LoRA/QLoRA | docentesDC | Qwen2.5-0.5B/1.5B-Instruct | *(a criar)* `pretreino_lora.py` adaptado p/ SFT | ⏳ Pendente |
| **Q4** | Destilação de conhecimento | sintético | Professor Qwen2.5-7B → Aluno 1.5B/0.5B | *(a criar)* `destilar.py` | ⏳ Pendente |
| **Q5** | RAG (Standard/HyDE/Reflexivo/Agêntico) | DOM-PI | G1 base · G2 DAPT · G3 qwen2.5:14b · embedder e5-base | `build_index.py`, `rag_core.py`, `run_eval.py`, `consolidar.py` | ✅ Concluída |
| **Q6** | Guardrails | DOM-PI | modelo da Q5 ou Q2 + guardrails-ai | *(a criar)* `guardrails_pipeline.py` | ⏳ Pendente |

## Modelos usados (nomes exatos)

| Rótulo | Modelo | Onde | Uso |
|---|---|---|---|
| Base 0.5B/1.5B | `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B` | HF | Q1 (DAPT), Q5 G1 |
| DAPT QLoRA | `Qwen2.5-1.5B` + adapter QLoRA NF4 (merged `best`) | cluster → `modelos/dapt_qlora_best` | Q1 melhor resultado, Q5 G2 |
| Inferência qualificada | `qwen2.5:14b` | Ollama (local) | Q5 G3 |
| Embedder | `intfloat/multilingual-e5-base` | HF (GPU) | Q5 recuperação |
| Auxiliar HyDE | `llama3.2:3b` | Ollama | Q5 doc hipotético (recall) |

---

## Etapas executadas (Q5 — referência de reprodução)

1. **Marcar benchmark** → `benchmark/dompi_qa_tagged.jsonl` (33 rag · 12 chat · 4 ambos)
2. **Indexar** → `scripts/build_index.py` → 175.924 chunks, embeddings e5-base (GPU, ~30 min)
3. **Avaliar recuperação** → `run_eval.py --task retrieval` (standard × HyDE)
4. **Comparar geradores** → `run_eval.py --task generation` (G1, G2, G3 × no_rag/standard)
5. **Traços avançados** → `run_eval.py --task traces` (hyde/reflexivo/agentico no 14b)
6. **Consolidar** → `consolidar.py` → `resultados/consolidado.json` + tabelas

### Resultado central (Q5)
Acurácia factual (perguntas `rag`), sem → com RAG: **G1 base 0%→11,1% · G2 DAPT 0%→16,7% · G3 14b 0%→33,3%**.
RAG é necessário (todos 0% sem ele); o ganho escala com a capacidade do gerador; o teto é a recuperação (recall ~42%).

---

## Artefatos pesados (NÃO versionados — regeneráveis)

Ignorados via `.gitignore` por tamanho; reconstrua localmente:

| Artefato | Tamanho | Como regenerar |
|---|---|---|
| `modelos/Qwen2.5-1.5B` | 3,1 GB | `huggingface_hub.snapshot_download('Qwen/Qwen2.5-1.5B')` |
| `modelos/dapt_qlora_best` | 1,16 GB | puxar `best/` de `treino/checkpoints_qlora_1.5b` no cluster |
| `rag/index/embeddings.npy` + `chunks.jsonl` | 773 MB | `python scripts/build_index.py` |

O índice fica em `rag/index/` (diretório de trabalho). Aqui versionamos apenas `index/meta.json` (config + estatísticas).
