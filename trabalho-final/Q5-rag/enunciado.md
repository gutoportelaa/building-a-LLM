# Q5 — RAG (Retrieval-Augmented Generation)

## Enunciado
Criar uma aplicação RAG (Standard, Agentic ou Self-Reflective) usando *diariosPrefeituras* ou *docentesDC*.
Construir um benchmark de **30 perguntas**. Analisar o **grau de contribuição do RAG**.

## Status: ✅ Concluída (implementadas as 4 técnicas: Standard, HyDE, Self-Reflective e Agêntico)

## Dataset escolhido: DOM-PI (diariosPrefeituras)
A Q1 mostrou que DAPT não resolve alucinação factual em perguntas sobre documentos específicos — exatamente
o que o RAG ataca. O DOM-PI tem a densidade factual ideal e reusa o benchmark da Q1.

## O que há nesta pasta
- `benchmark/dompi_qa_tagged.jsonl` — 49 perguntas marcadas (**33 rag · 12 chat · 4 ambos**)
- `scripts/build_index.py` — chunking + embeddings e5-base (GPU)
- `scripts/rag_core.py` — retriever + 4 modos (standard/hyde/reflexivo/agentico) + geradores
- `scripts/run_eval.py` — avaliação (retrieval / generation / traces)
- `scripts/consolidar.py` — tabelas consolidadas
- `index/meta.json` — config do índice (175.924 chunks; .npy/.jsonl regeneráveis, não versionados)
- `resultados/*.json` — saídas brutas (recuperação, 3 geradores, traços, consolidado)
- `relatorio_q5_rag.html` — **relatório dedicado da Q5**

## Modelos (nomes exatos)
- **G1 (antes):** `Qwen/Qwen2.5-1.5B` base
- **G2 (pré-treinado):** `Qwen2.5-1.5B` + adapter QLoRA DAPT (merged 4-bit) — melhor checkpoint da Q1
- **G3 (inferência qualificada):** `qwen2.5:14b` via Ollama
- **Embedder:** `intfloat/multilingual-e5-base` · **HyDE aux:** `llama3.2:3b`

## Resultado central
Acurácia factual (perguntas `rag`), sem → com RAG:

| Gerador | sem RAG | RAG standard |
|---|---|---|
| G1 base | 0% | 11,1% |
| G2 DAPT | 0% | 16,7% |
| G3 qwen2.5:14b | 0% | **33,3%** |

RAG é necessário (todos 0% sem ele); o ganho escala com a capacidade do gerador; o teto é o recall (~42%).

## Como reproduzir
```bash
# da raiz do repo, com o índice em rag/index/:
python build_index.py --out-dir ../index
PYTHONPATH=. python run_eval.py --task retrieval  --out resultados/retrieval.json --hyde-gen llama3.2:3b
PYTHONPATH=. python run_eval.py --task generation --gen "hf:modelos/Qwen2.5-1.5B|G1_base" --modes no_rag standard --out resultados/gen_G1_base.json
PYTHONPATH=. python run_eval.py --task generation --gen "ollama:qwen2.5:14b|G3_14b"     --modes no_rag standard --out resultados/gen_G3_14b.json
python consolidar.py
```

## Custo (medido, RTX 4070 Laptop local)
Latência/chamada: G1 ~7,5s · G2 ~15s (4-bit, 2× mais lento) · G3 14b ~27–38s. Chamadas/técnica: standard 1 ·
HyDE 2 · agêntico ~3,3 · reflexivo ~4. Indexação única ~30 min. Sem custo de API (tudo local).
