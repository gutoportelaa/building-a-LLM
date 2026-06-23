# Q4 — Pipeline de destilação (scripts)

White-box logit KD, **mesma família** Qwen2.5. Professor **Qwen2.5-14B-Instruct** (= G3 da Q5) → alunos
**Qwen2.5-0.5B** e **1.5B** (base pristino). Eixo experimental: **A (zerada) × B (RAG)** no grounding do professor.

Cluster: **gpunode01 = 2× L4 (46GB)** → professor 14B em vLLM `tensor_parallel_size=2`. O 14B só roda na
geração offline; o treino do aluno carrega só o aluno + o cache de logits do disco.

## Ordem de execução

| # | Script | Onde | Saída |
|---|---|---|---|
| 1 | `gerar_dataset_destilacao.py` (via `run_gerar_dataset.sbatch`) | gpunode01, 2 GPU | `dados/dataset_{A,B}.jsonl` + `logits_{A,B}.jsonl` |
| 2 | `destilar.py` *(a criar)* | 1 GPU | alunos destilados: {0.5B,1.5B} × {CE,KL,comb} × {A,B} |
| 3 | `avaliar_destilacao.py` *(a criar)* + `benchmark_destilacao_100.jsonl` *(a criar)* | 1 GPU | métricas antes/depois |

## Métodos de destilação (script 2)
- **CE / hard** — SFT na resposta do professor (`answer_token_ids`), prompt mascarado (−100).
- **KL / soft** — KL(aluno ‖ professor) sobre os **top-k logprobs** do cache, temperatura T; tokens fora do top-k
  ficam fora do suporte (softmax renormalizada no top-k).
- **Combinado** — α·CE + (1−α)·T²·KL (α=0,5).
- ⚠️ Lição da Q1: alinhar `labels = input_ids.clone()` (o HF já desloca interno) — **não** pré-deslocar (bug duplo-shift).

## Métricas (script 3) — benchmark 100 Q (50 DOM-PI + 50 docentesDC)
PPL/CE/token-acc (reusa `avaliacao/avaliar_modelo.py`) · ROUGE-L vs professor · acerto factual por entidade-chave.
Pergunta central: **aluno B supera aluno A?** (grounding do professor transfere conhecimento factual correto?)
E: aluno destilado (KL/comb) supera o SFT-puro (CE)? · quão perto o aluno-B chega do RAG-na-inferência (C, baseline Q5)?
