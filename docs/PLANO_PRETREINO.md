# Plano — Pré-treino continuado + avaliação (DOM-PI)

> Spec para a **próxima sessão** (modelagem). Pré-requisito pronto: corpus publicado no HF
> (`dom-pi-corpus-2025`, 13 territórios, 80.788 docs, configs `default`/`curated`/`raw`/`extraido`).
> Contexto de dados: [`CONTEXT.md`](../CONTEXT.md), [`CONTEXT_2.md`](../CONTEXT_2.md),
> [`RELATORIO_CORPUS_DOM-PI.md`](../RELATORIO_CORPUS_DOM-PI.md). Referência conceitual:
> `docs/referencia_raschka/ch04.ipynb` (arquitetura GPT) — mas na prática usaremos um modelo
> pronto da HF (não from-scratch).

## Objetivo

**Pré-treino continuado (domain-adaptive)** de um LLM **multilíngue** no corpus DOM-PI e
**avaliação antes × depois**. Não é from-scratch (o livro é didático); adaptamos um modelo
pré-treinado ao domínio jurídico-administrativo do Piauí.

## Modelo (multilíngue, ≠ GPT-2)

| Opção | Params | Licença | Encaixe | Nota |
|---|---|---|---|---|
| **Qwen2.5-0.5B** (recomendado) | 0,5B | Apache-2.0 | **full-FT cabe no RTX 4070 (12GB)** | forte em PT, licença limpa |
| Qwen2.5-1.5B | 1,5B | Apache-2.0 | LoRA no 4070 / full no lab | mais capaz |
| Llama-3.2-1B | 1B | Llama Community | LoRA | multilíngue, licença mais restritiva |

**Recomendação:** começar com **Qwen2.5-0.5B full-FT** (experimento rápido e barato);
escalar p/ 1.5B com LoRA depois.

## Dados

- **Treino:** config **`curated`** (Tier A+B, prosa limpa — evita ensinar tabela achatada).
- **Held-out:** reservar ~5% dos docs (ou um conjunto fixo por `id`) **fora do treino** para a
  avaliação intrínseca. Garantir que o held-out não vaze para o treino.
- Tokenizar com o tokenizer do próprio modelo; **empacotar** em blocos de 1024–2048 tokens
  (concat + split), objetivo **causal LM** (next-token).

## Método

- **Full-FT** (0.5B) ou **LoRA/PEFT** (1.5B+). lr ~2e-5 (full) / 1e-4 (LoRA), warmup, cosine,
  1–2 épocas num subconjunto, checkpoint + early stop por perplexidade no held-out.
- Stack: **HF Transformers + Datasets + Accelerate (+ PEFT p/ LoRA)**. Rodar no RTX 4070 local
  ou no lab (GPU L4). Usar `bf16`, gradient accumulation, `gradient_checkpointing` se faltar VRAM.

## Avaliação — antes × depois (held-out do domínio)

Métricas intrínsecas (as pedidas), medidas no **mesmo held-out** com o modelo **base** e o
**pós-treino**:
- **Entropia cruzada** = NLL média por token (objetivo causal LM).
- **Perplexidade** = `exp(cross_entropy)`.
- **Acurácia de previsão de token** = fração de posições em que `argmax(logits) == próximo token`.

> Expectativa: perplexidade/CE **caem** e a acurácia **sobe** no domínio DOM-PI após o pré-treino
> continuado — é o sinal de adaptação ao domínio.

## Benchmark de domínio (≥ 25 perguntas + respostas de referência)

Arquivo `benchmark/dompi_qa.jsonl` — cada linha `{"pergunta", "resposta_ref", "fonte_id", "tipo"}`.
Fatos extraídos de docs **Tier A** (licitações, contratos, nomeações, leis): vencedores de
pregão, valores de contrato, nomeados em portarias, datas/edições, ementas de decretos.

**Como avaliar o benchmark:**
- **Agora (pós-pré-treino):** perplexidade/NLL da `resposta_ref` **dado o prompt da pergunta**
  (proxy de "o modelo internalizou o fato?"). Comparar base × pós-treino.
- **Depois (pós-SFT):** geração + match (EM/F1 ou LLM-as-judge).

> **Caveat de MLOps:** pré-treino continuado melhora **perplexidade/conhecimento de domínio**,
> mas **responder perguntas** (geração) melhora mais com **SFT/instruction** (próxima fase, usando
> o `curated` como base e Q&A gerado por LLM — sem templar de metadados, conforme decidido).
> Por isso o benchmark serve às duas fases (perplexidade-da-resposta já; geração após SFT).

### Exemplos-semente (do próprio corpus, validar/expandir até ≥25)

- *"Qual empresa venceu o Pregão nº 040/2024 da Prefeitura de Pedro II?"* → **ANTARES Comércio Atacadista LTDA** (Termo de Homologação).
- *"Quem foi nomeado para a Coordenação de Solenidades Oficiais em Cocal pela Portaria nº 130/2025?"* → **Eulilia da Silva Carvalho Albuquerque**.
- *"De que trata a Portaria nº 130/2025 de Cocal?"* → nomeação para cargos em comissão e funções de confiança.

## Entregáveis da sessão

1. `benchmark/dompi_qa.jsonl` (≥25 Q&A com respostas de referência).
2. `treino/` — script de pré-treino continuado (HF Trainer/Accelerate) + config.
3. `avaliacao/` — script que mede CE, perplexidade e token-acc no held-out **e** no benchmark.
4. Relatório **antes × depois** (tabela de métricas + análise).

## Como iniciar a nova sessão

Carregar contexto: `MEMORY.md` + este plano + `CONTEXT_2.md` (qualidade/tiers) + `RELATORIO`.
Primeiro passo: baixar `curated` via `load_dataset("gutoportelaa/dom-pi-corpus-2025", "curated")`,
separar held-out, e rodar a avaliação **baseline** (Qwen2.5-0.5B) antes de treinar.
