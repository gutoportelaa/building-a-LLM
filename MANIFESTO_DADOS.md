# Manifesto de Dados — onde vive cada artefato

Este repositório versiona **código, resultados e relatórios**. Os dados pesados
(PDFs, corpus, checkpoints) vivem fora dele. Este documento mapeia cada
diretório que existia na máquina de desenvolvimento para o seu destino
definitivo, de modo que o trabalho continue reproduzível depois que aquela
máquina deixar de existir.

Auditoria de referência: 2026-08-12.

## Publicado no HuggingFace (`gutoportelaa`)

| Repositório | Conteúdo | Diretório local correspondente |
|---|---|---|
| [`dom-pi-pdfs-2025`](https://huggingface.co/datasets/gutoportelaa/dom-pi-pdfs-2025) | 73.979 PDFs-fonte (~66 GB), organizados por território/município | `territorios/`, `db_treino_carnaubais/` |
| [`dom-pi-corpus-2025`](https://huggingface.co/datasets/gutoportelaa/dom-pi-corpus-2025) | Corpus DOM-PI: 80.788 docs, ~178M tokens, 12 territórios, 4 configs (`default`, `curated`, `raw`, `extraido`) | `datalake/`, `staging_lab/`, `extraidos/`, `dados_limpos/`, `dados_brutos/` |
| [`dom-pi-teresina-2025`](https://huggingface.co/datasets/gutoportelaa/dom-pi-teresina-2025) | DOM-Teresina 2025: 250 PDFs + parquet | `diarios-teresina-2025/`, `territorios/teresina/`, `hf_teresina/` |
| [`DOMPI-2025`](https://huggingface.co/datasets/gutoportelaa/DOMPI-2025) | Parquets brutos por território (snapshot inicial) | — |
| [`qwen2.5-1.5b-dompi-teresina-v3`](https://huggingface.co/gutoportelaa/qwen2.5-1.5b-dompi-teresina-v3) | Modelo DAPT do Q1 (Teresina v3) | — |

### Verificação de cobertura dos PDFs

Contagem local × contagem no HF, por território (conferido em 2026-08-12):

| Território | Local | HF | Situação |
|---|---:|---:|---|
| carnaubais | 12.370 | 12.370 | replicado |
| tabuleiros_alto_parnaiba | 13.293 | 13.293 | replicado |
| vale_dos_rios_piaui_e_itaueiras | 9.800 | 9.800 | replicado |
| planice_litoran | 4.386 | 4.386 | replicado |
| entre_rios | 963 | 963 | replicado |
| vale_do_sambito | 555 | 555 | replicado |
| teresina | 250 | 250 | replicado |
| chapada_vale_do_rio_itaim | 0 | 14.104 | só no HF |
| cocais | 0 | 11.695 | só no HF |
| serra_da_capivara | 0 | 6.564 | só no HF |
| parnaiba | 0 | 0 | **sem coleta** |
| vale_do_caninde | 0 | 0 | **sem coleta** (corpus existe, PDFs não) |
| vale_do_rio_guaribas | 0 | 0 | **sem coleta** (corpus existe, PDFs não) |

Nenhum PDF existia apenas na máquina local. `db_treino_carnaubais/` era uma
duplicata exata de `territorios/carnaubais` em formato flat-hash (nome do
arquivo = hash da URL), formato legado que perde município e data de publicação.

## Artefatos que exigiram upload próprio

Não derivam de nada publicado e foram salvos separadamente antes do descarte da
máquina:

| Artefato | Tamanho | Observação |
|---|---:|---|
| `trabalho-final/Q2-pos-treino-sft/modelos/sft_full_0.5b` | 958 MB | full fine-tuning: os pesos são o próprio artefato |
| `trabalho-final/Q2-pos-treino-sft/modelos/sft_full_1.5b` | 2,9 GB | idem |
| `trabalho-final/Q2-pos-treino-sft/modelos/*/adapter/` (4×) | 209 MB | adapters LoRA/QLoRA dos quatro braços |
| `trabalho-final/Q4-destilacao/modelos_local/xfam_llama1b_B_ce` | 2,4 GB | aluno cross-família do Q4 |
| `data/train_corpus.jsonl` + `held_out.jsonl` | 219 MB | corpus de treino do Q1 (derivável do HF, mas o split não) |
| `datalake/extraido/` | 254 MB | OCR bruto pré-limpeza |

Os `model.safetensors` **merged** dos braços LoRA/QLoRA (sft_lora_1.5b 3,0 GB,
sft_qlora_1.5b 1,2 GB, sft_lora_0.5b 1007 MB, sft_qlora_0.5b 501 MB) não foram
salvos: são base + adapter fundidos, regeneráveis a partir dos 209 MB de adapter.

## Reprodutibilidade do split do Q1

`data/held_out_ids.txt` (2.860 ids, 94 KB) **está versionado neste repositório**,
como exceção explícita no `.gitignore`. Ele identifica exatamente os documentos
mantidos fora do treino, permitindo reconstruir o split a partir do corpus
público sem depender dos 219 MB de JSONL. O script que o gera é
`avaliacao/preparar_held_out.py`.

## Regeneráveis — não foram salvos

`.venv/`, `.venv-paddle/`, `dados/chroma_db/` (676 MB), `dados/bm25_index.pkl`
(121 MB), `rag/index/embeddings.npy`, `rag/index/chunks.jsonl`, todos os
`__pycache__/` e os logs de execução. Reconstruíveis pelos scripts versionados.

## Pendências conhecidas

- `PROBLEMAS_LOGICA_EXTRACAO.md` — P-03, P-08 e P-09 seguem em aberto.
- Três territórios sem coleta de PDFs: `parnaiba`, `vale_do_caninde` e
  `vale_do_rio_guaribas`. Os dois últimos possuem corpus extraído no datalake
  (a extração ocorreu no laboratório), mas os PDFs-fonte nunca foram
  centralizados. `to-do_territorios.txt` lista os municípios de cada território.
