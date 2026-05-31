# Benchmark de Extração OCR — PaddleOCR vs Docling (GPU/CPU)

Validação do "novo pipeline" de extração do DOM-PI usando **PaddleOCR (PP-Structure)**
e **Docling**, em GPU e CPU, com foco em desempenho, qualidade e **uso de memória**.

Data: 2026-05-31 · Máquina: lab (NFS home) · Executor: `src/dompi_scraper/bench_ocr.py`

> **Atualização (otimização do orquestrador).** Diagnóstico de um run real de `tabuleiros_alto_parnaiba`
> mostrou subutilização grave do hardware (~7.7 s/doc para PDFs de 1 página nativa). Causas e correções
> aplicadas no `orquestrador_extracao.py`:
> 1. **Docling rodava OCR em CPU** (RapidOCR/onnxruntime) mesmo no worker "GPU" → agora usa
>    **`do_ocr=False`** (lê camada de texto nativa, GPU só para layout/tabela). O flag é por-requisição
>    (`WorkerClient.extract(..., ocr=False|True)`; o worker mantém um conversor por modo).
> 2. **Escaneados** (sem texto nativo) vão para **PaddleOCR-GPU** (PP-Structure faz OCR+tabela na GPU),
>    não mais para o Docling-CPU.
> 3. **Dedup ANTES da extração**: hash do texto da triagem PyMuPDF; duplicatas são puladas sem pagar o
>    motor pesado (registro `registro_dla_dedup.txt`, persistido para retomada).
> 4. **Roteamento por presença de texto** (`--threshold` = corte de texto nativo, default 0.45):
>    escaneado→PaddleOCR-GPU; nativo+tabela→Docling `do_ocr=False`; nativo simples→PyMuPDF (instantâneo).
>
> Resultados do run de validação (200 docs, mesmo `output-dir`, sem apagar): ver §9.

---

## 1. Ambiente da máquina

| Item | Valor |
|------|-------|
| GPU | **2× NVIDIA L4** (24 GB cada, Compute Capability 8.9) |
| CUDA (driver) | API 13.0 |
| `nvidia-smi` / NVML | **QUEBRADO** — `Failed to initialize NVML: Driver/library version mismatch` (NVML 580.159) |
| CUDA (compute) | **Funciona** — `torch.cuda.is_available()=True`, `paddle.is_compiled_with_cuda()=True`, ops GPU OK em ambos |
| RAM | 61 GiB | 
| CPU | 16 cores |
| Disco | 1.8 TB livres (NFS) |

> **Telemetria ≠ Compute.** O NVML/`nvidia-smi` está fora do ar (mismatch de versão de driver),
> mas a pilha de compute CUDA opera normalmente. Como o NVML não funciona, **as medições de VRAM
> deste benchmark usam os contadores internos dos frameworks** (`torch.cuda.max_memory_reserved`,
> `paddle.device.cuda.max_memory_reserved`), não `nvidia-smi`.

---

## 2. Conflito de dependências (CRÍTICO)

`torch` (build **cu13**) e `paddlepaddle-gpu` (build **cu126**) **não coexistem no mesmo venv**:
ambos instalam bibliotecas no namespace compartilhado `nvidia/` e se sobrescrevem
(`libnccl.so.2`, `libcudnn.so.9`). Instalar `paddlepaddle-gpu` quebra o `import torch`
(`undefined symbol: ncclCommWindowDeregister`) e remove libs que o torch espera.

**Solução adotada — dois ambientes isolados:**

| venv | conteúdo | uso |
|------|----------|-----|
| `.venv` | torch (cu130) + docling + marker + pymupdf | engine **Docling** |
| `.venv-paddle` | paddlepaddle-gpu (cu126) + paddleocr 2.10 + pymupdf | engine **PaddleOCR** |

> **Importante:** não usar `uv run` para estes workers — o `uv.lock` lista paddleocr/paddlepaddle,
> então `uv run` re-sincroniza e reinstala o paddle no `.venv`, **requebrando o torch**.
> Invocar os interpretadores diretamente: `./.venv/bin/python`, `./.venv-paddle/bin/python`.

### Solução implementada — workers isolados por venv e por GPU
O `orquestrador_extracao.py` foi refatorado para **não importar paddle/torch no próprio
processo**. Cada engine pesada roda num **subprocesso worker dedicado** (`engine_worker.py`),
no seu venv, gerenciado por `worker_client.py` via protocolo de linhas JSON (stdin/stdout):

```
orquestrador (.venv: polars+fitz+torch p/ detectar GPU)
   ├─ triagem DLA (PyMuPDF, in-process)         → rota nativa: PyMuPDF fast path
   ├─ WorkerClient(paddle, .venv-paddle, GPU 0) → rota OCR simples: PaddleOCR-CUDA
   └─ WorkerClient(docling, .venv,        GPU 1) → rota complexa: Docling-CUDA
```

- **GPU por worker** (máquina tem 2× L4): `paddle → GPU 0`, `docling → GPU 1` (via
  `CUDA_VISIBLE_DEVICES`, definido pelo cliente). Com 1 só GPU, ambos compartilham a GPU 0
  (processos separados, sem conflito de runtime). Configurável por `--gpu-paddle/--gpu-docling`.
- O stderr de cada worker é drenado para o log do orquestrador (`[engine::worker] ...`),
  dando diagnóstico unificado (carga de modelos, VRAM, RSS, tempo por requisição, tracebacks).
- **Validado end-to-end** (2026-05-31): rotas PyMuPDF (0.1 s, in-process), PaddleOCR-CUDA
  (GPU 0, ~1 s) e Docling-CUDA (GPU 1, ~7–17 s) com 0 erros; workers encerram com rc=0.

```bash
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.orquestrador_extracao \
    --manifest <manifest.json> --output-dir dados_brutos_orquestrador \
    --corpus-output corpus_orquestrador.ndjson --limite 50 --verbose
# (--gpu-paddle 0 --gpu-docling 1 são o padrão "auto" com 2 GPUs)
```

> Observação de tuning: a heurística `is_complex` (keywords fiscais em `PALAVRAS_TABELA`) é
> agressiva — palavras comuns como "contrato/despesa/anexo" fazem a maioria dos documentos cair
> na rota **Docling** (lenta, `do_ocr=True`). Vale calibrar para reservar Docling a documentos
> com tabelas reais e mandar o restante para PaddleOCR/PyMuPDF.

---

## 3. Natureza dos dados

Amostragem de `territorios/` (13.293 PDFs): a **esmagadora maioria é PDF nativo digital**
(atos individuais já fatiados, mediana de 1 página, ~3–10 mil chars/página). PDFs
genuinamente escaneados são raros (0/400 numa amostra estrita por cobertura de imagem).

> Consequência: para a maior parte do corpus, **PyMuPDF (camada de texto) basta** — OCR pesado
> (Paddle/Docling/Marker) só é necessário na minoria escaneada e em páginas com tabelas fiscais.
> Rodar OCR/layout em PDFs nativos foi a causa provável dos estouros de memória anteriores.

---

## 4. Resultados — PDFs nativos pequenos (6× `teste_extractor`, 1–2 págs)

| engine-device | load (s) | s/pdf | RAM pico | VRAM pico | chars méd | score OCR |
|---------------|----------|-------|----------|-----------|-----------|-----------|
| docling-cpu | 14.5 | 2.9 | 2.0 GB | — | 9.671 | 0.618 |
| docling-gpu | 6.3 | 2.0 | 2.4 GB | 0.7 GB | 9.657 | 0.618 |
| paddle-cpu | 167¹ | 11.1 | **13.7 GB** | — | 4.956 | 0.683 |
| paddle-gpu | 5.1 | 1.2 | 2.5 GB | 0.5 GB | 4.954 | 0.683 |

¹ Primeira execução baixa os modelos PP-Structure; load real ≈ 5 s depois.

- **Docling lê a camada de texto nativa** e extrai ~2× mais caracteres que o PaddleOCR
  (que rasteriza e faz OCR, perdendo conteúdo) — para PDFs nativos, Docling/PyMuPDF é superior.
- **paddle-cpu consome 13.7 GB de RAM** (alocadores mkldnn/openblas + rasterização); na GPU cai para 2.5 GB.
- Em GPU, ambos são rápidos (1–2 s/pdf).

---

## 5. Teste de estresse de memória — PDF de 97 páginas / 58 MB (nativo, tabelas de PPA)

Cenário que causava OOM no Marker.

| engine-device | s/pdf | **RAM pico** | VRAM pico | chars | score |
|---------------|-------|----------|-----------|-------|-------|
| docling-cpu | 572.9 | **11.9 GB** | — | 1.160.687 | 0.633 |
| docling-gpu | 189.8 | **11.6 GB** | 1.8 GB | 1.164.481 | 0.633 |
| paddle-gpu | 52.9 | **2.8 GB** | 0.8 GB | 275.068 | 0.692 |

**Diagnóstico do estouro de memória:**
- O **Docling carrega o documento inteiro** (páginas + imagens + estruturas de célula + o
  documento montado de 1,16 M chars) → **~12 GB de RAM** num único PDF de 97 págs, e **cresce
  com o tamanho do documento**. Esse é o gatilho do OOM em edições consolidadas grandes.
- O `page_batch_size` do Docling (global, default 4) limita apenas as **imagens de página por
  lote** — não a acumulação do documento. **Não resolve sozinho.**
- **PaddleOCR é 4× mais econômico em RAM (2,8 GB) e 3,6× mais rápido**, mas extrai bem menos
  texto (OCR perde o detalhamento das tabelas que o Docling captura via camada nativa).

**Controle de memória que FUNCIONA — fatiar o PDF.** Validado com o módulo real do usuário
`worker_docling.extrair_com_docling(conv, pdf, pages=[0..5])`:

| entrada | chars | tempo | **RAM pico** |
|---------|-------|-------|----------|
| PDF inteiro (97 págs) | 1.164.481 | 189.8 s | 11.6 GB |
| Fatia de 6 páginas | 49.413 | 53.7 s | **3.4 GB** |

→ A estratégia de pré-chunking por município/página do orquestrador é o que mantém a memória
sob controle. **Recomendação: nunca enviar a edição inteira ao Docling; sempre fatiar.**

> Observação: o `worker_docling` usa `do_ocr=True` com **RapidOCR via onnxruntime (CPU)** — o
> reconhecimento de texto roda em CPU mesmo com a GPU ativa (só layout/tabela usam GPU). Para
> PDFs **nativos** isso é redundante e lento; considerar `do_ocr=False` quando a página tem
> camada de texto (score alto) e reservar `do_ocr=True` para páginas escaneadas.

---

## 6. Resultados — amostra escalonada de `territorios/` (12 PDFs, 1 por tipo de ato)

12 PDFs reais cobrindo Portaria, Decreto, Edital, Licitação, Lei, Ata, Extrato, Resolução,
Aviso, Contrato, Termo, Homologação. **0 falhas em todas as combinações.**

| engine-device | load (s) | s/pdf | RAM pico | VRAM pico | chars méd | score OCR |
|---------------|----------|-------|----------|-----------|-----------|-----------|
| docling-gpu | 6.8 | 1.9 | 2.6 GB | 1.8 GB | 15.638 | 0.626 |
| paddle-gpu | 4.9 | 1.4 | 2.9 GB | 0.8 GB | 6.642 | 0.701 |
| docling-cpu | 4.7 | 3.7 | 3.7 GB | — | 15.630 | 0.627 |
| paddle-cpu | 2.9 | **18.2** | **16.3 GB** | — | 6.642 | 0.701 |

- **Docling extrai ~2,4× mais texto** (15,6k vs 6,6k chars) lendo a camada nativa — confirmando
  que, para o corpus DOM-PI (majoritariamente nativo), Docling/PyMuPDF capturam mais conteúdo
  que o OCR por rasterização do PaddleOCR.
- **PaddleOCR tem score OCR mais alto** (0.701 vs 0.626): produz menos linhas, porém mais limpas.
- **GPU é decisiva para ambos**: paddle-cpu leva 18 s/pdf e **16,3 GB de RAM**; na GPU cai para
  1,4 s e 2,9 GB. docling-gpu é ~2× mais rápido que docling-cpu.
- Em GPU, RAM de ambos fica em ~2,6–2,9 GB para estes documentos pequenos/médios — seguro.

---

## 7. Recomendações

1. **Roteamento por necessidade** (já desenhado no orquestrador): PyMuPDF para nativo (a maioria);
   OCR só para escaneado/tabelas. Isso elimina a maior parte do custo e do risco de OOM.
2. **Fatiar sempre** antes de Docling/Paddle (mini-PDF por município/página). Cap de memória.
3. **Dois venvs + workers em subprocesso** para contornar o conflito torch×paddle-gpu.
4. **PaddleOCR-GPU** como motor OCR primário (rápido, leve em RAM); **Docling** reservado para
   páginas com tabelas onde a fidelidade estrutural compensa o custo de memória/tempo.
5. Ajustar `do_ocr` do Docling por score de página (evitar OCR redundante em nativo).

---

## 8. Como reproduzir

```bash
# Docling (CPU+GPU) — usa .venv (torch)
./.venv/bin/python -m dompi_scraper.bench_ocr \
    --pdfs "teste_extractor/*.pdf" --engines docling-cpu docling-gpu \
    --out dados_benchmark/docling.json

# PaddleOCR (CPU+GPU) — orquestrador dispara worker em .venv-paddle automaticamente
./.venv/bin/python -m dompi_scraper.bench_ocr \
    --pdfs "teste_extractor/*.pdf" --engines paddle-cpu paddle-gpu \
    --out dados_benchmark/paddle.json

# Stress de memória + anti-OOM por fatiamento
./.venv/bin/python -m dompi_scraper.bench_ocr \
    --pdfs "<PDF_grande>" --engines docling-gpu paddle-gpu --out dados_benchmark/stress.json
```

Artefatos JSON detalhados (por PDF) em `dados_benchmark/` (não versionado).

---

## 9. Resultado das otimizações (run real `tabuleiros_alto_parnaiba`)

Amostra de **200 documentos**, mesmo `output-dir` (nada apagado), `--threshold 0.45`.

| | Antes (run do usuário) | Depois (otimizado) |
|---|---|---|
| Tempo por documento | **~7.3 s/doc** | **~0.93 s/doc** (≈ **8× mais rápido**) |
| Docling por doc | 5–13 s (OCR em CPU) | ~2 s (`do_ocr=False`, GPU) |
| Dedup | só pós-extração (paga o motor) | **84/200 pulados ANTES de extrair** |
| Erros | 0 | 0 |

Distribuição do run otimizado (200 docs): PyMuPDF 26 · Docling 97 (`do_ocr=False`) ·
PaddleOCR 0 (não havia escaneados nesta amostra) · **Duplicatas 111 (84 puladas pré-extração)** ·
96 chunks salvos · 186 s total.

**Retomada / convergência do dedup** (re-executando o mesmo lote, sem apagar):

| Passada | Puladas pré-extração | Docling | Tempo |
|---|---|---|---|
| 1ª (registro vazio) | 84 | 97 | 186 s |
| 2ª | 180 | 27 | 62 s |
| 3ª | 194 | 13 | 51 s |
| 4ª | **207 / 207** | **0** | **27 s** (só triagem PyMuPDF) |

→ Após uma passada completa, re-execuções **pulam 100%** do que já foi processado, sem subir os
workers (registro `registro_dla_dedup.txt` persistido). Projeção para os 13.293 docs: de **~28 h**
para **poucas horas** (menos ainda graças à alta taxa de duplicatas das edições compartilhadas).

> Nota: nesta amostra a GPU 0 (PaddleOCR) ficou ociosa porque o corpus é todo nativo — o que está
> correto (PaddleOCR é reservado a escaneados). O motor de trabalho aqui é o Docling `do_ocr=False`
> na GPU 1. Em territórios com PDFs escaneados, a GPU 0 entra em ação automaticamente.
