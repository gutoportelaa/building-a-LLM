# Runbook Q2/Q3 — SFT full × LoRA/QLoRA (docentesDC)

Pipeline único e controlado: os mesmos pares e o mesmo benchmark alimentam Q2
(full FT) e Q3 (LoRA/QLoRA). Toda a engenharia foi validada localmente (RTX 4070);
só a execução pesada roda no cluster (L4 24 GB).

## Pré-requisitos no cluster
- Projeto em `~/building-a-LLM`.
- Venvs já existentes da Q4: `.venv-q4gen` (vLLM, professor 14B) e `.venv` (transformers/peft).
- Acesso ao HF (token já configurado) para baixar `vickminari/docentesDC` e os Qwen2.5.

## Ordem de execução (3 jobs encadeados)

```bash
cd ~/building-a-LLM
SCR=trabalho-final/Q2-pos-treino-sft/scripts

# 1) Geração dos pares (professor 14B, vLLM, gpunode01 2×L4) — ~2-4 h
#    Faz PILOTO (100, sem juiz) e depois o LOTE CHEIO (>=1000 limpos, com juiz).
sbatch $SCR/run_q2_gerar.sbatch

# 2) Treino (full/lora/qlora × 1.5B/0.5B) + avaliação antes×depois — ~3-6 h, 1 L4
sbatch $SCR/run_q2q3_treino_aval.sbatch

# 3) LLM-as-judge (14B, vLLM) sobre todas as gerações — ~1 h, 2×L4
sbatch $SCR/run_q2q3_juiz.sbatch

# 4) Consolidação (offline, .venv) — tabelas + figuras p/ o relatório
.venv/bin/python $SCR/consolidar_q2q3.py
```

## Saídas
- `dados/pares.jsonl`, `pares_train.jsonl`, `pares_heldout.jsonl`, `stats.json`
- `modelos/sft_{full,lora,qlora}_{1.5b,0.5b}/` (+ `treino_meta.json` com params/VRAM/tempo)
- `resultados/heldout_*.json` (PPL/CE/tok-acc), `bench_*.json` (EM/contains/F1 + gerações completas),
  `juiz_*.json` (nota 1-5)
- `resultados/resumo_q2q3.md` (tabelas), `resultados/fig_*.png`

## Validação já feita localmente (sem cluster)
- `gerar_pares_docentes.py`: chunking, parsing JSON robusto, validação, roteamento de tipos — OK em dados reais.
- `sft_docentes.py`: full/lora/qlora treinam com **loss inicial ~1.9–2.1** (sanidade contra o bug de
  duplo-shift da Q1: seria ~11). Adapter+merge salvos. full=100% params, lora=1.75%, qlora=2.72%.
- `avaliar_sft.py`: modos intrínseca (CE/PPL/tok-acc) e geração (EM/contains/F1 + geração completa) — OK.
- `consolidar_q2q3.py`: gera tabelas/figuras tolerando arquivos ausentes — OK.

## Smoke rápido no cluster (opcional, antes do lote cheio)
```bash
# valida o pipeline inteiro com poucos pares antes de gastar horas
.venv/bin/python $SCR/sft_docentes.py --method full --model Qwen/Qwen2.5-0.5B-Instruct \
    --train dados/piloto/pares.jsonl --out-dir modelos/_smoke --epochs 1 --grad-accum 4
```
