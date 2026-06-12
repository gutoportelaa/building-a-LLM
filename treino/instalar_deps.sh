#!/usr/bin/env bash
# Instala dependências de treino/avaliação no venv principal via uv pip.
# Executar uma vez antes de usar os scripts de treino/avaliação.
set -e

echo ">>> Instalando transformers, datasets, accelerate, peft..."
# huggingface-hub é fixado em <1.0 para compatibilidade com transformers 4.57.x
uv pip install "transformers>=4.51" datasets accelerate peft

echo ">>> Verificando compatibilidade..."
.venv/bin/python3 -c "
import torch
print(f'torch {torch.__version__} | cuda: {torch.cuda.is_available()}')
import transformers; print(f'transformers {transformers.__version__}')
import datasets; print(f'datasets {datasets.__version__}')
import accelerate; print(f'accelerate {accelerate.__version__}')
import peft; print(f'peft {peft.__version__}')
"

echo ""
echo ">>> Pronto. Fluxo completo de treino + avaliação:"
echo ""
echo "  1. Separar held-out:"
echo "     .venv/bin/python3 avaliacao/preparar_held_out.py"
echo ""
echo "  2. Avaliar baseline (ANTES do treino):"
echo "     .venv/bin/python3 avaliacao/avaliar_modelo.py \\"
echo "       --model Qwen/Qwen2.5-0.5B \\"
echo "       --held-out data/held_out.jsonl \\"
echo "       --output avaliacao/resultados_baseline.json"
echo ""
echo "  3. Pré-treino continuado:"
echo "     .venv/bin/python3 treino/pretreino_continuado.py \\"
echo "       --model Qwen/Qwen2.5-0.5B \\"
echo "       --train-data data/train_corpus.jsonl \\"
echo "       --output-dir treino/checkpoints"
echo ""
echo "  4. Avaliar modelo treinado (DEPOIS):"
echo "     .venv/bin/python3 avaliacao/avaliar_modelo.py \\"
echo "       --model treino/checkpoints/final \\"
echo "       --held-out data/held_out.jsonl \\"
echo "       --output avaliacao/resultados_postreino.json"
echo ""
echo "  5. Relatório antes × depois:"
echo "     .venv/bin/python3 avaliacao/comparar_resultados.py \\"
echo "       --baseline avaliacao/resultados_baseline.json \\"
echo "       --postreino avaliacao/resultados_postreino.json"
