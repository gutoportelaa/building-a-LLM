# Q6 — Guardrails

## Enunciado
Incluir camadas de **guardrails** em um dos modelos desenvolvidos. Avaliar com um benchmark de **30 perguntas**.
Quantificar o **grau de proteção** adicionado e analisar o trade-off Helpfulness × Harmlessness.

## Status: ⏳ Pendente

## Estratégia (resumo)
- **Biblioteca:** `guardrails-ai` (integra com HF; validators customizados)
- **Modelo a proteger:** pipeline RAG da Q5 (ou modelo SFT da Q2)
- **Camadas:** (1) mascaramento de PII (CPF/CNPJ); (2) validação de escopo (redireciona fora-de-DOM-PI);
  (3) detecção de prompt injection; (4) validação da resposta (mascara dados sensíveis)
- **Benchmark 30 perguntas:** 10 legítimas · 5 fora de escopo · 5 com PII · 5 prompt injection · 5 harmless fora de escopo (mede falso positivo)
- **Métricas:** taxa de proteção, taxa de falso positivo, latência adicionada

Scripts a criar: `scripts/guardrails_pipeline.py`.

Detalhamento: seção "Questão 6" em [`../relatorio_tecnico_completo.html`](../relatorio_tecnico_completo.html).
