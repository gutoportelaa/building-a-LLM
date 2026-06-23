# Validação Metodológica — Trabalho Final DOM-PI

Avaliação crítica do que foi feito, das etapas e dos resultados, cruzando com as fontes usadas
(principalmente **Sebastian Raschka — _Build a Large Language Model (From Scratch)_**) e literatura adicional.

## Veredito geral
As decisões metodológicas e as conclusões (após a correção do bug) estão **alinhadas com as fontes**. O ponto mais
importante: o **bug do duplo deslocamento de rótulos** e sua correção são corroborados pela documentação do
HuggingFace e pela convenção de Raschka; e a conclusão corrigida da Q1 (**full FT > PEFT em pré-treino continuado**)
coincide com o resultado canônico da literatura (_LoRA Learns Less and Forgets Less_).

---

## 1. Objetivo de treino / duplo deslocamento — ✅ VALIDADO
- **Raschka (cap. 5):** o `GPTDatasetV1` cria pares já deslocados (`input = tokens[i:i+n]`, `target = tokens[i+1:i+n+1]`)
  e o `calc_loss_batch` aplica `cross_entropy(logits, target)` **sem deslocamento adicional** — porque o `GPTModel`
  dele **não desloca internamente**.
- **HuggingFace CausalLM** (GPT-2/Llama/Qwen) faz o oposto: ao receber `labels`, desloca internamente
  (`shift_logits = logits[:-1]`, `shift_labels = labels[1:]`), **assumindo que os labels NÃO foram pré-deslocados**
  (issues HF #32944, #35066, fórum HF).
- **Nosso bug:** importamos a convenção de Raschka (pré-deslocar) para dentro de um modelo HF (que também desloca) →
  **duplo shift** → objetivo "prever 2 tokens à frente", loss ~11, PPL interna ~47.000. A correção (`labels =
  input_ids.clone()`) é exatamente o que a convenção HF exige. **Diagnóstico e correção corretos.**
- **Lição validada:** a métrica de sanidade (baseline interno PPL ~10, não ~47.000) teria exposto o bug cedo.

## 2. Full FT vs LoRA/QLoRA em pré-treino continuado — ✅ VALIDADO
- **_LoRA Learns Less and Forgets Less_** (Biderman et al., TMLR 2024): LoRA **subdesempenha** o full FT em
  **continued pretraining** (a lacuna não fecha nem com rank alto); em contrapartida, LoRA **esquece menos / regulariza**
  (preserva o modelo base). Recomendam LoRA para **instruction finetuning**, não para CPT.
- **Coincide com o nosso resultado corrigido:** full FT unificado **−11,3%** > QLoRA **−1,4%**; e o QLoRA é o mais
  "conservador/robusto" (muda pouco o modelo). O resultado **bugado** (QLoRA "vencendo") **contradizia** a literatura —
  outra confirmação de que era artefato do bug.
- **Implicação para Q2/Q3:** usar LoRA/QLoRA para **SFT** (Q3) é apropriado segundo a mesma fonte.

## 3. LR, scheduler e otimizador — ✅ VALIDADO
- **Raschka (Apêndice D):** warmup linear + cosine decay de meio-ciclo — exatamente o nosso scheduler manual.
- **lr conservador** (2e-6 full FT / 2e-5 PEFT): coerente com guias de CPT (lr ~2e-5 PEFT; menor para full FT em
  corpus pequeno/ruidoso). AdamW 8-bit (bitsandbytes) é padrão para caber 1.5B no L4.

## 4. Colapso do LoRA (PPL ×100) — ✅ CONSISTENTE
- Atribuído a "intruder dimensions" (Shuttleworth et al., 2024) + lr alto + corpus pequeno. Consistente com a
  literatura de instabilidade de LoRA em CPT. (Sob o objetivo corrigido o quadro melhora, mas LoRA segue o mais frágil.)

## 5. Métricas (CE, PPL, token-accuracy) — ✅ ADEQUADAS
- São exatamente as métricas de pré-treino de Raschka (cap. 5: cross-entropy e perplexidade) + token-accuracy. O
  benchmark Q&A por NLL da resposta condicionada ao prompt é uma escolha defensável (avalia preferência sem geração).

## 6. DAPT em corpus pequeno / qualidade > quantidade — ✅ CONSISTENTE
- ~50 M tokens é pequeno para DAPT, mas a literatura mostra DAPT efetivo em 50–119 M tokens com **corpus curado**
  (Juru; "DAPT com tokens mínimos"). Nosso achado (Teresina curado e, após correção, o unificado também melhoram)
  é coerente. A degradação do corpus ruidoso era amplificada pelo bug, não só pelo ruído.

## 7. RAG (Q5) — ✅ CONSISTENTE
- Standard/HyDE/Self-reflective(Self-RAG)/Agêntico(ReAct) são técnicas canônicas. Nosso achado de que **HyDE nem sempre
  ajuda** é reportado na literatura (depende da qualidade do documento hipotético). Ganho escalando com a capacidade do
  gerador e gargalo no recall são esperados.

## 8. Guardrails (Q6) — ✅ CONSISTENTE
- Defesa em camadas (entrada + saída), PII por regra determinística, juiz LLM para escopo/injection/groundedness, e a
  análise do **trade-off Helpfulness × Harmlessness** seguem as boas práticas (NeMo Guardrails, Llama Guard,
  guardrails-ai). O achado de que o groundedness over-recusa perguntas conceituais é um trade-off real e bem documentado.

---

## Pontos de atenção / limitações (honestas)
1. **Held-outs diferentes** entre unificado (PPL baseline 10,03) e Teresina (6,91): os Δ% são comparáveis, mas os
   valores absolutos não são diretamente (corpora distintos). Já está sinalizado no relatório.
2. **QLoRA não tem early stopping por held-out** (o `pretreino_lora.py` treina 1 época e avalia depois) — o QLoRA
   corrigido (−1,4%) pode ser um leve limite inferior.
3. **Benchmarks pequenos** (49 Q&A na Q1, 30 na Q6): adequados ao escopo do trabalho, mas as taxas têm variância;
   conclusões qualitativas são mais robustas que os pontos percentuais exatos.
4. **Juiz LLM nos guardrails** (qwen2.5:14b): a triagem é confiável no smoke test, mas um juiz pode errar; em produção,
   convém calibrar limiares e/ou combinar com classificador dedicado (ex.: Llama Guard).
5. **v2 do Teresina** (freeze+3 épocas) ficou registrado mesmo tendo piorado — útil como evidência do impacto do bug.

## Conclusão
O conjunto está **metodologicamente sólido e bem fundamentado**. A história científica é honesta (inclui o bug e a
inversão de conclusão que ele causava) e as conclusões finais batem com as fontes de referência. Recomenda-se manter
as limitações acima explícitas nos relatórios (já estão).

## Fontes
- Sebastian Raschka — *Build a Large Language Model (From Scratch)*, cap. 5 (treino, CE, PPL) e Apêndice D (warmup+cosine); repo `rasbt/LLMs-from-scratch`.
- HuggingFace Transformers — deslocamento interno de labels em CausalLM (issues #32944, #35066; fórum HF).
- Biderman et al. (2024), *LoRA Learns Less and Forgets Less*, TMLR (arXiv:2405.09673).
- Shuttleworth et al. (2024), *intruder dimensions* em LoRA (arXiv:2410.21228).
- Juru (arXiv:2403.18140); DAPT com tokens mínimos (arXiv:2507.02964); HyDE; Self-RAG; ReAct; NeMo Guardrails / Llama Guard.
