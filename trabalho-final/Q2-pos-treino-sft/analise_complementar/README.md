# Q2 — Análise complementar: o mesmo SFT sob perguntas de outra natureza

experimento complementar da Q2 **integrado a este repositório para reanálise**: SFT LoRA da mesma base
(`Qwen2.5-1.5B-Instruct`, 3 épocas, ~1.317 pares) avaliado com **perguntas de conhecimento geral de CC**
(QuickSort, TCP/UDP, HTTP…), em contraste com as nossas 30 perguntas ancoradas no corpus docentesDC.

- `dados_brutos/` — artefatos originais da execução (benchmark, avaliações antes/depois,
  `trainer_state.json` do treino, comparações por modelo). Execução conduzida pelo grupo de
  Pedro E. M. Carvalho ([repositório-fonte](https://github.com/PedroEmanuelMoreiraCarvalho/Trabalho_Final_IA_Fine_Tuning)).
- `analisar_complementar.py` — a nossa reanálise: agrega similaridade/comprimento antes×depois por rodada
  e extrai a curva train×eval loss por época.
- `figuras/` — saídas no padrão visual do relatório:
  - `q2c_curva_epocas.png` — validação satura após ~1,3 época (evidência da hipótese H3);
  - `q2c_sim_compressao.png` — rodada 3 replica a compressão de resposta com queda de qualidade (nosso achado central da §2.6).

Uso no relatório consolidado: `../../relatorio_apresentacao.html`, §2.10.

Reproduzir: `../../../.venv/bin/python3 analisar_complementar.py`
