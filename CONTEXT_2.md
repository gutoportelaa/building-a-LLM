# Relatório de Triagem e Planejamento para Processamento Visual (VL/OCR Estrutural)

## 1. Estatísticas e Impressões da Triagem Atual

> [!NOTE]
> **Contexto de Deduplicação Prévia**
> O volume inicial capturado no scraping do DOM-PI passava de **12.000 publicações**. Antes da etapa de limpeza, aplicamos um processo de deduplicação rigoroso para isolar arquivos únicos (usando verificações de hash e metadados cruzados), o que reduziu a base bruta consolidada para os **7.565 arquivos únicos** atuais.

Após rodarmos a versão atualizada do nosso script de limpeza com heurísticas de detecção de assinaturas e tabelas achatadas, obtivemos os seguintes resultados nos `dados_brutos`:
- **Total de documentos analisados:** 7.565
- **Documentos sinalizados para revisão (`needs_human_review: true`):** 7.172
- **Taxa de retenção para revisão:** ~94,8%

### O que isso significa?
Quase **95%** das publicações do DOM-PI (Diário Oficial dos Municípios do Piauí) não são apenas "textos corridos". O Diário é massivamente composto por:
1. Documentos fiscais e contábeis (Relatórios de Gestão Fiscal - RGF, Relatórios Resumidos de Execução Orçamentária - RREO).
2. Avisos de licitação e contratos (que contêm planilhas de itens e valores).
3. Portarias e Decretos (com assinaturas e validações CPF/CNPJ de gestores).

**A extração textual bruta (via PyMuPDF) "achata" essas tabelas**, transformando colunas estruturadas em linhas soltas de números misturados. Passar esses dados pelo modelo `nomic-embed-text` usando LangChain causaria perda severa de semântica, impossibilitando que a IA no RAG responda corretamente a perguntas complexas como *"Qual o valor do orçamento interno de Jatobá em 2024?"*.

---

## 2. Planejamento: Transição para Extração Visual/Estrutural

Para garantir que o nosso banco vetorial seja populado com dados semanticamente ricos e preservar a estrutura das tabelas do DOM-PI, precisamos substituir (ou aprimorar) a extração bruta por Modelos de Visão e Linguagem (VLM) ou pipelines de OCR estrutural.

### Opções Tecnológicas

1. **Marker / Surya (Recomendado para Volume)**:
   - Pipeline especializado que converte PDFs em Markdown de alta fidelidade.
   - Usa o Surya por baixo para detecção avançada de layout (identifica exatamente onde começa e termina uma tabela).
   - **Vantagem**: Funciona muito bem localmente (se houver GPU) e entrega o Markdown pronto para chunking.
2. **Modelos VLM (Qwen-VL, DeepSeek-VL ou LLaVA)**:
   - Podemos converter as páginas dos PDFs que falharam na triagem em imagens (PNG) e passá-las por um modelo VLM rodando via Ollama ou vLLM.
   - **Vantagem**: São excepcionais em reconstruir estruturas quebradas. Você pode instruí-los com um prompt direto: *"Transcreva a imagem em Markdown preservando as tabelas e ignorando cabeçalhos repetitivos de páginas"*.
   - **Desvantagem**: Processamento mais custoso e lento se comparado a pipelines focados em OCR.

### Novo Fluxo de Pipeline Proposto

**Fase 1: Triagem Leve e Identificação (Já Implementada)**
- Extração rápida com PyMuPDF.
- O `limpeza_textos.py` flagra os documentos problemáticos (com assinaturas, tabelas mal formatadas ou alto ruído).

**Fase 2: Processamento Profundo VLM/OCR (A Implementar)**
- Criar um **Orquestrador Secundário** que lê apenas os arquivos com a flag `needs_human_review: true`.
- Este orquestrador pega a página original do PDF daquele documento e aplica o **Marker** ou um modelo **VLM (ex: Qwen-VL)** para gerar o Markdown Estruturado (onde as tabelas tornam-se `| Coluna 1 | Coluna 2 |`).

**Fase 3: Ingestão Vetorial e RAG (A Implementar)**
- **Chunking Semântico**: Usar o `MarkdownHeaderTextSplitter` ou divisores que respeitem blocos de tabelas no Langchain.
- **Embeddings**: Inserir no ChromaDB usando o `nomic-embed-text`. Como a estrutura tabular estará preservada como texto Markdown organizado, o embedding capturará adequadamente a relação entre as colunas e os números.

## 3. Próximos Passos
1. Escolher qual abordagem usar para reconstruir os ~7.000 documentos problemáticos (Marker ou um VLM hospedado localmente/API).
2. Criar um script para converter os recortes originais do DOM-PI correspondentes aos arquivos flagados em imagens ou rodar diretamente sobre seus bytes no motor de extração estruturada.
3. Avaliar a qualidade de 10 amostras antes de reprocessar toda a base.
