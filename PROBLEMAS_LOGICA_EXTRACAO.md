# Problemas de Lógica — Etapa de Extração e Limpeza do Dataset DOM-PI

Documento de orientação técnica para revisão do pipeline. Cada problema inclui
a localização exata no código, o sintoma observável e uma proposta de correção.

---

## P-01 — Dedup global bloqueia chunking retroativo

**Arquivo:** `processar_pdfs.py`, linhas 641–651  
**Gravidade:** Alta

### Problema
`process_single_pdf()` calcula o MD5 do **PDF inteiro** (todos os blocos
concatenados) e verifica o `dedup_registry` antes de entrar no modo chunking.
Se o PDF já foi processado em modo padrão (sem `--modo-chunking`), a entrada
existe no registro e a função retorna `[]` imediatamente — nunca chega ao
`_process_chunks`. Resultado: re-executar o pipeline com `--modo-chunking`
sobre uma base já processada não separa as cidades.

```python
# processar_pdfs.py L642-651
raw_text = "\n".join(b["texto"] for b in all_blocks)
content_hash = compute_content_md5(raw_text)

if content_hash in dedup_registry:   # ← bloqueia ANTES do chunking
    ...
    return []
```

### Correção sugerida
O hash global deve ser verificado apenas no modo padrão. No modo chunking, a
deduplicação deve acontecer por chunk (já feita em `_gerar_documento`), não
pelo PDF inteiro. Mover a checagem global para depois do `if modo_chunking:`.

---

## P-02 — `extract_date_from_text()` captura a primeira data, não a de publicação

**Arquivo:** `shared_utils.py`, linhas 222–255  
**Arquivo afetado:** `processar_pdfs.py` (L705), `orquestrador_extracao.py` (L251)  
**Gravidade:** Alta

### Problema
A função pega a **primeira data encontrada** no corpo do texto. Documentos como
contratos, termos de posse e portarias com vigência frequentemente começam com
referências a datas anteriores ("conforme contrato firmado em 12/03/2023…") ou
datas de prazos futuros. O resultado é que `data_publicacao` no frontmatter e
na partição do Data Lake (`ano=`, `mes=`) fica com a data errada.

Exemplos de falso positivo:
- Portaria de 2025 referenciando Lei de 2019 → `data_publicacao: 2019-...`
- Contrato com vigência até 2026 → `data_publicacao: 2026-...`

**Nota:** O `extrair_territorio.py` mitiga isso usando `extrair_data_filename()`
(data do nome do arquivo), mas os outros módulos ainda usam o fallback textual.

### Correção sugerida
Priorizar sempre `extrair_data_filename()` quando disponível. Para o fallback
textual, buscar a data na **última ocorrência** do bloco de assinaturas (que
costuma ter "Cidade, DD de mês de AAAA") em vez da primeira no texto.

---

## P-03 — `data_publicacao` fica com apenas o ANO na partição do Data Lake

**Arquivo:** `extrair_territorio.py`, linhas 283–288  
**Gravidade:** Média

### Problema
`extrair_data_filename()` retorna somente o **ano** ("2025"), sem mês ou dia,
porque o nome do arquivo DOM-PI codifica apenas os 2 últimos dígitos do ano
(`-25_`). O registro JSONL recebe `"data_publicacao": "2025"` e o frontmatter
também. O particionamento cria `mes=sem_mes/` para todos esses documentos,
agrupando meses diferentes na mesma pasta.

```python
# extrair_territorio.py L288
data_publicacao_final = data_ano  # "2025" ou ""
```

Isso desfaz a utilidade do particionamento por mês para buscas híbridas no RAG
(filtros como `data_publicacao >= 2025-03-01` não funcionam).

### Correção sugerida
Cruzar o número de edição DOM (`DM_NNNN`) com uma tabela de mapeamento
edição→data (o portal DOM-PI expõe essa informação). Armazenar essa tabela
localmente ou enriquecer no momento do scraping (`pipeline.py`).

---

## P-04 — Falsos positivos no `detect_city_header()` para layout 4-em-1

**Arquivo:** `processar_pdfs.py`, linhas 264–276  
**Gravidade:** Média

### Problema
Para lidar com PDFs que comprimem 4 páginas em 1 (fontes de 5–8pt), a função
inclui a condição `is_4_in_1_size = 5.0 <= tamanho <= 8.0` que aceita qualquer
texto pequeno que combine com o regex `RE_CABECALHO_ENTIDADE`.

O regex é amplo:
```python
RE_CABECALHO_ENTIDADE = re.compile(
    r"(?:PREFEITURA|C[AÂ]MARA)\s+(?:MUNICIPAL\s+)?(?:DE\s+)?([A-ZÀ-Ÿ][A-ZÀ-Ÿ\s]{2,39})..."
)
```
Qualquer frase pequena contendo "PREFEITURA DE" no corpo do texto (ex: rodapés,
cabeçalhos repetidos de página) pode ser detectada como fronteira de cidade,
gerando chunks "DESCONHECIDO" ou fragmentando incorretamente o documento.

### Correção sugerida
Para o intervalo 4-em-1, exigir também que o bloco esteja no topo da página
(usar o campo `bbox` — `bbox[1]` pequeno indica posição vertical alta). Isso
filtra rodapés e referências internas.

---

## P-05 — `split_blocks_by_city()` não rastreia número de página por bloco

**Arquivo:** `processar_pdfs.py`, linhas 279–302  
**Gravidade:** Média

### Problema
`split_blocks_by_city()` recebe uma lista plana de blocos de todas as páginas
do PDF. Cada bloco não carrega a informação de qual página pertence. Consequências:

1. O `generate_frontmatter()` aceita o campo `paginas`, mas `_process_chunks`
   nunca o passa (sempre fica `None` no frontmatter).
2. O `orquestrador_extracao.py` precisaria saber quais páginas pertencem a qual
   cidade para criar o mini-PDF antes de enviar ao Marker. Ele reimplementa
   essa lógica separadamente em `analisar_e_fatiar_pdf()`, criando divergência.

### Correção sugerida
Adicionar `"pagina": page_num` ao dicionário de cada bloco em
`extract_rich_blocks()`. `split_blocks_by_city()` então pode derivar o intervalo
de páginas de cada chunk e populá-lo em `generate_frontmatter(paginas=...)`.

---

## P-06 — `_process_chunks` herda `data_iso/ano/mes` do manifesto, não do chunk

**Arquivo:** `processar_pdfs.py`, linhas 748–786  
**Gravidade:** Média

### Problema
Em PDFs consolidados com múltiplos municípios, todos os chunks recebem a mesma
`data_iso`, `ano` e `mes` herdados dos metadados do manifesto (que representam
o PDF inteiro). Se a data no manifesto está errada ou ausente (campo vazio),
todos os chunks do PDF ficam com data errada, e a partição do Data Lake fica
incorreta para todos eles.

```python
# _process_chunks L782
recs = _gerar_documento(blocos_chunk, len(all_blocks_raw), cidade, entidade,
                        edicao, sha256_pdf, url_origem, documento,
                        data_iso, ano, mes, ...)  # ← mesmo data para todos
```

### Correção sugerida
Dentro do loop de chunks, tentar `extract_date_from_text()` no texto do chunk
individual como refinamento, não usar cegamente a data do manifesto pai.

---

## P-07 — `orquestrador_extracao.py` não mantém registro de deduplicação

**Arquivo:** `orquestrador_extracao.py`, linhas 146–279  
**Gravidade:** Média

### Problema
O orquestrador usa `compute_content_md5` para nomear o arquivo `.md` de saída,
mas não mantém nenhum `dedup_registry` persistente entre execuções. Se o mesmo
PDF for re-processado (retomada por falha, por exemplo), o arquivo `.md` é
sobrescrito em disco (sem erro), mas o JSONL de saída acumula entradas
duplicadas porque não há verificação antes de escrever.

O `processar_pdfs.py` tem esse controle via `registro_dedup.json`. O
orquestrador não herdou esse mecanismo.

### Correção sugerida
Carregar/salvar um `registro_dedup_orquestrador.json` no `output_dir`, com
verificação antes de escrever cada chunk — igual ao padrão do `processar_pdfs.py`.

---

## P-08 — `RE_SIGNATURE` marca ~95% dos documentos para revisão humana

**Arquivo:** `limpeza_textos.py`, linha 44  
**Gravidade:** Média (impacto operacional)

### Problema
O regex detecta "PREFEITO", "SECRETÁRIO" e CPF/CNPJ como indicador de
assinatura, e todo documento com assinatura recebe `needs_human_review: true`.
Praticamente 100% dos documentos oficiais têm assinatura do prefeito — logo
a flag perdeu poder discriminatório (94.8% marcados, conforme CONTEXT_2.md).

A flag foi pensada para capturar documentos com **tabelas achatadas** e
**estrutura quebrada**, não documentos apenas com assinaturas.

### Correção sugerida
Separar as razões em severidades distintas:
- `assinaturas_detectadas` → informativa apenas, não leva a `needs_human_review`
- `tabela_achatada_detectada` → leva a `needs_human_review: true`
- `alto_indice_ruido_ocr` → leva a `needs_human_review: true`

---

## P-09 — `limpeza_textos.py` não re-calcula o hash de deduplicação após limpeza

**Arquivo:** `limpeza_textos.py`, linhas 133–189  
**Gravidade:** Baixa

### Problema
Após aplicar as transformações de limpeza (remoção de lixo OCR, junção de
letras isoladas, etc.), o conteúdo do `.md` muda. O `id_publicacao` no
frontmatter (MD5 do conteúdo original) fica desatualizado. Se dois documentos
que eram ligeiramente diferentes pré-limpeza se tornarem idênticos pós-limpeza,
o `id_publicacao` continuará diferente — gerando duplicatas no corpus limpo.

### Correção sugerida
Após limpar o corpo, recalcular `compute_content_md5(cleaned_body)` e atualizar
o campo `id_publicacao` no frontmatter antes de salvar em `dados_limpos/`.

---

## P-10 — `build_manifest_from_pdfs` usa `pdf_path.stem` como chave do manifesto

**Arquivo:** `extrair_territorio.py`, linhas 115–153  
**Gravidade:** Baixa

### Problema
O ID no manifesto é o **nome do arquivo sem extensão** (`fid = pdf_path.stem`).
Se dois PDFs em pastas de cidades diferentes têm o mesmo nome (o DOM-PI
reutiliza nomes como `DM_5234_001_...` para municípios diferentes dentro da
mesma edição), o segundo sobrescreve o primeiro no dicionário `manifest` sem
aviso, perdendo um dos arquivos silenciosamente.

```python
# extrair_territorio.py L118
fid = pdf_path.stem  # ← colisão possível
```

### Correção sugerida
Usar o SHA-256 do arquivo como chave (já calculado em `sha256_file()`), ou usar
o caminho relativo completo (`str(rel_path)`) que é garantidamente único.

---

## P-11 — `classify_act_type()` analisa apenas os primeiros 1000 chars do PDF inteiro

**Arquivo:** `shared_utils.py`, linha 131  
**Gravidade:** Baixa (mitigada pelo modo chunking)

### Problema
No modo padrão (sem chunking), `classify_act_type` recebe o texto concatenado
de todas as páginas do PDF, mas inspeciona apenas os primeiros 1000 chars:

```python
snippet = (text or "")[:1000]
```

Para PDFs consolidados multi-município, os 1000 primeiros caracteres são
tipicamente o cabeçalho do diário + o primeiro município. O tipo do segundo,
terceiro e demais municípios nunca é detectado. Todo o PDF fica classificado
com o tipo do ato que aparece primeiro.

No modo chunking isso é mitigado (cada chunk tem seu próprio `raw_text`), mas
o limite de 1000 chars pode ainda ser insuficiente para alguns documentos longos.

### Correção sugerida
No modo padrão, aumentar o snippet para 2000-3000 chars. Considerar também
buscar o tipo ao longo de TODO o texto quando o snippet retorna `"Não Identificado"`.

---

## Resumo Priorizado

| # | Problema | Gravidade | Esforço | Status |
|---|---|---|---|---|
| P-01 | Dedup global bloqueia chunking retroativo | Alta | Baixo | ✅ Corrigido |
| P-02 | Data captura referência, não publicação | Alta | Médio | ✅ Corrigido |
| P-03 | Data com apenas o ano no Data Lake | Média | Alto | 🔲 Pendente |
| P-04 | Falsos positivos no detect_city_header (4-em-1) | Média | Médio | ✅ Corrigido |
| P-05 | Número de página ausente nos blocos/chunks | Média | Médio | ✅ Corrigido |
| P-06 | Todos os chunks herdam data do manifesto pai | Média | Baixo | ✅ Corrigido |
| P-07 | Orquestrador sem deduplicação persistente | Média | Baixo | ✅ Corrigido |
| P-08 | RE_SIGNATURE satura needs_human_review | Média | Baixo | ✅ Corrigido |
| P-09 | Hash desatualizado após limpeza | Baixa | Baixo | ✅ Corrigido |
| P-10 | Colisão de chave no manifesto (stem) | Baixa | Baixo | ✅ Corrigido |
| P-11 | classify_act_type só olha 1000 chars | Baixa | Baixo | ✅ Corrigido |

## P-03 — Pendente: enriquecimento de data por mapeamento edição→data

A correção requer cruzar o número de edição DOM (`DM_NNNN`) com uma tabela
`{edicao: data_completa}` que não existe ainda localmente.

**Abordagem recomendada:** enriquecer o scraping no `pipeline.py` para gravar
a data do portal junto com a edição. O portal DOM-PI exibe a data de cada edição
na página de busca — capturá-la no momento do scraping é o caminho mais limpo.
