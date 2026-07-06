#!/usr/bin/env python3
"""
gerar_pares_docentes.py — Geração dos pares instruction/input/output da Q2 (SFT).

Estratégia: *grounded self-instruct*. Cada par nasce de um CHUNK real do
dataset `vickminari/docentesDC` (Ciência da Computação / UFPI) — o professor
LLM (Qwen2.5-14B-Instruct, servido com vLLM no cluster, idêntico à Q4) lê o
chunk e devolve UM par {instruction, input, output} ancorado naquele texto.
Isso reduz a alucinação típica do self-instruct puro (Alpaca), porque a
resposta tem de estar contida/derivável do material da disciplina.

Distribuição de tipos (alvo): explicação 30% · factual 25% · código 20% ·
resumo 15% · comparação 10%. Chunks com cara de código são roteados para o
tipo "código" preferencialmente.

Controle de qualidade (3 camadas):
  1. parsing/validação estrutural (JSON, campos não-vazios, tamanhos)
  2. dedup de instruções quase-idênticas (normalização)
  3. juiz LLM (mesmo professor) pontua 1-5 (utilidade+correção+ancoragem);
     descarta < --min-score.

Saídas (em --out-dir):
  pares_raw.jsonl       — tudo que o professor gerou (auditoria)
  pares.jsonl           — pares limpos e aprovados (>= --n-pairs)
  pares_train.jsonl     — split de treino (1-val_frac)
  pares_heldout.jsonl   — split held-out (val_frac, p/ PPL/CE antes×depois)
  stats.json            — contagens por tipo, taxas de descarte, distribuição de score

Roda NO CLUSTER (gpunode01, 2× L4, venv .venv-q4gen com vLLM). Ver run_q2_gerar.sbatch.

Exemplo (lote-piloto p/ validar prompt):
  python gerar_pares_docentes.py --teacher Qwen/Qwen2.5-14B-Instruct --tp 2 \
      --n-pairs 100 --pilot --out-dir ../dados

Exemplo (lote cheio):
  python gerar_pares_docentes.py --teacher Qwen/Qwen2.5-14B-Instruct --tp 2 \
      --n-pairs 1500 --out-dir ../dados
"""
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------- #
# Tipos e distribuição alvo                                                    #
# --------------------------------------------------------------------------- #

TIPOS = ["explicacao", "factual", "codigo", "resumo", "comparacao"]
DIST_ALVO = {"explicacao": 0.30, "factual": 0.25, "codigo": 0.20, "resumo": 0.15, "comparacao": 0.10}

# Instrução-tarefa por tipo. O professor recebe o CHUNK e deve produzir um par
# {instruction, input, output} ancorado nele, no estilo Alpaca.
TASK_POR_TIPO = {
    "explicacao": (
        "Crie uma pergunta que peça a EXPLICAÇÃO de um conceito de Ciência da Computação "
        "presente na passagem (como/por que funciona). A resposta deve explicar com clareza "
        "didática, usando apenas o que está na passagem."
    ),
    "factual": (
        "Crie uma pergunta FACTUAL e objetiva cuja resposta (um nome, definição, valor, "
        "propriedade, característica) esteja contida na passagem. A resposta deve ser curta e direta."
    ),
    "codigo": (
        "Crie uma instrução sobre o CÓDIGO ou estrutura de dados/algoritmo da passagem "
        "(o que o trecho faz, qual a saída, como implementar/corrigir/usar). A resposta deve "
        "explicar ou produzir o código, fiel à passagem."
    ),
    "resumo": (
        "Crie uma instrução pedindo um RESUMO dos pontos principais da passagem. "
        "A resposta deve sintetizar as ideias centrais em poucas linhas."
    ),
    "comparacao": (
        "Crie uma pergunta de COMPARAÇÃO entre dois conceitos/estruturas/abordagens mencionados "
        "na passagem (semelhanças, diferenças, vantagens). A resposta deve contrastá-los com base na passagem."
    ),
}

GEN_SYSTEM = (
    "Você é um professor de Ciência da Computação da UFPI criando exercícios de "
    "pergunta-e-resposta para treinar um assistente. A partir de uma PASSAGEM do material "
    "da disciplina, você gera UM par instrução/resposta ANCORADO na passagem.\n"
    "Regras: (1) escreva em português; (2) a resposta deve ser derivável da passagem, sem "
    "inventar fatos externos; (3) a pergunta deve fazer sentido SOZINHA, sem dizer 'segundo a "
    "passagem'; (4) responda APENAS com um objeto JSON válido com as chaves "
    "\"instruction\", \"input\", \"output\". Use \"input\" vazio (\"\") salvo quando um trecho de "
    "código/dado for necessário para a tarefa."
)

JUDGE_SYSTEM = (
    "Você avalia a QUALIDADE de um par instrução/resposta de Ciência da Computação. "
    "Dê uma nota inteira de 1 a 5 considerando: a resposta está correta e bem-formada? "
    "responde de fato à instrução? está ancorada na passagem (sem alucinação)? "
    "Responda APENAS com o número (1 a 5)."
)


# --------------------------------------------------------------------------- #
# Carregamento e chunking                                                      #
# --------------------------------------------------------------------------- #

def carregar_chunks(seed: int, min_chars: int, chunk_lo: int, chunk_hi: int,
                    max_chunks: int) -> list[str]:
    """Carrega docentesDC, filtra e quebra `text` em janelas de chunk_lo..chunk_hi chars."""
    from datasets import load_dataset

    ds = load_dataset("vickminari/docentesDC", split="train")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)

    chunks: list[str] = []
    vistos: set[str] = set()
    for i in idx:
        txt = (ds[i].get("text") or "").strip()
        if len(txt) < min_chars:
            continue
        # quebra em janelas respeitando limites de linha/sentença quando possível
        for ch in _split_chunks(txt, chunk_lo, chunk_hi):
            key = _norm(ch)[:200]
            if key in vistos:
                continue
            vistos.add(key)
            chunks.append(ch)
            if len(chunks) >= max_chunks:
                rng.shuffle(chunks)
                return chunks
    rng.shuffle(chunks)
    return chunks


def _split_chunks(txt: str, lo: int, hi: int) -> list[str]:
    """Divide em pedaços de ~lo..hi chars, cortando em quebra de linha/ponto quando der."""
    if len(txt) <= hi:
        return [txt]
    out, start = [], 0
    n = len(txt)
    while start < n:
        end = min(start + hi, n)
        if end < n:
            # tenta recuar até um separador natural dentro da janela [lo, hi]
            janela = txt[start + lo:end]
            corte = max(janela.rfind("\n"), janela.rfind(". "), janela.rfind("; "))
            if corte > 0:
                end = start + lo + corte + 1
        ch = txt[start:end].strip()
        if len(ch) >= lo // 2:
            out.append(ch)
        start = end
    return out


CODE_HINT = re.compile(r"(#include|\bvoid\b|\bint main\b|\bdef \w+\(|\{|\};|printf|System\.out|public class|->|==)")


def parece_codigo(ch: str) -> bool:
    return len(CODE_HINT.findall(ch)) >= 2


# --------------------------------------------------------------------------- #
# Normalização / validação                                                     #
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    return s


def extrair_json(texto: str) -> dict | None:
    """Extrai o primeiro objeto JSON da saída do modelo (tolerante a cercas ```json)."""
    texto = texto.strip()
    texto = re.sub(r"^```(json)?", "", texto).strip()
    texto = re.sub(r"```$", "", texto).strip()
    m = re.search(r"\{.*\}", texto, re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # conserto leve: aspas curvas, vírgula sobrando
        blob2 = blob.replace("“", '"').replace("”", '"').replace("\n", " ")
        blob2 = re.sub(r",\s*}", "}", blob2)
        try:
            return json.loads(blob2)
        except json.JSONDecodeError:
            return None


def validar_par(par: dict, chunk: str) -> tuple[bool, str]:
    if not isinstance(par, dict):
        return False, "nao_dict"
    ins = (par.get("instruction") or "").strip()
    out = (par.get("output") or "").strip()
    inp = (par.get("input") or "").strip()
    if len(ins) < 10:
        return False, "instrucao_curta"
    if len(out) < 15:
        return False, "resposta_curta"
    if len(out) > 2000:
        return False, "resposta_longa"
    if _norm(ins) == _norm(out):
        return False, "ins_igual_out"
    # heurística anti-eco: a instrução não pode ser uma cópia literal do chunk
    if _norm(ins) in _norm(chunk) and len(ins) > 60:
        return False, "instrucao_e_copia"
    par["instruction"], par["output"], par["input"] = ins, out, inp
    return True, "ok"


# --------------------------------------------------------------------------- #
# vLLM helpers                                                                 #
# --------------------------------------------------------------------------- #

def montar_chat(tok, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera pares instruction/input/output (Q2 SFT)")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--n-pairs", type=int, default=1500, help="alvo de pares LIMPOS")
    ap.add_argument("--oversample", type=float, default=1.6,
                    help="fator de sobre-geração p/ absorver descartes")
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--chunk-lo", type=int, default=800)
    ap.add_argument("--chunk-hi", type=int, default=1200)
    ap.add_argument("--min-score", type=int, default=3, help="nota mínima do juiz (1-5)")
    ap.add_argument("--no-judge", action="store_true", help="pula a camada de juiz LLM")
    ap.add_argument("--pilot", action="store_true", help="lote pequeno p/ validar prompt (sem juiz)")
    ap.add_argument("--out-dir", default="trabalho-final/Q2-pos-treino-sft/dados")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--quantization", default=None,
                    help="quantização do vLLM (ex.: 'awq' p/ rodar o 14B-AWQ em 1 L4)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.pilot:
        args.no_judge = True

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ---- 1. chunks + roteamento de tipos -----------------------------------
    n_gen = int(args.n_pairs * args.oversample)
    print(f"[1/4] Carregando e fatiando docentesDC (alvo {n_gen} chunks)...", flush=True)
    chunks = carregar_chunks(args.seed, args.min_chars, args.chunk_lo, args.chunk_hi,
                             max_chunks=n_gen * 2)
    if not chunks:
        raise SystemExit("Nenhum chunk carregado — verifique acesso ao dataset.")

    # roteia chunks para tipos respeitando a distribuição alvo; código vai p/ chunks-código
    cota = {t: int(round(DIST_ALVO[t] * n_gen)) for t in TIPOS}
    cod_chunks = [c for c in chunks if parece_codigo(c)]
    txt_chunks = [c for c in chunks if not parece_codigo(c)]
    rng.shuffle(cod_chunks); rng.shuffle(txt_chunks)

    plano: list[tuple[str, str]] = []  # (tipo, chunk)
    # código primeiro, dos chunks-código
    for _ in range(min(cota["codigo"], len(cod_chunks))):
        plano.append(("codigo", cod_chunks.pop()))
    # restante dos tipos, dos chunks de texto (sobra de código vira texto)
    pool = txt_chunks + cod_chunks
    rng.shuffle(pool)
    for t in ["explicacao", "factual", "resumo", "comparacao"]:
        for _ in range(cota[t]):
            if pool:
                plano.append((t, pool.pop()))
    rng.shuffle(plano)
    print(f"      {len(plano)} tarefas planejadas: "
          f"{dict(Counter(t for t, _ in plano))}", flush=True)

    if args.pilot:
        plano = plano[:args.n_pairs]
        print(f"      [PILOTO] reduzido para {len(plano)} tarefas", flush=True)

    # ---- 2. vLLM: gerar pares ---------------------------------------------
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"[2/4] Subindo professor {args.teacher} (TP={args.tp}) no vLLM...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    llm_kwargs = dict(model=args.teacher, tensor_parallel_size=args.tp,
                      gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len,
                      enforce_eager=True, trust_remote_code=True, seed=args.seed)
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization  # AWQ traz seu próprio dtype
    else:
        llm_kwargs["dtype"] = "bfloat16"
    llm = LLM(**llm_kwargs)

    prompts = [
        montar_chat(tok, GEN_SYSTEM,
                    f"TAREFA: {TASK_POR_TIPO[t]}\n\nPASSAGEM:\n{ch}\n\nResponda só com o JSON:")
        for t, ch in plano
    ]
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=args.max_new_tokens, seed=args.seed)
    outs = llm.generate(prompts, sp)

    raw_f = open(out / "pares_raw.jsonl", "w", encoding="utf-8")
    cand: list[dict] = []
    descartes = Counter()
    for (tipo, ch), o in zip(plano, outs):
        gen = o.outputs[0].text
        par = extrair_json(gen)
        raw_f.write(json.dumps({"tipo": tipo, "raw": gen[:2000]}, ensure_ascii=False) + "\n")
        if par is None:
            descartes["json_invalido"] += 1
            continue
        ok, motivo = validar_par(par, ch)
        if not ok:
            descartes[motivo] += 1
            continue
        par["tipo"] = tipo
        cand.append(par)
    raw_f.close()
    print(f"      {len(cand)} pares válidos / {len(plano)} ({dict(descartes)} descartados)", flush=True)

    # ---- 3. dedup de instruções quase-idênticas ----------------------------
    vistos, dedup = set(), []
    for p in cand:
        k = _norm(p["instruction"])[:120]
        if k in vistos:
            descartes["dup_instrucao"] += 1
            continue
        vistos.add(k)
        dedup.append(p)
    cand = dedup
    print(f"      {len(cand)} após dedup de instruções", flush=True)

    # ---- 4. juiz LLM (opcional) -------------------------------------------
    scores_hist = Counter()
    if not args.no_judge and cand:
        print("[3/4] Juiz LLM pontuando 1-5...", flush=True)
        jprompts = [
            montar_chat(tok, JUDGE_SYSTEM,
                        f"INSTRUÇÃO: {p['instruction']}\n"
                        f"INPUT: {p.get('input','')}\n"
                        f"RESPOSTA: {p['output']}\n\nNota (1-5):")
            for p in cand
        ]
        jsp = SamplingParams(temperature=0.0, max_tokens=4, seed=args.seed)
        jouts = llm.generate(jprompts, jsp)
        aprovados = []
        for p, jo in zip(cand, jouts):
            m = re.search(r"[1-5]", jo.outputs[0].text)
            score = int(m.group(0)) if m else 0
            p["score"] = score
            scores_hist[score] += 1
            if score >= args.min_score:
                aprovados.append(p)
            else:
                descartes[f"score_{score}"] += 1
        cand = aprovados
        print(f"      {len(cand)} aprovados (score>={args.min_score}); "
              f"distribuição={dict(sorted(scores_hist.items()))}", flush=True)
    else:
        print("[3/4] (juiz pulado)", flush=True)

    # ---- 5. corta no alvo, embaralha, splita -------------------------------
    rng.shuffle(cand)
    cand = cand[:args.n_pairs]
    n_val = int(len(cand) * args.val_frac)
    held, train = cand[:n_val], cand[n_val:]

    def dump(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(out / "pares.jsonl", cand)
    dump(out / "pares_train.jsonl", train)
    dump(out / "pares_heldout.jsonl", held)

    stats = {
        "n_total": len(cand), "n_train": len(train), "n_heldout": len(held),
        "por_tipo": dict(Counter(p["tipo"] for p in cand)),
        "descartes": dict(descartes),
        "scores": dict(sorted(scores_hist.items())) if scores_hist else None,
        "teacher": args.teacher, "min_score": args.min_score,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"[4/4] OK — {len(cand)} pares ({len(train)} train / {len(held)} held-out) em {out}/", flush=True)
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
