#!/usr/bin/env python3
"""
rag_core.py — Núcleo do sistema RAG para o corpus DOM-PI.

Componentes:
  • E5Embedder      — embeddings multilingual-e5-base na GPU (query/passage).
  • Retriever       — busca top-k por similaridade de cosseno (numpy, brute force).
  • HFGenerator     — geração local (Qwen2.5-1.5B base ou DAPT) via transformers.
  • OllamaGenerator — geração via Ollama (ex.: qwen2.5:14b — inferência qualificada).
  • Modos de RAG    — no_rag, standard, hyde, reflexivo, agentico.

Cada modo retorna um dict:
  {"answer": str, "mode": str, "retrieved": [{doc_id, score, preview}], "trace": [...]}

Decisões:
  - e5 é assimétrico: consultas levam prefixo "query: " e documentos "passage: ".
  - HyDE gera um documento hipotético e o embeda no espaço de passagens.
  - Reflexivo usa o próprio gerador como crítico de fundamentação (grounding).
  - Agêntico é um laço ReAct com a ferramenta BUSCAR[...] / RESPONDER[...].
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

# ───────────────────────── Prompts ─────────────────────────
SYS_RAG = (
    "Você é um assistente que responde perguntas sobre documentos oficiais do "
    "Diário Oficial dos Municípios do Piauí (DOM-PI). Use SOMENTE as informações "
    "das PASSAGENS fornecidas. Se a resposta não estiver nas passagens, responda "
    "exatamente: 'Não encontrado no corpus.' Responda de forma direta e objetiva."
)
SYS_CHAT = (
    "Você é um assistente especializado em administração pública municipal e no "
    "Diário Oficial dos Municípios do Piauí (DOM-PI). Responda de forma direta."
)


def _fmt_ctx(passages: list[dict]) -> str:
    return "\n\n".join(
        f"[{i+1}] {p['texto']}" for i, p in enumerate(passages)
    )


# ───────────────────────── Embedder ─────────────────────────
class E5Embedder:
    def __init__(self, model="intfloat/multilingual-e5-base", device="cuda"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, device=device)

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode([f"query: {text}"], normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")[0]

    def encode_passage(self, text: str) -> np.ndarray:
        return self.model.encode([f"passage: {text}"], normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")[0]


# ───────────────────────── Retriever ────────────────────────
class Retriever:
    def __init__(self, index_dir="rag/index"):
        d = Path(index_dir)
        self.emb = np.load(d / "embeddings.npy")              # N×D, já normalizado
        self.chunks = [json.loads(l) for l in open(d / "chunks.jsonl", encoding="utf-8")]
        self.meta = json.load(open(d / "meta.json", encoding="utf-8"))
        assert len(self.chunks) == self.emb.shape[0], "índice inconsistente"

    def search_vec(self, qvec: np.ndarray, k: int = 5) -> list[dict]:
        scores = self.emb @ qvec                              # cosseno (vetores normalizados)
        idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        idx = idx[np.argsort(-scores[idx])]
        out = []
        for i in idx:
            c = self.chunks[int(i)]
            out.append({"doc_id": c["doc_id"], "chunk_id": c["chunk_id"],
                        "score": float(scores[int(i)]), "texto": c["texto"]})
        return out


# ───────────────────────── Geradores ────────────────────────
class HFGenerator:
    """Gerador local via transformers (Qwen2.5-1.5B base ou DAPT merged)."""
    def __init__(self, model_path: str, name: str, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=device)
        self.model.eval()
        self.device = device

    def generate(self, messages: list[dict], max_new_tokens=256, temperature=0.0) -> str:
        import torch
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                pad_token_id=self.tok.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


class OllamaGenerator:
    """Gerador via Ollama (ex.: qwen2.5:14b)."""
    def __init__(self, model_name: str, name: Optional[str] = None):
        import ollama
        self.ollama = ollama
        self.model_name = model_name
        self.name = name or model_name

    def generate(self, messages: list[dict], max_new_tokens=256, temperature=0.0) -> str:
        r = self.ollama.chat(
            model=self.model_name, messages=messages,
            options={"num_predict": max_new_tokens, "temperature": temperature})
        return r["message"]["content"].strip()


# ───────────────────────── Modos de RAG ─────────────────────
def _preview(passages, n=160):
    return [{"doc_id": p["doc_id"], "score": round(p["score"], 3),
             "preview": p["texto"][:n].replace("\n", " ")} for p in passages]


def no_rag(query, gen, *, max_new_tokens=256, **_):
    """Baseline sem recuperação — o gerador responde do próprio conhecimento."""
    msgs = [{"role": "system", "content": SYS_CHAT},
            {"role": "user", "content": query}]
    return {"answer": gen.generate(msgs, max_new_tokens=max_new_tokens),
            "mode": "no_rag", "retrieved": [], "trace": []}


def rag_standard(query, gen, retr, emb, *, k=5, max_new_tokens=256, **_):
    """RAG clássico: embeda a consulta → top-k → gera com contexto."""
    qv = emb.encode_query(query)
    passages = retr.search_vec(qv, k=k)
    ctx = _fmt_ctx(passages)
    msgs = [{"role": "system", "content": SYS_RAG},
            {"role": "user", "content": f"PASSAGENS:\n{ctx}\n\nPERGUNTA: {query}\nRESPOSTA:"}]
    return {"answer": gen.generate(msgs, max_new_tokens=max_new_tokens),
            "mode": "standard", "retrieved": _preview(passages), "trace": []}


def rag_hyde(query, gen, retr, emb, *, k=5, max_new_tokens=256, **_):
    """HyDE: gera documento hipotético, embeda no espaço de passagens e recupera."""
    hyde_msgs = [
        {"role": "system", "content":
         "Escreva um trecho curto de documento oficial (estilo Diário Oficial) que "
         "responderia à pergunta. Invente detalhes plausíveis. Máx. 4 frases."},
        {"role": "user", "content": f"PERGUNTA: {query}"}]
    hypo = gen.generate(hyde_msgs, max_new_tokens=160)
    hv = emb.encode_passage(hypo)                  # embeda no espaço de documentos
    passages = retr.search_vec(hv, k=k)
    ctx = _fmt_ctx(passages)
    msgs = [{"role": "system", "content": SYS_RAG},
            {"role": "user", "content": f"PASSAGENS:\n{ctx}\n\nPERGUNTA: {query}\nRESPOSTA:"}]
    return {"answer": gen.generate(msgs, max_new_tokens=max_new_tokens),
            "mode": "hyde", "retrieved": _preview(passages),
            "trace": [{"step": "hyde_doc", "content": hypo}]}


def rag_reflexivo(query, gen, retr, emb, *, k=5, max_new_tokens=256, max_iter=2, **_):
    """Self-Reflective: gera, critica a fundamentação e re-busca se necessário."""
    qv = emb.encode_query(query)
    passages = retr.search_vec(qv, k=k)
    trace = []
    answer = ""
    for it in range(max_iter):
        ctx = _fmt_ctx(passages)
        msgs = [{"role": "system", "content": SYS_RAG},
                {"role": "user", "content": f"PASSAGENS:\n{ctx}\n\nPERGUNTA: {query}\nRESPOSTA:"}]
        answer = gen.generate(msgs, max_new_tokens=max_new_tokens)

        crit_msgs = [
            {"role": "system", "content":
             "Você é um crítico rigoroso. Avalie se a RESPOSTA está fundamentada nas "
             "PASSAGENS. Responda SOMENTE em JSON: "
             '{"fundamentada": true|false, "nova_busca": "<consulta reformulada ou vazio>"}'},
            {"role": "user", "content":
             f"PASSAGENS:\n{ctx}\n\nPERGUNTA: {query}\nRESPOSTA: {answer}"}]
        verdict_raw = gen.generate(crit_msgs, max_new_tokens=120)
        trace.append({"step": f"critica_{it}", "content": verdict_raw})
        m = re.search(r'\{.*\}', verdict_raw, re.DOTALL)
        verdict = {}
        if m:
            try:
                verdict = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        if verdict.get("fundamentada", True):
            break
        nova = (verdict.get("nova_busca") or "").strip()
        if not nova:
            break
        # re-busca e funde passagens (sem duplicar chunks)
        nv = emb.encode_query(nova)
        novos = retr.search_vec(nv, k=k)
        vistos = {p["chunk_id"] for p in passages}
        passages = passages + [p for p in novos if p["chunk_id"] not in vistos]
        passages = sorted(passages, key=lambda p: -p["score"])[:k + 2]
        trace.append({"step": f"rebusca_{it}", "content": nova})
    return {"answer": answer, "mode": "reflexivo",
            "retrieved": _preview(passages), "trace": trace}


def rag_agentico(query, gen, retr, emb, *, k=4, max_new_tokens=256, max_steps=3, **_):
    """Agêntico (ReAct): o LLM decide quando BUSCAR e quando RESPONDER (multi-hop)."""
    sys = (
        "Você é um agente de busca em documentos oficiais (DOM-PI). A cada passo, "
        "escreva uma linha 'Pensamento:' e uma linha de Ação. Ações disponíveis:\n"
        "  BUSCAR[consulta]   — procura passagens nos documentos\n"
        "  RESPONDER[texto]   — entrega a resposta final\n"
        f"Use no máximo {max_steps} buscas. Quando tiver evidência suficiente, RESPONDA. "
        "Se nada for encontrado, RESPONDER[Não encontrado no corpus.]")
    history = f"PERGUNTA: {query}\n"
    trace, retrieved_all = [], []
    for step in range(max_steps + 1):
        msgs = [{"role": "system", "content": sys},
                {"role": "user", "content": history + "\nPróximo passo:"}]
        act = gen.generate(msgs, max_new_tokens=200)
        trace.append({"step": f"acao_{step}", "content": act})
        mb = re.search(r'BUSCAR\[(.+?)\]', act, re.DOTALL)
        mr = re.search(r'RESPONDER\[(.+?)\]', act, re.DOTALL)
        if mr:
            return {"answer": mr.group(1).strip(), "mode": "agentico",
                    "retrieved": _preview(retrieved_all[:k]), "trace": trace}
        if mb and step < max_steps:
            q = mb.group(1).strip()
            passages = retr.search_vec(emb.encode_query(q), k=k)
            retrieved_all += passages
            obs = _fmt_ctx(passages[:k])
            history += f"\nAção: BUSCAR[{q}]\nObservação:\n{obs}\n"
        else:
            # forçar resposta com o que houver
            ctx = _fmt_ctx(retrieved_all[:k]) if retrieved_all else "(nenhuma passagem)"
            msgs = [{"role": "system", "content": SYS_RAG},
                    {"role": "user", "content": f"PASSAGENS:\n{ctx}\n\nPERGUNTA: {query}\nRESPOSTA:"}]
            return {"answer": gen.generate(msgs, max_new_tokens=max_new_tokens),
                    "mode": "agentico", "retrieved": _preview(retrieved_all[:k]), "trace": trace}
    return {"answer": "Não encontrado no corpus.", "mode": "agentico",
            "retrieved": _preview(retrieved_all[:k]), "trace": trace}


MODES = {"no_rag": no_rag, "standard": rag_standard, "hyde": rag_hyde,
         "reflexivo": rag_reflexivo, "agentico": rag_agentico}


def run_mode(mode, query, gen, retr=None, emb=None, **kw):
    fn = MODES[mode]
    if mode == "no_rag":
        return fn(query, gen, **kw)
    return fn(query, gen, retr, emb, **kw)


# ───────────────────────── CLI de demonstração ──────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--mode", default="standard", choices=list(MODES))
    ap.add_argument("--gen", default="ollama:qwen2.5:14b",
                    help="ollama:<modelo> ou hf:<caminho>")
    ap.add_argument("--index", default="rag/index")
    args = ap.parse_args()

    if args.gen.startswith("ollama:"):
        gen = OllamaGenerator(args.gen.split(":", 1)[1])
    else:
        gen = HFGenerator(args.gen.split(":", 1)[1], name="hf")

    retr = emb = None
    if args.mode != "no_rag":
        retr = Retriever(args.index)
        emb = E5Embedder()
    res = run_mode(args.mode, args.query, gen, retr, emb)
    print(json.dumps(res, ensure_ascii=False, indent=2))
