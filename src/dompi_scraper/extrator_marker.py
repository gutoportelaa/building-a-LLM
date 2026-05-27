#!/usr/bin/env python3
"""
extrator_marker.py — Extrator GPU-Otimizado via Marker (Etapa 3-B)
-------------------------------------------------------------------
Lê o manifesto de downloads (download_manifest.json) e extrai texto
estruturado dos PDFs via Marker (GPU), gerando Markdown com frontmatter
YAML no Data Lake e JSONL para fine-tuning — mesmo formato do processar_pdfs.py.

Otimizações:
  - Modelos Marker carregados UMA vez por sessão (não por PDF)
  - Quality gate em memória via PyMuPDF (sem PNGs no disco)
  - DocTR e pdf2image removidos completamente
  - Extração de imagens desativada (--disable_image_extraction)
  - Incremental: consulta registro_dedup_marker.json e pula já processados

GPU: RTX 4070 Laptop (8GB) — Marker usa ~3.17 GB VRAM, 1 worker seguro.

Uso:
    # Teste com 3 PDFs
    uv run python src/dompi_scraper/extrator_marker.py \\
        --manifest db_treino_carnaubais/pdfs_arquivos/download_manifest.json \\
        --output-dir dados_brutos_marker \\
        --jsonl-output corpus_marker.jsonl \\
        --limite 3 --verbose

    # Produção completa (PDFs escaneados do DOM-PI)
    uv run python src/dompi_scraper/extrator_marker.py \\
        --manifest db_treino_carnaubais/pdfs_arquivos/download_manifest.json \\
        --output-dir dados_brutos_marker \\
        --jsonl-output corpus_marker.jsonl \\
        --force-ocr
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

try:
    import torch
except ImportError:
    print("Erro: 'torch' necessário. Instale com: uv add torch")
    sys.exit(1)

try:
    import fitz  # PyMuPDF — quality gate em memória
except ImportError:
    print("Erro: 'pymupdf' necessário. Instale com: uv add pymupdf")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("Erro: 'opencv-python' e 'numpy' necessários. Instale com: uv add opencv-python numpy")
    sys.exit(1)

try:
    from .shared_utils import compute_content_md5, classify_act_type, extract_date_from_text
except ImportError:
    from shared_utils import compute_content_md5, classify_act_type, extract_date_from_text

# Importa utilitários de formatação do processar_pdfs (evita duplicação)
try:
    from .processar_pdfs import generate_frontmatter, build_datalake_path
except ImportError:
    from processar_pdfs import generate_frontmatter, build_datalake_path

# ==============================================================================
# LOGGING
# ==============================================================================

log = logging.getLogger("extrator_marker")


def _configure_logging(verbose: bool = False, log_file: str | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log.setLevel(level)
    log.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(level)
    log.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)


# ==============================================================================
# QUALITY GATE EM MEMÓRIA (PyMuPDF, sem PNGs no disco)
# ==============================================================================

def quality_gate_inmemory(
    pdf_path: str,
    sample_pages: int = 2,
    min_variance: float = 50.0,
) -> tuple[bool, float]:
    """
    Avalia qualidade do PDF renderizando páginas de amostra em memória (72 DPI,
    escala de cinza) via PyMuPDF e calculando variância do Laplaciano com cv2.

    Sem gravação de PNGs em disco. Dependências: fitz, cv2, numpy (já existentes).

    Returns:
        (aprovado, max_variancia_encontrada)
    """
    try:
        doc = fitz.open(pdf_path)
        total = len(doc)
        if total == 0:
            doc.close()
            return False, 0.0

        # Amostra primeira e página do meio
        indices = sorted({0, total // 2})[:sample_pages]
        max_var = 0.0

        for idx in indices:
            page = doc.load_page(idx)
            # 72 DPI (Matrix 1x) em cinza — rápido, suficiente para desfoque
            pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), colorspace=fitz.csGRAY)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
            variance = float(cv2.Laplacian(arr, cv2.CV_64F).var())
            if variance > max_var:
                max_var = variance

        doc.close()
        return max_var >= min_variance, max_var

    except Exception as e:
        log.warning(f"  Quality gate falhou ({os.path.basename(pdf_path)}): {e}")
        return True, -1.0  # Permite processamento em caso de erro

def is_garbage_text(pdf_path: str, max_garbage_ratio: float = 0.25) -> tuple[bool, float]:
    """
    Identifica se um PDF é majoritariamente um scan complexo/manuscrito
    analisando a taxa de caracteres não-alfanuméricos (lixo de OCR do PyMuPDF).
    Retorna (is_garbage, garbage_ratio).
    """
    try:
        import re
        doc = fitz.open(pdf_path)
        total_text = ""
        for page in doc:
            total_text += page.get_text()
        doc.close()
        
        if not total_text.strip():
            return True, 1.0 # Sem texto = scan puro
            
        text_no_space = re.sub(r'\s+', '', total_text)
        if not text_no_space:
            return True, 1.0
            
        alnum_count = sum(c.isalnum() for c in text_no_space)
        garbage_ratio = 1.0 - (alnum_count / len(text_no_space))
        
        return garbage_ratio > max_garbage_ratio, garbage_ratio
    except Exception:
        return False, 0.0

# ==============================================================================
# SESSÃO MARKER — MODELOS CARREGADOS UMA VEZ
# ==============================================================================

def create_marker_session(
    force_ocr: bool = False,
    disable_images: bool = True,
) -> tuple[dict, object]:
    """
    Cria a sessão Marker carregando todos os modelos UMA única vez por execução.
    O custo de ~30s de inicialização ocorre apenas aqui, não por PDF.

    Returns:
        (modelos_dict, config_parser) — reutilizados em cada chamada ao converter.
    """
    try:
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser
    except ImportError:
        log.error("Marker não instalado. Instale com: uv add marker-pdf")
        raise

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["TORCH_DEVICE"] = device
    log.info(f"Dispositivo: {device.upper()}")

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"  VRAM: {props.total_memory / 1e9:.1f} GB total")
    else:
        log.warning("  GPU não disponível — processamento em CPU (lento)")

    config: dict = {
        "disable_image_extraction": disable_images,
        "output_format": "markdown",
    }
    if force_ocr:
        config["force_ocr"] = True
        log.info("  Modo force_ocr: ATIVADO")

    config_parser = ConfigParser(config)

    log.info("Carregando modelos Marker (operação única por sessão)...")
    t0 = time.time()
    modelos = create_model_dict()
    elapsed = time.time() - t0
    log.info(f"  Modelos prontos em {elapsed:.1f}s")

    if device == "cuda":
        used_gb = torch.cuda.memory_allocated() / 1e9
        log.info(f"  VRAM alocada: {used_gb:.2f} GB")

    return modelos, config_parser


def extract_with_marker(
    modelos: dict,
    config_parser: object,
    pdf_path: str,
) -> str | None:
    """
    Extrai Markdown de um único PDF reutilizando os modelos já carregados.
    Cria um novo PdfConverter por PDF (leve — os modelos são passados por referência).
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.output import text_from_rendered

        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=modelos,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )
        rendered = converter(pdf_path)
        text, _, _ = text_from_rendered(rendered)
        del rendered, converter
        gc.collect()
        return text

    except Exception as e:
        log.error(f"  Marker falhou em {os.path.basename(pdf_path)}: {e}")
        return None


# ==============================================================================
# PERSISTÊNCIA
# ==============================================================================

def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json_atomic(data: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _append_jsonl(records: list[dict], path: str) -> None:
    if not records:
        return
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_pdfs_com_tabela_achatada(limpos_dir: str) -> set[str]:
    """Retorna um set com os sha256_pdf dos arquivos marcados com tabela_achatada_detectada."""
    target_shas = set()
    limpos_path = Path(limpos_dir)
    if not limpos_path.exists():
        return target_shas
        
    for md_file in limpos_path.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            if "needs_human_review: true" in content and "tabela_achatada_detectada" in content:
                for line in content.split("\n"):
                    if line.startswith("sha256_pdf:"):
                        sha = line.split('"')[1] if '"' in line else line.split(":")[1].strip()
                        target_shas.add(sha)
                        break
        except Exception:
            pass
    return target_shas


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def run_marker_pipeline(
    manifest_path: str,
    output_dir: str,
    jsonl_output: str,
    dedup_path: str,
    limite: int,
    force_ocr: bool,
    min_variance: float,
    checkpoint_every: int,
    filter_limpos_dir: str | None = None,
) -> dict:
    """
    Pipeline completo: manifesto → quality gate → Marker → Data Lake + JSONL.
    Incremental: pula PDFs já processados via registro_dedup_marker.json.
    Pode filtrar os PDFs lendo os markdowns sinalizados em filter_limpos_dir.
    """
    manifest = _load_json(manifest_path)
    ok_entries = {fid: e for fid, e in manifest.items() if e.get("status") == "OK"}
    
    if filter_limpos_dir:
        log.info(f"Escaneando {filter_limpos_dir} para identificar PDFs com tabelas achatadas...")
        shas_tabelas = get_pdfs_com_tabela_achatada(filter_limpos_dir)
        filtered = {}
        for fid, e in ok_entries.items():
            if e.get("sha256") in shas_tabelas or fid in shas_tabelas:
                filtered[fid] = e
            elif Path(e.get("path", "")).stem in shas_tabelas:
                filtered[fid] = e
        ok_entries = filtered
        log.info(f"Filtro aplicado: {len(ok_entries)} PDFs mapeados para reprocessamento de tabelas.")
        
    log.info(f"PDFs com status OK no manifesto (após filtros): {len(ok_entries)}")

    dedup_registry = _load_json(dedup_path)
    log.info(f"Hashes já registrados: {len(dedup_registry)}")

    if not ok_entries:
        log.error("Nenhum PDF disponível para processar.")
        return {"total": 0, "gerados": 0, "rejeitados": 0, "duplicatas": 0, "erros": 0}

    # Carrega modelos Marker UMA VEZ
    modelos, config_parser = create_marker_session(force_ocr=force_ocr, disable_images=True)

    os.makedirs(output_dir, exist_ok=True)
    Path(jsonl_output).parent.mkdir(parents=True, exist_ok=True)
    jsonl_meta_path = str(Path(jsonl_output).with_suffix(".meta.jsonl"))

    stats = {"total": 0, "gerados": 0, "rejeitados": 0, "rejeitados_scan": 0, "duplicatas": 0, "erros": 0}
    buf_jsonl: list[dict] = []
    buf_meta: list[dict] = []
    processed = 0

    for fid, entry in ok_entries.items():
        if processed >= limite:
            log.info(f"Limite de {limite} PDFs atingido.")
            break

        pdf_path = entry.get("path", "")
        if not pdf_path or not os.path.exists(pdf_path):
            log.warning(f"PDF não encontrado em disco: {pdf_path}")
            stats["erros"] += 1
            continue

        stats["total"] += 1
        processed += 1

        municipio  = entry.get("municipio", "DESCONHECIDO")
        entidade   = entry.get("entidade", "")
        categoria  = entry.get("categoria", "")
        data_pub   = entry.get("data_publicacao", "")
        edicao     = entry.get("edicao", "")
        url_orig   = entry.get("url", "")
        documento  = entry.get("documento", "")
        sha256_pdf = entry.get("sha256", "")

        log.info(f"\n[{processed}/{min(limite, len(ok_entries))}] {os.path.basename(pdf_path)}")
        log.info(f"  Município: {municipio} | Entidade: {entidade}")

        # 1. Quality gate em memória (PyMuPDF, sem disco)
        t0 = time.time()
        aprovado, variancia = quality_gate_inmemory(pdf_path, min_variance=min_variance)
        log.debug(f"  Quality gate: var={variancia:.1f} limiar={min_variance} ({time.time()-t0:.2f}s)")
        if not aprovado:
            log.warning(f"  ⚠️  Rejeitado — variância={variancia:.1f} < {min_variance}")
            stats["rejeitados"] += 1
            continue

        # 1.5 Fatiar PDF apenas para as páginas do município
        import tempfile
        t_slice = time.time()
        try:
            from dompi_scraper.orquestrador_extracao import analisar_e_fatiar_pdf
        except ImportError:
            from orquestrador_extracao import analisar_e_fatiar_pdf
            
        chunks = analisar_e_fatiar_pdf(pdf_path, municipio)
        paginas_alvo = []
        if municipio in chunks:
            paginas_alvo = chunks[municipio]["paginas"]
        elif chunks:
            # Fallback para a maior cidade mapeada
            maior_cidade = max(chunks.keys(), key=lambda k: len(chunks[k]["paginas"]))
            paginas_alvo = chunks[maior_cidade]["paginas"]
            
        if not paginas_alvo:
            doc_temp = fitz.open(pdf_path)
            paginas_alvo = list(range(len(doc_temp)))
            doc_temp.close()
            
        doc = fitz.open(pdf_path)
        doc.select(paginas_alvo)
        fd, tmp_pdf = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        doc.save(tmp_pdf)
        doc.close()
        log.debug(f"  Fatiamento ({len(paginas_alvo)} págs) concluído em {time.time()-t_slice:.2f}s")

        # 1.6 Detectar se é um scan denso/manuscrito (lixo de OCR)
        is_garbage, garbage_ratio = is_garbage_text(tmp_pdf)
        if is_garbage:
            log.warning(f"  ⚠️  Rejeitado — Scan complexo detectado (garbage_ratio={garbage_ratio:.2f})")
            stats["rejeitados_scan"] += 1
            
            with open(os.path.join(output_dir, "scans_complexos.jsonl"), "a", encoding="utf-8") as f_scan:
                f_scan.write(json.dumps({
                    "manifest_id": fid,
                    "municipio": municipio,
                    "entidade": entidade,
                    "pdf_path": pdf_path,
                    "garbage_ratio": garbage_ratio,
                    "motivo": "Excesso de caracteres não-alfanuméricos (provável manuscrito/scan denso)"
                }, ensure_ascii=False) + "\n")
                
            if os.path.exists(tmp_pdf):
                os.remove(tmp_pdf)
            continue

        # 2. Extração via Marker (modelos reutilizados) no mini-PDF
        t0 = time.time()
        markdown_text = extract_with_marker(modelos, config_parser, tmp_pdf)
        elapsed_ext = time.time() - t0
        
        # Limpeza do PDF temporário
        if os.path.exists(tmp_pdf):
            os.remove(tmp_pdf)

        if not markdown_text or len(markdown_text.strip()) < 50:
            log.warning(f"  ⚠️  Texto insuficiente extraído ({elapsed_ext:.1f}s)")
            stats["erros"] += 1
            continue

        log.debug(f"  Marker: {elapsed_ext:.1f}s | {len(markdown_text)} chars")

        # 3. Deduplicação textual
        content_hash = compute_content_md5(markdown_text)
        if content_hash in dedup_registry:
            log.info(f"  ♻️  DUPLICATA (hash={content_hash[:12]}...)")
            stats["duplicatas"] += 1
            continue

        # 4. Classificação e data
        tipo_ato = classify_act_type(markdown_text[:1000], fallback_category=categoria)
        if not data_pub:
            data_pub = extract_date_from_text(markdown_text) or ""

        ano, mes = "sem_ano", "sem_mes"
        if data_pub and len(data_pub) >= 7:
            parts = data_pub.split("-")
            if len(parts) >= 2:
                ano, mes = parts[0], parts[1]

        # 5. Frontmatter YAML (mesmo formato do processar_pdfs.py)
        frontmatter = generate_frontmatter(
            content_hash=content_hash,
            municipio=municipio,
            entidade=entidade,
            tipo_ato=tipo_ato,
            data_publicacao=data_pub,
            url_origem=url_orig,
            edicao=edicao,
            sha256_pdf=sha256_pdf,
        )
        full_md = frontmatter + markdown_text

        # 6. Data Lake particionado
        md_path = build_datalake_path(output_dir, ano, mes, municipio, f"{content_hash}.md")
        os.makedirs(os.path.dirname(md_path), exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(full_md)

        # 7. Registro de dedup (persiste extrator=marker para rastreabilidade)
        dedup_registry[content_hash] = {
            "path": md_path,
            "municipio": municipio,
            "entidade": entidade,
            "tipo_ato": tipo_ato,
            "data_publicacao": data_pub,
            "url_origem": url_orig,
            "documento": documento,
            "extrator": "marker",
        }

        # 8. Buffers JSONL
        buf_jsonl.append({"text": markdown_text})
        buf_meta.append({
            "text": markdown_text,
            "metadata": {
                "id_publicacao": content_hash,
                "municipio": municipio,
                "entidade": entidade,
                "tipo_ato": tipo_ato,
                "data_publicacao": data_pub,
                "edicao": edicao,
                "extrator": "marker",
            },
        })

        stats["gerados"] += 1
        log.info(
            f"  ✅ {municipio} / {tipo_ato} → {content_hash[:12]}... "
            f"({len(markdown_text)} chars, {elapsed_ext:.1f}s)"
        )

        # Checkpoint periódico (tolerância a interrupções)
        if stats["gerados"] % checkpoint_every == 0:
            _append_jsonl(buf_jsonl, jsonl_output)
            _append_jsonl(buf_meta, jsonl_meta_path)
            _save_json_atomic(dedup_registry, dedup_path)
            log.info(f"  [CHECKPOINT] {stats['gerados']} documentos persistidos")
            buf_jsonl.clear()
            buf_meta.clear()

    # Flush final dos buffers
    _append_jsonl(buf_jsonl, jsonl_output)
    _append_jsonl(buf_meta, jsonl_meta_path)
    _save_json_atomic(dedup_registry, dedup_path)

    # Libera VRAM
    del modelos
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.debug(f"VRAM liberada. Alocada atual: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    return stats


# ==============================================================================
# CLI
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrator GPU-Otimizado via Marker — Etapa 3-B do pipeline DOM-PI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--manifest", required=True,
        help="Caminho do download_manifest.json (saída do download_pdfs.py)."
    )
    parser.add_argument(
        "--output-dir", default="dados_brutos_marker",
        help="Diretório raiz do Data Lake de saída (padrão: dados_brutos_marker)."
    )
    parser.add_argument(
        "--jsonl-output", default="corpus_marker.jsonl",
        help="Arquivo JSONL de saída (padrão: corpus_marker.jsonl)."
    )
    parser.add_argument(
        "--dedup-registry", default=None,
        help="Registro de dedup. Padrão: <output-dir>/registro_dedup_marker.json"
    )
    parser.add_argument(
        "--limite", type=int, default=999999,
        help="Máx PDFs a processar (padrão: ilimitado)."
    )
    parser.add_argument(
        "--force-ocr", action="store_true",
        help="Força re-OCR completo via Marker (recomendado para PDFs escaneados do DOM-PI)."
    )
    parser.add_argument(
        "--min-variance", type=float, default=50.0,
        help="Limiar de variância do Laplaciano para quality gate (padrão: 50.0)."
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="Salva JSONL e dedup a cada N docs (padrão: 10)."
    )
    parser.add_argument(
        "--filter-limpos-dir", default=None,
        help="Diretório dados_limpos para ler arquivos e processar apenas os que possuem tabelas achatadas."
    )
    parser.add_argument(
        "--log-file", default=None,
        help="Caminho para arquivo de log em disco (opcional)."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Ativa logs DEBUG."
    )

    args = parser.parse_args()
    _configure_logging(verbose=args.verbose, log_file=args.log_file)

    if not os.path.exists(args.manifest):
        log.error(f"Manifesto não encontrado: {args.manifest}")
        sys.exit(1)

    dedup_path = args.dedup_registry or os.path.join(
        args.output_dir, "registro_dedup_marker.json"
    )

    log.info("=" * 65)
    log.info("EXTRATOR MARKER — DOM-PI (GPU-Otimizado, Etapa 3-B)")
    log.info("=" * 65)
    log.info(f"Manifesto:     {args.manifest}")
    log.info(f"Data Lake:     {args.output_dir}")
    log.info(f"JSONL:         {args.jsonl_output}")
    log.info(f"Dedup:         {dedup_path}")
    log.info(f"Force OCR:     {args.force_ocr}")
    log.info(f"Min variance:  {args.min_variance}")
    log.info(f"Limite:        {args.limite}")
    log.info("-" * 65)

    t_start = time.time()
    stats = run_marker_pipeline(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        jsonl_output=args.jsonl_output,
        dedup_path=dedup_path,
        limite=args.limite,
        force_ocr=args.force_ocr,
        min_variance=args.min_variance,
        checkpoint_every=args.checkpoint_every,
        filter_limpos_dir=args.filter_limpos_dir,
    )
    elapsed = time.time() - t_start

    print("\n" + "=" * 65)
    print("✅  EXTRAÇÃO MARKER CONCLUÍDA")
    print("=" * 65)
    print(f"  ⏰ Tempo total:          {elapsed:.1f}s")
    print(f"  📊 Total processado:     {stats['total']}")
    print(f"  📝 Markdown gerados:     {stats['gerados']}")
    print(f"  ♻️  Duplicatas:           {stats['duplicatas']}")
    print(f"  ⚠️  Rejeitados (QG):      {stats['rejeitados']}")
    print(f"  📸 Scans Complexos:      {stats['rejeitados_scan']}")
    print(f"  ❌ Erros:                {stats['erros']}")
    print(f"  📄 JSONL:                {args.jsonl_output}")
    print(f"  🗂️  Data Lake:            {args.output_dir}/")
    if stats["gerados"] > 0:
        avg = elapsed / stats["gerados"]
        print(f"  ⚡ Média/documento:      {avg:.1f}s")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
