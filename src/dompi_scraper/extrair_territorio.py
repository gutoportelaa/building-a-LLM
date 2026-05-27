#!/usr/bin/env python3
"""
extrair_territorio.py — Script único de extração por território DOM-PI
-----------------------------------------------------------------------
Lê os PDFs da pasta territorios/<slug>/pdfs/, executa o pipeline
de extração (Orquestrador Híbrido) e grava os resultados em
extraidos/<slug>/.

Uso:
    uv run python src/dompi_scraper/extrair_territorio.py --territorio carnaubais
    uv run python src/dompi_scraper/extrair_territorio.py --territorio parnaiba --limite 5 --verbose
    uv run python src/dompi_scraper/extrair_territorio.py --territorio vale_do_sambito --force-ocr

Territórios válidos:
    planice_litoran, cocais, carnaubais, entre_rios, vale_do_sambito,
    vale_do_rio_guaribas, chapada_vale_do_rio_itaim, vale_do_caninde,
    serra_da_capivara, vale_dos_rios_piaui_e_itaueiras,
    tabuleiros_alto_parnaiba, teresina, parnaiba
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger("extrair_territorio")

# ==============================================================================
# MAPEAMENTO SLUG → NOME CANÔNICO
# ==============================================================================

TERRITORIOS: dict[str, str] = {
    "planice_litoran":               "Planície Litorânea",
    "cocais":                        "Cocais",
    "carnaubais":                    "Carnaubais",
    "entre_rios":                    "Entre Rios",
    "vale_do_sambito":               "Vale do Sambito",
    "vale_do_rio_guaribas":          "Vale do Rio Guaribas",
    "chapada_vale_do_rio_itaim":     "Chapada Vale do Rio Itaim",
    "vale_do_caninde":               "Vale do Canindé",
    "serra_da_capivara":             "Serra da Capivara",
    "vale_dos_rios_piaui_e_itaueiras": "Vale dos Rios Piauí e Itaueiras",
    "tabuleiros_alto_parnaiba":      "Tabuleiros do Alto Parnaíba e Chapada das Mangabeiras",
    "teresina":                      "Teresina",
    "parnaiba":                      "Parnaíba",
}


def _configure_logging(verbose: bool = False, log_file: str | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
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


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest_from_pdfs(
    pdfs_dir: Path,
    territorio_nome: str,
) -> dict:
    """
    Gera um download_manifest.json em memória a partir de PDFs
    na pasta de drop-zone, sem precisar do pipeline de scraping.
    """
    manifest = {}
    # Use rglob to find PDFs recursively in subdirectories (like city/entity)
    pdfs = sorted(pdfs_dir.rglob("*.pdf"))
    log.info(f"  Escaneando {len(pdfs)} PDFs em {pdfs_dir} (recursivo)")

    for pdf_path in pdfs:
        log.debug(f"    Calculando SHA-256: {pdf_path.name}")
        sha = sha256_file(str(pdf_path))
        fid = pdf_path.stem  # Usa nome do arquivo como ID

        # Tenta inferir município e entidade pela estrutura de pastas:
        # Ex: pdfs/marcos_parente/Prefeitura/file.pdf -> municipio="marcos_parente", entidade="Prefeitura"
        municipio_inferido = territorio_nome
        entidade_inferida = ""
        
        # O caminho relativo ao diretório base de pdfs
        rel_path = pdf_path.relative_to(pdfs_dir)
        parts = rel_path.parts
        
        if len(parts) >= 2:
            # Pelo menos um subdiretório (cidade)
            # Substitui '_' por espaço e capitaliza (ex: marcos_parente -> Marcos Parente)
            cidade_dir = parts[0].replace("_", " ").title()
            # Ajuste simples de preposições para nomes mais limpos (opcional mas útil)
            for prep in [" Do ", " Da ", " De ", " Dos ", " Das "]:
                cidade_dir = cidade_dir.replace(prep, prep.lower())
            municipio_inferido = cidade_dir
            
            if len(parts) >= 3:
                # Tem subdiretório de entidade também (cidade/entidade)
                entidade_inferida = parts[1].replace("_", " ").title()

        manifest[fid] = {
            "path": str(pdf_path.resolve()),
            "sha256": sha,
            "status": "OK",
            "municipio": municipio_inferido,  # Inferido da pasta ou fallback pro território
            "entidade": entidade_inferida,
            "data_publicacao": "",
            "edicao": "",
            "url": "",
            "documento": pdf_path.name,
        }

    return manifest


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[2]  # raiz do projeto
    slug = args.territorio
    territorio_nome = TERRITORIOS[slug]

    pdfs_dir = root / "territorios" / slug / "pdfs"
    output_dir = root / "extraidos" / slug
    log_file = root / "logs" / slug / f"extracao_{time.strftime('%Y%m%d_%H%M%S')}.log"

    _configure_logging(verbose=args.verbose, log_file=str(log_file))

    log.info("=" * 65)
    log.info(f"DOM-PI — Extração de Território: {territorio_nome}")
    log.info("=" * 65)
    log.info(f"  Slug:       {slug}")
    log.info(f"  PDFs:       {pdfs_dir}")
    log.info(f"  Saída:      {output_dir}")
    log.info(f"  Log:        {log_file}")
    log.info("-" * 65)

    if not pdfs_dir.exists():
        log.error(
            f"Pasta não encontrada: {pdfs_dir}\n"
            f"Execute primeiro: bash setup_territorios.sh"
        )
        sys.exit(1)

    pdfs = list(pdfs_dir.rglob("*.pdf"))
    if not pdfs:
        log.error(
            f"Nenhum PDF encontrado em {pdfs_dir}\n"
            f"Copie os PDFs do território '{territorio_nome}' para essa pasta e tente novamente."
        )
        sys.exit(1)

    log.info(f"  PDFs encontrados: {len(pdfs)}")

    # Constrói manifesto em memória
    manifest = build_manifest_from_pdfs(pdfs_dir, territorio_nome)

    # Salva manifesto temporário para uso pelo orquestrador
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "download_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info(f"  Manifesto gerado: {manifest_path}")

    # Importa e executa o pipeline
    try:
        if args.force_ocr:
            # Marker puro com force-ocr para PDFs escaneados pesados
            from dompi_scraper.extrator_marker import run_marker_pipeline

            dedup_path = str(output_dir / "registro_dedup_marker.json")
            jsonl_output = str(output_dir / f"corpus_{slug}.jsonl")

            t0 = time.time()
            stats = run_marker_pipeline(
                manifest_path=str(manifest_path),
                output_dir=str(output_dir / "datalake"),
                jsonl_output=jsonl_output,
                dedup_path=dedup_path,
                limite=args.limite,
                force_ocr=True,
                min_variance=args.min_variance,
                checkpoint_every=10,
            )
            elapsed = time.time() - t0
            engine = "Marker (force-ocr)"

        else:
            # Orquestrador Híbrido (padrão): PyMuPDF fast + Marker slow
            from dompi_scraper.orquestrador_extracao import run_orquestrador_pipeline

            t0 = time.time()
            stats = run_orquestrador_pipeline(
                manifest_path=str(manifest_path),
                output_dir=str(output_dir / "datalake"),
                limite=args.limite,
                threshold=args.threshold,
            )
            elapsed = time.time() - t0
            engine = f"Orquestrador Híbrido (threshold={args.threshold})"

    except ImportError as e:
        log.error(f"Erro de importação: {e}\nVerifique se executou: uv sync")
        sys.exit(1)

    # Relatório final
    print("\n" + "=" * 65)
    print(f"✅  EXTRAÇÃO CONCLUÍDA — {territorio_nome}")
    print("=" * 65)
    print(f"  ⏰  Tempo total:      {elapsed:.1f}s")
    print(f"  🔧  Motor:            {engine}")
    print(f"  📂  Saída:            {output_dir}/")
    if "gerados" in stats:
        print(f"  📝  Gerados:          {stats.get('gerados', 'N/A')}")
        print(f"  ♻️   Duplicatas:       {stats.get('duplicatas', 'N/A')}")
        print(f"  ⚠️   Rejeitados:       {stats.get('rejeitados', 'N/A')}")
        print(f"  ❌  Erros:            {stats.get('erros', 'N/A')}")
    else:
        print(f"  📝  Chunks salvos:    {stats.get('total', 'N/A')}")
        print(f"  ⚡  Fast Path:        {stats.get('fast_path', 'N/A')} (PyMuPDF)")
        print(f"  🐢  Slow Path:        {stats.get('slow_path', 'N/A')} (Marker)")
    print("=" * 65 + "\n")

    log.info(f"Log completo salvo em: {log_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrator padronizado por território — DOM-PI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--territorio", required=True,
        choices=list(TERRITORIOS.keys()),
        metavar="SLUG",
        help=(
            "Slug do território. Valores aceitos:\n" +
            "\n".join(f"  {k} → {v}" for k, v in TERRITORIOS.items())
        ),
    )
    parser.add_argument(
        "--limite", type=int, default=999_999,
        help="Máx. PDFs a processar (padrão: ilimitado). Use valores pequenos para testes.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.70,
        help="Score OCR: acima → PyMuPDF (rápido), abaixo → Marker GPU (padrão: 0.70).",
    )
    parser.add_argument(
        "--force-ocr", action="store_true",
        help="Usa Marker com force-ocr para todos os PDFs (ignora threshold). Mais lento, máxima qualidade.",
    )
    parser.add_argument(
        "--min-variance", type=float, default=50.0,
        help="Limiar de nitidez (Laplacian) para rejeitar páginas em branco/corruptas (padrão: 50.0).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Ativa logs detalhados (DEBUG).",
    )
    parser.add_argument(
        "--listar", action="store_true",
        help="Lista todos os territórios disponíveis e sai.",
    )

    args = parser.parse_args()

    if args.listar:
        print("\nTerritórios disponíveis:\n")
        for slug, nome in TERRITORIOS.items():
            print(f"  {slug:<40} → {nome}")
        print()
        sys.exit(0)

    run(args)


if __name__ == "__main__":
    main()
