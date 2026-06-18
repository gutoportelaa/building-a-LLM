#!/bin/bash
# ============================================================================
# download_demais.sh — Download organizado dos 6 territórios novos (roda NO LAB).
# Por território: baixa PDFs (flat-hash) em territorios/<slug>/pdfs_arquivos/,
# loga em logs/download_<slug>.log e gera relatório de falhas legível
# logs/download_<slug>_FALHAS.txt a partir do download_manifest.json (status FAILED).
# Incremental/resumível: re-rodar pula os OK e só retenta FAILED/ausentes.
# ============================================================================
cd "$(dirname "$0")"
PY=./.venv/bin/python
# menor → maior (resultados completos mais cedo)
SLUGS=(vale_do_caninde chapada_vale_do_rio_itaim vale_do_rio_guaribas serra_da_capivara mangabeiras cocais)

echo "===== INÍCIO DOWNLOAD $(date '+%F %T') | ${#SLUGS[@]} territórios ====="
for slug in "${SLUGS[@]}"; do
  IN="dados/scraping_results/scraping_${slug}_2025_deduplicados.json"
  OUT="territorios/${slug}/pdfs_arquivos"
  LOG="logs/download_${slug}.log"
  echo ""
  echo "########## $slug | início $(date '+%F %T') ##########"
  if [ ! -f "$IN" ]; then echo "!! JSON ausente: $IN — pulando"; continue; fi

  $PY -m dompi_scraper.download_pdfs --input "$IN" --output-dir "$OUT" > "$LOG" 2>&1
  rc=$?

  # Relatório de falhas a partir do manifest
  $PY - "$OUT/download_manifest.json" "$slug" <<'PYEOF'
import json, sys
from collections import Counter
manifest, slug = sys.argv[1], sys.argv[2]
try:
    m = json.load(open(manifest))
except Exception as e:
    print(f"  [{slug}] manifest ilegível: {e}"); sys.exit(0)
c = Counter(v.get("status") for v in m.values())
ok, fail = c.get("OK", 0), c.get("FAILED", 0)
falhas = [v for v in m.values() if v.get("status") == "FAILED"]
print(f"  [{slug}] total={len(m)} OK={ok} FAILED={fail}")
falhas_path = f"logs/download_{slug}_FALHAS.txt"
with open(falhas_path, "w", encoding="utf-8") as f:
    f.write(f"# Falhas de download — {slug} — {fail} de {len(m)}\n")
    by_mun = Counter(v.get("municipio", "?") for v in falhas)
    f.write("## por município:\n")
    for mun, n in by_mun.most_common():
        f.write(f"  {mun}: {n}\n")
    f.write("\n## URLs que falharam:\n")
    for v in falhas:
        f.write(f"  {v.get('municipio','?')}\t{v.get('url','')}\n")
print(f"  [{slug}] relatório de falhas → {falhas_path}")
PYEOF
  echo "########## $slug | fim $(date '+%F %T') (download rc=$rc) ##########"
done
echo ""
echo "===== FIM DOWNLOAD $(date '+%F %T') ====="
