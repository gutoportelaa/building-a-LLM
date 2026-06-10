#!/bin/bash
# Scrape sequencial dos territórios restantes (metadados, sem download).
cd "$(dirname "$0")"
export PYTHONPATH=src/dompi_scraper
SLUGS=(chapada_vale_do_rio_itaim vale_do_caninde serra_da_capivara cocais vale_do_rio_guaribas mangabeiras)
echo "===== INÍCIO $(date '+%F %T') | ${#SLUGS[@]} territórios ====="
for slug in "${SLUGS[@]}"; do
  echo ""; echo "########## $slug | $(date '+%T') ##########"
  ./.venv/bin/python scraper_isolado.py --territorio "$slug" --ano 2025 --so-json \
      --saida "dados/scraping_results/scraping_${slug}_2025" 2>&1 \
    | grep -vE "^\s+\S+ / (Prefeitura|Camara): [0-9]+ doc" \
    || echo "!! falha em $slug (rc=$?)"
done
echo ""; echo "===== FIM $(date '+%F %T') ====="
