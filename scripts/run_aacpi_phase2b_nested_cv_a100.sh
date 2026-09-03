#!/usr/bin/env bash
set -euo pipefail

SEARCH_SPACE="docs/protocols/AACPI_V2_PHASE2_SEARCH_SPACE.yaml"

run_pair() {
  local pair_id="$1"
  python scripts/train_aacpi_advantage_nested_cv.py \
    --utility-table "outputs/aacpi/utility_tables/${pair_id}_dev_utility_table.csv.gz" \
    --search-space "$SEARCH_SPACE" \
    --output-dir "outputs/aacpi/phase2b/${pair_id}" \
    --device cuda \
    --overwrite
}

if [[ $# -gt 0 ]]; then
  for pair_id in "$@"; do
    run_pair "$pair_id"
  done
else
  run_pair mkgw_mhyper_native
  run_pair mkgw_mhyper_adamf
  run_pair mkgw_native_adamf
  run_pair db15k_mhyper_native
  run_pair db15k_mhyper_adamf
  run_pair db15k_native_adamf
fi
