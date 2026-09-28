#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_root=${1:?Usage: build.sh <disk-backed-scratch-build-dir>}
bridge_root=${BRIDGESTAN:-/home/n/.bridgestan/bridgestan-2.9.0}
mkdir -p -- "$build_root"
cp -- "$task_root/unit_bdf.stan" "$build_root/unit_bdf.stan"
make --no-print-directory -j1 -f "$bridge_root/Makefile" \
  "BS_ROOT=$bridge_root" "STANC=${STANC_PATH:-$bridge_root/bin/stanc}" \
  "$build_root/unit_bdf_model.so"
