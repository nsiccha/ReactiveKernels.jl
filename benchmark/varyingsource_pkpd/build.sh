#!/usr/bin/env bash
set -euo pipefail
source_model=${1:?Usage: build.sh <original-varyingsource3.stan> <scratch-build-dir>}
build_root=${2:?Usage: build.sh <original-varyingsource3.stan> <scratch-build-dir>}
bridge_root=${BRIDGESTAN:-/home/n/.bridgestan/bridgestan-2.9.0}
mkdir -p -- "$build_root"
python3 - "$source_model" "$build_root" <<'PY'
from pathlib import Path
import hashlib
import sys
source, target = Path(sys.argv[1]), Path(sys.argv[2])
body = source.read_bytes()
# The Bruno 896137dd source; the caller supplies its checked-out file.
expected = 'b203d15ecdb2940a3503a162a3ce88415b1fa763d163af2684e14ecefb2d7c2e'
assert hashlib.sha256(body).hexdigest() == expected, 'original source differs from the pinned model'
text = body.decode()
old = '1.0e-6, 1.0e-6, 10000'
assert text.count(old) == 2, 'expected exactly two BDF tolerance call sites'
def write_if_changed(path, content):
    if not path.exists() or path.read_bytes() != content:
        path.write_bytes(content)
write_if_changed(target/'varyingsource3.stan', body)
for label, tolerance in [('tight', '1.0e-10'), ('reference', '1.0e-12')]:
    write_if_changed(target/f'varyingsource3_{label}.stan',
        text.replace(old, f'{tolerance}, {tolerance}, 10000').encode())
print('source_sha256=' + expected)
PY
for name in varyingsource3 varyingsource3_tight varyingsource3_reference; do
  make --no-print-directory -j1 -f "$bridge_root/Makefile" \
    "BS_ROOT=$bridge_root" "STANC=${STANC_PATH:-$bridge_root/bin/stanc}" \
    "$build_root/${name}_model.so"
done
