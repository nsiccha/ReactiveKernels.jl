"""Check the reported toy's timing ratios in a saved measurement run."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("measurements", type=Path)
parser.add_argument("--tag", default="candidate")
parser.add_argument("--max-gradient-ratio", type=float, default=8.0)
parser.add_argument("--max-batch-ratio", type=float, default=1.5)
args = parser.parse_args()
report = json.loads(args.measurements.read_text())
failures = []
for n, variants in report["cases"].items():
    measured = variants[args.tag]
    for name, ratio, limit in (
        ("gradient/value", measured["gradient"]["min_ms"] / measured["value"]["min_ms"],
         args.max_gradient_ratio),
        ("batch per point/single", measured["batch16"]["min_ms"] / 16 /
         measured["gradient"]["min_ms"], args.max_batch_ratio),
    ):
        print(f"N={n} {args.tag} {name}: {ratio:.3f} <= {limit:.3f}")
        if not 0 < ratio <= limit:
            failures.append(f"N={n} {name}")
if failures:
    raise SystemExit("Timing regression: " + ", ".join(failures))
