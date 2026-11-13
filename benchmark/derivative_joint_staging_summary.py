"""Summarize paired benchmark rounds; bootstrap intervals are within-run only."""
import json
import random
import re
import statistics
import sys
from pathlib import Path

pattern = re.compile(
    r"stage = :warm, label = :(\w+), n = (\d+), round = (\d+), "
    r"arm = :(\w+), ns = ([\d.eE+-]+), bytes = (\d+), allocations = (\d+)"
)
output = []
for filename in sys.argv[1:]:
    groups = {}
    for line in Path(filename).read_text().splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        label, n, rnd, arm, ns, size, allocs = match.groups()
        groups.setdefault((label, int(n)), {}).setdefault(arm, {})[int(rnd)] = (
            float(ns), int(size), int(allocs)
        )
    for (label, n), arms in sorted(groups.items()):
        pairs = [("shared", "independent")]
        if "adapter_shared" in arms:
            pairs += [("adapter_shared", "adapter_independent"),
                      ("adapter_shared", "ordinary")]
        row = {"source": filename, "family": label, "n": n, "comparisons": {}}
        for numerator, denominator in pairs:
            assert set(arms[numerator]) == set(arms[denominator])
            rounds = sorted(arms[numerator])
            assert len(rounds) == 12
            ratios = [arms[numerator][r][0] / arms[denominator][r][0] for r in rounds]
            rng = random.Random(20261113)
            draws = sorted(statistics.median(rng.choices(ratios, k=len(ratios)))
                           for _ in range(20000))
            row["comparisons"][numerator + "/" + denominator] = {
                "median_ratio": statistics.median(ratios),
                "paired_round_bootstrap_95": [draws[499], draws[19499]],
                "ratios": ratios,
            }
        row["arms"] = {
            arm: {"median_ns": statistics.median(v[0] for v in trials.values()),
                  "bytes": sorted(set(v[1] for v in trials.values())),
                  "allocations": sorted(set(v[2] for v in trials.values()))}
            for arm, trials in arms.items()
        }
        output.append(row)
json.dump(output, sys.stdout, indent=2)
print()
