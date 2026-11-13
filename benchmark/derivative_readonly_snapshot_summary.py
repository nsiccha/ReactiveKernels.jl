"""Summarize within-process readonly matrix rounds; modes run separately."""
import json
import random
import re
import statistics
import sys
from pathlib import Path

pattern = re.compile(
    r"n = (\d+), k = (\d+), round = (\d+), arm = :(\w+), "
    r"ns = ([\d.eE+-]+), bytes = (\d+), allocations = (\d+)"
)
output = []
for filename in sys.argv[1:]:
    groups = {}
    text = Path(filename).read_text()
    assert "READONLY_SNAPSHOT_COSTS_COMPLETE" in text
    mode = re.search(r"mode = :(\w+)", text)[1]
    for match in pattern.finditer(text):
        n, k, rnd, arm, ns, size, allocs = match.groups()
        groups.setdefault((int(n), int(k)), {}).setdefault(arm, {})[int(rnd)] = (
            float(ns), int(size), int(allocs)
        )
    assert set(groups) == {(128, 8), (2048, 32), (8192, 64)}
    for (n, k), arms in sorted(groups.items()):
        assert set(arms["rule"]) == set(arms["native"]) == set(range(1, 13))
        ratios = [arms["rule"][r][0] / arms["native"][r][0] for r in range(1, 13)]
        rng = random.Random(20261113)
        draws = sorted(statistics.median(rng.choices(ratios, k=12))
                       for _ in range(20000))
        output.append({
            "source": filename, "mode": mode, "n": n, "k": k,
            "rule_over_native": statistics.median(ratios),
            "within_process_paired_bootstrap_95": [draws[499], draws[19499]],
            "ratios": ratios,
            "arms": {
                arm: {"median_ns": statistics.median(v[0] for v in rounds.values()),
                      "bytes": sorted(set(v[1] for v in rounds.values())),
                      "allocations": sorted(set(v[2] for v in rounds.values()))}
                for arm, rounds in arms.items()
            },
        })
json.dump(output, sys.stdout, indent=2)
print()
