"""Check exported before/after values, gradients and actual GP counts."""
import csv
import json
import sys


def table(path, key):
    with open(path) as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    indexed = {tuple(row[k] for k in key): row for row in rows}
    assert len(indexed) == len(rows), path
    return indexed


def compare(before_path, after_path, gradient=False):
    key = ("subjects", "case", "path")
    before, after = table(before_path, key), table(after_path, key)
    assert before.keys() == after.keys()
    summary = []
    for identity, old in before.items():
        new = after[identity]
        assert old["RK_coordinates"] == new["RK_coordinates"], identity
        assert float(old["density"]) == float(new["density"]), identity
        row = dict(zip(key, identity))
        if gradient:
            assert old["reference_RK_gradient"] == new["reference_RK_gradient"], identity
            g0 = list(map(float, old["RK_gradient"].split(",")))
            g1 = list(map(float, new["RK_gradient"].split(",")))
            assert len(g0) == len(g1)
            row["max_gradient_change"] = max(abs(x-y) for x, y in zip(g0, g1))
            assert row["max_gradient_change"] < 1e-12, identity
            assert new["accuracy_matched"] == "true", identity
        allocation = "Julia_allocated_bytes" if gradient else "Julia_bytes"
        row.update(before_us=float(old["median_us"]), after_us=float(new["median_us"]),
                   before_bytes=int(old[allocation]), after_bytes=int(new[allocation]))
        summary.append(row)
    return summary


def counts(before_path, after_path):
    before, after = table(before_path, ("subjects",)), table(after_path, ("subjects",))
    assert before.keys() == after.keys()
    summary = []
    for identity, old in before.items():
        new = after[identity]
        assert float(old["density"]) == float(new["density"]), identity
        row = {"subjects": int(identity[0])}
        for phase in ("prepare", "evaluate"):
            for math in ("weights", "normalizers", "placebos"):
                name = f"{phase}_{math}"
                row[f"before_{name}"] = int(old[name])
                row[f"after_{name}"] = int(new[name])
        assert all(row[f"{side}_prepare_{math}"] == 0
                   for side in ("before", "after")
                   for math in ("weights", "normalizers", "placebos"))
        assert row["after_evaluate_weights"] == row["after_evaluate_normalizers"] == 1
        assert row["after_evaluate_placebos"] == 2
        assert row["before_evaluate_weights"] == row["before_evaluate_normalizers"] > 1
        assert row["before_evaluate_placebos"] > 2
        summary.append(row)
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit("Usage: compare_shared_gp.py <primal-before> <primal-after> "
                 "<gradient-before> <gradient-after> <counts-before> <counts-after>")
    print(json.dumps({"primal": compare(*sys.argv[1:3]),
                      "gradient": compare(*sys.argv[3:5], gradient=True),
                      "counts": counts(*sys.argv[5:7])}, indent=2))
