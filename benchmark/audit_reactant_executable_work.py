#!/usr/bin/env python3
"""Report work in complete executable HLO, including loop callees and array work.

This is an inspection aid, not an acceptance oracle. A surviving while may do
unrelated work; an equivalent array computation may implement the intended work.
Read the report with the source contract, numerical checks and growth evidence.
Unsupported or incomplete text is rejected instead of silently undercounted.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


HEADER = re.compile(r"^(ENTRY )?%([\w.-]+) \(.*\) -> .* \{$")
ASSIGNMENT = re.compile(r"^\s+(?:ROOT\s+)?%[\w.-]+ = ")
INSTRUCTION = re.compile(
    r"^\s+(ROOT\s+)?%([\w.-]+) = (.*?) ([A-Za-z][A-Za-z0-9_-]*)\((.*?)\)(.*)$"
)
REFERENCE = re.compile(
    r"(?:calls|to_apply|condition|body|true_computation|false_computation)=%([\w.-]+)"
)
# Report these operations; none is independently a correctness requirement.
WORK_OPS = {
    "log", "log-plus-one", "divide", "exponential", "power", "sqrt",
    "dynamic-slice", "slice", "gather", "reduce", "dot", "map",
}


def audit(path):
    raw = path.read_bytes()
    text = raw.decode()
    if not text.startswith("HloModule "):
        raise ValueError(f"{path}: expected a complete executable HloModule")
    computations = {}
    entry = None
    current = None
    for line in text.splitlines()[1:]:
        header = HEADER.fullmatch(line)
        if header:
            if current is not None or header[2] in computations:
                raise ValueError(f"{path}: duplicate or nested computation header")
            current = header[2]
            computations[current] = {"instructions": [], "targets": set()}
            if header[1]:
                if entry is not None:
                    raise ValueError(f"{path}: multiple ENTRY computations")
                entry = current
        elif line == "}":
            if current is None:
                raise ValueError(f"{path}: unmatched computation close")
            roots = sum(i["root"] for i in computations[current]["instructions"])
            if roots != 1:
                raise ValueError(f"{path}: computation {current} has {roots} roots")
            current = None
        elif line.strip():
            instruction = INSTRUCTION.fullmatch(line)
            if current is None or instruction is None:
                raise ValueError(f"{path}: unsupported HLO line: {line}")
            root, name, shape, opcode, operands, attributes = instruction.groups()
            targets = set(REFERENCE.findall(attributes))
            for group in re.findall(r"branch_computations=\{([^}]+)\}", attributes):
                targets.update(re.findall(r"%([\w.-]+)", group))
            comp = computations[current]
            if any(i["name"] == name for i in comp["instructions"]):
                raise ValueError(f"{path}: duplicate instruction {name}")
            comp["targets"].update(targets)
            comp["instructions"].append({
                "name": name, "opcode": opcode, "shape": shape,
                "root": bool(root), "operands": operands,
                "attributes": attributes,
            })
    if current is not None or entry is None:
        raise ValueError(f"{path}: missing ENTRY or incomplete computation")
    if sum(len(c["instructions"]) for c in computations.values()) != sum(
        bool(ASSIGNMENT.match(line)) for line in text.splitlines()
    ):
        raise ValueError(f"{path}: instruction parser did not cover every assignment")

    def reach(start):
        seen, todo = set(), [start]
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            if name not in computations:
                raise ValueError(f"{path}: undefined computation {name}")
            seen.add(name)
            todo.extend(computations[name]["targets"])
        return seen

    def inventory(names):
        return dict(sorted(Counter(
            i["opcode"] for n in names for i in computations[n]["instructions"]
        ).items()))

    def work(names):
        return [
            {"computation": n, "instruction": i["name"],
             "opcode": i["opcode"], "shape": i["shape"]}
            for n in sorted(names) for i in computations[n]["instructions"]
            if i["opcode"] in WORK_OPS
        ]

    reachable = reach(entry)
    loops = []
    body_union = set()
    for name in sorted(reachable):
        for instruction in computations[name]["instructions"]:
            if instruction["opcode"] != "while":
                continue
            attributes = instruction["attributes"]
            body = re.search(r"body=%([\w.-]+)", attributes)
            condition = re.search(r"condition=%([\w.-]+)", attributes)
            if body is None or condition is None:
                raise ValueError(f"{path}: while lacks body/condition reference")
            body_reachable = reach(body[1])
            body_union.update(body_reachable)
            loops.append({
                "computation": name, "instruction": instruction["name"],
                "body": body[1], "condition": condition[1],
                "body_computations": sorted(body_reachable),
                "body_inventory": inventory(body_reachable),
                "body_work": work(body_reachable),
            })
    # Shapes expose vector/array work outside loop bodies. They do not prove
    # which source inputs contribute or whether transformations are equivalent.
    outside_work = work(reachable - body_union)
    array_work = [i for i in outside_work if re.search(r"\[\d", i["shape"])]
    conditionals = []
    for name in sorted(reachable):
        for instruction in computations[name]["instructions"]:
            if instruction["opcode"] != "conditional":
                continue
            attributes = instruction["attributes"]
            branches = re.findall(
                r"(true_computation|false_computation)=%([\w.-]+)", attributes
            )
            for group in re.findall(r"branch_computations=\{([^}]+)\}", attributes):
                branches.extend(
                    (f"branch_{i}", target)
                    for i, target in enumerate(re.findall(r"%([\w.-]+)", group))
                )
            if not branches:
                raise ValueError(f"{path}: conditional lacks branch references")
            conditionals.append({
                "computation": name, "instruction": instruction["name"],
                "branches": [{
                    "label": label, "computation": target,
                    "reachable_inventory": inventory(reach(target)),
                    "work": work(reach(target)),
                } for label, target in branches],
            })
    return {
        "path": str(path), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "entry": entry,
        "inventory": inventory(computations),
        "reachable_inventory": inventory(reachable),
        "unreachable_computations": sorted(set(computations) - reachable),
        "loops": loops, "conditionals": conditionals,
        "work_outside_loop_bodies": outside_work,
        "array_work_outside_loop_bodies": array_work,
        "computation_targets": {
            n: sorted(computations[n]["targets"]) for n in sorted(computations)
        },
        "computation_inventory": {
            n: inventory([n]) for n in sorted(computations)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    reports = [audit(path) for path in args.files]
    args.out.write_text(json.dumps(reports, indent=2) + "\n")
    for report in reports:
        body_ops = sorted({
            i["opcode"] for loop in report["loops"] for i in loop["body_work"]
        })
        array_ops = sorted({
            i["opcode"] for i in report["array_work_outside_loop_bodies"]
        })
        print(f"{Path(report['path']).name}: "
              f"instructions={sum(report['inventory'].values())} "
              f"reachable_whiles={len(report['loops'])} "
              f"body_work={body_ops} array_work_outside={array_ops}")
    print("Inspection only: loop presence and opcode lists do not certify "
          "retained intended work, equivalence, resource use or performance.")


if __name__ == "__main__":
    main()
