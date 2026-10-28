"""Audit full default/protected HLO, including called fusions and branch regions.

Usage: python3 benchmark/audit_reactant_lazy_executable.py <module-directory>
Rejects unparsed assignments or unresolved calls; a conditional count alone
does not establish that logarithms/divisions remain inside its regions.
"""
import collections
import hashlib
import json
import pathlib
import re
import sys

ASSIGNMENT = re.compile(r"^\s*(?:ROOT\s+)?%?[\w.-]+ = ")
INSTRUCTION = re.compile(
    r"^\s*(?:ROOT\s+)?%?([\w.-]+) = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\(")
COMPUTATION = re.compile(r"^(ENTRY )?%?([\w.-]+) \(.*\) -> .* \{$")
CALLEE = re.compile(
    r"\b(calls|to_apply|true_computation|false_computation|condition|body)=%?([\w.-]+)")
SENSITIVE = {"log", "divide", "sqrt", "rsqrt"}


def audit(path):
    computations, entry, current = {}, None, None
    counts = collections.Counter()
    for line in path.read_text().splitlines():
        header = COMPUTATION.match(line)
        if header:
            current = header[2]
            assert current not in computations, current
            computations[current] = []
            if header[1]:
                assert entry is None
                entry = current
        elif ASSIGNMENT.match(line):
            instruction = INSTRUCTION.match(line)
            assert instruction is not None and current is not None, line
            name, op = instruction.groups()
            counts[op] += 1
            callees = CALLEE.findall(line)
            # This corpus uses Boolean conditionals, not indexed branches.
            assert "branch_computations=" not in line, line
            computations[current].append((name, op, callees))
        elif line == "}":
            current = None
    assert entry is not None
    pending, visited, unguarded = [(entry, False)], set(), set()
    while pending:
        name, guarded = pending.pop()
        if (name, guarded) in visited:
            continue
        visited.add((name, guarded))
        for instruction, op, callees in computations[name]:
            if op in SENSITIVE and not guarded:
                unguarded.add(f"{name}/{instruction}:{op}")
            for attribute, callee in callees:
                assert callee in computations, callee
                branch = op == "conditional" and attribute in {
                    "true_computation", "false_computation"}
                pending.append((callee, guarded or branch))
    result = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "complete_inventory": dict(sorted(counts.items())),
        "computations": len(computations),
        "reachable_computations": len({name for name, _ in visited}),
        "unguarded_sensitive_instructions": sorted(unguarded),
    }
    if ".protected." in path.name:
        assert counts["conditional"] > 0, path
        assert not unguarded, (path, sorted(unguarded))
    return result


if __name__ == "__main__":
    files = sorted(pathlib.Path(sys.argv[1]).glob("*.executable.hlo"))
    assert files, "No complete executable modules found"
    print(json.dumps({path.name: audit(path) for path in files}, indent=2))
