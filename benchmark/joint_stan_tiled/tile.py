"""Tile the oracle stan_data.json K times (subject replicas).

All index arrays in the SB emission are within-subject relative (per-subject
plate cells), so tiling = replicate each subject's segment, recompute ends.
No index remapping needed. Exact replication keeps standardized covariates
(mean/std) and category levels identical.
"""
import json
import sys

K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SRC = "/home/n/scratch/kb-agent-tmp/ReactiveKernels-brm-tgi/joint-oracle/continuous/stan_data.json"
DST = f"/tmp/bs-bench/stan_data_tiled{K}.json"

d = json.load(open(SRC))
N_SUBJ = d["n_subject"]


def segments(vals, ends):
    segs, prev = [], 0
    for e in ends:
        segs.append(vals[prev:e])
        prev = e
    assert len(segs) == N_SUBJ, f"{len(segs)} != {N_SUBJ}"
    return segs


def tile_ragged(obj):
    vals, ends = obj["1"], obj["2"]
    segs = segments(vals, ends)
    out, new_ends, pos = [], [], 0
    for _ in range(K):
        for s in segs:
            out.extend(s)
            pos += len(s)
            new_ends.append(pos)
    return {"1": out, "2": new_ends}


# op-segment structure shared by op_log_dose and PHI rows
op_ends = d["op_dt"]["2"]
op_segs = segments(list(range(d["op_log_dose_n"])), op_ends)

out = {}
for k, v in d.items():
    if k == "n_subject":
        out[k] = N_SUBJ * K
    elif k == "kernel_nsub_pk_loc":
        out[k] = d[k] * K
    elif k == "PHI_hsgp_op_log_dose_m":
        out[k] = v * K
    elif k.endswith("_ends_n"):
        out[k] = v * K
    elif k.endswith("_mem_n"):
        out[k] = v * K
    elif k.endswith("_n") and k not in (
        "indication_n_levels",
        "n_terms_p_subject",
        "n_terms_tb_subject",
        "n_terms_tg_subject",
        "PHI_hsgp_op_log_dose_n",
        "omega2_hsgp_op_log_dose_m",
        "omega2_hsgp_op_log_dose_n",
    ):
        out[k] = v * K
    elif isinstance(v, dict) and set(v) == {"1", "2"}:
        out[k] = tile_ragged(v)
    elif k == "op_log_dose":
        out[k] = [x for _ in range(K) for s in op_segs for x in [v[i] for i in s]]
    elif k == "PHI_hsgp_op_log_dose":
        out[k] = [row for _ in range(K) for s in op_segs for row in [v[i] for i in s]]
    elif isinstance(v, list) and len(v) == N_SUBJ and k != "subject_idx":
        out[k] = v * K
    elif k == "subject_idx":
        out[k] = list(range(1, N_SUBJ * K + 1))
    else:
        out[k] = v  # scalars, level counts, omega2, PHI_n

json.dump(out, open(DST, "w"))
print(f"wrote {DST} n_subject={out['n_subject']}")
