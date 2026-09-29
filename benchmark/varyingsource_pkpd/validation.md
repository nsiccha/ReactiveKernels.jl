# Verification of the full synthetic posterior

The runtime tested by this benchmark is `ceccb43d8d66d02c3a83e4f7f3ceb7cd4d451c77`,
including structured-bound native AD support `00d7f1f4`. The original Bruno
model is from `896137dd`; its SHA-256 is
`b203d15ecdb2940a3503a162a3ce88415b1fa763d163af2684e14ecefb2d7c2e`.
The original, `1e-10` and `1e-12` builds all exited 0; only the two BDF
tolerance pairs differ. Complete build logs are in `build-receipt.txt`.

The final benchmark run `kb-run-compact.MovNOF` exited 0 after 402 seconds at
2026-09-29T01:32:50Z, holding one compute token. It checked all original
constrained names, independent unconstraining, complete transported reverse
gradients, BDF convergence, the analytic density-offset constant and an
independent likelihood sum at all six points before recording timings.
One prepared native sampler served all three points in each workload.
The run receipt reports actual selected-source Gamma shapes, including the
corrected mode-56 stress point around shape eight.
The repository receipt normalizes trailing whitespace; the unmodified log
remains in managed scratch.

An independent table check verified all 12 rows, finite complete coordinate
and gradient vectors of lengths 232/583, paired coordinate/reference identity,
positive medians, and accuracy matching. The post-timing exported gradients
recompute every reported difference metric within rounding. Raw-table SHA-256:
`e9f73910e5a927b0b8c92eb8862397b3b289cdc60c26f30bccf841ced6851dba`.

Native regression evidence under the same preserved scratch environment:

| Check | Observed outcome |
| --- | --- |
| Core partial evaluation and AD, including structured bound records/views | 282 assertions passed (`mej69R`) |
| Existing/new PPL layout, varying/LKJ, centered draws, PK/PD grids, batch and emitter; bound and sampled innovations | 732 assertions passed; `1NwEcC` exited 0 after 514 seconds at 2026-09-28T23:53:09Z |
| Changed dose-axis/gather/censor checks plus existing QT/joint/declaration/MO/SB parity | 872 assertions passed in `FNoQ65`, including all 15 new axis cases; the scratch runner then exited 1 because it omitted the helper for the final legacy PK file |
| Corrected remaining legacy PK file | All 35 assertions passed; `OdNDcO` exited 0 after 120 seconds at 2026-09-29T01:01:02Z; 17 bind assertions overlap the previous run |
| Raw grid parity against the actual Bruno builder | 86 exact array comparisons plus 40 durable boundary assertions passed (`0RYf1K`, exit 0) |
| Native full cell and centered prior at 3/30 groups | Identical LLVM structure after fresh JIT-symbol normalization; loop/branch instructions retained |

The final expression retains one full batched cell and one centered-prior call
at both sizes. Raw node counts are 10453/10480; the sole difference is the
bound integer grouping-level table (3/30 entries), proved by diagnostic
`K5HRaO` (exit 0). Counting that data payload once yields 10450 code nodes
for both models. Preparation folds the data-only grouping encoder.

These are native checks. The scratch PPL driver excluded the Reactant seam;
the full native family explicitly rejects Reactant execution. Watson-eight
truncation, the reported BDF reference convergence and shared-host timing
variation remain limitations. Public synthetic data and these parameter points
do not establish real-fit/Mac performance or joint BRM extraction delivery.
