# Retained documentation models: investigation boundaries

The unchanged sources at BRM `c0158b4fc44faf03bc490955293a78e602bff07c`
reproduce the four reported failures. The investigation uses Julia 1.10.12-kb3,
RK 0.1.169 / RKPPL 0.1.158 as the baseline, BRM 0.2.278 and StanBlocks
0.2.36. The final probe uses RK 0.1.170 / RKPPL 0.1.160 and reaches JAX
emission in all four cases. No model has yet passed full native/JAX
acceptance. No data, prior or model substitution was used to obtain the
results below.

| Model | Original boundary | Boundary after the local repairs |
| --- | --- | --- |
| Mesh space-time field | Layout axis unavailable to JAX `keep` | Data-derived range `geometry.previous_ptr[j]:geometry.previous_ptr[j + 1] - 1` inside sparse factorization's history scan |
| Station field | Layout metadata unavailable to JAX `keep` | Host field read `geometry` on `BayesianRegressionModels._BRMGMRFFootprintSupport{Vector{Int64}}`, while preparing the packed endpoint `4 + support.geometry.ncoordinates` |
| OU state-space model | `fieldcount(Union{})` during JAX emission | `ou_path` has no applicable Julia method at the original source pin |
| Public wastewater model | Binding the latent epsilon count reads an unavailable sampled predictor | Artifact translation and kernel construction succeed; `gather_exp` has no applicable Julia method during JAX emission |

The local repairs give retained metadata its own data-only plan, preserve
structural aliases, use bound shape information consistently when binding
generated cell-latent definitions, and export their counts as runtime reads of
authored data. Fully empty ragged rows retain zero width, and empty numeric
plates return before JAX traces indexed reads in their cell bodies. Error
reports preserve the underlying unsupported operation. Fixing these faults
exposes the later boundaries; it does not establish support for the complete
models.

Focused regressions cover retained aliases and inverse-graph producers,
uniform cell sizes at 2 and 5 rows, ragged sizes `[2, 0, 4]` and
`[0, 1, 3, 0, 5]`, empty inner cells and an empty outer domain. The cell tests
compare native values and Enzyme gradients with an independent density and
finite differences, then reuse the first emitted Python module for JAX values,
ordinary reverse gradients and layout dimensions at every binding. The
growing cases retain the same recipe inventory and Python source; the empty
gather test also checks that a real nonempty invalid read still fails.

The focused tests pass 26 retained-metadata/diagnostic checks, 9 empty-plate
checks and 60 cell-extent checks. The existing core JAX suite passes 513
checks (one existing broken case); the PPL JAX suite passes 400 and the
bound-array extent suite passes 734. Existing layout, latent-plate axis and
cell-vector latent suites also pass, retaining their existing broken cases.
The PPL suites run in `Main` with the package's continuing-test runner and
shared generator helpers. The original-model probe exits 1, with all four
failures at `jax_source`, exactly as the table records. These are regression
results and identified capability boundaries, not full-model acceptance.

## Capability work required for complete model support

The mesh requires staging pure data-only metadata inside a prepared helper
and a history-scan step. The existing host split covers top-level plate cells;
the scan carries live numerical state. A general implementation must carry
prepared per-step rows and their lengths across the helper and scan boundaries,
then execute the numerical work with runtime iteration. Its scope spans the
host split, prepared-helper calls, scan inputs and ragged extent propagation.
The first independently testable increment is a small history scan with a
data-derived ragged neighborhood and live carry; acceptance must include
changed row counts, empty neighborhoods, lazy invalid-access guards, native
values and ordinary reverse derivatives. The original mesh model is the
subsequent consumer acceptance. A literal-length unroll or an eagerly masked
invalid read would not implement the required operation.

The station support value also needs a portable representation and property
semantics at the Julia/Python boundary. The current host translator handles
named tuples, but this retained model supplies a BRM support object. A general
increment must cover structured host inputs and their accessor behavior,
including any source needed to compute the properties; merely renaming the
field or converting the object to a string does not preserve its meaning.
Acceptance must vary geometry sizes and compare the resulting layout and
model evaluation. This boundary needs coordination with BRM's support-value
export, in addition to RK's host translation.

The OU and wastewater helpers are in plain StanBlocks `@deffun` blocks.
StanBlocks' documented `@juliacompat` annotation supplies native Julia methods.
The BRM helper investigation verified the OU annotation with byte-identical
Stan output and 25 native checks, but then reached the generated-loop
source-provider boundary in JAX. That source-provider capability is already
tracked by StanBlocks task `0h74hbt`; consumer occurrence
`juliacompat-ou-s-113eaac3` records the OU case. The wastewater `gather_exp`
occurrence is attached to BRM snag `docs-ou-stan-onl-fef60921`, so its helper
block can be audited within the same work. Full original-model JAX acceptance
remains necessary after those dependencies are delivered.
