# Core constraints

These requirements govern every lowering, extension and example, including bound data. Existing nonconforming paths are limitations to fix or reject; they do not waive a requirement.

## Generated-code displays

A display labeled as generated or prepared code prints the actual `code_expr` or pre-build `kernel_expr` with Julia's ordinary expression printer. It may remove `LineNumberNode` objects from a copy. No other display rewrite is permitted: do not substitute authored source, rename or hide bindings, inline a call for presentation, or add aliases or explanatory comments to that code.

A separately labeled structural view may explain provenance. Repair a complaint about generated code in the underlying representation or lowering and verify its execution; do not repair its appearance in a printer.

## Ownership

General numerical and compiler machinery belongs in ReactiveKernels. General probabilistic authoring, binding and lowering belongs in ReactiveKernelsPPL. Reusable statistical model construction and backend emission belongs in BayesianRegressionModels, with statistical implementations shared by its backends. Domain-specific models belong in downstream packages; StanBlocks owns Stan transpilation.

Historical placement and a shared repository do not change ownership. Temporary compatibility requires a concrete migration task and a consumer-adoption condition.

## Data-dependent iteration

Do not statically unroll a loop whose trip count derives from data. Runtime inputs, bound arrays, shapes, observation counts, operation tables, ragged ranges, dose counts and derived capacities all remain data. Knowing a length during preparation or tracing does not make it structural.

Retain a runtime loop, scan or equivalent batched-array structure, with the necessary carried state and output storage. Fixed structural algebra is distinct from data-derived iteration. A backend may specialize and optimize its own executable, but RK emission must not duplicate bodies according to data values or sizes.

Inspect emitted loops and their reachable bodies, callees and equivalent array work. A surviving loop by itself does not establish the intended semantics. Movement of pure invariant arithmetic is permitted when it preserves semantics; operation counts are diagnostic evidence rather than an independent policy.

## Lazy branches

Do not replace required lazy control flow with eager `ifelse`, evaluation of both arms or equivalent predication to bypass a compiler failure. Preserve selected values, ordinary derivatives, observable effects, exceptions and invalid-access safety. Clamped indices and dummy buffers do not make an unsafe rewrite valid.

Elementwise selection between already valid values remains permitted. Pure arithmetic may be simplified or speculated when the selected result, derivative and observable behavior remain unchanged. An unused floating-point nonfinite value alone is not a correctness failure. There is no requirement for a particular conditional instruction or a blanket ban on computing unused pure intermediates.

Retain required guards in the authored graph. If a backend cannot preserve their semantics, isolate and report the unsupported case. Data-bound branch partitioning must evaluate only the taken arm for each lane.

## Concrete programs

RK lowering must not introduce a non-concrete value or runtime-dispatched call into a prepared program whose client code is concretely typed.

Non-concrete client code is discouraged with a warning that identifies the value and recipe. Refusal is opt-in through `type_stability=:error` or the corresponding project preference. The default audit covers the prepared program and its callees except Julia's own code. The narrower `:rk` and `:values` audit levels are explicit choices, not the default.

## Runtime values and sizes

Keep sizes and numerical values, including floating-point prior and configuration settings, as runtime model inputs. Do not splice them into source or source specialization. Packed positions, axis lengths, scan lengths, ordered level counts and data-derived transform bounds come from the data.

Backend compilation for a concrete shape is separate from model-source generation. A source program must not become tied to that shape. Remaining extent-dependent or branch-occupancy-dependent paths are limitations to fix, not precedents.

## Derivative authority

Author a numerical primitive's derivatives once in a pure mathematical graph: a stable primal with named partials, or forward and reverse branches in one multi-output graph. Generate every backend adapter from activity-selected cuts of that graph and attach it to an owned callable.

Do not maintain separately authored backend JVP/VJP rules, hand-written EnzymeRules or ChainRules methods, activity annotations, or finite-difference substitutions as durable fixes. Never attach a rule to a foreign function. Acceptance differentiates the authored primal through the backend's ordinary reverse path.

A backend failure remains a limitation until the backend supports the case or a generated rule on an owned callable covers it. Keep a backend-only reproducer under `benchmark/`. If it aborts the process, identify any temporarily skipped acceptance cases by name.

## Acceptance evidence

A lowering change must show that increasing relevant data lengths or capacities does not replicate bodies or control-flow regions in RK emission, including traces. Check primal and ordinary-AD agreement with native Julia, inactive branches, and supported empty and ragged inputs. A small Julia expression alone does not prove a trace retains iteration.

| Evidence | Interpretation |
| --- | --- |
| Changed-input results, derivatives, effects, ownership and access safety | Semantic correctness; do not relax it to accommodate a compiler. |
| Retained iteration and bounded emitted-body growth | Required RK structure; subsequent backend optimization is outside this structural requirement. |
| Operation inventories and missing conditionals in pure arithmetic | Diagnostics whose significance depends on semantics and scope. |
| Runtime, allocation and data movement | Workload-specific measurements with direction, comparison and conditions stated. |

Raw operation inventories need not be identical across shapes. Optimizers can simplify constants and singleton structures without changing the intended program. Record what an acceptance criterion establishes rather than treating an engineering heuristic as a user requirement.

Historical backend investigations and reproduction commands belong under `benchmark/`, outside the reader navigation.
