# Compiler capability and limits

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

ReactiveKernels has one small exact planner and several lowering surfaces built
around it. This page is the capability contract: it says which decisions the
compiler makes, which facts an author must supply, which work disappears from a
prepared hot path, and which source shapes are deliberately rejected.

The word *compiler* is used here for two related pipelines:

- the public stateless have→want compiler, which selects recipes and emits a
   straight-line Julia callable; and
- the public reactive compiler, which fixes the same selected graph into typed
   slots, validity bits, dependency closures, and generated lazy getters.

Those surfaces share graph identities, producer selection, and currentness
rules. A stateless recipe may contain arbitrary ordinary Julia behind an opaque
recipe call.

## Pure kernels

ReactiveKernels compiles pure kernels: everything a stateless graph needs, from
authoring to execution:

- function-shaped `@kernel` specifications (`KernelSpec`) and pure
   method-bearing kernel objects (`KernelObjectSpec`), including every
   DistributionKernels family;
- have→want planning and lowering: `plan`, `prepare`, `explain`, `code_expr`,
   `compose`, `extract` and preparation-time partial evaluation;
- plates and scans: `plate`, `scan`, `@plate`, `@scan`, `plate_cell`,
   `plate_body`, `scan_body` and `replica`;
- position batching with `vectorize` and `prepare_batched`;
- differentiation: `prepare_ad` with its gradient, pullback, pushforward and
   Hessian-vector entry points, and scalar and vector derivative rules;
- `prepare_nonallocating`, and the owned primitives such as `rk_expm`,
   `rk_cholesky_lower` and `rk_ode_bdf_tol`.

ReactiveKernelsPPL is built on this. So are the reactive execution layers,
`ReactiveState` and `prepare_reactive` with `CompiledReactiveState`: they run
prepared pure kernels with provenance-aware invalidation.

## Numerical metadata and its consumers

OutputSignatures owns numerical transfer rules for compiler-visible calls.
RK's internal `src/value_metadata.jl` adapter resolves callable identities and
translates indexing, broadcast and literal-power syntax into metadata queries.
It does not execute a helper, manufacture an example input, or ask Julia to
infer an arbitrary helper body. Descriptors are ordinary data; graph contents,
array lengths and literal exponents are not metadata type parameters.

The first consumer is PPL's pointwise arithmetic emission. Its former
`_PointwiseValue.rank` and `.length` rules have been removed: indexing,
scalar/vector arithmetic, broadcast axes, `sum` and numeric `vcat` now obtain
their numerical facts from OS. PPL still owns the separate fact that an
expression reads a sampled value. Density roles, conditioning, observation
domains and parameter layout remain PPL responsibilities. A scalar numerical
result does not imply any of those roles.

The second consumer is PPL's declaration and observation shape inference,
`_value_axes` (`packages/ReactiveKernelsPPL/src/arrays.jl`). Its own broadcast,
product, reduction, concatenation and adjoint rules have been removed; every
numerical axis fact now comes from OS through the same adapter, which also
reports a call's status and any sound partial facts. A literal size enters as a
known extent and any other declared dimension (`levels(g)`, `axes(M, d)`) as a
named `SymbolicAxis`, which comes back in its declared form. Bound data and a
caller's values enter with their actual types; sampled values are `Float64`.
Two PPL rules remain on top of the OS facts:

- A broadcast keeps a declared axis that an operand carries, a level axis
  first, so that a later level gather on the result stays possible.
- A per-observation gather reads its bound index column as positions, whatever
  the column's values (a level gather reads labels).

Where OS proves a shape conflict, the shape is unavailable and Julia's own error
appears when the value is computed. The exception is a declared axis sized from
the value (`axes(M, d)`), which raises the `DimensionMismatch` at once. Where OS
has no rule, the shape stays unknown: a whole reduction of an operand whose
facts are unknown is not assumed to be a scalar.

The pointwise adapter accepts a supported OS result only. Unknown or incompatible
analysis retains the original Julia expression; it does not invent a scalar
or copy an input's shape/type. A supported result is conditional on ordinary
Julia execution succeeding, so bounds and broadcast compatibility checks must
remain. A descriptor for an abstract vector is not permission to assume Base
semantics for an arbitrary custom array. Literal `x .^ -1` is queried as
`literal_pow`, preserving its distinction from `broadcast(^, x, -1)`.

Further compiler changes should follow these rules:

- Add numerical operation rules to OS, with independent Julia result checks;
  keep source binding/normalization in RK and statistical meaning in PPL.
- Preserve unknown facts and sampled dependence independently. Partial axes
  alone do not authorize fusion of an unsupported callable.
- Replace and delete one duplicated rule at a time. Each slice needs numerical,
  observation-scope and ordinary-Reverse checks, plus separate construction,
  preparation and first-call measurements on exact source/dependency pins.
- Keep metadata as cold data structures and retain runtime loops over data.
  Do not expand graph-wide compiler inference or encode observation lengths
  in generated program structure to make an analysis appear precise.

These consumers do not replace RK's general recipe typing or PPL's whole-model
shape classifier. Scans, planner migration and a general opaque-call tracing
protocol remain separate work. The next slice should name
its numerical duplicate and acceptance examples before expanding the interface.
The runnable [`benchmark/compiler_metadata_slice.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/compiler_metadata_slice.jl)
shows the complete model, query/gradient interaction and separate timing stages;
`test/test_value_metadata.jl` exercises the corresponding kernel directly.
The [recorded comparison](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/results/compiler_metadata_slice_20261015.json)
pins the isolated before/after trees and dependency versions. Both runs pass
32 value/gradient checks. Emitted size is unchanged at 238/252 nodes for the
two expression widths, at both 8 and 256 observations. These are single-run
compiler-cost checks, not evidence of a speedup.

## API map

```@eval
Main.ReactiveKernelsDocs.render_compiler_api_map()
```

## The stateless compiler

### 1. Graph construction

A `Value` has a stable identity and a declared Julia value type. The identity
is the namespace that allocated the value plus a number within it. Every
process, and every package image, allocates in a namespace of its own, so
values from different images and from the running process never coincide, and
a value that one image refers to from another keeps its identity after loading.
Its name is only a diagnostic hint. Two values with the same name are distinct;
generated local variables therefore never use a name as identity.

A `Recipe` records ordered input and output values, a callable operation, a
finite non-negative planning cost, an optional structural-CSE key, and an
effectful flag. Graph construction records producers but executes nothing. A
`KernelSpec` adds named ports, a default HAVE/WANT boundary, and an optional
ordinary call-signature adapter; it does not introduce another planner or
runtime.

Stateless `@kernel` authoring supports:

- short or long function-shaped definitions;
- required arguments, trailing positional defaults, and fixed keyword
  arguments, including required keywords;
- defaults evaluated left to right at call time, with references to earlier
  arguments;
- forward declarations and forward producer references;
- one assignment per recipe, tuple assignment for a multi-output recipe, and
  repeated assignment for alternative producers;
- an explicit return boundary or implicit derived sink outputs;
- bare same-type identities, which become canonical aliases rather than copy
  recipes;
- recipe costs and opt-in structural-CSE keys; and
- arbitrary ordinary Julia inside a recipe right-hand side, including local
  control flow, exceptions, comprehensions, closures, and `do` blocks.

The last item works because the recipe body is one opaque callable. The graph
knows its free-port inputs and declared outputs, but it does not infer the
callable's internal effects, branches, allocations, or mathematics.

Stateless signatures do not support positional or keyword splats. A port may be
untyped metadata, but a typed port is checked at the generated function
boundary. A defaulted or keyword argument remains a real HAVE port after the
signature adapter has supplied its value.

### 2. Exact have→want planning

Planning treats HAVE values as authoritative cut points. Even if a HAVE value
also has producers, it is never recomputed. Repeated or structurally aliased
HAVE entries collapse to one canonical input while the first-seen positional
order is retained.

The planner then performs these steps:

- **Candidate frontier.** Starting from every wanted value not already in HAVE, walk producers
   backward until reaching HAVE. This forms the candidate-recipe frontier.
   Effectful recipes are excluded.
- **Exact subset search.** Search exact subsets of that frontier with branch-and-bound. A search state
   contains selected recipes and excluded alternatives. HAVE plus every selected
   output gives syntactic coverage; its unresolved frontier is the WANTS and
   selected inputs outside that coverage. Coverage alone does not establish
   executable availability.
- **Groundability.** For a value with alternative producers, test forward
   reachability from HAVE while withholding that value. A recipe whose only
   non-HAVE output is that value cannot be its first producer if one of its
   inputs remains unreachable. Such a recipe adds no value after another
   producer has run, so discard it. Multi-output recipes that can supply other
   non-HAVE values remain candidates.
- **Branching.** Pick the first unresolved value and branch over every remaining
   candidate recipe that can produce it. Non-negative costs allow pruning any
   branch already more expensive than the incumbent. Each branch excludes the
   alternatives explored before it, partitioning subsets instead of revisiting
   them in different selection orders.
- **Ranking.** Rank complete selections lexicographically by total declared cost and then
   by number of recipes. Recipe identifiers provide deterministic traversal
   order; the cost is an author-supplied planning weight, not a timing estimate.
- **Execution check.** Accept a complete selection only if availability-based Kahn ordering can
   execute it. Each ready recipe adds all of its outputs to the available set.
   This avoids inventing a cycle when overlapping multi-output recipes admit a
   valid order through a different producer.
- **Grounding a stalled selection.** If outputs cover every obligation but
   execution stalls, branch over external producers of any still-unavailable
   WANT or selected input. A cycle can be grounded at different inputs, so
   choosing only the first unavailable value would lose valid plans. Forward
   reachability with every non-excluded candidate rules out a branch if even
   those candidates cannot supply the stalled frontier.
- **Recorded result.** Record the topological recipe order and one selected owner recipe for each
   produced canonical value. Collateral outputs from a non-owner recipe may be
   computed, but they never overwrite HAVE or an earlier selected owner.

Groundability pruning removes inverse routes that depend on their own result,
without choosing a heuristic plan or executing any recipe body. Exact
optimization with arbitrary shared alternatives can still require exponential
search; topology and declared costs determine that work, rather than parameter
or observation counts alone. The pass preserves authored plates, scans and
scalar laws.

The result minimizes total declared cost, then recipe count, over executable
subsets of a finite graph with additive finite non-negative recipe costs.
The graph may contain cycles. For example, with `x → a` at cost 1,
`a → (a, b)` at cost 0 and `x → b` at cost 2, HAVE `x` / WANT `b` selects
the first two recipes at cost 1. The independent producer of `a` grounds the
self-output recipe. Without a grounding route, that recipe cannot execute.

Completeness follows from the frontier obligations. An uncovered value needs
a producer in any executable completion. At a cyclic stall, some additional
recipe must produce a currently unavailable frontier value before all selected
recipes can run. Branching over those producers retains every possible
completion, partitioned by its first included alternative. The reachability
check uses all non-excluded candidates and therefore cannot discard an
executable completion. Every accepted subset still passes the full execution
check; the greedy incumbent also passes that check before it can prune search.

Search remains exponential in the worst case. An impossible query, including
a cycle with no executable grounding or an effectful-only route, fails with a
`PlanningError`; impossible-query diagnostics report the missing producer frontier.

The planner does not do algebraic simplification, constant folding, common
subexpression discovery from Julia code, shape inference, symbolic
differentiation, or runtime profiling.

### 3. Structural common-subexpression elimination

Structural CSE is explicit and conservative. A new recipe coalesces with an
earlier recipe only when both are non-effectful, both have the same non-null
CSE key, their canonical input identities match in order, and their output
arities and corresponding declared types match. The new outputs become aliases
of the old outputs.

The graph validates the complete alias mapping before mutation, including
conflicting and cyclic multi-output mappings. Composition replays recipes
through the same CSE rules, so equivalent fragments may coalesce.

A matching function name, equal output name, syntactically similar expression,
or coincidentally equal runtime value is not CSE evidence. Effectful recipes are
never CSE candidates. CSE is not an algebra system: commutativity,
associativity, distributivity, and floating-point reassociation are never
assumed.

### 4. Lowering and emitted ABI

Lowering turns the selected plan into a straight-line anonymous Julia function.
It creates collision-free locals from canonical value identities and takes the
selected operations in a positional tuple, one slot per recipe. A recipe whose
operation is its authored source runs that source in the body, as a `let` that
binds the slot first, then the source's parameters to the recipe's arguments
and its captures to the fields of the authored closure the slot keeps:

```julia
a = let var"#op1" = __ops__[1], var"#x#op1" = x, var"#c#op1" = getfield(__ops__[1], :c)
    <the recipe's source over x and c>
end
```

The source's own names are resolved in the module that defined it, as the
rest of the body resolves globals. An operation whose code is not its source
is called through its literal tuple index: a raw function, a plate, a scan, a
branch, a bound constant, a source that reads a module binding the module can
reassign (its generated source function names the value the binding held when
the kernel was defined), or a source outside the emitter's grammar. HAVE
values appear in their preserved boundary order; WANT values appear in their
requested return order. A single WANT returns one value and multiple WANTS
return a tuple.

Multi-output recipes use one tuple destructure. If a later recipe emits a value
already owned by HAVE or an earlier selected producer, lowering binds that
collateral result to a discard local rather than changing the logical value.
The generated function closes over no graph object and does not resolve
operations through module globals.

`RuntimeGeneratedFunctions` compiles the expression. Lowering's scratch locals
(plate axes and indices, scan carries, the renamed locals of a spliced embedded
kernel) are alpha-renamed to canonical names before the expression reaches its
content-addressed cache, so an unchanged plan lowers to one callable type:
preparing the same graph shape again — including a `bound=` rebinding to data
of the same types — reuses that type and compiles once per process rather than
once per preparation. A program compiled while a package image is generated
belongs to that image's cache instead, and an image records the generated
function it made for each program under the identity the same expression has
in ReactiveKernels' cache: its argument names and content id. Loading the image
registers that record, so preparing an identical program afterwards returns the
image's generated function and runs the code the image compiled for it, rather
than compiling the program again in every process. The cache, not the callable,
retains the expression:
the callable is identified by its type alone, so a prepared callable holds no
mutable `Expr` for Enzyme to prove read-only where its call is not inlined, as
for a prepared lazy-arm child. The resulting
`PreparedKernel` stores the compiled callable, its concrete operation tuple,
and inspection metadata. Calling it performs no graph traversal, producer
selection, topological sort, cache lookup, or dynamic scheduling. Julia still
specializes and executes the recipe callables normally; their own dispatch,
allocations, exceptions, and side effects are not erased by the graph compiler.

Fresh authored definitions retain distinct provenance even when their source is
identical. At prepared-kernel assembly, execution uses the existing generated
source companion and its concrete lexical captures. The execution copy carries
no definition token: the original operations, with their tokens, remain in the
plan, source recipes and `lowered_recipes`. This lets identical fresh builds
reuse operation and derivative types without merging their graph identities or
caching bound data. The execution copy holds no provenance value either, so a
capture-free operation table stays free of heap references that Enzyme would
otherwise have to prove read-only in the differentiated callable. Bodies with
local functions, generators, exception scopes, or boxed lexical cells retain
Julia's native closure path. A companion compiled while a package image is
built, before the image's generated method is visible, is wrapped for a
latest-world call. Once that method is visible, as in every process that loads
the image, the execution copy holds the generated function itself, so a graph
defined in a package image runs its cells as one built at run time does.
Before then, preparation keeps the guarded closure, whose method query selects
the native closure after loading. Private generated
names in the derivative operand pack are canonicalized in that actual pack;
the operands and their positional reads are unchanged. Construction, planning,
and preparation still perform work, so reusing a build remains useful.
Colliding local names receive suffixes from their stable plan order, rather
than fresh graph IDs, so repeated child composition can also reuse code.

One rewrite reaches into those callables. Natively, a generator fold over a
data-length iterator, `sum(term for i in iter [if condition]; init = x)` with
`iter` an `eachindex`, `axes`, `range` or `a:b` call, runs as its
explicit-accumulator loop in every recipe, plate cell, scan step and
`@traceable` method body. A call head resolved to its binding, as `@rkppl`
lowering emits `1:P` (a `GlobalRef` to Base's `:` in the model module), is the
same iterator. Base evaluates `sum` with `init` as the left fold
`foldl(add_sum, …; init)` on Julia 1.10 through 1.13, so the loop returns
Base's value bitwise, in Base's order: the iterator, then `init`, then the
condition and the term for each element. The rewrite exists for Julia 1.10's
inference. When a Base fold is inferred inside another fold's term (a nested
sum, or a sum in a helper that the term calls), it re-enters `mapfoldl` with a
larger signature, inference widens that call to `Any`, and every element then
dispatches dynamically and boxes. A 4 × 4 nested sum allocates 10 times (about
300 B) per call, and a 14-step scan whose step evaluates one allocates 4,256 B
beside its 176 B result vector. Julia 1.12 infers the same nest concretely. Some folds keep the
authored call: one whose term or condition assigns, declares, returns, jumps,
loops or holds a macro call, one without `init`, one over another iterator (a
tuple's fold is Base's unrolled one), and one whose `sum` is not Base's. A
plain helper's own nested sum also keeps it, because RK never sees that body:
define the helper with `@traceable`, or write it with loops.

`transform` applies user-supplied expression passes between lowering and
compilation. The compiler preserves pass order but cannot prove that a pass
preserves semantics. A transformed expression is therefore part of the user's
trusted compilation boundary.

`PreparationCache` is keyed by graph identity and version, canonical ordered
HAVE/WANT signatures, and pass identities. Mutating a graph increments its
version, so a cache entry for the old graph cannot be returned. Cache lookup is
preparation-time work only.

### Visible throw policy

`prepare(spec; on_error = :ignore)` opts into a source transform that replaces
visible `throw(...)` sites with `nothing`, including the throws in expanded
`@assert`s. The default `on_error = nothing` retains normal throwing behavior.
The same option is available for graphs, plans and `prepare!`; caches keep
the two policies separate. Bound-data mathematics uses the selected policy
before its first evaluation.

The transform sees captured `@kernel` source and the bodies of `@traceable`
helpers. It keeps the original methods and selects separate stripped bodies
only for an opted-in prepared call. Ordinary opaque function bodies stay
opaque and their errors remain errors. It preserves unrelated lazy branches
and retained loops, and does not evaluate a removed throw's message arguments.

This first policy simply continues execution after a removed throw. It does
not map failure to a fallback value, provide exception recovery, or make
invalid mathematical inputs valid. Further policies are not implemented yet.

### Concretely typed programs

RK tries to prepare programs in which every value infers a concrete type.
When one value is not concretely typed, every recipe and plate cell that reads
it dispatches dynamically on each evaluation, and native reverse-mode AD can
refuse the program after a long derivative compilation.

RK's own lowering is held to this by a gate test, `test/test_type_stability.jl`.
Its corpus covers plates, nested and chained plates, scans, embedded kernels
and endpoints, design matrices, concatenation and bound data, all with
concretely typed client code. For each kernel the test infers the prepared
native body and the prepared gradient call for concrete arguments, follows
every statically invoked RK callee, and requires that no used value is
non-concrete and that no call is dispatched at run time, outside paths that
only throw. A small union of concrete types, such as the `Union{Nothing,Tuple}`
of Julia's iteration protocol, is accepted. A kernel calling a function whose
result is uninferred is the test's negative control.

An RK-introduced residue that cannot be removed would be exempted with its
reason in `src/type_stability_audit.jl` (`_CODE_FINDINGS_JUSTIFIED`); none is
exempt.

Client code is checked when it is prepared, with the same audit.
`prepare` checks from the declared types of the HAVE ports where they are
concrete and from bound data; `prepare_ad`, and so `prepare_sampler`, checks
from the exemplar arguments and audits the callable its backend
differentiates. How far the check reaches is `type_stability_level`:

| level | checks |
|---|---|
| `:values` | every value of the program: a recipe whose inputs are concretely typed but whose result Julia does not infer concretely (`Any`, an abstract type, a `Union`, or an array of such elements) is reported, naming the value and its recipe. Values computed from such a value, or from a HAVE port whose type is not known, are not reported again. |
| `:rk` | also the code of the prepared program: the body ReactiveKernels generates and the functions of ReactiveKernels' packages it calls. A used value Julia does not infer concretely, or a call dispatched at run time, is reported with its method. Every other function is a leaf: its call must dispatch statically and return a concrete type. |
| `:all` (default) | also every function that code calls, a client helper's body included, except Julia's own Base, Core and standard libraries, which stay leaves. |

So at `:all`, a helper that dispatches at run time but asserts its result
(`f(x) = (g(x))::Float64` over an uninferred `g`) is reported, and so is a
declared local (`mu::Vector{Float64} = f(x)`) over a helper whose result is
not inferred; at `:values` neither is.

The code that Julia's logging macros (`@debug`, `@info`, `@warn`, `@error`,
`@logmsg`) write into a function's body is Julia's own and is not reported at
any level, the record's message and fields included. It dispatches at run
time by design: the logger is a dynamically scoped `AbstractLogger`, and
`shouldlog` and `handle_message` are called in the latest world. It runs only
when its log statement is reached, and dispatches only when a logger accepts
the record. So a dependency that logs on a cold branch, as SpecialFunctions'
`beta_inc` does with `@debug` on underflow, passes `:all`, while a run-time
dispatch of that dependency's own code is still reported.

`type_stability` is what preparation does with a finding. By default (`:warn`)
it logs a warning once per process and succeeds, so the primal of such a
kernel keeps working. `:error` refuses the program: every preparation throws a
`TypeInstabilityError` listing each finding. Both settings are keywords of
`prepare`, `prepare!` and the `prepare_ad*` functions, which take precedence
for one call and the preparations nested in it, and ReactiveKernels
preferences of the same name, which a project sets once:

```julia
using Preferences, ReactiveKernels
Preferences.set_preferences!(ReactiveKernels, "type_stability" => "error")
Preferences.set_preferences!(ReactiveKernels, "type_stability_level" => "rk")
```

This writes `[ReactiveKernels] type_stability = "error"` (and the level) to the
project's `LocalPreferences.toml`. Every preparation in that project then
follows it, including preparations that other packages make for it (RKPPL's
`prepare_sampler`), its tests (`Pkg.test` copies the preferences), scripts and
precompile workloads. The preferences are read at the first preparation of a
Julia session; a new session picks up a change without recompiling anything.
Cached preparations are kept per setting, so a kernel prepared under `:warn`
or a lower level is never returned to a stricter call.

When an abstract HAVE declaration hides a concrete input type needed to infer
an authored plate's rank, `prepare(...; exemplars = (x, ...))` opts the native
lowering into type evidence from one exemplar per remaining positional HAVE
port. Calls with different input types are refused; prepare again for those
types. No exemplar length or value is baked into the loop. RKPPL's
`prepare_sampler` uses the converted `Vector{Float64}` exemplar by default,
and its public calls convert other real vectors to that type. A graph prepared with exemplars
compiles afresh rather than reusing its bound-preparation cache.
The authored `AbstractVector{Float64}` boundary remains broad; for example,
a coefficient slice can be a concrete `SubArray` view. Native plate inference
uses that concrete inferred subtype when available and keeps the declaration
when inference is weaker.

```julia
uninferred_curve(t, a) = Base.inferencebarrier(t .* exp(a))

@kernel location(a::Float64, s::Float64, t::Vector{Float64},
                 y::Vector{Float64}) = begin
    mu = uninferred_curve(t, a)
    pointwise = plate(y, mu) do yi, mi
        -0.5 * ((yi - mi) / s)^2 - log(s)
    end
    total::Float64 = sum(pointwise)
end

prepare(location)
# ┌ Warning: ReactiveKernels: `mu` is not concretely typed. Its recipe
# │ `mu = uninferred_curve(t, a)` infers `Any` from input types
# │ (Vector{Float64}, Float64). …
```

The remedy is in the client function: make its result inferable, or assert it
inside the function (`f(x) = (...)::Vector{Float64}`). In a kernel that
contains a plate, a scan or an embedded kernel, a concrete declaration on the
assignment (`mu::Vector{Float64} = uninferred_curve(t, a)`) also asserts the
value, because those kernels declare their typed locals. Each warning is
issued once per process. It is logged in the group `:type_stability`, so a
logger can filter it.

### Preparation-time partial evaluation

When some HAVE values stay fixed across many calls, `bound` opts into a
Plan→Plan partial-evaluation pass. For an authored `KernelSpec`, bind them by
port name with a NamedTuple such as `bound = (; observations)`. The resulting
kernel accepts only the remaining inputs, for example `kernel(parameters)`.

For a low-level `Plan` or `Graph`, the corresponding form is one
`Value => data` pair or an iterable of pairs. Bound ports must belong to the
selected HAVE boundary. They are removed from the returned callable's inputs;
unbound ports retain their relative order.

Partial evaluation partitions the already selected recipe order. A recipe is
hoisted only when all of its transitive inputs come from bound ports (or from
earlier hoisted recipes). That data-only prefix executes exactly once during
preparation. Values needed by the residual body re-enter it through zero-input
constant recipes, so downstream lowering sees an ordinary `Plan` and the hot
path performs no graph traversal or data-only recomputation.

Binding freezes values for that preparation; changing or mutating the data is
not an invalidation signal. Rebind by preparing again from the original graph
or plan. The pass does not rewrite mathematics or mutate the shared graph; it
splits a recipe only along the structure its author wrote (see
[Data-only subexpressions](#Data-only-subexpressions)). The default
`bound = ()`/`bound = (;)` path is unchanged.

The same named `bound` contract is available on `prepare_nonallocating`, whose
lowering rewrites only the residual work. Optional downstream preparation
surfaces consume that same residual plan rather than implementing another
partial evaluator.

#### Rebinding

Most of what a bound preparation does is independent of the bound values:
planning, the prefix's lowering and compilation, and the residual's lowering
and compilation. When one graph is bound again and again with fresh data — a
request-time binding per user interaction — that fixed cost can exceed the
bound-data mathematics itself. So `prepare(spec; have, want, bound)` (and the
`Graph` form) keeps that value-independent work per graph version, boundary,
bound port set and passes: the plan, the prepared prefix kernel, and the
compiled residual. A later binding of new values to the same ports runs the
prefix on those values and returns the residual kernel around the same
compiled callable with a fresh constant table; only what is recomputed per
binding changes. The entry is keyed on the bound ports, not on the bound
values or their types: a later binding of differently typed data on the same
ports (a different plan struct, an integer vector for a float one) reuses the
entry, and its generic compiled bodies specialize per concrete type on first
use as any Julia call does; a lazy branch on a hoisted constant remains a
per-call branch and follows each binding's value.

The graph holds these entries itself, so they live exactly as long as the
graph: a graph authored as a module constant keeps them for the process, a
graph built per request drops them with it, and a mutation of the graph starts
afresh. A binding made while a package image is produced — a module-level
`const K = prepare(graph; bound = …)` or a precompile workload — is kept in
that image, so the first binding after loading it compiles nothing. Passes
that are not singletons (closures) are not retained: such a binding is prepared
afresh, like a `Plan`, which is always prepared afresh.

`prepare!(cache, spec; have, want, bound)` does the same through a
caller-owned `PreparationCache`, for a caller who wants the entries' lifetime
decoupled from the graph's (dropped with the cache, or shared across graphs).

Position batching of a rebound kernel is reused the same way. `vectorize`
lowers and compiles position drivers for the whole graph: the owning driver,
and with `reuse = true` or a `NativeScheduling` the borrowed driver, the
shared prefix and the per-worker residual. For a kernel rebound from a cached
residual, the first `vectorize` keeps those drivers per batched ports,
`reuse`, scheduling and operation-table type, without the binding's
constants. A later binding's `vectorize` with the same key rebuilds them
around its own constants, without lowering or compiling, so binding and
vectorizing again costs what the rebinding `prepare` costs. Each call still
returns an independent instance with its own buffers, and the kept drivers
retain no binding's values. A residual specialized per binding (below) is
vectorized afresh for each binding. (Before, every `vectorize` of a rebound
kernel lowered and compiled the graph again: 52–68 ms and 26–30 MB per
binding on a 12-port synthetic composed graph whose rebinding `prepare` took
0.3 ms. On a large composed simulation graph whose plates the inner-plate
pass rewrites, binding a design and vectorizing its reader took 490–520 ms
and 170 MB, 400–440 ms of it in `vectorize`; it now takes 4.5–4.9 ms and
5–6 MB, with `vectorize` at 0.3–0.5 ms. Snag `rebinding-a-comp-d03576ab`.)

For a residual containing authored plates the inner-plate pass below runs on
every binding (it declines before evaluating anything when a plate has
nothing bound to specialize). A plate it rewrites to read bound-only cell
values cached at preparation follows from the pass's caching decisions, not
from the cached values, so the entry keeps that rewrite: a later binding that
decides alike has the same residual up to its cached values and shares the
compiled residual, as a binding the pass leaves alone does. A plate the pass
partitions by a data-bound branch, or replaces by a bind-time value, bakes
that binding's structure in: such a binding is still specialized and compiled
per binding and reuses only the plan and prefix. (Before, every binding the
pass rewrote was prepared, lowered and compiled afresh: 63–88 ms per design
on the composed simulation graph above, now 1.6–4.1 ms.) Every other binding shares
the compiled residual of its class and plate rewrites, whatever the bound
values' element types or array lengths: an inline plate over a bound domain
rebinds to a new domain length at the cost of the prefix, exactly as the same
plate in a separately prepared child does. A plate that reads a bound
argument lowers for that argument's class — scalar, vector or range, rank-N
array, or tuple — as a port declaring that class would, even when the port is untyped: a bound
`1:n` domain has no runtime axis guard or per-coordinate bookkeeping, and a
recipe that reads only a bound scalar runs once above the loop. So each class
has its own compiled residual; a rank-0 array or any other bound value keeps
the undeclared port's runtime guard.
(Before, such a residual was templated per bound-value type and array-shape
signature, so each new bound array length lowered the residual again and kept
another template: 2.6 ms and 1.16 MB more per new schedule length on the
ShinyRK simulation graph, and about 43 kB retained per length.) The entries
never observe bound values after preparation
and keep none of them: a template holds the compiled callable and the
value-independent tail of the operation table, not the kernel of the binding it
was built from. Binding freshness remains the caller's responsibility as for
`prepare`.

Rebinding itself is not specialized on the graph's kernel types. Its code is
compiled once, for every graph, into the ReactiveKernels package image, so the
first rebinding of a graph in a process compiles nothing and costs what any
later one does. (Before, the first rebinding of each graph in a process
compiled the rebinding path for that graph's prefix and residual kernel types:
290 ms and 40 MB for the ShinyRK simulation graph, whose steady-state
rebinding costs about 1 ms. Later, while that untyped path was compiled only
incidentally with the first binding, a change to its arguments moved its
compilation back to the first rebinding in each process: about 40 ms and
0.55 MB on the same graph.)

#### Data-bound branches in plate cells

A plate cell statement whose right-hand side is a lazy branch (`c ? a : b`,
`if`/`elseif`/`else`, `a && b`, `a || b`) is split during preparation when its
condition reads only bound data. The condition may read a bound plate argument,
or a bound-only cell value that the pass has already evaluated per lane.

1. The condition is evaluated once per lane, and the lanes are grouped by
   outcome.
2. The plate becomes one plate per taken arm, each over that arm's lanes.
3. The plate's arguments follow the arm's lanes:
   - bound arguments are gathered during preparation;
   - live arguments are gathered at run time by constant lane indices; native
     execution reads an array argument's lanes through a view of it rather
     than a copy;
   - a singleton or scalar argument broadcasts over the lanes.
4. A nested arm is split again over its own lanes only, so an inner condition
   is never evaluated for a lane that does not reach it.
5. The results are combined:
   - a sole `sum(pointwise)` consumer becomes a sum of the arm sums;
   - any other use of the pointwise vector reassembles it in lane order with
     one gather. Like the lane indices, its constant lane order is a bound
     operand rather than data held by the operation, so a backend receives it
     as an inactive array. The arm vectors are concatenated as the native
     kernel body concatenates (see the [core constraints](constraints.md)),
     so an arm that reads no live value, such as a constant fallback, can sit
     beside a live arm under native reverse.

A lane-invariant condition selects its arm without splitting.

An arm with its own graph (a plain arm expression, authored as a cell of its
own as described below) becomes that graph's recipes in its plate, as in the
native shared-condition lowering. Its work over shared plate arguments
therefore runs once per call, above the lane loop, and only when some lane
takes the arm. An arm without a graph, or one reading no lane value, runs as
its closure on each of its lanes.

The prepared program contains no branch for that condition, so no backend
receives one:

- every lane evaluates exactly the arm it takes;
- gathers inside an arm are in bounds by construction.

An empty bound domain has no lanes to group. It runs no cell and evaluates no
condition. Its result is empty, with the element type inferred from the
authored native cell, as in Julia's empty broadcast.

The number of arm plates is bounded by the cell's branch structure, never by
the data. A condition that reads a live value stays an ordinary lazy branch.
This includes the shape of a live array, such as `length(c)` of a runtime
input. Only one-dimensional plate domains are split.

An arm may contain a constructed object endpoint, such as
`normal(lpi, 1.5).logpdf(yf)` under a missingness guard. The endpoint's
recipes are rendered as source inside the arm, so each arm plate evaluates its
own endpoint copy and no backend receives the branch.

Without a binding, the cell runs its authored lazy branch unchanged.

#### Data-only subexpressions

A recipe that reads bound data and a live value runs whole on every call, and
in a plate cell, in every cell. Authoring therefore records, beside each
recipe's ordinary body, the strict subexpressions that some binding could make
data-only while the expression around them is not: calls, field reads and
indexing that read fewer of the recipe's arguments than the subexpression
enclosing them, and that call more than cheap arithmetic (comparison,
`+`/`-`/`*`/`/`, `min`/`max`/`abs`, conversions and literal integer powers).
Under `bound=`, each maximal one whose arguments are all bound becomes a recipe
of its own. At top level it joins the data-only prefix and runs once per
binding; in a plate cell it is cached per lane like any bound-only cell value.
The rest of the recipe reads its value.

```julia
@kernel binomial_loglik(logit, y, n) = begin
    pointwise = plate(y, n, logit) do yi, ni, li
        loggamma(ni + 1.0) - loggamma(yi + 1.0) - loggamma(ni - yi + 1.0) +
            yi * li - ni * log1pexp(li)
    end
    total::Float64 = sum(pointwise)
end
```

With `bound = (; y, n)` the log binomial coefficient is one cached value per
lane. Each cell computes only `yi * li - ni * log1pexp(li)` and adds it. With
`y` alone bound, `loggamma(yi + 1.0)` is the cached value instead.

- Only strict positions are entered: call and broadcast arguments, field reads,
  indexing and tuple elements. Branch arms, short-circuit operands,
  generators, closures and macro calls keep their own evaluation, although
  the whole call that holds one can be a cached node. A dotted call is cached
  only whole, which keeps Base's broadcast fusion.
- A data-bound branch arm is split after the plate partitions its lanes, so
  each lane caches only what its own arm computes.
- An arm of a branch whose condition reads a live value — a validity check
  such as `(y >= 0) & (phi > 0)` around a count likelihood — runs only where
  the condition holds on that call. Its data-only nodes are computed for every
  lane at preparation all the same; the branch stays lazy, and the arm reads
  the cached value where it is taken. A node whose computation throws on any
  lane stays in its arm, so preparation never raises an error the original
  program would not.
- The same split applies to sequential bindings and constructed endpoints in
  those arms. Their retained arm graphs supply the dependencies of local
  bindings, including live locals inside an inline `let`. Preparation caches
  the pure data-only prefix and keeps the selected residual plan beneath its
  original guards. A speculative value that cannot cross the cache boundary
  also stays inside its arm. This includes RKPPL's parameter-support guards
  around distribution endpoints, without rewriting individual families.
- The parts apply the authored operations to the same operands in the same
  order, so values and derivatives are unchanged; nothing is reassociated or
  simplified.

`code_expr` contains a cached part as `bound_plate_<name>` in a cell, or as a
hoisted constant at top level, and the remaining recipe reading it by name.

### 5. Composition

Low-level `compose` preserves globally stable `Value` identities when graph
fragments already share them. Named `KernelSpec` composition instead creates a
fresh graph, unifies same-name ports only when their declared types agree,
copies both recipe sets and aliases, and reruns structural CSE. Its default
call boundary remains the base spec's boundary; adopting the fragment boundary
must be explicit.

Composition does not resolve same-name type disagreements, rename ports, infer
adapter recipes, or merge incompatible call signatures semantically.

A direct stateless `KernelSpec` call in another stateless `@kernel` is the
call-site-oriented composition form. Each call clones the nested graph with
fresh value identities and aliases its typed default HAVE/WANT boundary to the
arguments and assignment outputs. This permits repeated calls without internal
name collisions and leaves no residual runtime call: planning, CSE, lowering,
reactive preparation, visualization, and batching all operate on the one
fused graph.

Computed arguments and a single-output child used inside a larger expression
compose through named, aliased, qualified (at any depth, `Pkg.Sub.child`) and
`GlobalRef` bindings. A builder
that authors such a graph with `Core.eval` can immediately prepare and call it
from its existing world; callers need no `invokelatest` barrier. Computed scalar
compositions also retain ordinary Enzyme reverse from that builder. Lexical
captures stay live and lazy arms evaluate only the selected branch.

## Batch and replica lowering

### Plate: fuse a scalar density across observations

`plate` starts from one pure scalar plan. An authored `plate(...) do` body is an
ordinary RK graph: transparent nested `KernelSpec` and kernel-object calls are
spliced before planning, so their internal dependencies remain visible. An
ordinary opaque Julia operation keeps the normal `Recipe` contract—pure unless
explicitly marked otherwise—without implementation inspection. Effectful
recipes are excluded by the planner and cannot form a plate plan.

Plated HAVE values follow Julia broadcasting. Compatible dimensions zip,
singleton dimensions expand, scalars repeat, and `Ref(value)` makes an
array-valued input atomic. An authored `plate(...) do` cell shares a value
across cells by closing over it, never through a `Ref` argument
([Plate-cell scope](batched.md#Plate-cell-scope)). The complete broadcast shape
is instantiated before any recipe executes or output is mutated.

Lowering propagates each selected recipe's transitive plated HAVE dependencies.
Native Cartesian traversal recomputes a recipe only when a broadcast dimension
kept by one of those roots changes. This is the dynamic equivalent of placing
the recipe at its narrowest valid nested-loop boundary: work depending only on
an outer scale dimension runs once per scale coordinate, with its last scalar
result reused through inner dimensions and no axis-sized cache. Recipes with no
plated dependency are emitted once above the traversal.

An authored cell is split into recipes before planning. Besides its
statements, each maximal subexpression that reads only shared values (closure
captures, `Ref` operands, plate arguments whose port is annotated with a
`Number` type, and cell locals computed from them) becomes a recipe of its own.
In `f(t, exp(log_k))`, `exp(log_k)` therefore runs once, above the traversal,
exactly as in `k = exp(log_k); f(t, k)`. Such an annotated argument stays a
zipped operand; its split-off recipe runs once because a number has no plate
dimension. An unannotated plate argument is not known to be a number when the
cell is split: annotate its port, or name the value as a cell local, which the
traversal runs once when the argument is a number. Only positions that Julia
evaluates whenever the cell runs are split: call and broadcast arguments, field
reads, indexing and tuple elements. A branch arm, loop, generator term,
closure or macro call keeps its own evaluation, as do the arguments of a
mutating (`!`) call and an index expression reading `end`. Like a named
invariant, a split-off one runs once per call even when the plate has no
cells. A `scan` step is not split: its native lowering evaluates every step
statement at every step, named or not.

A cell statement whose right-hand side is a lazy branch keeps each arm's own
evaluation, but a branch whose condition reads only shared values (closure
captures, `Ref` operands and `Number` plate arguments, including scalar results
inferred from unannotated caller recipes) takes the same arm at
every cell. The native lowering decides such a condition once per call, above
the traversal, and then runs the cells of the taken arm alone. Each plain arm
expression is also authored as a cell of its own, so its shared work is
lifted like any other invariant: the support guard
`sigma > 0 ? normal(mu, sigma).logpdf(y) : -Inf` computes `log(sigma)` once
per call, and only when `sigma > 0`. An empty domain evaluates neither the
condition nor an arm. A condition that reads the cell, including a combined
one such as `s > 0 && x > 0`, keeps its branch in every cell. An arm with
shared work is inlined into the cell loop, and that work runs at the first
cell that takes the arm and is reused by the later ones: an arm no cell takes
evaluates nothing, and a shared value that throws does so at the same cell as
before. An arm without shared work keeps its own evaluation. Constructed
endpoints spliced into a plate cell get the same treatment: their authored
branch-arm recipes are exposed to the plate lowering, so `loggamma(shape)` in
a Gamma density or `log1p(-zi)` in a zero-inflated count density runs once at
the first lane that takes its arm when its input is shared. Local bindings
inside an arm remain separate recipes. The graph keeps the authored branch.

A cell may return a tuple of its locals, `(m, s, d)`, that other plates read
entry by entry (`c[2]`). Each query then runs only the cell statements of the
entries its plan reads: an unread entry takes the value of a read one, so the
cell keeps its arity and its readers are unchanged, and statements that only
unread entries need leave the cell. A query of that entry still computes it,
and a reader that takes the tuple other than by a literal entry (`first(c)`, a
`Ref(cell)` operand) keeps every entry. RKPPL's shared `@plate` cells are such
plates, so a per-index value that no density reads costs the sampler and its
gradient nothing, and adds none of its AD requirements to them.

The per-element WANT is
either accumulated with a reducer (sum by default) or collected in the one
requested broadcast-shaped output.

The native reducing body materializes no per-observation result vector.

The current plate domain is deliberately narrow:

- at least one non-`Ref` HAVE value must instantiate a broadcast axis;
- every recipe the cell lowers has exactly one output: a destructuring
  `(a, b) = f(x)` in the cell, or in a kernel the cell calls, is planned as one
  recipe with several outputs and then lowered as one recipe computing `f(x)`
  plus one element read per name, as Julia's `Base.indexed_iterate` binds it;
- there must be exactly one scalar WANT, and it must transitively depend on a
  batched port;
- the native dependency scheduler assumes the scalar recipe contract is pure;
  known effectful recipes cannot be selected; and
- a non-null reducer is spliced as a bare callable, with the documented surface
  intended for Base reducer symbols.

This is dependency-stratified map-plus-reduce/collect. It is not a general scan,
fold with loop-carried state, segmented reduction, parallel reduction,
associative tree reduction, or an effect-ordering construct.

### Nested observation plates

An authored plate cell can contain another authored plate. Native lowering
retains each axis as a runtime loop and fuses a selected `sum` into its plate;
a total-only reader needs no intermediate pointwise arrays. A plate read only
by `LogExpFunctions.logsumexp` streams the same way, folding each cell into the
running log-sum-exp with `logaddexp` from `-Inf`, the empty plate's value. Ragged groups,
empty groups, and array views use the same graph and generated loop structure.
A cell reads an enclosing value whole, as a Julia closure does: the observation
cells below read the kernel's `scale` directly, with no `Ref` argument.

The example below keeps the complete normalized scalar Normal law visible in
the inner graph. Its source is shared with native acceptance; the documentation
build executes it and renders that reader's generated kernel and compute DAG.

```@eval
Main.ReactiveKernelsDocs.render_nested_plate_example(@__MODULE__)
```

`plate_body` exposes the nested scalar plans, and `recipe_inventory` lists a
program's plates and scans recursively (see [Inspecting
structure](#Inspecting-structure)). `prepare` supports bound group
data with a live scalar scale, and `prepare_ad` with ordinary native Enzyme
Reverse differentiates the scale through both loops. No custom derivative
rule is required. The native acceptance suite also replays the displayed
authoring source in a fresh module without its producer's operation table.

A cell containing another plate computes work that reads only shared values (captured or `Ref` arrays,
scalars, and values computed from them alone) once before its outer loop. The
cell reads those results whole; the rest of its graph, including inner loops,
runs per outer coordinate.

Work that varies per outer coordinate cannot move out of the loop. When such a
cell computes an `hcat` or a broadcast whose only readers index it or query
its shape, including readers in an embedded kernel's nested plates, native
lowering reads its entries through those readers instead of storing it per
cell. A value computed before the loop and read only that way in its cells is
composed the same way. For example, `present = p .* offsets[event]` followed by
`components = hcat(present, absent)` builds neither array per event. The graph
and its queries keep both authored values, and ordinary Enzyme Reverse
differentiates the composed reads without a derivative rule. Shape checks and
declarations still run at the authored statement, and scalar reads keep bounds
checks. A reader that indexes the value with an integer vector, a range, a
colon or a mask computes the entries it selects into a fresh array of Julia's
result shape, so a group cell's `x[rows[g]]` computes that group's rows only.
Before ReactiveKernels 0.1.4, such a read computed the whole value at every
read, once per group. The composition covers dense `Float32`/`Float64` arrays and real
scalars, `hcat` of vectors and matrices, and one-dimensional broadcasts of
total arithmetic (`+ - * /`, `exp`, `abs`, `abs2`, `logistic`, `log1pexp`,
`identity`). Other values,
whole-array readers, custom storage, broadcasts that may throw (such as `log.`)
and effectful regions keep their stored arrays. A higher-rank broadcast is also
stored: on Julia 1.10, ordinary Enzyme Reverse of composed matrix-broadcast
reads fails with an internal error in Enzyme 0.13.216
([EnzymeAD/Enzyme.jl#3822](https://github.com/EnzymeAD/Enzyme.jl/issues/3822)).
`test/test_cell_arrays.jl` covers these cases and the event-count-independent
allocations.

An empty native plate evaluates no cell work, including shared expressions.
For inferable cells, pointwise readers still return a concretely typed empty
array; total readers return the reduction identity. This applies separately at each nesting level:
an expression written in an outer cell runs when that cell is visited, even if
its inner plate is empty; an expression written inside the inner cell runs only
when the inner domain has cells. Expressions authored outside a plate retain
their ordinary evaluation. Nonempty plates retain once-per-call shared work,
and visited cells retain ordinary bounds and domain errors.

### One cell of a plate

`plate_cell(spec, :pointwise; index = :i)` returns a kernel whose value
`pointwise_cell` is ONE cell of the authored plate producing `pointwise`, at the
position held by the new HAVE port `i` (a linear position or a
`CartesianIndex`). The cell runs the plate's own scalar body over the one-cell
slices of the plate's arguments, through the same native plate lowering, so it
equals the plate's element at that position and differentiates with ordinary
native Enzyme Reverse. A plate whose only consumer is this one composes into
the cell, as plate chains compose, and runs at that cell only. So does a plate
read only by cells at one position (several `plate_cell` values sharing an
index port): each of those cells runs it at that position. A position
outside the domain throws `BoundsError`. Several plates can share one index
port by passing the same `index` to successive `plate_cell` calls.

A whole array the cell captures and reads only at its own position, `v[i]`
with `i` an integer element of a batched argument, is computed at that row
only when its producer computes it row by row. So is one the cell reads at
several rows: `v[r]` with `r` an integer-vector element of a batched argument,
or `v[rows[k]]` through a shared index array `rows` (one integer vector per
cell, as a total over a group of rows reads them), and a matrix the cell
reads a row of, `M[i, :]` (every later index the whole axis `:`), computed at
that row. A capture whose index is not provably an integer position or a
vector of them (an undeclared, unbound port) is read whole, and the locality
report names it (`:untyped_index`). The native lowering supports these
producers:

- a dotted expression of inputs, numeric literals, gathers `A[I]` or
  `A[I, k]` (an integer vector input `I`, integer literals `k`), column reads
  `A[:, k]` (also written as the view `view(A, :, k)`, which RKPPL reads a
  per-observation value's column through), products `A * w` of two inputs,
  and literal-position reads `a[1]` of an array of numbers (read whole, as in
  `z .* tau[1]`). One source may be
  gathered by several index vectors, as in `exp.(t[r1]) .+ exp.(t[r2])`; it
  is then computed at the rows of every index;
- such a gather alone, also written as the read-only view
  `_readonly_gather(A, I)` that RKPPL reads a bare gathered definition
  through, and such a column read alone;
- `hcat(a, b, ...)` of inputs, such as a design matrix with a computed column;
- `vcat(a, b, ...)` of inputs and number literals, such as a completed
  covariate `vcat(free, grouped)[lookup]` or a padded field `vcat(phi, 0.0)`.
  A row of it is a row of exactly one operand, so each operand is computed at
  the rows that fall in it, and an operand none falls in is computed at no
  rows. Matrices stack along their first axis; their other dimensions must
  agree, as `vcat` requires (`DimensionMismatch`);
- a typed alias `b::T = a` whose declaration converts no elements (`T` an
  array type without an element type such as `AbstractVector`, or no
  declaration): its rows are `a`'s rows. A declaration such as
  `b::Vector{Float64} = a` converts the whole value and keeps it whole;
- a product `A * W` whose left factor is an array input, a column read or an
  `hcat` (row `r` is `A[r:r, :] * W`). A product of two operands whose right
  operand is an array or a number, such as `X * beta`, can be read like any
  other value; other products only by gathers `R[g, k]`, since their full
  shape is not known before they run;
- an authored plate. Row `r` of its value is the plate over row `r` of each
  batched argument, so the cell runs that plate's body at row `r` only. A
  shared array the plate reads only at its own position, `u[j]` with `j` an
  integer element of one of its batched arguments, is in turn computed at that
  row by these rules. So a later plate reading `pred[i]` from
  `pred = plate(rows) do r; f(log_k[r]) end` computes `pred` and `log_k` at
  row `i` only.

These producers can also be written inside one expression, such as a gather
straight from an inline `vcat`, `completed = vcat(observed, drawn)[order]`, or
`observed .+ mask .* vcat(free, grouped)[lookup]`. The cell lowering outlines
the inner call into a value of its own, so both are computed at the rows read,
as with the two-statement spelling `stacked = vcat(observed, drawn)` and
`completed = stacked[order]`. A call is outlined where the producers above read
a value by name: a gather's source or index, an operand of `vcat`, `hcat` or
`*`, or an operand of a dotted expression that is not itself dotted (a nested
dotted call stays fused). Only strict subexpressions are outlined, never a
branch arm, a generator or a closure body, and the parts apply the authored
operations to the same operands in the same order, so values are unchanged. A
recipe is outlined only when the cell then computes it at its rows; otherwise
it is lowered as written. Cheap expressions are not outlined: a gather `x[pos]`
written as a `vcat` operand (`vcat(x[pos], u)`) still makes that `vcat` read
whole.

The producer's own operation runs on views of its row-aligned operands at the
rows read; an operand that broadcasting extrudes is passed whole, or repeated
at several rows. A value it gathers from is computed at the gathered rows by
the same rule. So for `R = Z * F'` and `log_k = a .+ b .* w .+ R[g, 1]` read
as `log_k[i]`, the cell computes `log_k` at `i` and `R` at row `g[i]`, at a
cost that does not grow with the domain; a population term `pop = X * beta`
in `log_k = pop .+ …` is computed at row `i` the same way, also when
`X = hcat(one, c, year)` has a computed column `c`. A stratified group effect
written by a per-row plate as one matrix column and read back as `b[:, 1]`,
and a group block `z .* tau[1]` gathered as `b[g, 1]`, are computed at the
rows read. A value is computed at rows only when every reader reads it this
way, and it is computed once per cell, at all the rows its readers read: a
predictor gathered by two index vectors in two definitions, or by a
definition and a plate, runs once at both sets of rows, with everything
upstream of it. A value read whole anywhere, wanted, or produced any other way (including
through an opaque function call) is computed whole as before, also when it is
an operand of a `vcat` whose read rows all fall in other operands.

The full broadcast shape of each such expression, and the full shape of each
`hcat`, is still checked from its operands' axes (`DimensionMismatch`), and the
cell's own reads and the factor rows are bounds-checked. An error that only
another row would raise, such as an out-of-range gather index at another
position, is not raised, just as the cell does not run other cells.

Native Enzyme Reverse through an `hcat` read at several rows of a cell fails
with `IllegalTypeAnalysisException` in Base's `_typed_hcat`: those rows are
views, and Enzyme rejects Base `hcat` of views beside dense arrays (Enzyme snag
`native-reverse-t-6d51cae6`). Values are exact; the gradient waits on that
Enzyme fix (`test_plate_cell.jl` pins it broken).

A plate body that reads a captured value through its window (a struct of
active data beside constant rows, `M[q, :]` of a shared matrix) differentiates
under registered Enzyme with `Enzyme.set_runtime_activity(Enzyme.Reverse)`:
static activity refuses the constant rows a call returns stored beside active
data (Enzyme snag `native-reverse-r-285478e2`), which the Enzyme dogfood
branch repairs.

Two more producers run at a cell's rows: `view(A, lo:hi)` or `A[lo:hi]` and
`reshape(v, n, k...)`, with the bounds and dimensions integer literals or
scalar inputs. These are the shapes ReactiveKernelsPPL unpacks a packed
parameter vector's blocks with, so a block read at a cell's rows reads only
its own packed coordinates.

### Locality: named, not silent

A value a cell reads at its rows but whose producer has no row form was
computed whole in every cell, silently. The rule is now checked at
preparation: a value that every reader of the evaluation needs only at some
rows is computed only at those rows, or it is a violation, named. A value the
evaluation genuinely needs whole, such as a wanted value, `sum(x)`, the right
operand of `A * w`, an index array or a scalar, is shared and never refused.
`plate_cell(spec, …; locality)` says what preparation does with a violation:
`:error` throws a `PartitionLocalityError` listing each value with its
producer's authored source, the reason and the reader that asked for rows;
`:warn` names each once per process; `:ignore` says nothing; the default
`nothing` reads the project preference `cell_locality` (`"warn"` unless set,
as `type_stability` is). Only the root of a cascade is refused: a value read
whole by another refused value is marked as such in the report.
`partition_locality(kernel)` lists every value a cell reads with its status
(`:rows`, `:input`, `:data`, `:scalar`, `:shared`, `:whole`) and reason. A
scalar's type must be known, by declaration, bound data or inference from its
inputs: an undeclared value with no row form may be an array computed whole,
and is refused as `:untyped`, naming the declaration (`s::Float64 = …`) that
settles it.

### A partition of plates by a key

`plate_partition(spec, plates; by, index = :partition, name = :partition_total,
locality = :error, partitioned = ())` returns a kernel whose value `name` is
the sum of the cells of the named authored plates (one or several) whose key
equals the label held by the new HAVE port `index`: a partition, such as one
subject's observations. `by` gives each plate's key, a value of `spec` with
one entry per cell in the plate's linear order (usually bound data), as one
name for every plate or `(plate = :key, …)`.

A partition runs as one compiled pass over its cells: each named plate loops
once over the positions holding the label, through its own scalar body and
the native total loop, and every value the partition's cells read at their
rows is computed once per partition at the union of those rows, by the row
forms above, with the plate's batched arguments sliced as views. Values the
partition reads whole are computed once per call. Under `bound=` each label's
positions are computed from the key once per binding, and a label's lookup
is one dictionary read per call. A label no named plate has throws
`ArgumentError` naming the keys; a plate without a cell of the label
contributes nothing. The partition's default `locality` is `:error`: it is
prepared only when every value its cells read at rows is computed at those
rows. Native Enzyme Reverse differentiates it with the label a constant, and
a vector input's gradient is nonzero only on the positions the partition
reads. A plate whose domain is not one-dimensional is refused at evaluation.

`partition_footprint(kernel, args...)` returns, for each HAVE port, the
sorted linear positions the evaluation with those arguments reads, or
`:whole` for a port read whole: the body is lowered once more in a mode that
records every read of a port (the rows a value is sliced to, the positions
gathers and captures read, the constant positions of a bound-indexed read, a
lane of one row as that row) and returns them instead of the value; a
lowering nested in it (a plate body, an embedded kernel) runs blind to the
recording. For a packed parameter vector this is the exact support of a
partition's gradient; a lazy branch contributes both arms.
`partition_footprint` keeps the footprint kernel it prepares for the process;
`partition_footprint_kernel(kernel)` prepares one for a caller to hold and
evaluate itself. `partitioned` adds the separability rule, pairs
`port => labels` giving the label of each position of a per-partition port
(`missing` for a shared position): `partition_separability(kernel, args...;
footprint)` checks that the partition reads no other label's positions of
those ports, throwing `PartitionSeparabilityError` otherwise.
ReactiveKernelsPPL builds its partition queries and compact gradients on
these (`prepare_partition`, `prepare_partition_sampler`).

`plate_partition(…; plan = port)` declares one more HAVE port holding the
partition's *plan*: the integer bookkeeping of a label (its positions, every
pushed value's rows, the positions of gathers and captures in their windows,
the shapes it checks), as a named tuple `partition_plan(kernel, args...)`
computes once per label from the bound data (`nothing` for the plan port
among `args`); the body then reads it from the port instead of recomputing it
on every call, and a derivative carries none of it. The plan holds index
bookkeeping only: a bound value enters it only as a partition table, an
index column, a mask or a count, never a data column. A statement whose value
is a shape of a data value is computed in the plan from the exemplar and
checked against the value's shape on every call: arguments of another shape
throw `ArgumentError` naming `partition_plan`. The plan's type is the port's
declared type (`plan = port => T`), found by preparing the plan kernel first
(`partition_plan_kernel(spec; have, want, bound)`, the same arguments as
`prepare`, returns the kernel that computes a label's plan; `T` is the type of
its result), or, for an untyped port, by evaluating the plan on `prepare`'s
`exemplars` (one value per HAVE port, `nothing` for the plan port); a plan
port that cannot be typed is refused naming the remedy, never silently
dropped. `partition_plan_kernel(kernel)` returns the plan kernel of a prepared
partition; `partition_plan` keeps one per kernel for the process.

A partition's windows hold their rows sorted and unique, the union of what
their readers ask for, and a read finds its row by bisection: a partition of
r cells reading a value row by row costs O(r log r), never O(r²), and a value
gathered at the rows of many cells is computed once per distinct row. A
single cell's windows hold its few rows as read. A literal
range `A[lo:hi]` or a reshape computed at rows reads a vector source only
(linear positions of a matrix are not its rows): a matrix source is refused
by name (`:operands`), an untyped one checked at evaluation.

### Inspecting structure

`recipe_kind(recipe)` classifies a recipe as `:plate`, `:scan` or `:ordinary`
without throwing; `plate_body(recipe)` and `scan_body(recipe)` return the body
plan of a plate or scan. `recipe_inventory` walks a `KernelSpec` (its graph), a
`Graph`, a `Plan` or a `PreparedKernel` (its plan; with `bound=` data, the
residual plan) and every plate and scan body below it, returning `(; kind,
depth, parent, recipe)` entries in depth-first order. `depth` counts the
enclosing plates and scans, and `parent` is the index of the enclosing entry, or
`0` at top level. Structural acceptance can then state the retained nesting
without depending on internal operation types:

```julia
[(entry.kind, entry.depth) for entry in recipe_inventory(prepare(spec; bound))
 if entry.kind !== :ordinary]   # for example [(:plate, 0), (:scan, 1)]
```

The inventory only reads its argument. The `Graph` text listing renders the
same walk.

### Prepared-kernel composition is flattened

If an outer `@kernel` calls a prepared RK kernel, `prepare` treats it as
compiler-owned recipe code: it alpha-renames and splices the inner generated
statements into the outer function, remaps its operation-table slots, and emits
one flat executable operation table. The inner
`PreparedKernel`, `Plan`, and generated `Expr` are not runtime operations.

For example, an author can prepare `reduced = plate(scalar_density; have =
(:x, :μ, :logσ), want = :ld, batched = :x)` and call `reduced(x, μ, logσ)`
inside an outer `@kernel`. The outer prepared function contains the plated
statements directly rather than a runtime call through `reduced`.

Nested plates lower to the native fused loop, so the PPL examples' consumers
see one flat generated function.

### Position batching: lift the entire scalar callable

`vectorize` / `prepare_batched` (also available under the lower-level name
`replica`) keep a complete scalar `PreparedKernel` as the mathematical source
of truth and map it over one trailing replica axis on selected HAVE ports.
Scalar batched ports become vectors, array ports gain one final dimension, and
outputs are stacked along the same final dimension. Reductions inside the
scalar kernel keep their original dimensions; they do not accidentally reduce
across replicas.

Selected input and output types must be numbers or arrays. Native execution
validates the extra rank and equal replica counts, evaluates scalar replicas,
copies array slices, and stacks results, so it is a semantic transform rather
than a native zero-allocation promise.

The same transform accepts a scalar `PreparedADKernel`. `replica(ad; batched =
:position)` slices each selected HAVE, runs the scalar reverse pass, and stacks
the objective and active-port gradient along the replica axis. The scalar AD
preparation remains the derivative authority; it is not re-prepared per replica.
See [Position batching](position-batching.md) for the full public contract.

## Incremental and compiled reactive execution

### Open-ended `ReactiveState`

`ReactiveState` is orchestration above the stateless planner. Each stored value
has a monotonic version, a policy (`source`, `reactive`, or `frozen`), and—for a
reactive materialization—the versions of the actual selected HAVE leaves that
produced it.

On a demand, the state performs these operations in dependency order:

- **Effective HAVE.** Form effective HAVE from sources, frozen cut points, and recursively
   provenance-valid materializations;
- **Initial plan.** Plan the missing request;
- **Boundary extension.** Extend the request only with nominated materialization boundaries that this
   plan actually produces;
- **Prepared execution.** Replan the extended boundary, reuse or prepare its stateless kernel, and
   call it with only the HAVE leaves the plan consumes; and
- **Stored result.** Store nominated results with provenance from the selected producer paths,
   not from unused alternatives.

Validation uses a recursion-stack cycle check, so shared provenance subgraphs
are valid while true recursive provenance is not. Sources and frozen values
stop validation and remain authoritative. A checkpoint refuses an absent or
stale value.

This mode supports arbitrary later WANTS but deliberately performs dictionary,
planning, provenance, and cache work at the demand boundary. Only the prepared
kernel call inside it has the stateless hot-path guarantee.

### Closed `ReactiveProgram`

`prepare_reactive` selects one fixed HAVE/WANT plan once. It assigns each
reachable canonical value a typed `Ref` slot and a `ReactiveValue` whose slot
index is a type parameter. It precomputes direct dependents and compiles one
lazy getter per reachable value.

A getter checks the target validity bit. If stale, it recursively ensures the
selected recipe's inputs, calls that recipe once, stores every owned output,
and marks the stored outputs valid. A source with no producer must already be
valid. Setting or touching a HAVE slot walks the precomputed dependent closure
with a reusable stack and clears validity bits, stopping at frozen derived cut
points.

The plan, tuple layout, recipe selection, slot types, and getter expressions
are shared by all program instances. Runtime state is only slots, valid/frozen
bit vectors, and invalidation scratch. Mutating the source graph after
preparation causes an actionable version error rather than silently using a
stale program.

`set!`, `touch!`, and public `mutate!` are restricted to the declared HAVE
boundary. `mutate!` invalidates in a `finally` block because a throwing mutation
may already have changed its object. Derived values may be frozen as explicit
cut points and checkpointed into a later instance. State copies duplicate
array storage; `copy_group!` copies selected HAVE slots left to right and is
order-dependent when source and destination groups overlap.

A compiled state is mutable and not thread-safe. A generated getter can be
inferred and allocation-free for a compatible operation set, but the compiler
does not promise that arbitrary recipe operations allocate nothing.

## Allocation-reusing preparation

The optional MutatingFunctions extension changes call lowering, not graph
semantics or producer selection.

`prepare_nonallocating` requires every selected recipe to have one output. It
gives a step a persistent typed cache only where a destination-passing method
fills storage the kernel owns: its own decomposed steps, authored plates and
scans, and operations with a registered `apply!!` method for their concrete
types. The first invocation seeds the cache; later invocations offer it to
`MutatingFunctions.apply!!`. Every other operation is called directly, because
its first result may be caller-owned storage that the generic fallback would
overwrite. Zero allocation is a property of the complete selected operation
set and runtime types, never a planner theorem. Exemplar arguments
(`prepare_nonallocating(spec, args...)`) type the step program from runtime
argument types where HAVE ports are undeclared.

The returned result is borrowed. A directly called operation may return
caller-owned input storage, and the next call may overwrite the prior returned
object. A `NonAllocatingKernel` instance is not reentrant, thread-safe, or
suitable for concurrent callers.

`prepare_reactive_nonallocating` instead uses each compiled-state output slot as
the per-instance cache for selected mutable single-output recipes that have a
registered `apply!!` method for their declared port types. Invalidation clears
validity without clearing storage, allowing a later recomputation to reuse the
buffer. Multi-output, immutable-output and other recipes remain on the pure
path. Independent state instances own independent buffers, but a returned
mutable value is still borrowed from its state.

## Definitive support matrix

```@eval
Main.ReactiveKernelsDocs.render_compiler_capabilities()
```

## How to read a rejection

Rejection is part of the compiler contract, not a request to guess at hidden
semantics. A `PlanningError` says the dataflow query is impossible or cyclic.
An authoring error says the declared graph/signature is malformed. A
compiled-state error says the fixed graph/program/handle boundary was
violated.

None of those paths silently switches to dynamic interpretation. If a
computation belongs behind an opaque stateless recipe, put it there explicitly.

The [API reference](api.md) lists exported names; this page defines the
algorithmic meaning and limits behind them.
