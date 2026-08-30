# Compiler capability and limits

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

ReactiveKernels has one small exact planner and several lowering surfaces built
around it. This page is the capability contract: it says which decisions the
compiler makes, which facts an author must supply, which work disappears from a
prepared hot path, and which source shapes are deliberately rejected.

The word *compiler* is used here for three related but different pipelines:

- the public stateless have→want compiler, which selects recipes and emits a
   straight-line Julia callable;
- the public reactive compiler, which fixes the same selected graph into typed
   slots, validity bits, dependency closures, and generated lazy getters; and
- the source-captured state-machine compiler, which analyzes method-bearing
   `@kernel` definitions using captured and validated effect metadata. Its most
   complete consumer is the external sealed native NUTS acceptance artifact.

Those surfaces share graph identities, producer selection, and currentness
rules. They do **not** share the same public stability or source-language
boundary. In particular, a stateless recipe may contain arbitrary ordinary
Julia behind an opaque recipe call, while a source-captured state-machine method
must stay inside the compiler's inspectable and registered subset.

## API map

```@eval
Main.ReactiveKernelsDocs.render_compiler_api_map()
```

Most method-bearing source machinery and its general control factory remain
implementation surfaces rather than a general exported `prepare` API. The
narrow exported `compile_state_transition` seam described below is the
exception: it combines a stateless endpoint `KernelSpec` with a bound free
update method. NUTS, log-density, and
PPL artifacts are external compilation examples and acceptance evidence, not
domain APIs owned by ReactiveKernels. The NUTS runtime and domain surface live
under `examples/nuts_runtime/`; a bare `using ReactiveKernels` does not load or
export them. See [What the NUTS proof does and does not
establish](#what-the-nuts-proof-does-and-does-not-establish).

## The stateless compiler

### 1. Graph construction

A `Value` has a stable process-unique identity and a declared Julia value type.
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
   contains selected recipes. Its available set is HAVE plus every selected
   output; its unresolved frontier is the unsatisfied WANTS plus inputs of
   selected recipes that are not yet available.
- **Branching.** Pick the first unresolved value and branch over every candidate recipe that
   can produce it. Non-negative costs allow pruning any branch already more
   expensive than the incumbent.
- **Ranking.** Rank complete selections lexicographically by total declared cost and then
   by number of recipes. Recipe identifiers provide deterministic traversal
   order; the cost is an author-supplied planning weight, not a timing estimate.
- **Execution check.** Accept a complete selection only if availability-based Kahn ordering can
   execute it. Each ready recipe adds all of its outputs to the available set.
   This avoids inventing a cycle when overlapping multi-output recipes admit a
   valid order through a different producer.
- **Recorded result.** Record the topological recipe order and one selected owner recipe for each
   produced canonical value. Collateral outputs from a non-owner recipe may be
   computed, but they never overwrite HAVE or an earlier selected owner.

The result is exact for the implemented domain: a finite, acyclic graph with
additive finite non-negative recipe costs. Search is exponential in the worst
case and is intended for modest kernel graphs, not large-scale mathematical
optimization. Cycles, impossible wants, and an effectful-only route fail with a
`PlanningError`; impossible-query diagnostics report the missing producer
frontier.

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
It creates collision-free locals from canonical value identities, takes the
selected operations in a positional tuple, and calls each operation through a
literal tuple index. HAVE values appear in their preserved boundary order;
WANT values appear in their requested return order. A single WANT returns one
value and multiple WANTS return a tuple.

Multi-output recipes use one tuple destructure. If a later recipe emits a value
already owned by HAVE or an earlier selected producer, lowering binds that
collateral result to a discard local rather than changing the logical value.
The generated function closes over no graph object and does not resolve
operations through module globals.

`RuntimeGeneratedFunctions` compiles the expression. The resulting
`PreparedKernel` stores the compiled callable, its concrete operation tuple,
and inspection metadata. Calling it performs no graph traversal, producer
selection, topological sort, cache lookup, or dynamic scheduling. Julia still
specializes and executes the recipe callables normally; their own dispatch,
allocations, exceptions, and side effects are not erased by the graph compiler.

`transform` applies user-supplied expression passes between lowering and
compilation. The compiler preserves pass order but cannot prove that a pass
preserves semantics. A transformed expression is therefore part of the user's
trusted compilation boundary.

`PreparationCache` is keyed by graph identity and version, canonical ordered
HAVE/WANT signatures, and pass identities. Mutating a graph increments its
version, so a cache entry for the old graph cannot be returned. Cache lookup is
preparation-time work only.

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
reactive preparation, visualization, batching, and AD all operate on the one
fused graph.

## Batch and replica lowering

### Plate: fuse a scalar density across observations

`plate` starts from one scalar plan. It marks nominated HAVE ports as batched,
then propagates batch-dependence through selected recipes in topological order.
Recipes with no transitive dependency on a batched port are emitted once above
the loop. Dependent recipes are emitted once per index. The single per-element
WANT is either accumulated with a reducer (sum by default) or collected into a
new vector.

The native reducing body materializes no per-observation result vector. A
second eager tensor body expresses dependent operations as broadcasts and the
final reduction as a tensor operation; an optional array-compiler extension
selects it for traced arrays while ordinary Julia arrays retain the native
loop.

The current plate domain is deliberately narrow:

- at least one named HAVE port must be batched;
- every selected recipe must have exactly one output;
- there must be exactly one scalar WANT, and it must transitively depend on a
  batched port;
- all batched ports are indexed over the first nominated port's `eachindex`, so
  compatible axes are the caller's responsibility; and
- a non-null reducer is spliced as a bare callable, with the documented surface
  intended for Base reducer symbols.

This is map-plus-reduce/collect with invariant hoisting. It is not a general
scan, fold with loop-carried state, segmented reduction, parallel reduction,
associative tree reduction, or automatic prepared-kernel batching transform.

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

Nested plates retain both compiler products: ordinary arrays select the
native fused loop, while traced arrays select the tensorized broadcast/reduce
body. This is the static-friendly density boundary used by the PPL examples:
plain reverse Enzyme sees one flat generated function, and Reactant still sees
the array-native plate form.

### Replica: lift the entire scalar callable

`replica` keeps a complete scalar `PreparedKernel` as the mathematical source
of truth and maps it over one trailing replica axis on selected HAVE ports.
Scalar batched ports become vectors, array ports gain one final dimension, and
outputs are stacked along the same final dimension. Reductions inside the
scalar kernel keep their original dimensions; they do not accidentally reduce
across replicas.

Selected input and output types must be numbers or arrays. Native execution
validates the extra rank and equal replica counts, evaluates scalar replicas,
copies array slices, and stacks results, so it is a semantic transform rather
than a native zero-allocation promise. Optional array-compiler integration may
lower the same mapping as a backend batch primitive.

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

### `@reactive` object grammar

`@reactive` is a facade over `ReactiveProgram`. Signature arguments become
mutable HAVE sources, top-level body assignments become derived recipes, and
inner methods become ordinary type-stable methods over a generated
`ReactiveObject`. Reads demand the appropriate slot; whole-field and in-place
writes route through compiled-state assignment and invalidation.

Its method rewriter supports straight-line code, `if`/`elseif`, `&&`/`||`,
`for`/`while`, indexing, property chains, `@.` broadcast, compound and
destructuring assignment, and sibling-method calls. It tracks straight-line
aliases of reactive fields and rejects a later mutation when control-flow paths
could make the alias refer to different roots.

It rejects `let`, `try`/`catch`, comprehensions and generators, `do` blocks,
anonymous functions, and nested function definitions inside a rewritten
method. Those forms create bindings, deferred execution, or exceptional paths
for which the rewriter cannot prove which reactive root to invalidate. Move
that logic into supported control flow, a sibling method, or an ordinary
wrapper that supplies a port.

`specialize=true` builds a program in the constructor with HAVE port types
derived from runtime values. The default reuses one definition-level program.
An injected `prepare=` hook must return a `ReactiveProgram` from the same graph
with the exact signature boundary and all exposed ports; the macro validates
those facts.

## Allocation-reusing preparation

The optional MutatingFunctions extension changes call lowering, not graph
semantics or producer selection.

`prepare_nonallocating` requires every selected recipe to have one output. It
gives each recipe a persistent typed cache. The first invocation seeds the
cache; later invocations offer it to `MutatingFunctions.apply!!`. The generic
fallback may still allocate, so zero allocation is a property of the complete
selected operation set and runtime types, never a planner theorem.

The cached result is borrowed. An aliasing operation may retain caller-owned
input storage, and the next call may overwrite the prior returned object. A
`NonAllocatingKernel` instance is not reentrant, thread-safe, or suitable for
concurrent callers.

`prepare_reactive_nonallocating` instead uses each compiled-state output slot as
the per-instance cache for selected mutable single-output recipes. Invalidation
clears validity without clearing storage, allowing a later recomputation to
reuse the buffer. Multi-output and immutable-output recipes remain on the pure
path. Independent state instances own independent buffers, but a returned
mutable value is still borrowed from its state.

## Source-captured method compiler

Method-bearing `@kernel` definitions enter a stricter compiler. They are not
opaque recipe closures and are not analyzed through Julia's compiled IR.

### Authoritative input is captured source

At definition time, the macro stores detached, structurally frozen source ASTs,
definition-module identity, port metadata, and exact callee-registration
snapshots. Later analysis walks that stored source. It does not call
`code_lowered`, `code_typed`, `CodeInfo`, `Core.Compiler`, method inference, or
arbitrary function-body inspection. Rebinding an authored callee slot to a
different identity is detected and rejected.

This boundary makes the compiler's claim finite: it compiles the language it
captured and the effects explicitly registered for it. It makes no claim to
understand arbitrary Julia hidden behind an unregistered call.

### Two stateful authoring modes

A body containing nested methods is an object kernel. Its receiver is
implicit: bare unshadowed port names are owner fields, while formals and locals
shadow fields lexically. Nested methods declare no `self` or `__self__` formal.
The token `__self__` is legal only as the first positional actual of a sibling
method call; using it as a field qualifier or ordinary value is rejected.

A methodless definition that mutates a field of its first positional argument,
or whose name ends in `!!`, is a free update method. Its first positional
argument is the explicit subject. The `!!` spelling is an explicit strong
same-object update contract, not a heuristic about a hidden implementation.

Everything else remains the stateless `KernelSpec` path. These modes are
selected from syntax before any recipe runs.

### Normalized `MethodIR`

Captured methods normalize into immutable `MethodIR` values. The IR represents:

- literals, locals, formals, owner-field paths, indexes, property access,
  tuples, named tuples, short-circuit values, and conditional values;
- local assignment, direct/indexed/nested place writes, place swaps, and
  write-then-return;
- branches, guards, `for` and `while` loops, `break`, `continue`, and returns;
- sibling calls, subject-method calls, registered callable-field calls,
  registered primitive/intrinsic calls, and unresolved external calls; and
- positional/keyword formals, defaults, overload candidates, source-order
  reads, branch groups, and loop-carried access events.

Each method records three independent facts: control shape (straight, branch,
loop, or recursive), local effects, and unresolved dependencies. Structured
access events keep branch alternatives separate and mark loops that may execute
zero times. Every call records its actual-value reads. This prevents a flat
source scan from inventing definite assignment or hiding a read behind a
branch.

Normalization rejects malformed or unsoundly ambiguous source, including a
computed/dynamic callee, a positional splat at a call site, an unresolved
qualified callee, a maybe-bound local read, unsupported tuple-place mutation,
and a mutation whose owner root cannot be represented. An unregistered external
may be represented as opaque for diagnostics, but executable ownership/effect
lowering must resolve it or reject it.

### Effects, ownership, and physical storage

Calls are authorized by exact identity, not by spelling. Built-in compiler
descriptors describe arity, reads, writes, result aliasing/borrowing, and RNG
position. The former `@rk_pure`, `@rk_borrows`, and `@rk_rng` declarations have
been removed. External fixtures use ordinary visible arithmetic/control or a
captured sibling `@kernel` method. Undeclared ordinary helpers remain unsupported and are
not made pure because they have an operator-like name or receive an argument
named `rng`. `@node` is unrelated to this removal boundary and remains part of
graph authoring.

For compiler-known primitives, the implementation also checks the concrete
specialization domain. Sanctioned Base numeric scalars, dense arrays, and the
specific linear-algebra wrappers required by the compiled kernels are
accepted. A user subtype that dispatches a custom overload through the same
generic function is rejected unless it is admitted by an exact, validated
compiler rule. This prevents generic-function identity from smuggling unknown
effects into compiled code.

The factory computes an interprocedural ownership fixed point over direct
writes, sibling calls, callable fields, borrowed aliases, and strong-copy
operations. It separates producer-owned endpoint storage from shared
read-authority. Its immutable plan records canonical slots, role and physical
index, authoritative HAVE, selected producer recipes, dependents, entry-current
bits, captured operation handles, and recipe input/output ownership.

Storage is concrete tuples/structs and validity masks; author symbols are only
labels. Copied child endpoints get distinct owned storage but may share one
read-only authority object. Two paths using the same graph value identity remain
distinct because schedule state is keyed by physical owner/view path plus
canonical value identity.

### Currentness and exception semantics

A direct write makes its target current and kills the transitive dependents of
that target in the selected producer graph. Before a derived read, lowering
recursively ensures the selected producer's inputs and executes that producer
once if needed. One recipe execution produces its selected owned output set
atomically; collateral outputs owned by another recipe are not blessed.

A method may directly write only an authoritative source slot. Treating one
value as both directly written and plan-produced is rejected. A stale derived
read with no selected producer is also rejected. Entry currentness is an
explicit factory contract, never inferred by assuming that all derived values
are valid.

Validity changes are exception-safe but values are not transactionally rolled
back. Dependents are invalidated before or with a mutation. A throwing recipe
does not bless its outputs. If several writes have already executed before a
later throw, their invalidations remain, so a retry recomputes from the actual
partially mutated state rather than reading stale cached values.

### Control lowering

The general control compiler distinguishes acyclic work from recursive or
suspending work. Acyclic sibling methods and their early returns are inlined
into the caller's continuation. Non-suspending loops remain native Julia loops.
Recursive strongly connected components and callers that suspend across them
are defunctionalized into finite method/PC dispatch, concrete frame stores, and
an isbits control stack. Values live across a suspension are stored explicitly;
the generated dispatcher contains no boxed closure or runtime method table.

The implemented generic control surface is narrower than Julia. In the PC
machine, a suspending `for` loop currently requires a two-bound unit range;
non-suspending loops can retain an arbitrary native iterable. Exact overload
narrowing is required before emission. Unsupported statements or unresolved
effects reject instead of falling back to interpretation.

### Functional free state transitions

`compile_state_transition(spec, transition, endpoint_args;
endpoint_kwargs=(;))` is the public, backend-neutral subset of the
method-bearing compiler. `spec` supplies the endpoint's authoritative sources
and have→want recipes. `transition` is a registered free mutating `@kernel`, or
a `partial` of one that binds all of its required keyword controls. Construction
uses the original endpoint and transition signature binders; it does not match
the kernel's name, field names, or a program-counter census.

The result is a `CompiledStateTransition`. `initial_transition_state(result)` returns an
isolated, fully materialized named state while preserving external callable
authorities by identity. Calling the result with that state is functional: an
authored array write produces a new array value, invalidates the written
source's transitive derived closure, and recomputes a stale derived field only
at its next source-ordered read. The same generated program is ordinary Julia
and static program metadata to optional array compilers such as Reactant.

The initial public source subset is intentionally finite. It admits direct
owned-field writes, exact captured `map(copy, tuple)`, tuple and named
destructuring, bound non-Boolean numeric controls, and integer `Base.Colon`
loops whose bounds are entirely static. Static loops are unrolled during
lowering so validity is propagated through every authored iteration; this is
not a host loop around traced execution. Indexed destinations,
data-dependent branches/loops, arbitrary higher-order calls, and opaque
callbacks reject. Callback computations used by endpoint recipes must be
explicit endpoint ports so their authority is auditable and their identity is
preserved.

The six ReactiveHMC phase-point examples exercise this one compiler path for
Gaussian and relativistic Euclidean, Riemannian, and diagonal-SoftAbs geometry.
Both generalized leapfrog and implicit midpoint remain their ordinary authored
mathematical kernels. Those examples validate a reusable compiler capability;
they do not add geometry or integrator cases to compiler code.

## What the NUTS proof does and does not establish

The source-captured compiler's strongest end-to-end external exemplar is the
reviewed eight-spec NUTS fixture shown on the [NUTS sampling](nuts.md) page. It exercises
alternative recipe planning, destination-bound potential/gradient production,
owned and shared storage, endpoint copies, mutation kills, recomputation,
recursive tree control, loops, early returns, RNG effects, optional statistics,
dual averaging, and Welford accumulation.

The sealed native path cold-compiles detached `MethodIR` into a registry-free
type tree. Callable values live in one concrete tuple indexed from type
metadata. The native encoder admits exactly the source forms needed by that
fixture and rejects anything else, including opaque calls, unresolved overload
sets, call-site keyword splats, formal splats, typed native-NUTS formals,
chained comparisons, raw `@node` payloads in control metadata, unsupported
structural literals, and subject-method calls inside the NUTS object methods.

Construction produces fixed-depth concrete endpoint/tree/proposal storage and a
certificate binding owner token, root token, immutable plan type/key, native
program type, control fingerprint, physical roles, selected recipes, emitted
operations, and frame/root/scratch types. The acceptance evidence for that
artifact verifies same-object mutation, inferred concrete return, exact
zero allocation after construction, Float32/Float64 paths, two RNG types,
exception/currentness behavior, and an instrumented emitted-operation census.
The sealed acceptance root consults no graph, planner, mutable registry, or later-world
method lookup.

That proof is deliberately **not** a general Julia-recursion compiler proof. It
establishes the captured fixture and the encoder's rejection boundary. It does
not establish arbitrary recursion, arbitrary containers or numeric subtypes,
arbitrary integrators, automatic differentiation, PPL semantics, or a stable
exported constructor for every method-bearing `@kernel`.

There is also a package-boundary distinction. The sealed `_build_nuts_sampler`
compiler, fixture entry, certificate accessors, and the compiled-reactive
`nuts_state` / `CompiledNUTSState` comparison path all live in the explicitly
loaded external exemplar. None is loaded or exported by `ReactiveKernels`.

Therefore “the sealed native compiler runs the NUTS fixture” must not be read as
“ReactiveKernels exports or promises a NUTS API.” The two external paths are
implemented and tested as compiler evidence, but neither
turns NUTS, log-density, or PPL models into the package's domain contract.

## Definitive support matrix

```@eval
Main.ReactiveKernelsDocs.render_compiler_capabilities()
```

## How to read a rejection

Rejection is part of the compiler contract, not a request to guess at hidden
semantics. A `PlanningError` says the dataflow query is impossible or cyclic.
An authoring error says the declared graph/signature is malformed. A
source-compiler rejection says captured syntax, callee identity, effect,
ownership, concrete specialization domain, or currentness could not be proved.
A compiled-state error says the fixed graph/program/handle boundary was
violated.

None of those paths silently switches to dynamic interpretation. If a
computation belongs behind an opaque stateless recipe, put it there explicitly.
If its mutations must participate in the source compiler, register the exact
effect contract and remain within the corresponding concrete domain. If a
reactive method needs unsupported lexical or exceptional control flow, keep
that logic in ordinary Julia and pass its result through a declared port.

The [API reference](api.md) lists exported names; this page defines the
algorithmic meaning and limits behind them.
