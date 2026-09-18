# Observable tail sentinel: counts how many times `_evaluate_ppl_source` has
# executed the source's DEMO/SELF-CHECK TAIL — i.e. evaluated any source
# expression AFTER `model` is defined (the `prepare(...)` + kernel execution +
# `@assert`s that build `docs_example`). `model_only=true` stops at `model` and
# therefore never bumps this; the full path (default) bumps it by exactly one.
# It is the regression guard for the startup contract: every model module's
# `__init__` builds its template with `model_only=true`, so this stays 0 across
# package load. If an `__init__` regressed to the FULL evaluator, or a
# `model_only` build failed to stop before the tail (even while returning only
# `.model`), package load would bump it — caught by
# test/test_startup_initialization.jl. Not thread-safe; the example inits and
# tests are single-threaded.
const _DEMO_TAIL_EXECUTIONS = Ref(0)

"""
    _evaluate_ppl_source(source, owner; bindings=(), model_only=false)

Evaluate one PPL walkthrough source string in a fresh module. `owner` is bound
under its module name and `bindings` exposes the example's public data/types,
so the same source bytes can be executed by tests and displayed/executed by the
documentation build without a second model body.

With `model_only=true` the evaluation STOPS as soon as the source has defined
its `model` (the `@kernel model(...) = …` `KernelSpec`) and returns just
`(; model, source, sandbox)` — skipping the source's trailing demo/self-check
(the `prepare(...)`, kernel execution, and `@assert`s that build `docs_example`).
That tail is redundant for a graph TEMPLATE: it only needs `.model`, and every
consumer runs its own `prepare` on the composed graph. This is what each model
module's `__init__` uses to build its `_<MODEL>_GRAPH_TEMPLATE`, so package load
pays only the graph construction, not a full prepare+evaluate per model. The
full path (default) still runs the whole source and is what tests and the docs
build use, so the demo/self-check is exercised there. Measured (strato2,
2026-09-08): the demo tail is ~76–88% of a per-model source eval (eight_schools
1.83s of 2.4s, rate_2 5.35s of 6.1s, pipeline-warm), which is the bulk of the
old ~600s `using ReactiveKernelsPPLExamples` load storm across ~89 modules.
"""
function _evaluate_ppl_source(source::AbstractString, owner::Module;
                              bindings = (), model_only::Bool = false)
    # Match the docs renderer's framing rule exactly: triple-quoted authorities
    # carry a terminal newline, while the displayed/executed panel bytes do not.
    displayed = strip(String(source), '\n')
    # Keep package loading in the source owner's dependency context. A top-level
    # anonymous module instead resolves `using` against the caller's active
    # project, forcing runners to repeat dependencies named by the source.
    sandbox_name = gensym(Symbol(nameof(owner), :Source))
    Core.eval(owner, Expr(:module, true, sandbox_name, Expr(:block)))
    sandbox = getfield(owner, sandbox_name)
    Core.eval(sandbox, :(using ReactiveKernels))
    owner_name = nameof(owner)
    Core.eval(sandbox, :(const $(owner_name) = $owner))

    for name in bindings
        name isa Symbol || throw(ArgumentError("PPL source binding must be a Symbol"))
        isdefined(owner, name) || error("$(nameof(owner)) does not define source binding $name")
        value = getfield(owner, name)
        Core.eval(sandbox, :(const $(name) = $value))
    end

    parsed = Meta.parseall(displayed; filename = "$(nameof(owner))-docs-source.jl")
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    model_defined = false
    tail_ran = false
    for expression in expressions
        expression isa LineNumberNode && continue
        # An expression evaluated AFTER `model` is defined is part of the demo/
        # self-check tail — the observable the sentinel below records.
        model_defined && (tail_ran = true)
        Core.eval(sandbox, expression)
        if !model_defined && isdefined(sandbox, :model)
            model_defined = true
            # Template build: the graph is complete once `model` is defined; stop
            # before the source's prepare/execute/@assert demo tail runs.
            model_only && break
        end
    end
    tail_ran && (_DEMO_TAIL_EXECUTIONS[] += 1)

    if model_only
        isdefined(sandbox, :model) || error(
            "$(nameof(owner)) model_only source did not define a `model`",
        )
        model = Core.eval(sandbox, :model)
        model isa KernelSpec || error(
            "$(nameof(owner)) source did not provide its KernelSpec as `model`",
        )
        return (; model, source = displayed, sandbox)
    end

    artifact = Core.eval(sandbox, :docs_example)
    artifact.model isa KernelSpec || error(
        "$(nameof(owner)) docs source did not provide its KernelSpec as `model`",
    )
    artifact.kernel isa PreparedKernel || error(
        "$(nameof(owner)) docs source did not provide a PreparedKernel",
    )
    merge(artifact, (; source = displayed, sandbox))
end
