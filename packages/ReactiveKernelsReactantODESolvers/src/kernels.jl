# The Tsit5 stage block as a standard ReactiveKernels `@kernel`.
#
# `tsit5_stage` is a line-by-line mirror of [`tsit5_step`](@ref): the same
# seven stage evaluations, propagator row, and embedded error estimate, in
# the same operation order, expressed as a kernel graph over explicit ports.
# `test_kernels.jl` proves bit-for-bit parity with the plain step across
# RHS shapes, parameter shapes, and float types.
#
# Deliberate differences from the plain step, all load-bearing for kernels:
# - No validation branches: kernels admit no data-dependent control flow, so
#   the `_stage` length/eltype checks are omitted here; valid inputs only.
# - `inv_n` (the reciprocal state count in the compute float type) is a port
#   rather than a computed `inv(length)`, so the error norm keeps the
#   state's element type exactly as the plain `rms_norm` does.
# - The stage tuple is bound to the named port `kk`: only named ports can be
#   `want` targets.
#
# Execution boundary, honestly: the graph above is the single source of truth
# for the step, executed through two standard-derived executors.
# - Native (`solve_ode` via `tsit5_step`): the `lower`ed plan body evaluated
#   functionally (`stage_functional` below). Plain Enzyme reverse works
#   through it; routing native execution through the prepared kernel instead
#   breaks Enzyme (`IllegalTypeAnalysisException` on the executor's
#   `Union{Missing,Bool}` restart bookkeeping, plus mutation discipline the
#   contract forbids working around).
# - Traced (the Reactant ext): the prepared kernel itself, called per step.
#   Prepared calls lower through Reactant — primal and reverse — via the
#   core traced-slot machinery.
# The adaptive loop itself (accept/reject, PI control, guards) is
# driver-level in both paths: kernels express straight-line dataflow, not
# data-dependent control.

"""
    tsit5_stage

Standard-kernel expression of one FSAL Tsit5 step: a [`ReactiveKernels.KernelSpec`](@ref)
whose ports are the step inputs (`f`, `uprev`, `k1`, `p`, `t`, `dt`, `tab`,
`atol`, `rtol`, `inv_n`) and whose `(:u, :kk, :EEst)` outputs mirror
[`tsit5_step`](@ref)'s `(u, k, EEst)` return bit-for-bit on valid inputs.
Prepare with [`prepare_tsit5_stage`](@ref); the driver validates `f` once.
"""
@kernel tsit5_stage(f::Function, uprev::AbstractVector, k1::AbstractVector, p,
        t::Number, dt::Number, tab::Tsit5Tableau, atol::Number, rtol::Number,
        inv_n::Number) = begin
    k2::AbstractVector = f(uprev .+ dt .* (tab.a21 .* k1), p, t + tab.c1 * dt)
    k3::AbstractVector = f(uprev .+ dt .* (tab.a31 .* k1 .+ tab.a32 .* k2), p,
        t + tab.c2 * dt)
    k4::AbstractVector = f(uprev .+ dt .* (tab.a41 .* k1 .+ tab.a42 .* k2 .+
                                     tab.a43 .* k3), p, t + tab.c3 * dt)
    k5::AbstractVector = f(uprev .+ dt .* (tab.a51 .* k1 .+ tab.a52 .* k2 .+
                                     tab.a53 .* k3 .+ tab.a54 .* k4), p,
        t + tab.c4 * dt)
    k6::AbstractVector = f(uprev .+ dt .* (tab.a61 .* k1 .+ tab.a62 .* k2 .+
                                     tab.a63 .* k3 .+ tab.a64 .* k4 .+
                                     tab.a65 .* k5), p, t + tab.c5 * dt)
    # FSAL propagator row: the fifth-order solution itself.
    u::AbstractVector = uprev .+ dt .* (tab.a71 .* k1 .+ tab.a72 .* k2 .+
        tab.a73 .* k3 .+ tab.a74 .* k4 .+ tab.a75 .* k5 .+ tab.a76 .* k6)
    k7::AbstractVector = f(u, p, t + tab.c6 * dt)
    utilde::AbstractVector = dt .* (tab.btilde1 .* k1 .+ tab.btilde2 .* k2 .+
        tab.btilde3 .* k3 .+ tab.btilde4 .* k4 .+ tab.btilde5 .* k5 .+
        tab.btilde6 .* k6 .+ tab.btilde7 .* k7)
    scales::AbstractVector = atol .+ max.(abs.(uprev), abs.(u)) .* rtol
    EEst::Number = sqrt(sum(abs2, utilde ./ scales) * inv_n)
    kk = (k1, k2, k3, k4, k5, k6, k7)
    return (u=u, k=kk, EEst=EEst)
end

"""
    prepare_tsit5_stage() -> PreparedKernel

Prepare [`tsit5_stage`](@ref) over its full have/want boundary. The
Reactant-traced driver calls this kernel per step; prepared calls lower
through Reactant (primal and reverse) via the core traced-slot machinery.
"""
function prepare_tsit5_stage()
    prepare(tsit5_stage;
        have=(:f, :uprev, :k1, :p, :t, :dt, :tab, :atol, :rtol, :inv_n),
        want=(:u, :kk, :EEst))
end

# Functional evaluation of the same graph for the native path. The stateful
# prepared executor is Enzyme-hostile (see the boundary note above), so the
# native driver executes the `lower`ed plan body instead: `lower` turns the
# plan into straight-line code over `(__ops__, ports...)`, and the generator
# below bakes the ops in as constants and drops `__ops__`, yielding a plain
# Julia function with static dispatch — which is why plain Enzyme reverse
# works through it. Generated (not `eval`ed) so the product is always fresh:
# no `__init__`, no precompilation staleness, no load order. The ops tuple
# has no public accessor, so it comes from the private `_lower_with_ops` —
# pinned by the ReactiveKernels compat entry, guarded by the parity tests and
# the fail-fast signature check below, and filed as a core snag requesting a
# public functional-lowering surface.
const TSIT5_STAGE_PORTS = (:f, :uprev, :k1, :p, :t, :dt, :tab, :atol, :rtol,
    :inv_n)

function _bake_stage_ops!(ex::Expr, ops::Tuple)
    # Replace every `__ops__[i]` (literal index) with the constant op value.
    # Anything else shaped like an ops reference is a core-shape change that
    # must fail loudly here, not miscompile silently downstream.
    for (i, a) in enumerate(ex.args)
        if a === :__ops__
            error("bare __ops__ reference (not a literal ref): $ex")
        elseif a isa Expr
            if a.head === :ref && length(a.args) == 2 && a.args[1] === :__ops__
                idx = a.args[2]
                idx isa Integer || error("non-literal __ops__ index: $idx")
                ex.args[i] = QuoteNode(ops[idx])
            else
                _bake_stage_ops!(a, ops)
            end
        end
    end
    ex
end

"""
    stage_functional(f, uprev, k1, p, t, dt, tab, atol, rtol, inv_n)

[`tsit5_stage`](@ref) evaluated functionally: the standard-planned,
`lower`ed graph body with baked-in ops. Bitwise identical to the prepared
kernel; Enzyme-clean. Valid inputs only (no validation branches).
"""
@generated function stage_functional(f, uprev, k1, p, t, dt, tab, atol,
        rtol, inv_n)
    spec_plan = plan(tsit5_stage; have=TSIT5_STAGE_PORTS,
        want=(:u, :kk, :EEst))
    lowered, ops, _ = ReactiveKernels._lower_with_ops(spec_plan;
        inline_embedded=false)
    sig = lowered.args[1]
    sig.args[1] === :__ops__ ||
        error("lower() signature changed shape: $sig")
    names = map(sig.args[2:end]) do a
        a isa Symbol ? a : (a isa Expr && a.head === :(::) ? a.args[1] :
            error("lower() signature changed shape: $sig"))
    end
    Tuple(names) == TSIT5_STAGE_PORTS ||
        error("lower() port order changed: $names")
    body = _bake_stage_ops!(deepcopy(lowered.args[2]), ops)
    body
end
