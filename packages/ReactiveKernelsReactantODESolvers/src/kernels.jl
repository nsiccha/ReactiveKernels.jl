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
# Kernel boundary, honestly: this mirror is proven but NOT on either hot
# path, and both reasons are structural, not incidental.
# - Native: routing `solve_ode` through the prepared kernel breaks
#   plain-Enzyme reverse through the solve (`IllegalTypeAnalysisException`
#   on the executor's `Union{Missing,Bool}` restart bookkeeping), which the
#   native-gradient contract forbids working around. The leaner
#   `prepare_nonallocating` executor needs the optional MutatingFunctions
#   extension — a new dependency, out of scope.
# - Traced: the prepared executor is in-place (0-alloc); Reactant traces
#   only the functional out-of-place form, so the traced driver consumes
#   the plain step.
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

Prepare [`tsit5_stage`](@ref) over its full have/want boundary. Used by the
parity tests; see the kernel-boundary note above for why the hot paths stay
on the plain step.
"""
function prepare_tsit5_stage()
    prepare(tsit5_stage;
        have=(:f, :uprev, :k1, :p, :t, :dt, :tab, :atol, :rtol, :inv_n),
        want=(:u, :kk, :EEst))
end
