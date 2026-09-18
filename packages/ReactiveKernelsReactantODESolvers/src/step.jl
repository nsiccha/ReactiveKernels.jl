# One FSAL Tsit5 step, functional and vectorized.
#
# Stage structure follows OrdinaryDiffEq's `perform_step!` for
# `Tsit5ConstantCache` (out-of-place RHS): `k1` enters as the FSAL first
# stage, six fresh RHS evaluations produce `k2..k6` and the FSAL last stage
# `k7 = f(u)`, and the embedded error `utilde = dt*Σbtildeᵢkᵢ` drives the
# controller. Straight-line broadcasting with no scalar indexing, no
# mutation, and no branches, so the native driver and the Reactant-traced
# driver share this code.

"""
    tsit5_step(f, uprev, k1, p, t, dt, tableau, abstol, reltol)

Take one Tsit5 step from `(uprev, t)` with step `dt` and FSAL first stage
`k1`. Returns `(u = u_proposed, k = (k1, …, k7), EEst = error_estimate)`.

Performs exactly six RHS evaluations. All stage combinations allocate fresh
vectors; nothing is mutated. The state arguments share one element type
(concrete floats natively, traced numbers under Reactant); the tableau stays
concrete. `f` must return the stage input's element type at the state
length.
"""
function tsit5_step(f, uprev::AbstractVector, k1::AbstractVector, p, t::Number,
        dt::Number, tab::Tsit5Tableau{<:Number}, abstol::Number,
        reltol::Number)
    n = length(uprev)
    length(k1) == n ||
        throw(DimensionMismatch("first stage must match the state length $n"))

    k2 = _stage(f, uprev .+ dt .* (tab.a21 .* k1), p, t + tab.c1 * dt, n, 2)
    k3 = _stage(f, uprev .+ dt .* (tab.a31 .* k1 .+ tab.a32 .* k2), p,
        t + tab.c2 * dt, n, 3)
    k4 = _stage(f, uprev .+ dt .* (tab.a41 .* k1 .+ tab.a42 .* k2 .+
                                     tab.a43 .* k3), p, t + tab.c3 * dt, n, 4)
    k5 = _stage(f, uprev .+ dt .* (tab.a51 .* k1 .+ tab.a52 .* k2 .+
                                     tab.a53 .* k3 .+ tab.a54 .* k4), p,
        t + tab.c4 * dt, n, 5)
    k6 = _stage(f, uprev .+ dt .* (tab.a61 .* k1 .+ tab.a62 .* k2 .+
                                     tab.a63 .* k3 .+ tab.a64 .* k4 .+
                                     tab.a65 .* k5), p, t + tab.c5 * dt, n, 6)
    # FSAL propagator row: the fifth-order solution itself.
    u = uprev .+ dt .* (tab.a71 .* k1 .+ tab.a72 .* k2 .+ tab.a73 .* k3 .+
                        tab.a74 .* k4 .+ tab.a75 .* k5 .+ tab.a76 .* k6)
    k7 = _stage(f, u, p, t + tab.c6 * dt, n, 7)
    utilde = dt .* (tab.btilde1 .* k1 .+ tab.btilde2 .* k2 .+
                    tab.btilde3 .* k3 .+ tab.btilde4 .* k4 .+
                    tab.btilde5 .* k5 .+ tab.btilde6 .* k6 .+
                    tab.btilde7 .* k7)
    EEst = error_estimate(utilde, uprev, u, abstol, reltol)
    (u=u, k=(k1, k2, k3, k4, k5, k6, k7), EEst=EEst)
end

function _stage(f, tmp::AbstractVector, p, t::Number, n::Integer,
        stage::Integer)
    k = f(tmp, p, t)
    k isa AbstractVector ||
        throw(ArgumentError("RHS must return a vector at stage $stage"))
    length(k) == n || throw(DimensionMismatch(
        "RHS must return a vector of length $n at stage $stage, got $(length(k))"))
    eltype(k) == eltype(tmp) || throw(ArgumentError(
        "RHS must return element type $(eltype(tmp)) at stage $stage, " *
        "got $(eltype(k))"))
    k
end
