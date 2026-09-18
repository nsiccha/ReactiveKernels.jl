# One FSAL Tsit5 step, executed from the standard kernel graph.
#
# The stage structure lives in `tsit5_stage` (`kernels.jl`); this function is
# the native driver's validated entry point to its functional evaluation.
# Same contract as before the refactor: six RHS evaluations, `(u, k, EEst)`
# return, `f` checked to return the state element type at the state length.

"""
    tsit5_step(f, uprev, k1, p, t, dt, tableau, abstol, reltol)

Take one Tsit5 step from `(uprev, t)` with step `dt` and FSAL first stage
`k1`. Returns `(u = u_proposed, k = (k1, …, k7), EEst = error_estimate)`.

The step itself is the [`tsit5_stage`](@ref) kernel graph evaluated
functionally; this wrapper only validates `f`'s contract (same exception
types as the retired hand-written step) and adapts the signature.
Performs exactly six RHS evaluations. `f` must return the stage input's
element type at the state length.
"""
function tsit5_step(f, uprev::AbstractVector, k1::AbstractVector, p, t::Number,
        dt::Number, tab::Tsit5Tableau{<:Number}, abstol::Number,
        reltol::Number)
    n = length(uprev)
    length(k1) == n ||
        throw(DimensionMismatch("first stage must match the state length $n"))
    T = eltype(uprev)
    u, kk, EEst = stage_functional(f, uprev, k1, p, t, dt, tab, abstol,
        reltol, T(inv(n)))
    _check_step_output(u, kk, T, n)
    (u=u, k=kk, EEst=EEst)
end

function _check_step_output(u::AbstractVector, kk::Tuple, T::Type, n::Int)
    length(u) == n || throw(DimensionMismatch(
        "RHS must return a vector of length $n, got $(length(u))"))
    eltype(u) == T || throw(ArgumentError(
        "RHS must return element type $T, got $(eltype(u))"))
    for (j, kj) in enumerate(kk)
        length(kj) == n || throw(DimensionMismatch(
            "RHS must return a vector of length $n at stage $j, got $(length(kj))"))
        eltype(kj) == T || throw(ArgumentError(
            "RHS must return element type $T at stage $j, got $(eltype(kj))"))
    end
    nothing
end
