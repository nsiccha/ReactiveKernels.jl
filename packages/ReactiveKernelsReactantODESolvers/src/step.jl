# One FSAL Tsit5 step on preallocated buffers.
#
# Stage structure follows OrdinaryDiffEq's `perform_step!` for
# `Tsit5ConstantCache` (out-of-place RHS): `k1` enters as the FSAL first
# stage, six fresh RHS evaluations produce `k2..k6` and the FSAL last stage
# `k7 = f(u)`, and the embedded error `utilde = dt*Σbtildeᵢkᵢ` drives the
# controller. All stage combinations accumulate into `tmp` so only the seven
# stage buffers plus two scratch vectors are live.

"""
    Tsit5Buffers{T}

Preallocated stage/scratch storage for [`tsit5_step!`](@ref): the seven
stage derivatives `k1..k7`, the proposed step `u`, the error combination
`utilde`, and one accumulation scratch `tmp`. All vectors have length `n`.
"""
struct Tsit5Buffers{T<:AbstractFloat}
    k1::Vector{T}
    k2::Vector{T}
    k3::Vector{T}
    k4::Vector{T}
    k5::Vector{T}
    k6::Vector{T}
    k7::Vector{T}
    u::Vector{T}
    utilde::Vector{T}
    tmp::Vector{T}
end

function Tsit5Buffers{T}(n::Integer) where {T<:AbstractFloat}
    n > 0 || throw(ArgumentError("state dimension must be positive"))
    Tsit5Buffers{T}([Vector{T}(undef, n) for _ in 1:10]...)
end

Tsit5Buffers(n::Integer, ::Type{T}=Float64) where {T<:AbstractFloat} =
    Tsit5Buffers{T}(n)

"""
    tsit5_step!(buffers, f, uprev, p, t, dt, tableau, abstol, reltol) -> EEst

Take one Tsit5 step from `(uprev, t)` with step `dt`, reading the FSAL first
stage from `buffers.k1`. Writes the proposed state to `buffers.u`, the
embedded error combination to `buffers.utilde`, all seven stages to
`buffers.k1..k7`, and returns the scaled error estimate `EEst`.

Performs exactly six RHS evaluations.
"""
function tsit5_step!(buffers::Tsit5Buffers{T}, f, uprev::AbstractVector{T}, p,
        t::T, dt::T, tab::Tsit5Tableau{T}, abstol::T,
        reltol::T) where {T<:AbstractFloat}
    (; k1, k2, k3, k4, k5, k6, k7, u, utilde, tmp) = buffers
    n = length(uprev)
    (length(k1) == n && length(tmp) == n && length(u) == n &&
     length(utilde) == n) ||
        throw(DimensionMismatch("Tsit5 buffers must match the state length $n"))

    @inbounds for i in 1:n
        tmp[i] = uprev[i] + dt * (tab.a21 * k1[i])
    end
    copyto!(k2, f(tmp, p, t + tab.c1 * dt))

    @inbounds for i in 1:n
        tmp[i] = uprev[i] + dt * (tab.a31 * k1[i] + tab.a32 * k2[i])
    end
    copyto!(k3, f(tmp, p, t + tab.c2 * dt))

    @inbounds for i in 1:n
        tmp[i] = uprev[i] + dt * (tab.a41 * k1[i] + tab.a42 * k2[i] + tab.a43 * k3[i])
    end
    copyto!(k4, f(tmp, p, t + tab.c3 * dt))

    @inbounds for i in 1:n
        tmp[i] = uprev[i] +
                 dt * (tab.a51 * k1[i] + tab.a52 * k2[i] + tab.a53 * k3[i] +
                       tab.a54 * k4[i])
    end
    copyto!(k5, f(tmp, p, t + tab.c4 * dt))

    @inbounds for i in 1:n
        tmp[i] = uprev[i] +
                 dt * (tab.a61 * k1[i] + tab.a62 * k2[i] + tab.a63 * k3[i] +
                       tab.a64 * k4[i] + tab.a65 * k5[i])
    end
    copyto!(k6, f(tmp, p, t + tab.c5 * dt))

    # FSAL propagator row: the fifth-order solution itself.
    @inbounds for i in 1:n
        u[i] = uprev[i] +
               dt * (tab.a71 * k1[i] + tab.a72 * k2[i] + tab.a73 * k3[i] +
                     tab.a74 * k4[i] + tab.a75 * k5[i] + tab.a76 * k6[i])
    end
    copyto!(k7, f(u, p, t + tab.c6 * dt))

    @inbounds for i in 1:n
        utilde[i] = dt * (tab.btilde1 * k1[i] + tab.btilde2 * k2[i] +
                           tab.btilde3 * k3[i] + tab.btilde4 * k4[i] +
                           tab.btilde5 * k5[i] + tab.btilde6 * k6[i] +
                           tab.btilde7 * k7[i])
    end
    error_estimate(utilde, uprev, u, abstol, reltol)
end
