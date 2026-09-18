# Free fourth-order dense output for Tsit5 (Tsitouras): with the seven stage
# derivatives of an accepted step, `y(t+θ*dt) = uprev + dt*Σbᵢ(θ)kᵢ` where
# `b1(θ) = θ*evalpoly(θ, r11..r14)` and `bᵢ(θ) = θ²*evalpoly(θ, rᵢ2..rᵢ4)`
# for stages 2..7. Matches OrdinaryDiffEq's `_ode_interpolant` for
# `Tsit5ConstantCache`.

"""
    tsit5_dense_weights(θ, dense) -> NTuple{7}

Dense-output weights `(b1(θ), …, b7(θ))` for a fraction `θ` across the step.
"""
function tsit5_dense_weights(θ::T,
        dense::Tsit5DenseCoefficients{T}) where {T<:AbstractFloat}
    θ2 = θ * θ
    b1 = θ * (dense.r11 + θ * (dense.r12 + θ * (dense.r13 + θ * dense.r14)))
    b2 = θ2 * (dense.r22 + θ * (dense.r23 + θ * dense.r24))
    b3 = θ2 * (dense.r32 + θ * (dense.r33 + θ * dense.r34))
    b4 = θ2 * (dense.r42 + θ * (dense.r43 + θ * dense.r44))
    b5 = θ2 * (dense.r52 + θ * (dense.r53 + θ * dense.r54))
    b6 = θ2 * (dense.r62 + θ * (dense.r63 + θ * dense.r64))
    b7 = θ2 * (dense.r72 + θ * (dense.r73 + θ * dense.r74))
    (b1, b2, b3, b4, b5, b6, b7)
end

"""
    tsit5_dense_eval(uprev, stages, dt, θ, dense)

Evaluate the dense output at fraction `θ` of the accepted step:
`uprev + dt*Σbᵢ(θ)kᵢ`. `stages` is the 7-tuple of stage-derivative vectors
`(k1, …, k7)` of that step. Returns a fresh vector; nothing is mutated.
"""
function tsit5_dense_eval(uprev::AbstractVector{T},
        stages::NTuple{7,AbstractVector{T}}, dt::T, θ::T,
        dense::Tsit5DenseCoefficients{T}) where {T<:AbstractFloat}
    b1, b2, b3, b4, b5, b6, b7 = tsit5_dense_weights(θ, dense)
    k1, k2, k3, k4, k5, k6, k7 = stages
    uprev .+ dt .* (k1 .* b1 .+ k2 .* b2 .+ k3 .* b3 .+ k4 .* b4 .+
                    k5 .* b5 .+ k6 .* b6 .+ k7 .* b7)
end
