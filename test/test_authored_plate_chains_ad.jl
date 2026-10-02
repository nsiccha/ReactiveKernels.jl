using ReactiveKernels, DifferentiationInterface, Test
import Enzyme
isdefined(@__MODULE__, :AuthoredPlateChains) ||
    include("fixtures/authored_plate_chains.jl")

_chain_gradient_allocated(ad::A, gradient, q, x, y) where {A} =
    @allocated ad_value_and_gradient!(ad, gradient, q, x, y)

@testset "Authored plate chains under plain reverse Enzyme" begin
    C = AuthoredPlateChains
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    q = [0.7]
    for n in (32, 4096)
        x = collect(range(-1.0, 1.0; length = n))
        y = fill(0.3, n)
        kernel = prepare(C.chain)
        reference = prepare(C.flat)
        ad = prepare_ad(kernel, backend, q, x, y; active = :q)
        flat_ad = prepare_ad(reference, backend, q, x, y; active = :q)
        gradient, flat_gradient = zeros(1), zeros(1)
        v, _ = ad_value_and_gradient!(ad, gradient, q, x, y)
        flat_v, _ = ad_value_and_gradient!(flat_ad, flat_gradient, q, x, y)
        @test v == flat_v
        @test gradient == flat_gradient
        @test only(gradient) ≈ -sum((only(q) .* x .- y) .* x)
        _chain_gradient_allocated(ad, gradient, q, x, y)
        _chain_gradient_allocated(flat_ad, flat_gradient, q, x, y)
        @test _chain_gradient_allocated(ad, gradient, q, x, y) <=
              _chain_gradient_allocated(flat_ad, flat_gradient, q, x, y) + 64
        bound = prepare(C.chain; bound = (; x, y))
        bound_ad = prepare_ad(bound, backend, q; active = :q)
        @test ad_gradient(bound_ad, q) == gradient
    end
end

# Bound raw-data axis arrays with the only live HAVE (`q`) captured solely via an
# atomic `Ref(q)` whole-vector capture: the shape whose `prepare` used to throw.
# Guards that AD survives the marker-selection fix — bound gradient must equal the
# unbound gradient and the analytic gradient under plain reverse Enzyme.
@testset "Ref-atomic-array bound chain under plain reverse Enzyme" begin
    C = AuthoredPlateChains
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    q = [0.7, -0.3]
    for n in (8, 64)
        x = collect(range(-1.0, 1.0; length = n))
        y = fill(0.3, n)
        m = 2 .* x .+ q[1]
        # total = Σ (yᵢ + mᵢ² + q₂·mᵢ), m = 2x + q₁ ⇒ ∂/∂q₁ = Σ(2m + q₂), ∂/∂q₂ = Σ m
        g_analytic = [sum(2 .* m .+ q[2]), sum(m)]
        unbound = prepare(C.ref_atomic_chain)
        bound = prepare(C.ref_atomic_chain; bound = (; x, y))
        ad = prepare_ad(unbound, backend, q, x, y; active = :q)
        bound_ad = prepare_ad(bound, backend, q; active = :q)
        gradient = zeros(2)
        v, _ = ad_value_and_gradient!(ad, gradient, q, x, y)
        @test v ≈ unbound(q, x, y)
        @test gradient ≈ g_analytic
        @test ad_gradient(bound_ad, q) ≈ gradient
    end
end

# A gathered generator sum lowers dose-outer (`_KernelReduction`); plain
# reverse Enzyme differentiates that loop nest. Its gradients equal those of
# the same cell read through an alias of `get`, which keeps the cell loop.
module DoseOuterAD
using ReactiveKernels
struct Lattice
    shifts::Vector{Int}
    nobs::Int
end
domain(plan::Lattice) = 1:plan.nobs
lag(t, plan::Lattice, j) = t - plan.shifts[j]
const fetch = Base.get
@kernel cell(plan, units::Vector{Float64}, weights::Vector{Float64}) = begin
    observations = domain(plan)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum(w[j] * get(u, lag(t, p, j), 0.0) for j in eachindex(w); init = 0.0)
    end
    objective::Float64 = sum(abs2, concentration)
    return objective
end
@kernel control(plan, units::Vector{Float64}, weights::Vector{Float64}) = begin
    observations = domain(plan)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum(w[j] * fetch(u, lag(t, p, j), 0.0) for j in eachindex(w); init = 0.0)
    end
    objective::Float64 = sum(abs2, concentration)
    return objective
end
end

@testset "Dose-outer gathered generator sum under plain reverse Enzyme" begin
    D = DoseOuterAD
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    plan = D.Lattice([0, 40, 100, 300], 257)
    units = collect(range(0.5, 2.0; length = 257))
    weights = [3.0, 1.5, 0.25, 2.0]
    kernel, control = prepare(D.cell), prepare(D.control)
    @test count(op -> op isa ReactiveKernels._KernelSourceOp &&
                      op.f isa ReactiveKernels._KernelReduction, kernel.ops) == 1
    for active in (:units, :weights, (:units, :weights))
        v, g = ad_value_and_gradient(prepare_ad(kernel, backend, plan, units, weights; active),
                                     plan, units, weights)
        v0, g0 = ad_value_and_gradient(prepare_ad(control, backend, plan, units, weights; active),
                                       plan, units, weights)
        @test v === v0
        @test g == g0
    end
    # d/dweights of Σ c² with c = Σ_j w_j u[t - s_j] is 2 Σ_t c_t u[t - s_j].
    c = [sum((w * get(units, t - s, 0.0) for (s, w) in zip(plan.shifts, weights)); init = 0.0)
         for t in 1:plan.nobs]
    _, g = ad_value_and_gradient(prepare_ad(kernel, backend, plan, units, weights;
                                            active = :weights), plan, units, weights)
    @test g ≈ [2 * sum(c[t] * get(units, t - s, 0.0) for t in 1:plan.nobs) for s in plan.shifts]
end

module DenseColumnDoseOuterAD
using ReactiveKernels
@kernel cell(observations::UnitRange{Int}, shifts::Vector{Int},
             units::AbstractVector{Float64}, weights::Vector{Float64}) = begin
    response::Vector{Float64} = plate(observations, Ref(shifts), Ref(units), Ref(weights)) do t, s, u, w
        sum(w[j] * get(u, t - s[j], 0.0) for j in eachindex(w); init=0.0)
    end
    loss::Float64 = sum(abs2, response)
end
end

@testset "Dense column dose-outer source under plain reverse Enzyme" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    U = [sin(0.01i) + l for i in 1:128, l in 1:2]
    units = view(U, :, 2)
    shifts, weights = [-7, 0, 300], [0.7, -0.3, 1.1]
    original_U, original_shifts, original_weights = copy(U), copy(shifts), copy(weights)
    kernel = prepare(DenseColumnDoseOuterAD.cell)
    for n in (32, 257)
        args = (1:n, shifts, units, weights)
        reference_args = (1:n, shifts, copy(units), weights)
        @test count(op -> op isa ReactiveKernels._KernelSourceOp &&
                          op.f isa ReactiveKernels._KernelReduction, kernel.ops) == 1
        @test kernel(args...) == kernel(reference_args...)
        ad = prepare_ad(kernel, backend, args...; active=:weights)
        reference = prepare_ad(kernel, backend, reference_args...; active=:weights)
        value, gradient = ad_value_and_gradient(ad, args...)
        @test gradient ≈ ad_gradient(reference, reference_args...) rtol=1e-12
        @test value ≈ kernel(args...)
        response = [sum(w * get(units, t - s, 0.0) for (s, w) in zip(shifts, weights))
                    for t in 1:n]
        analytic = [2 * sum(response[t] * get(units, t - s, 0.0) for t in 1:n)
                    for s in shifts]
        @test gradient ≈ analytic rtol=1e-12
        # A prepared gradient must read each call's constant view, including
        # its current parent contents, without capturing or mutating them.
        replacement_U = U .+ 0.4
        replacement_args = (1:n, shifts, view(replacement_U, :, 2), weights)
        replacement_reference = (1:n, shifts, copy(replacement_args[3]), weights)
        @test ad_gradient(ad, replacement_args...) ≈
              ad_gradient(reference, replacement_reference...) rtol=1e-12
        @test replacement_U == original_U .+ 0.4
    end
    @test U == original_U
    @test shifts == original_shifts
    @test weights == original_weights
end
