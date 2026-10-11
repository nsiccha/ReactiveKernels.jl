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
        # The bound kernel runs the same authored arithmetic, but it is a
        # different compiled program. Enzyme generates derivatives with fast
        # math by default (`Enzyme.API.fast_math!`), so LLVM may reassociate and
        # FMA-contract the n-term adjoint sum of `q` differently in the two
        # programs (hosted CI: 1 ulp at n = 32). Each evaluation lies within
        # (n + 2) * eps / 2 * magnitude of the exact sum, so two lie within the
        # bound below.
        magnitude = sum(abs.(x) .* (abs.(only(q) .* x) .+ abs.(y)))
        @test isapprox(ad_gradient(bound_ad, q), gradient;
                       rtol = 0, atol = 2 * n * eps() * magnitude)
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
    concentration::Vector{Float64} = plate(observations) do t
        sum(weights[j] * get(units, lag(t, plan, j), 0.0) for j in eachindex(weights); init = 0.0)
    end
    objective::Float64 = sum(abs2, concentration)
    return objective
end
@kernel control(plan, units::Vector{Float64}, weights::Vector{Float64}) = begin
    observations = domain(plan)
    concentration::Vector{Float64} = plate(observations) do t
        sum(weights[j] * fetch(units, lag(t, plan, j), 0.0) for j in eachindex(weights); init = 0.0)
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

# An `evalpoly(x, c)` cell over shared coefficients runs coefficient-outer;
# plain reverse Enzyme differentiates that loop nest. Its gradients match the
# same cell called through a function the lowering does not recognize.
module EvalpolyCellAD
using ReactiveKernels
horner(x, c) = evalpoly(x, c)
@kernel cell(xs::Vector{Float64}, c::Vector{Float64}) = begin
    ys::Vector{Float64} = plate(xs) do x
        evalpoly(x, c)
    end
    objective::Float64 = sum(abs2, ys)
    return objective
end
@kernel control(xs::Vector{Float64}, c::Vector{Float64}) = begin
    ys::Vector{Float64} = plate(xs) do x
        horner(x, c)
    end
    objective::Float64 = sum(abs2, ys)
    return objective
end
end

@testset "Coefficient-outer evalpoly cell under plain reverse Enzyme" begin
    E = EvalpolyCellAD
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    xs = collect(range(-1.5, 1.5; length = 300))
    c = [1.0, -0.25, 0.125, 0.3, -0.07, 0.011]
    kernel, control = prepare(E.cell), prepare(E.control)
    @test occursin("_plate_evalpoly_ready", string(ReactiveKernels.code_expr(kernel)))
    for active in (:xs, :c, (:xs, :c))
        v, g = ad_value_and_gradient(prepare_ad(kernel, backend, xs, c; active), xs, c)
        v0, g0 = ad_value_and_gradient(prepare_ad(control, backend, xs, c; active), xs, c)
        @test v == v0
        @test all(map((a, b) -> isapprox(a, b; rtol = 1e-12), g isa Tuple ? g : (g,),
                      g0 isa Tuple ? g0 : (g0,)))
    end
    # d/dc_k of Σ p(x)² is 2 Σ_t p(x_t) x_t^(k-1).
    p = evalpoly.(xs, Ref(c))
    _, g = ad_value_and_gradient(prepare_ad(kernel, backend, xs, c; active = :c), xs, c)
    @test g ≈ [2 * sum(p .* xs .^ (k - 1)) for k in eachindex(c)]
end

# Plates that only feed a scan run strip by strip with it; plain reverse
# Enzyme differentiates the strip loops. The tuple domain keeps the
# materialized plates and is the control.
module StripScanAD
using ReactiveKernels
using ReactiveKernels: scan
@kernel recurrence(steps, pa::Vector{Float64}, decay::Float64) = begin
    xs = plate(steps) do k
        1 / (k + 0.5)
    end
    sa::Vector{Float64} = plate(xs) do x
        evalpoly(x, pa)
    end
    out::Vector{Float64} = scan(sa; init = 0.0) do carry, s
        next = muladd(decay, carry, s)
        (next, next)
    end
    objective::Float64 = sum(abs2, out)
    return objective
end
end

@testset "Strip-fused plates and scan under plain reverse Enzyme" begin
    S = StripScanAD
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    kernel = prepare(S.recurrence)
    @test occursin("_plate_strip_ready", string(ReactiveKernels.code_expr(kernel)))
    pa, decay = [1.0, 0.3, -0.2, 0.05], 0.9
    steps = 0:299
    for active in (:pa, :decay, (:pa, :decay))
        v, g = ad_value_and_gradient(prepare_ad(kernel, backend, steps, pa, decay; active),
                                     steps, pa, decay)
        v0, g0 = ad_value_and_gradient(prepare_ad(kernel, backend, Tuple(steps), pa, decay; active),
                                       Tuple(steps), pa, decay)
        @test isapprox(v, v0; rtol = 1e-13)
        @test all(map((a, b) -> isapprox(a, b; rtol = 1e-10), g isa Tuple ? g : (g,),
                      g0 isa Tuple ? g0 : (g0,)))
    end
end

# Hoisted carry-independent step statements under plain reverse Enzyme.
module FissionScanAD
using ReactiveKernels
using ReactiveKernels: scan
@kernel recurrence(steps, pa::Vector{Float64}, decay::Float64) = begin
    out::Vector{Float64} = scan(steps; init = 0.0) do carry, k
        x = 1 / (k + 0.5)
        s = evalpoly(x, pa)
        next = muladd(decay, carry, s)
        (next, next)
    end
    objective::Float64 = sum(abs2, out)
    return objective
end
end

@testset "Hoisted scan-step statements under plain reverse Enzyme" begin
    F = FissionScanAD
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    kernel = prepare(F.recurrence)
    @test occursin("scan_hoisted", string(ReactiveKernels.code_expr(kernel)))
    pa, decay = [1.0, 0.3, -0.2, 0.05], 0.9
    steps = 0:299
    function reference(pa, decay)
        carry, total = 0.0, 0.0
        for k in steps
            carry = muladd(decay, carry, evalpoly(1 / (k + 0.5), pa))
            total += abs2(carry)
        end
        total
    end
    v0, gp = value_and_gradient(p -> reference(p, decay), backend, pa)
    _, gd = value_and_gradient(d -> reference(pa, d), backend, decay)
    v, g = ad_value_and_gradient(prepare_ad(kernel, backend, steps, pa, decay;
                                            active = (:pa, :decay)), steps, pa, decay)
    @test isapprox(v, v0; rtol = 1e-13)
    @test isapprox(g[1], gp; rtol = 1e-10)
    @test isapprox(g[2], gd; rtol = 1e-10)
end

# An arithmetic generator-sum cell runs fold-outer; plain reverse Enzyme
# differentiates its tile passes. Its gradients match the same cell spelled
# through a function the lowering does not recognize.
module FoldSumAD
using ReactiveKernels
using ReactiveKernels: scan
divide(a, b) = a / b
@kernel cell(ts::Vector{Float64}, w::Vector{Float64}, s::Vector{Float64}) = begin
    out::Vector{Float64} = plate(ts) do t
        sum(w[j] / (t + s[j]) for j in eachindex(w); init = 0.0)
    end
    objective::Float64 = sum(abs2, out)
    return objective
end
@kernel control(ts::Vector{Float64}, w::Vector{Float64}, s::Vector{Float64}) = begin
    out::Vector{Float64} = plate(ts) do t
        sum(divide(w[j], t + s[j]) for j in eachindex(w); init = 0.0)
    end
    objective::Float64 = sum(abs2, out)
    return objective
end
@kernel recurrence(steps, w::Vector{Float64}, s::Vector{Float64}, decay::Float64) = begin
    out::Vector{Float64} = scan(steps; init = 0.0) do carry, k
        r = sum(w[j] / (k + s[j]) for j in eachindex(w); init = 0.0)
        next = muladd(decay, carry, r)
        (next, next)
    end
    objective::Float64 = sum(abs2, out)
    return objective
end
end

@testset "Fold-outer arithmetic sum cell under plain reverse Enzyme" begin
    F = FoldSumAD
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    ts = collect(range(0.1, 5.0; length = 300))
    w, s = [0.5, -0.25, 2.0], [1.0, 3.5, 6.0]
    kernel, control = prepare(F.cell), prepare(F.control)
    @test occursin("_plate_one_axis", string(ReactiveKernels.code_expr(kernel)))
    for active in (:ts, :w, :s, (:w, :s))
        v, g = ad_value_and_gradient(prepare_ad(kernel, backend, ts, w, s; active), ts, w, s)
        v0, g0 = ad_value_and_gradient(prepare_ad(control, backend, ts, w, s; active), ts, w, s)
        @test v == v0
        @test all(map((a, b) -> isapprox(a, b; rtol = 1e-12), g isa Tuple ? g : (g,),
                      g0 isa Tuple ? g0 : (g0,)))
    end
    # d/dw_j of Σ_t c_t² is 2 Σ_t c_t / (t + s_j).
    c = [sum(w[j] / (t + s[j]) for j in eachindex(w); init = 0.0) for t in ts]
    _, g = ad_value_and_gradient(prepare_ad(kernel, backend, ts, w, s; active = :w), ts, w, s)
    @test g ≈ [2 * sum(c ./ (ts .+ s[j])) for j in eachindex(w)]
    # The same sum in a scan step runs in the strip region.
    recurrence = prepare(F.recurrence)
    @test occursin("scan_hoisted", string(ReactiveKernels.code_expr(recurrence)))
    steps, decay = 0:299, 0.9
    # The reference spells the sum as a loop and reads `s` as a `Constant`
    # context: a generator holding the active `w` beside the constant `s`, or
    # a closure capturing `s`, meets Enzyme's activity checks (§7aj).
    function reference(w, s, decay)
        carry, total = 0.0, 0.0
        for k in steps
            r = 0.0
            for j in eachindex(w)
                r = Base.add_sum(r, w[j] / (k + s[j]))
            end
            carry = muladd(decay, carry, r)
            total += abs2(carry)
        end
        total
    end
    v0, gw = value_and_gradient(reference, backend, w, Constant(s), Constant(decay))
    _, gd = value_and_gradient((d, w, s) -> reference(w, s, d), backend, decay,
                               Constant(w), Constant(s))
    v, g = ad_value_and_gradient(prepare_ad(recurrence, backend, steps, w, s, decay;
                                            active = (:w, :decay)), steps, w, s, decay)
    @test isapprox(v, v0; rtol = 1e-13)
    @test isapprox(g[1], gw; rtol = 1e-10)
    @test isapprox(g[2], gd; rtol = 1e-10)
end

module DenseColumnDoseOuterAD
using ReactiveKernels
@kernel cell(observations::UnitRange{Int}, shifts::Vector{Int},
             units::AbstractVector{Float64}, weights::Vector{Float64}) = begin
    response::Vector{Float64} = plate(observations) do t
        sum(weights[j] * get(units, t - shifts[j], 0.0) for j in eachindex(weights); init=0.0)
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
    for n in (0, 1, 32, 257, 4096)
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
        analytic = [2 * sum((response[t] * get(units, t - s, 0.0) for t in 1:n); init=0.0)
                    for s in shifts]
        @test gradient ≈ analytic rtol=1e-12
        for bound in ((; units), (; observations=args[1], shifts, units),
                      (; observations=args[1], shifts))
            residual = prepare(DenseColumnDoseOuterAD.cell; bound)
            values = (; observations=args[1], shifts, units, weights)
            residual_args = Tuple(getproperty(values, port.name) for port in inputs(residual))
            bound_ad = prepare_ad(residual, backend, residual_args...; active=:weights)
            bound_value, bound_gradient = ad_value_and_gradient(bound_ad, residual_args...)
            @test bound_value ≈ value
            @test bound_gradient ≈ gradient rtol=1e-12
        end
        # Consumer package images wrap the native body for their first use.
        # After loading, that wrapper must also enter the body without packing
        # a constant view beside the active weights.
        warmed = ReactiveKernels._PrecompileWarmFunction(kernel.f.native)
        warm_call = ReactiveKernels._ADNativeKernelCall{
            4,typeof(warmed),typeof(kernel.ops)}(warmed, kernel.ops)
        @test DifferentiationInterface.gradient(warm_call, backend, weights,
            Constant(args[1]), Constant(shifts), Constant(units)) ≈ gradient rtol=1e-12
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
