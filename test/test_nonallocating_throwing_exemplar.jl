# A non-allocating gradient is prepared at an exemplar where the program throws
# (snag `prepare-sampler-55203665`: RKPPL's `prepare_sampler`, whose default
# non-allocating preparation evaluates the density once at `u0`, failed with
# the ODE adapter's "native DP5 failed: Unstable" at `u0 = zeros`). The
# exemplar is a type/shape exemplar: preparation warns, naming the error, and
# the operations its seeding call did not reach allocate on every gradient,
# as an untaken branch's do. Values and gradients equal the ordinary
# preparation's at every point where the program evaluates. `warm = true`,
# which evaluates the gradient at the exemplar, likewise keeps its
# preparation. Runs through `test/run_nonallocating_integration.jl`, whose
# environment pins the unregistered MutatingFunctions.
using ReactiveKernels
using ReactiveKernelsPPL
using MutatingFunctions
using DifferentiationInterface: AutoEnzyme
import Enzyme
using Test

const TE_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
const TE_SEED_WARNING = r"program threw there"
const TE_WARM_WARNING = r"`warm = true` evaluates the gradient"

# Diverges at `s >= 1`, as an ODE solve fails past a blow-up.
te_scale(s) = s < 1 ? 1 / (1 - s) :
    throw(DomainError(s, "the scale diverges at s >= 1"))

# A cached product before the throwing step and a cached broadcast after it.
@kernel te_model(u::Vector{Float64}, X::Matrix{Float64}, y::Vector{Float64}) = begin
    eta = X * u
    k = te_scale(u[1])
    mu = k .* eta
    total = -0.5 * sum(abs2, y .- mu)
    return total
end

const TE_X = [sin(0.7i + j) for i in 1:40, j in 1:3]
const TE_Y = [0.3 + 0.2 * cos(1.3i) for i in 1:40]
const TE_THROWING = [1.0, 0.0, 0.0]
const TE_POINTS = ([0.2, -0.4, 0.7], [-0.5, 0.3, 0.1], [0.6, 1.2, -0.8])

@testset "a non-allocating gradient prepared where the program throws" begin
    kernel = prepare(te_model)
    # The positive control: the exemplar really throws.
    @test_throws DomainError kernel(TE_THROWING, TE_X, TE_Y)
    ordinary = prepare_ad(kernel, TE_BACKEND, TE_THROWING, TE_X, TE_Y; active = :u)
    na = prepare_nonallocating(kernel, TE_THROWING, TE_X, TE_Y)
    ad = @test_logs (:warn, TE_SEED_WARNING) match_mode = :any prepare_ad(
        na, TE_BACKEND, TE_THROWING, TE_X, TE_Y; active = :u)
    # Seeded where the program evaluates: the same values, for comparison.
    seeded = prepare_ad(prepare_nonallocating(kernel, TE_POINTS[1], TE_X, TE_Y),
        TE_BACKEND, TE_POINTS[1], TE_X, TE_Y; active = :u)
    for u in TE_POINTS
        g, gna, gseeded = zeros(3), zeros(3), zeros(3)
        value, _ = ad_value_and_gradient!(ordinary, g, u, TE_X, TE_Y)
        value_na, _ = ad_value_and_gradient!(ad, gna, u, TE_X, TE_Y)
        value_seeded, _ = ad_value_and_gradient!(seeded, gseeded, u, TE_X, TE_Y)
        @test value_na ≈ value rtol = 1e-12
        @test gna ≈ g rtol = 1e-10 atol = 1e-12
        @test value_seeded ≈ value rtol = 1e-12
        @test gseeded ≈ g rtol = 1e-10 atol = 1e-12
    end
    # The prepared gradient still throws the program's error where it throws.
    @test_throws DomainError ad_value_and_gradient!(ad, zeros(3), TE_THROWING,
                                                    TE_X, TE_Y)
    g = zeros(3)
    value, _ = ad_value_and_gradient!(ad, g, TE_POINTS[2], TE_X, TE_Y)
    @test g ≈ last(ad_value_and_gradient!(ordinary, zeros(3), TE_POINTS[2],
                                           TE_X, TE_Y)) rtol = 1e-10

    @testset "warm = true" begin
        warmed = @test_logs (:warn, TE_WARM_WARNING) match_mode = :any prepare_ad(
            kernel, TE_BACKEND, TE_THROWING, TE_X, TE_Y; active = :u, warm = true)
        warmed_na = @test_logs (:warn, TE_SEED_WARNING) (:warn, TE_WARM_WARNING) match_mode = :any prepare_ad(
            prepare_nonallocating(kernel, TE_THROWING, TE_X, TE_Y), TE_BACKEND,
            TE_THROWING, TE_X, TE_Y; active = :u, warm = true)
        for u in TE_POINTS
            g, gw, gwna = zeros(3), zeros(3), zeros(3)
            value, _ = ad_value_and_gradient!(ordinary, g, u, TE_X, TE_Y)
            @test first(ad_value_and_gradient!(warmed, gw, u, TE_X, TE_Y)) ≈ value rtol = 1e-12
            @test first(ad_value_and_gradient!(warmed_na, gwna, u, TE_X, TE_Y)) ≈ value rtol = 1e-12
            @test gw ≈ g rtol = 1e-10 atol = 1e-12
            @test gwna ≈ g rtol = 1e-10 atol = 1e-12
        end
    end
end

# An objective neither declared nor inferred as a `Number` is checked by one
# call at the exemplar. Where that call throws, nothing proves the objective a
# scalar, so preparation refuses with the program's error as its cause; at a
# point where the program evaluates the same kernel prepares. A declared
# objective type needs no call.
te_opaque_scale(u) = Base.inferencebarrier(te_scale(u[1]))
@kernel te_opaque(u::Vector{Float64}) = begin
    total = -0.5 * sum(abs2, u) + log(te_opaque_scale(u))
    return total
end
@kernel te_declared(u::Vector{Float64}) = begin
    total::Float64 = -0.5 * sum(abs2, u) + log(te_scale(u[1]))
    return total
end

@testset "the objective check where the program throws" begin
    opaque = prepare(te_opaque)
    for kernel in (opaque, prepare_nonallocating(opaque, TE_THROWING))
        err = try
            prepare_ad(kernel, TE_BACKEND, TE_THROWING; active = :u)
            nothing
        catch caught
            caught
        end
        @test err isa ArgumentError
        @test occursin("total::Float64", sprint(showerror, err))
        # At a point where it evaluates, the call proves the objective scalar.
        @test prepare_ad(kernel, TE_BACKEND, TE_POINTS[1]; active = :u) isa
              ReactiveKernels.PreparedADKernel
    end
    declared = prepare(te_declared)
    for kernel in (declared, prepare_nonallocating(declared, TE_THROWING))
        ad = prepare_ad(kernel, TE_BACKEND, TE_THROWING; active = :u)
        for u in TE_POINTS
            @test last(ad_value_and_gradient(ad, u)) ≈
                  -u .+ [1 / (1 - u[1]), 0.0, 0.0] rtol = 1e-12
        end
    end
end

# The reporter's shape: a plate whose cells throw at `u0 = zeros`. At
# `logk = 0` the cells with `t > 1` throw (`log1p` of a value below -1), so the
# seeding call fills the first cells and throws in a later one; at
# `logk = -3` every cell evaluates.
te_cell(k, t) = log1p(-k * t)

_te_plate = @rkppl begin
    logk ~ Normal(-3, 1)
    sigma ~ HalfNormal(1)
    @plate for i in eachindex(t)
        mu[i] = te_cell(exp(logk), t[i])
    end
    y .~ Normal.(mu, sigma)
end

@testset "prepare_sampler at a u0 where the density throws" begin
    ts = collect(range(0.1, 2.0; length = 12))
    ys = [log1p(-0.05 * t) + 0.01 * sin(3t) for t in ts]
    plan = _te_plate(; t = ts) | (; y = ys)
    built = build_kernel(plan)
    n = built.layout.total
    u0 = zeros(n)
    points = [[-3.0 + 0.2s, -1.0 + 0.3s] for s in 0:2]
    q = prepare_sampler(built, plan, points[1]; backend = TE_BACKEND,
                        nonallocating = false)
    @test_throws DomainError q(u0)
    for warm in (false, true)
        qna = @test_logs (:warn, TE_SEED_WARNING) match_mode = :any prepare_sampler(
            built, plan, u0; backend = TE_BACKEND, warm)
        @test qna.kernel isa NonAllocatingKernel
        for u in points
            g, gna = zeros(n), zeros(n)
            value, _ = sampler_value_and_gradient!(q, g, u)
            value_na, _ = sampler_value_and_gradient!(qna, gna, u)
            @test value_na ≈ value rtol = 1e-12
            @test gna ≈ g rtol = 1e-10 atol = 1e-12
            @test qna(u) ≈ value rtol = 1e-12
        end
    end
end
