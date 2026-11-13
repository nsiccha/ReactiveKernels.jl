# Matched targets for the PB/PBR mechanism investigation. This is a benchmark,
# not a production dependency on PBR. Run in an environment containing RK,
# PBR, Enzyme, MutatingFunctions and BenchmarkTools. Sources/pins are printed.
# Both targets omit theta-independent normalizers, as PBR's sampler does.
# No data summaries or handwritten model derivative are used by the RK modes.
using ReactiveKernels, MutatingFunctions, DifferentiationInterface
using LinearAlgebra, Random, Statistics, Test, BenchmarkTools, LogExpFunctions
import PracticalBayesianRegression as PBR
import Enzyme
const RK = ReactiveKernels
const LDP = PBR.LogDensityProblems
BLAS.set_num_threads(1)

module MatchedTargets
using ReactiveKernels, LinearAlgebra, LogExpFunctions

# PBR's transferable mechanism: stable softplus and its local partial share
# one exponential. This is a scalar numerical graph, independent of a model.
@kernel softplus_graph(x::Float64) = begin
    positive = x >= 0
    a::Float64 = ifelse(positive, -x, x)
    q::Float64 = exp(a)
    value::Float64 = ifelse(positive, x, 0.0) + log1p(q)
    partial::Float64 = ifelse(positive, 1.0, q) / (1 + q)
    return value, partial
end
const shared_softplus = scalar_derivative_rule(softplus_graph; primal=:value,
    partials=(x=:partial,), name=:shared_softplus)

@inline function softplus_primal(x)
    positive = x >= 0
    q = exp(ifelse(positive, -x, x))
    ifelse(positive, x, zero(x)) + log1p(q)
end

@kernel gaussian(theta::Vector{Float64}, X::Matrix{Float64}, y::Vector{Float64}) = begin
    p = size(X, 2)
    beta = view(theta, 1:p)
    s = theta[p + 1]
    eta::Vector{Float64} = X * beta
    invsigma = exp(-s)
    terms = plate(eta, y) do ei, yi
        r = (yi - ei) * invsigma
        -r * r / 2 - s
    end
    total::Float64 = sum(terms) - sum(abs2, beta) / 8 - s * s / 2
    return total
end

@kernel bernoulli(theta::Vector{Float64}, X::Matrix{Float64}, y::Vector{Float64}) = begin
    eta::Vector{Float64} = X * theta
    terms = plate(eta, y) do ei, yi
        yi * ei - log1pexp(ei)
    end
    total::Float64 = sum(terms) - sum(abs2, theta) / 8
    return total
end

@kernel bernoulli_shared(theta::Vector{Float64}, X::Matrix{Float64}, y::Vector{Float64}) = begin
    eta::Vector{Float64} = X * theta
    terms = plate(eta, y) do ei, yi
        yi * ei - shared_softplus(ei)
    end
    total::Float64 = sum(terms) - sum(abs2, theta) / 8
    return total
end

@kernel bernoulli_shared_primal(theta::Vector{Float64}, X::Matrix{Float64}, y::Vector{Float64}) = begin
    eta::Vector{Float64} = X * theta
    terms = plate(eta, y) do ei, yi
        yi * ei - softplus_primal(ei)
    end
    total::Float64 = sum(terms) - sum(abs2, theta) / 8
    return total
end

# Ordinary authored Julia + ordinary Enzyme is an independent compiler control.
function native_gaussian(theta, X, y)
    p = size(X, 2)
    beta = view(theta, 1:p)
    s = theta[p + 1]
    eta = X * beta
    invsigma = exp(-s)
    total = -sum(abs2, beta) / 8 - s * s / 2
    @inbounds @simd for i in eachindex(y)
        r = (y[i] - eta[i]) * invsigma
        total += -r * r / 2 - s
    end
    total
end
function native_bernoulli(theta, X, y)
    eta = X * theta
    total = -sum(abs2, theta) / 8
    @inbounds @simd for i in eachindex(y)
        total += y[i] * eta[i] - log1pexp(eta[i])
    end
    total
end

# A separate numerical oracle, never a timed implementation.
function reference(family, theta, X, y)
    p = size(X, 2)
    beta = view(theta, 1:p)
    eta = X * beta
    g = -theta / 4
    if family === :gaussian
        s = theta[p + 1]
        r = (y - eta) * exp(-s)
        value = -dot(r, r) / 2 - length(y) * s - dot(beta, beta) / 8 - s*s/2
        g[1:p] += transpose(X) * (r * exp(-s))
        g[p + 1] = dot(r, r) - length(y) - s
    else
        value = sum(y .* eta - log1pexp.(eta)) - dot(beta, beta)/8
        g += transpose(X) * (y - logistic.(eta))
    end
    value, g
end
end

struct NativeGradient{F}
    f::F
end
function (call::NativeGradient)(g, theta, X, y)
    fill!(g, 0)
    result = Enzyme.autodiff(Enzyme.ReverseWithPrimal, call.f, Enzyme.Active,
        Enzyme.Duplicated(theta, g), Enzyme.Const(X), Enzyme.Const(y))
    result[2], g
end

function pbr_model(family, X, y)
    data = (x=X[:, 1], z=X[:, 2], y=y)
    priors = [PBR.prior("normal(0, 2)"; class=:b)]
    family === :gaussian && push!(priors, PBR.prior("lognormal(0, 1)"; class=:sigma))
    fam = family === :gaussian ? PBR.gaussian() : PBR.bernoulli()
    ld, _ = PBR.build_model(PBR.@bf(y ~ 0 + x + z), data; family=fam, prior=priors)
    ld
end

function measurement(label, f)
    f()
    trial = @benchmark $f() samples=1000 evals=1 seconds=0.3
    m = median(trial)
    println("MEASURE label=",label," ns=",m.time," bytes=",m.memory," allocs=",m.allocs)
    flush(stdout)
end

# Function barriers give every timed closure immutable, concretely typed
# captures; construction and cache sizing are outside the timed operation.
measure_pbr(label, ld, g, theta) = begin
    measurement("$label/pbr_inplace", () -> PBR.logdensity_and_gradient!(g, ld, theta))
    measurement("$label/pbr_fresh", () -> LDP.logdensity_and_gradient(ld, theta))
end
measure_native(label, native, g, args) = measurement(label, () -> native(g, args...))
measure_rk(label, op, g, args) = measurement(label, () -> ad_value_and_gradient!(op, g, args...))

function prepare_mode(spec, backend, args, nonallocating, plate_on)
    with_plate_reverse(plate_on) do
        kernel = nonallocating ? prepare_nonallocating(spec, args...) : prepare(spec)
        prepare_ad(kernel, backend, args...; active=:theta)
    end
end

function run_family(family)
    println("BEGIN family=", family); flush(stdout)
    rng = Xoshiro(713)
    X = randn(rng, 128, 2)
    y = family === :gaussian ? randn(rng, 128) : Float64.(rand(rng, 128) .< 0.5)
    theta = family === :gaussian ? [0.2, -0.15, 0.3] : [0.2, -0.15]
    spec = getfield(MatchedTargets, family)
    native = NativeGradient(getfield(MatchedTargets, Symbol(:native_, family)))
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    modes = Any[]
    for nonallocating in (false, true), plate_on in (false, true)
        label = "rk_na$(nonallocating)_plate$(plate_on)"
        println("PREP ", family, " ", label); flush(stdout)
        op = prepare_mode(spec, backend, (theta, X, y), nonallocating, plate_on)
        push!(modes, (label, op, nonallocating, plate_on))
    end
    for n in (3, 128, 2048, 8192)
        X = randn(rng, n, 2)
        y = family === :gaussian ? randn(rng, n) : Float64.(rand(rng, n) .< 0.5)
        ld = pbr_model(family, X, y)
        @test ld.dim == length(theta)
        expected = MatchedTargets.reference(family, theta, X, y)
        g = zero(theta)
        v = PBR.logdensity_and_gradient!(g, ld, theta)
        @test v ≈ expected[1]
        @test g ≈ expected[2]
        @test LDP.logdensity(ld, theta) ≈ expected[1]
        original = deepcopy((theta, X, y))
        measure_pbr("$family/n$n", ld, g, theta)
        value, grad = native(g, theta, X, y)
        @test value ≈ expected[1]
        @test grad ≈ expected[2]
        measure_native("$family/n$n/native_enzyme", native, g, (theta, X, y))
        for (label, op, nonallocating, plate_on) in modes
            value, grad = ad_value_and_gradient!(op, g, theta, X, y)
            @test value ≈ expected[1]
            @test grad ≈ expected[2]
            # Changed parameters exercise the same prepared call and cached storage.
            alt = theta .+ 0.17
            refalt = MatchedTargets.reference(family, alt, X, y)
            va, ga = ad_value_and_gradient!(op, g, alt, X, y)
            @test va ≈ refalt[1]
            @test ga ≈ refalt[2]
            # PBR prepares its workspace for n; use the same cache policy here.
            # The original n=128 operation above still checks changed lengths.
            sized = prepare_mode(spec, backend, (theta, X, y), nonallocating, plate_on)
            vs, gs = ad_value_and_gradient!(sized, g, theta, X, y)
            @test vs ≈ expected[1]
            @test gs ≈ expected[2]
            measure_rk("$family/n$n/$label", sized, g, (theta, X, y))
        end
        if family === :bernoulli
            for plate_on in (false, true)
                shared = prepare_mode(MatchedTargets.bernoulli_shared, backend,
                    (theta, X, y), true, plate_on)
                vs, gs = ad_value_and_gradient!(shared, g, theta, X, y)
                @test vs ≈ expected[1]
                @test gs ≈ expected[2]
                measure_rk("$family/n$n/shared_softplus_plate$plate_on", shared, g, (theta, X, y))
            end
        end
        @test (theta, X, y) == original
    end
    # Empty RK/native input, independent of the PBR formula builder's support.
    X = zeros(0,2); y = Float64[]
    expected = MatchedTargets.reference(family, theta, X, y)
    for (_, op, _, _) in modes
        result = ad_value_and_gradient(op, theta, X, y)
        @test result[1] ≈ expected[1]
        @test result[2] ≈ expected[2]
    end
    println("END family=",family); flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("PROVENANCE Julia=", VERSION, " RK=", pkgdir(RK), " PBR=", pkgdir(PBR),
        " Enzyme=", pkgdir(Enzyme), " library=", Enzyme.Compiler.Enzyme_jll.libEnzyme_path)
    @testset "Matched RK and PBR targets" begin
        for x in (-1000.0, -40.0, -1.0, -0.0, 0.0, 1.0, 40.0, 1000.0)
            v, d = derivative_cut(MatchedTargets.shared_softplus, (true,), x)
            @test v ≈ log1pexp(x)
            @test d ≈ logistic(x)
        end
        for family in (isempty(ARGS) ? (:gaussian, :bernoulli) : Tuple(Symbol.(ARGS)))
            run_family(family)
        end
    end
    println("COMPLETE")
end
