# Diagnostic arithmetic/scheduling controls. Handwritten derivatives below
# are benchmark references, never production backend rules. The old select
# graph remains in pbr_matched_targets.jl for same-process comparisons.
include("pbr_matched_targets.jl")

module ArithmeticTargets
using ReactiveKernels

@kernel current(theta::Vector{Float64}, X::Matrix{Float64}, y::Vector{Float64}) = begin
    eta::Vector{Float64} = X * theta
    terms = plate(eta, y) do ei, yi
        yi * ei - rk_log1pexp(ei)
    end
    total::Float64 = sum(terms) - sum(abs2, theta) / 8
    return total
end

@inline function direct_parts(x)
    q = exp(-abs(x))
    max(x, 0.0) + log1p(q), ifelse(x >= 0, 1.0, q) / (1 + q)
end
function fused!(g, x, y, f)
    value = 0.0
    @inbounds @simd for i in eachindex(x, y, g)
        v, d = f(x[i])
        value += y[i] * x[i] - v
        g[i] = y[i] - d
    end
    value, g
end
function primal(x, y, f)
    value = 0.0
    @inbounds @simd for i in eachindex(x, y)
        value += y[i] * x[i] - f(x[i])
    end
    value
end
end

function arithmetic_reverse!(g, x, y, f)
    fill!(g, 0.0)
    result = Enzyme.autodiff(Enzyme.ReverseWithPrimal,
        ArithmeticTargets.primal, Enzyme.Active,
        Enzyme.Duplicated(x, g), Enzyme.Const(y), Enzyme.Const(f))
    result[2], g
end
arithmetic_cut(f) = x -> derivative_cut(f, (true,), x)
arithmetic_thunk(f, g, x, y, p) = () -> f(g, x, y, p)
arithmetic_rk_thunk(q, g, args) = () -> ad_value_and_gradient!(q, g, args...)
arithmetic_pbr_thunk(q, g, x) = () -> (PBR.logdensity_and_gradient!(g, q, x), g)

function arithmetic_rounds(label, cases, rng)
    # Warm every case before the randomized rounds, and keep every result.
    foreach(c -> c[2](), cases)
    for round in 1:12, idx in randperm(rng, length(cases))
        name, f = cases[idx]
        t = median(@benchmark $f() samples=500 evals=1 seconds=0.2)
        println("MEASURE label=", label, " round=", round, " mode=", name,
            " ns=", t.time, " julia_bytes=", t.memory)
        flush(stdout)
    end
end

@testset "Softplus arithmetic and full-target controls" begin
    @assert Base.get_extension(RK, :ReactiveKernelsEnzymeExt) !== nothing
    @assert Base.get_extension(RK, :ReactiveKernelsMutatingFunctionsExt) !== nothing
    println("PROVENANCE Julia=", VERSION, " RK=", pkgdir(RK),
        " Enzyme=", pkgdir(Enzyme), " library=", Enzyme.Compiler.Enzyme_jll.libEnzyme_path)
    println("ALLOCATION_METRIC Julia heap only; excludes native AD tape mallocs")
    for n in (0, 3, 2048, 8192)
        rng = Xoshiro(819)
        X = randn(rng, n, 2)
        theta = [.2, -.15]
        x = X * theta
        y = Float64.(rand(rng, n) .< .5)
        original = deepcopy((X, theta, x, y))
        old = MatchedTargets.shared_softplus
        cases = [
            ("direct", arithmetic_thunk(ArithmeticTargets.fused!, zero(x), x, y,
                ArithmeticTargets.direct_parts)),
            ("owned_select", arithmetic_thunk(ArithmeticTargets.fused!, zero(x), x, y,
                arithmetic_cut(old))),
            ("owned_current", arithmetic_thunk(ArithmeticTargets.fused!, zero(x), x, y,
                arithmetic_cut(rk_log1pexp))),
            ("reverse_select", arithmetic_thunk(arithmetic_reverse!, zero(x), x, y, old)),
            ("reverse_current", arithmetic_thunk(arithmetic_reverse!, zero(x), x, y, rk_log1pexp)),
        ]
        for (_, f) in cases
            v, g = f()
            @test v ≈ sum(y .* x - log1pexp.(x); init=0.0)
            @test g ≈ y - logistic.(x)
        end
        n >= 2048 && arithmetic_rounds("likelihood/n$n", cases, rng)

        full = Any[]
        backend = AutoEnzyme(; mode=Enzyme.Reverse)
        for on in (false, true), (label, spec) in (
                ("select", MatchedTargets.bernoulli_shared),
                ("current", ArithmeticTargets.current))
            println("PREP ", label, " plate=", on, " n=", n)
            flush(stdout)
            q = prepare_mode(spec, backend, (theta, X, y), true, on)
            g = zero(theta)
            f = arithmetic_rk_thunk(q, g, (theta, X, y))
            v, got = f()
            expected = MatchedTargets.reference(:bernoulli, theta, X, y)
            @test v ≈ expected[1]
            @test got ≈ expected[2]
            @test got === g
            push!(full, ("$(label)_plate$on", f))
        end
        if n > 0
            q = pbr_model(:bernoulli, X, y)
            f = arithmetic_pbr_thunk(q, zero(theta), theta)
            v, g = f()
            expected = MatchedTargets.reference(:bernoulli, theta, X, y)
            @test v ≈ expected[1]
            @test g ≈ expected[2]
            push!(full, ("pbr", f))
        end
        n >= 2048 && arithmetic_rounds("full/n$n", full, rng)
        @test (X, theta, x, y) == original
    end
end
println("COMPLETE")
