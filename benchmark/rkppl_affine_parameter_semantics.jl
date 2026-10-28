# Compare this script under a consumer environment developed against the
# canonical checkout and under one developed against the candidate checkout.
# Both environments need ReactiveKernels, ReactiveKernelsPPL, Enzyme and
# DifferentiationInterface as direct dependencies. Do not change this repo's
# Project/Manifest to run the comparison.
#
# The AST, constrained parameter values and observation counts are identical
# on both revisions. The coordinate mapping admits both the historical
# predictor-owned pack and ordinary declaration-owned coordinates. Report the
# source path, density, warmed median time and allocation count for each case.
# Shared-host timing needs corroboration; equal allocations establish neither
# a speedup nor a complete performance comparison.

using ReactiveKernels, ReactiveKernelsPPL, Enzyme, DifferentiationInterface
using Statistics

function measure(f, u; iterations=10000)
    f(u)
    f(u)
    allocation = @allocated f(u)
    times = Float64[]
    for _ in 1:7
        GC.gc()
        elapsed = @elapsed for _ in 1:iterations
            f(u)
        end
        push!(times, elapsed * 1e9 / iterations)
    end
    return (; ns=median(times), bytes=allocation)
end

function bench(kind, n)
    x = collect(range(-1.0, 1.0; length=n))
    y = 0.2 .+ 0.3 .* x
    ast = kind === :scalar ? quote
        a ~ Normal(0, 5)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1)
        mu = a .+ b .* x
        y .~ Normal.(mu, sigma)
    end : quote
        X = hcat(1, x)
        b[axes(X, 2)] .~ Normal.(0, 1)
        sigma ~ Exponential(1)
        mu = X * b
        y .~ Normal.(mu, sigma)
    end
    plan = RKPPLModel(ast, @__MODULE__)(; x, y)
    built = build_kernel(plan)
    names = coordinate_names(built.layout)
    modern = isempty(plan.population_priors)
    u = kind === :scalar || !modern ? [0.2, -0.3, log(0.8)] :
        [log(0.8), 0.2, -0.3]
    post = prepare_query(built, plan, :sampler)
    sampler = prepare_sampler(built, plan, u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    grad = similar(u)
    primal(w) = Base.invokelatest(post, w)
    derivative(w) = Base.invokelatest(sampler_value_and_gradient!, sampler, grad, w)[1]
    println("BENCH kind=", kind, " n=", n, " names=", names,
        " value=", primal(u), " primal=", measure(primal, u),
        " gradient=", measure(derivative, u; iterations=1000))
end

println("SOURCE ", pathof(ReactiveKernelsPPL))
for kind in (:scalar, :matrix), n in (32, 1024)
    bench(kind, n)
end
