module PredictorValueDependencyTests
using Test, ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme

const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
normal(x, s) = -0.5 * (x / s)^2 - log(s) - 0.5 * log(2pi)

function finite_gradient(f, u)
    h = cbrt(eps(Float64))
    return map(eachindex(u)) do i
        hi, lo = copy(u), copy(u)
        hi[i] += h
        lo[i] -= h
        (f(hi) - f(lo)) / (2h)
    end
end

function values_at(nt, data)
    base = nt.w[data.i] .* data.x
    left = sin.(base) .- nt.c[data.j]
    right = left .+ base .- nt.c[data.k]
    return (; base, left, right)
end

function reference(layout, data, u; reversed = false)
    nt = constrain(layout, u)
    v = values_at(nt, data)
    total = sum(x -> normal(x, 1.3), nt.c) + sum(x -> normal(x, 0.8), nt.w)
    for i in eachindex(data.y)
        a, b = reversed ? (v.right[i], v.left[i]) : (v.left[i], v.right[i])
        total += (data.y[i] == 1 ? 0.0 : data.y[i] == 2 ? a : b) -
            log(1 + exp(a) + exp(b))
    end
    return total
end

function program(spelling)
    reader = spelling === :expanded ? :(sin.(base) .- c[j]) :
        spelling === :alias ? :alias : :left
    response = spelling === :reversed ?
        :(y .~ CategoricalLogit.(right, left)) :
        :(y .~ CategoricalLogit.(left, right))
    return quote
        c[1:K] .~ Normal.(0, 1.3)
        w[1:K] .~ Normal.(0, 0.8)
        base = w[i] .* x
        left = sin.(base) .- c[j]
        alias = left
        right = $reader .+ base .- c[k]
        $response
    end
end

function check_case(spelling, n, K, shift)
    data = (; K, i = [mod1(t + shift, K) for t in 1:n],
        j = [mod1(2t + shift, K) for t in 1:n],
        k = [mod1(t + 1 + shift, K) for t in 1:n],
        x = [0.2 * cos(t + shift) for t in 1:n],
        y = [mod1(t + shift, 3) for t in 1:n])
    saved = deepcopy(data)
    plan = bind_data(lower_rkppl(program(spelling), data; conditioned = (:y,)), data)
    built = build_kernel(plan)
    u = [0.2 * sin(t) for t in 1:built.layout.total]
    sampler = prepare_sampler(built, plan, u; backend = BACKEND,
        native_forward_batch = false)
    @test sampler.ad isa ReactiveKernels.PreparedADOperator
    query = prepare(built.spec; have = built.spec.have_names,
        want = (:base, :left, :right, :alias), bound = NamedTuple(plan.columns))
    for point in (u, -1.7 .* u)
        before = copy(point)
        f = v -> reference(built.layout, data, v; reversed = spelling === :reversed)
        value, grad = sampler_value_and_gradient!(sampler, similar(point), point)
        @test value ≈ f(point) rtol = 2e-12
        @test grad ≈ finite_gradient(f, point) rtol = 2e-7 atol = 2e-8
        expected = values_at(constrain(built.layout, point), data)
        actual = Base.invokelatest(query, point)
        @test actual[1] ≈ expected.base
        @test actual[2] ≈ expected.left
        @test actual[3] ≈ expected.right
        @test actual[4] ≈ expected.left
        @test point == before
        @test data == saved
    end
    return (; sampler, plan, built, data, u)
end

@testset "predictor locations remain values for column readers" begin
    # Naming, aliasing, or changing which category is lowered first must
    # preserve the same dependency chain and ordinary reverse derivatives.
    for spelling in (:named, :alias, :expanded, :reversed), (n, K, shift) in
            ((3, 2, 0), (6, 3, 1), (9, 4, 2))
        @testset "$spelling / n=$n / K=$K" begin
            check_case(spelling, n, K, shift)
        end
    end
end

function factor_case(mode, n, m)
    declaration = mode === :matrix ? quote
        X = hcat(x, ones(length(x)))
        c[axes(X, 2)] .~ Normal.(0, 1.3)
        mu = X * c
    end : quote
        c[1:K] .~ Normal.(0, 1.3)
        mu = c[1] .+ a .* x
    end
    factor = mode === :positive ? quote
        total = 1 - sum(c .^ 2)
        total ~ Exponential(factor_scale)
    end : quote
        total = -sum(c)
        total ~ Normal(0, factor_scale)
    end
    # Flatten only the fixed authoring blocks, never a data-derived loop.
    ast = Expr(:block, :(a ~ Normal(0, 0.8)), declaration.args..., factor.args...,
        :(y .~ Normal.(mu, 1)), :(z .~ Normal.(a .* w, 1.4)))
    data = (; K = 3, factor_scale = 1.7,
        x = [0.2 * sin(i) for i in 1:n], w = [0.3 * cos(i) for i in 1:m],
        y = [0.1 * cos(i) for i in 1:n], z = [0.2 * sin(i) for i in 1:m])
    saved = deepcopy(data)
    plan = bind_data(lower_rkppl(ast, data; conditioned = (:y, :z)), data)
    built = build_kernel(plan)
    @test length(plan.factors) == 1
    @test plan.n_obs == n + m
    @test built.layout.total == 1 + (mode === :matrix ? 2 : data.K)
    @test ReactiveKernelsPPL._uses_structured_observation_axes(plan) == (mode === :matrix)
    u = [0.1 * sin(i) for i in 1:built.layout.total]
    sampler = prepare_sampler(built, plan, u; backend = BACKEND,
        native_forward_batch = false)
    @test sampler.ad isa ReactiveKernels.PreparedADOperator
    prior = prepare_query(built, plan, :prior)
    function reference(point; prior_only = false)
        nt = constrain(built.layout, point)
        f = mode === :positive ?
            -(1 - sum(abs2, nt.c)) / data.factor_scale - log(data.factor_scale) :
            normal(-sum(nt.c), data.factor_scale)
        p = normal(nt.a, 0.8) + sum(x -> normal(x, 1.3), nt.c) + f
        prior_only && return p
        mu = mode === :matrix ? hcat(data.x, ones(n)) * nt.c :
            nt.c[1] .+ nt.a .* data.x
        return p + sum(normal.(data.y .- mu, 1); init = 0.0) +
            sum(normal.(data.z .- nt.a .* data.w, 1.4); init = 0.0)
    end
    for point in (u, -1.7 .* u)
        value, grad = sampler_value_and_gradient!(sampler, similar(point), point)
        @test value ≈ reference(point) rtol = 2e-12
        @test grad ≈ finite_gradient(reference, point) rtol = 2e-7 atol = 2e-8
        @test Base.invokelatest(prior, point) ≈ reference(point; prior_only = true)
        @test data == saved
    end
    if mode === :positive
        invalid = unconstrain(built.layout, (; a = 0.2, c = fill(2.0, data.K)))
        @test sampler(invalid) == -Inf
        @test Base.invokelatest(prior, invalid) == -Inf
    end
    return (; sampler, plan, built, data, u)
end

@testset "scalar factors are independent of response axes" begin
    for mode in (:normal, :matrix, :positive), (n, m) in ((0, 3), (1, 4), (5, 2))
        @testset "$mode / $n and $m rows" begin
            factor_case(mode, n, m)
        end
    end
end

end
