using DifferentiationInterface, Distributions, Enzyme, Random, ReactiveKernels, ReactiveKernelsPPL, Test

_aps_add(x, y) = x + y
_aps_never_call(x, y) = error("shape discovery must not evaluate the cell")
_aps_nodes(x) = 0
_aps_nodes(x::Expr) = 1 + sum(_aps_nodes, x.args; init = 0)
_aps_nodes(x::ReactiveKernels._PlateProgram) = _aps_nodes(x.ast)
_aps_loops(x) = 0
_aps_loops(x::Expr) = (x.head in (:for, :while)) + sum(_aps_loops, x.args; init = 0)
_aps_loops(x::ReactiveKernels._PlateProgram) = _aps_loops(x.ast)

const _APS_HELPER = quote
    a ~ Normal(0, 1)
    mu = a .+ x
    @plate for i in eachindex(x)
        eta[i] = _aps_add(mu[i], 0.0 * x[i])
    end
    y .~ BernoulliLogit.(eta)
end

@testset "absent responses retain collected scalar plate extents" begin
    emitted_sizes = Dict{Int,Tuple{Int,Int,Int}}()
    for n in (0, 1, 2, 5), asdict in (false, true)
        data = (; x = collect(1.0:n))
        supplied = merge(data, (; y = zeros(Int, n)))
        container(d) = asdict ? Dict{Symbol,Any}(pairs(d)) : d
        plan = bind_data(lower_rkppl(_APS_HELPER, container(data);
            conditioned = (:y,), mod = @__MODULE__), container(data))
        fit = bind_data(lower_rkppl(_APS_HELPER, container(supplied);
            conditioned = (:y,), mod = @__MODULE__), container(supplied))
        built, fitbuilt = build_kernel(plan), build_kernel(fit)
        @test predictive_targets(built, plan).y.dims == (n,)
        pred, fitted = prepare_predictive(built, plan), prepare_predictive(fitbuilt, fit)
        if n in (2, 5) && !asdict
            emitted = ReactiveKernels.code_expr(pred.kernel)
            emitted_sizes[n] = (_aps_nodes(kernel_expr(plan, assign_layout(plan))),
                _aps_nodes(emitted), _aps_loops(emitted))
            @test _aps_loops(emitted) > 0
        end
        for a in (-0.7, 0.3)
            laws = predictive_laws(pred, [a]).y
            @test size(laws) == (n,)
            @test [p.p for p in laws] ≈ [1 / (1 + exp(-(a + x))) for x in data.x]
            @test Base.invokelatest(pred, [a], Xoshiro(17)) ==
                Base.invokelatest(fitted, [a], Xoshiro(17))
        end
        # Rebinding a fitted program must recover each new design's extent.
        rebound = bind_data(fit, container(data))
        @test predictive_targets(fitbuilt, rebound).y.dims == (n,)
        @test data.x == collect(1.0:n)
        if n == 5 && !asdict
            sampler = prepare_sampler(fitbuilt, fit, [0.3];
                backend = AutoEnzyme(; mode = Enzyme.Reverse))
            value, gradient = sampler_value_and_gradient!(sampler, zeros(1), [0.3])
            probabilities = [1 / (1 + exp(-(0.3 + x))) for x in data.x]
            @test value ≈ logpdf(Normal(), 0.3) + sum(log1p.(-probabilities))
            @test only(gradient) ≈ -0.3 - sum(probabilities)
        end
    end
    # Data lengths change storage and runtime iteration, not emitted bodies.
    @test emitted_sizes[2] == emitted_sizes[5]
end

@testset "plate shape discovery does not evaluate user cells" begin
    source = quote
        a ~ Normal(0, 1)
        @plate for i in eachindex(x)
            eta[i] = _aps_never_call(a, x[i])
        end
        y .~ BernoulliLogit.(eta)
    end
    data = (; x = zeros(5))
    plan = bind_data(lower_rkppl(source, data; conditioned = (:y,),
        mod = @__MODULE__), data)
    @test size(plan.columns[:y]) == (5,)
    matrix_source = quote
        a ~ Normal(0, 1)
        @plate for i in axes(x, 1)
            eta[i] = _aps_never_call(a, x[i, 1])
        end
        y .~ BernoulliLogit.(eta)
    end
    for n in (0, 2, 5)
        data = (; x = zeros(n, 3))
        plan = bind_data(lower_rkppl(matrix_source, data; conditioned = (:y,),
            mod = @__MODULE__), data)
        @test size(plan.columns[:y]) == (n,)
    end
end

@testset "computed plate iterators beside ragged responses" begin
    source = quote
        a ~ Normal(0, 1)
        b[axes(G, 2)] .~ Normal.(0, 1)
        mu = a .+ G * b
        @plate for i in eachindex(mu)
            eta[i] = _aps_add(mu[i], 0.1)
        end
        @plate for i in eachindex(eta)
            logits[i] = _aps_add(eta[i], -0.1)
        end
        flag .~ BernoulliLogit.(logits)
        @plate for i in eachindex(times)
            y[i] .~ Normal.(logits[i] .+ times[i], 1.0)
        end
    end
    for n in (1, 2, 5)
        data = (; G = hcat(ones(n), collect(1.0:n)),
            times = [collect(1.0:mod(i, 3)) for i in 1:n])
        supplied = merge(data, (; flag = zeros(Int, n), y = zero.(data.times)))
        fit = bind_data(lower_rkppl(source, supplied;
            conditioned = (:flag, :y), mod = @__MODULE__), supplied)
        built = build_kernel(fit)
        absent = bind_data(fit, data)
        @test predictive_targets(built, absent).flag.dims == (n,)
        pred, fitted = prepare_predictive(built, absent), prepare_predictive(built, fit)
        u = unconstrain(built.layout, (; a = 0.3, b = [0.2, -0.1]))
        laws = predictive_laws(pred, u)
        expected = 0.3 .+ data.G * [0.2, -0.1]
        @test [law.p for law in laws.flag] ≈ 1 ./ (1 .+ exp.(-expected))
        @test length.(laws.y) == length.(data.times)
        @test all([mean(law) for law in laws.y[i]] ≈ expected[i] .+ data.times[i]
            for i in 1:n)
        @test Base.invokelatest(pred, u, Xoshiro(27)) ==
            Base.invokelatest(fitted, u, Xoshiro(27))
    end
end
