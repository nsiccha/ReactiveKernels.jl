using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Statistics, Test

# Absent observations (snag rkppl-bind-a-des-e6cfca75; user pick on design
# decision `1spnf7v`: omit the value). A binding that supplies no value for
# an observation binds the same program, leaving that value out of every
# query's inputs: values that do not read it evaluate on the graph built
# from a binding that supplied it, and queries reading it are refused,
# naming it. A prediction design binds without placeholder responses.

const _AO_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
_ao_x(n) = collect(range(-1.2, 1.5; length = n))
_ao_prepare(built, plan, want) = ReactiveKernelsPPL._prepare_wants(built, plan, want)
_ao_eval(q, u) = Base.invokelatest(q, u)
_ao_refused(f, name) = try
    f()
    false
catch e
    msg = sprint(showerror, e)
    e isa ContractValidationError && occursin(Regex("leaves .*`$name`.* absent"), msg)
end

const _AO_REPORTER = quote
    a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1)
    xc = Statistics.mean(x_fit); xs = Statistics.std(x_fit)
    z = (x .- xc) ./ xs
    mu = a .+ b .* z
    y .~ Normal.(mu, s)
end

@testset "absent observations: a prediction design on the retained graph" begin
    X = _ao_x(9)
    Y = 0.4 .+ 0.9 .* X .+ 0.05 .* sin.(1:9)
    data = Dict{Symbol,Any}(:x => X, :x_fit => X, :y => Y)
    fit = bind_data(lower_rkppl(_AO_REPORTER, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(fit)
    for (rows, label) in ((2:5, "design rows"), (3:3, "one row"), (1:9, "every row"))
        design = Dict{Symbol,Any}(:x => X[rows], :x_fit => X)
        before = deepcopy(design)
        d = bind_data(fit, design)
        @test design == before                       # caller data untouched
        @test ReactiveKernelsPPL._absent_observations(d) == [:y]
        for (a, b, s) in ((0.3, -0.7, 0.8), (-1.1, 0.4, 2.5))
            u = unconstrain(built.layout, (; a, b, s))
            mu = a .+ b .* (X[rows] .- mean(X)) ./ std(X)
            @test _ao_eval(_ao_prepare(built, d, (:mu,)), u) ≈ mu rtol = 1e-12
            obs = _ao_eval(prepare_query(built, d, :observations), u)
            @test obs.y.location ≈ mu rtol = 1e-12
            @test obs.y.scale ≈ s rtol = 1e-12
            @test _ao_eval(prepare_query(built, d, :prior), u) ≈
                logpdf(Normal(0, 1), a) + logpdf(Normal(0, 1), b) +
                logpdf(Exponential(1), s) rtol = 1e-12
            @test _ao_eval(prepare_query(built, d, :log_jacobian), u) ≈ log(s) rtol = 1e-12
        end
        for preset in (:sampler, :likelihood, :pointwise)
            @test _ao_refused(() -> prepare_query(built, d, preset), :y)
        end
        u0 = unconstrain(built.layout, (; a = 0.1, b = 0.2, s = 1.0))
        @test _ao_refused(() -> prepare_sampler(built, d, u0; backend = _AO_BACKEND), :y)
        @test _ao_refused(() -> prepare_cell_query(built, d, :y), :y)
        @test _ao_refused(() -> prepare_cell_sampler(built, d, :y, u0; backend = _AO_BACKEND), :y)
    end
end

@testset "absent observations: every spelling binds the same program" begin
    X = _ao_x(7)
    Y = sin.(X)
    data = Dict{Symbol,Any}(:x => X, :x_fit => X, :y => Y)
    fit = bind_data(lower_rkppl(_AO_REPORTER, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(fit)
    design = Dict{Symbol,Any}(:x => X[1:4], :x_fit => X)
    m = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1)
        xc = Statistics.mean(x_fit); xs = Statistics.std(x_fit)
        z = (x .- xc) ./ xs
        mu = a .+ b .* z
        y .~ Normal.(mu, s)
    end
    spellings = (
        "rebinding the fit" => bind_data(fit, design),
        "names-only lowering" => bind_data(lower_rkppl(_AO_REPORTER, (:x, :x_fit);
            conditioned = (:y,), mod = @__MODULE__), design),
        "value lowering" => bind_data(lower_rkppl(_AO_REPORTER, design;
            conditioned = (:y,), mod = @__MODULE__), design),
        "model call without the response" => m(; x = X[1:4], x_fit = X),
    )
    u = unconstrain(built.layout, (; a = 0.2, b = -0.5, s = 1.3))
    expected = 0.2 .+ -0.5 .* (X[1:4] .- mean(X)) ./ std(X)
    for (label, d) in spellings
        @testset "$label" begin
            @test _ao_eval(prepare_query(built, d, :observations), u).y.location ≈ expected rtol = 1e-12
            @test _ao_refused(() -> prepare_query(built, d, :sampler), :y)
        end
    end
    # The model call observes once the response is supplied, as before.
    @test _ao_eval(prepare_query(built, m(; x = X, x_fit = X) | (; y = Y), :sampler), u) ≈
        _ao_eval(prepare_query(built, fit, :sampler), u) rtol = 1e-12
end

@testset "absent observations: building from a binding without its response" begin
    X = _ao_x(6)
    Y = 0.2 .- X
    data = Dict{Symbol,Any}(:x => X, :x_fit => X, :y => Y)
    fit = bind_data(lower_rkppl(_AO_REPORTER, data; conditioned = (:y,), mod = @__MODULE__), data)
    design = bind_data(fit, Dict{Symbol,Any}(:x => X[2:3], :x_fit => X))
    absent_built = build_kernel(design)
    own = build_kernel(fit)
    @test coordinate_names(absent_built.layout) == coordinate_names(own.layout)
    u = unconstrain(own.layout, (; a = -0.4, b = 0.9, s = 0.7))
    # The fit binding is the same program: its posterior on the graph built
    # without the response equals its own build's.
    @test _ao_eval(prepare_query(absent_built, fit, :sampler), u) ≈
        _ao_eval(prepare_query(own, fit, :sampler), u) rtol = 1e-12
    # Supplying the response again rebinds it as an observation.
    rebound = bind_data(design, data)
    @test isempty(ReactiveKernelsPPL._absent_observations(rebound))
    @test _ao_eval(prepare_query(own, rebound, :sampler), u) ≈
        _ao_eval(prepare_query(own, fit, :sampler), u) rtol = 1e-12
end

@testset "absent observations: ordinary Reverse gradients of the available values" begin
    X = _ao_x(8)
    data = Dict{Symbol,Any}(:x => X, :x_fit => X, :y => cos.(X))
    fit = bind_data(lower_rkppl(_AO_REPORTER, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(fit)
    d = bind_data(fit, Dict{Symbol,Any}(:x => X[1:3], :x_fit => X))
    u = unconstrain(built.layout, (; a = 0.6, b = -0.3, s = 1.7))
    ad = prepare_query_ad(prepare_ad, built, d, :prior, _AO_BACKEND, u)
    value, g = ad_value_and_gradient!(ad, zeros(length(u)), u)
    nt = constrain(built.layout, u)
    @test value ≈ logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b) +
        logpdf(Exponential(1), nt.s) rtol = 1e-12
    # d/du of log p(a) + log p(b) + log p(exp(u3)) (Exponential(1): -s).
    names = string.(coordinate_names(built.layout))
    @test g[findfirst(==("a"), names)] ≈ -nt.a rtol = 1e-12
    @test g[findfirst(==("b"), names)] ≈ -nt.b rtol = 1e-12
    @test g[findfirst(==("s"), names)] ≈ -nt.s rtol = 1e-12
end

@testset "absent observations: response families and shapes" begin
    X = _ao_x(8)
    cases = (
        (name = "Poisson counts", src = quote
            a ~ Normal(0, 1); b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ Poisson.(exp.(eta))
        end, fit = Dict{Symbol,Any}(:x => X, :y => [0, 2, 1, 4, 0, 3, 1, 2]),
            design = Dict{Symbol,Any}(:x => X[1:3]), value = :eta,
            oracle = (nt, d) -> nt.a .+ nt.b .* d[:x]),
        (name = "Bernoulli Bool", src = quote
            a ~ Normal(0, 1); b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ Bernoulli.(logistic.(eta))
        end, fit = Dict{Symbol,Any}(:x => X, :y => Bool[1, 0, 1, 1, 0, 0, 1, 0]),
            design = Dict{Symbol,Any}(:x => X[2:6]), value = :eta,
            oracle = (nt, d) -> nt.a .+ nt.b .* d[:x]),
        (name = "Binomial with trials", src = quote
            a ~ Normal(0, 1)
            eta = a .+ x
            y .~ Binomial.(n, logistic.(eta))
        end, fit = Dict{Symbol,Any}(:x => X, :n => fill(5, 8), :y => [0, 5, 2, 3, 1, 4, 2, 2]),
            design = Dict{Symbol,Any}(:x => X[1:2], :n => [4, 6]), value = :eta,
            oracle = (nt, d) -> nt.a .+ d[:x]),
        (name = "factor gather", src = quote
            a ~ Normal(0, 1); s ~ Exponential(1)
            c[levels(g)] .~ Normal.(0, 1)
            mu = a .+ c[g]
            y .~ Normal.(mu, s)
        end, fit = Dict{Symbol,Any}(:g => repeat(1:3, 3)[1:8], :y => X),
            design = Dict{Symbol,Any}(:g => [3, 1, 3, 2]), value = :mu,
            oracle = (nt, d) -> nt.a .+ nt.c[d[:g]]),
        (name = "ordered logistic", src = quote
            b ~ Normal(0, 1)
            cuts ~ Ordered(Normal(0, 1), 2)
            eta = b .* x
            y .~ OrderedLogistic.(eta, Ref(cuts))
        end, fit = Dict{Symbol,Any}(:x => X, :y => [1, 3, 2, 2, 1, 3, 3, 2]),
            design = Dict{Symbol,Any}(:x => X[4:5]), value = :eta,
            oracle = (nt, d) -> nt.b .* d[:x]),
        (name = "@plate cell over the response", src = quote
            a ~ Normal(0, 1); s ~ Exponential(1)
            @plate for i in eachindex(y)
                y[i] ~ Normal(a + sqrt(abs(x[i])), s)
            end
        end, fit = Dict{Symbol,Any}(:x => X, :y => X .^ 2),
            design = Dict{Symbol,Any}(:x => X[1:5]), value = nothing, oracle = nothing),
        (name = "@plate cell over another array", src = quote
            a ~ Normal(0, 1); s ~ Exponential(1)
            @plate for i in eachindex(x)
                r[i] = a + x[i]
                y[i] ~ Normal(r[i], s)
            end
        end, fit = Dict{Symbol,Any}(:x => X, :y => X .+ 0.1),
            design = Dict{Symbol,Any}(:x => X[2:4]), value = :r,
            oracle = (nt, d) -> nt.a .+ d[:x]),
        (name = "joint outcomes", src = quote
            a1 ~ Normal(0, 1); a2 ~ Normal(0, 1)
            sd[1:2] .~ Exponential.(1)
            C ~ LKJCholesky(2, 2)
            L = sd .* C
            mu1 = a1 .+ x
            mu2 = a2 .- x
            [y1, y2] ~ MvNormalCholesky([mu1, mu2], L)
        end, fit = Dict{Symbol,Any}(:x => X, :y1 => sin.(X), :y2 => cos.(X)),
            design = Dict{Symbol,Any}(:x => X[1:3]), value = :mu2,
            oracle = (nt, d) -> nt.a2 .- d[:x]),
    )
    for c in cases
        @testset "$(c.name)" begin
            observed = Tuple(k for k in (:y, :y1, :y2) if haskey(c.fit, k))
            fit = bind_data(lower_rkppl(c.src, c.fit; conditioned = observed,
                mod = @__MODULE__), c.fit)
            built = build_kernel(fit)
            d = bind_data(fit, c.design)
            @test ReactiveKernelsPPL._absent_observations(d) == collect(observed)
            u = [0.1 * i - 0.35 for i in 1:built.layout.total]
            @test _ao_eval(prepare_query(built, d, :prior), u) ≈
                _ao_eval(prepare_query(built, fit, :prior), u) rtol = 1e-12
            @test _ao_eval(prepare_query(built, d, :log_jacobian), u) ≈
                _ao_eval(prepare_query(built, fit, :log_jacobian), u) rtol = 1e-12
            @test _ao_refused(() -> prepare_query(built, d, :likelihood), first(observed))
            if c.value !== nothing
                nt = constrain(built.layout, u)
                @test _ao_eval(_ao_prepare(built, d, (c.value,)), u) ≈ c.oracle(nt, c.design) rtol = 1e-12
            end
        end
    end
end

_ao_shift_cells(t, a) = [a .+ ti for ti in t]

@testset "absent observations: one array per index" begin
    X = [[0.1, 0.2], [0.3], [0.4, 0.5, 0.6]]
    Y = [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]
    # Data read per index (`x[i]`) give each index's cell its shape.
    src = quote
        a ~ Normal(0, 1); s ~ Exponential(1)
        @plate for i in eachindex(y)
            y[i] .~ Normal.(a .+ x[i], s)
        end
    end
    fit = bind_data(lower_rkppl(src, Dict{Symbol,Any}(:x => X, :y => Y);
        conditioned = (:y,), mod = @__MODULE__), Dict{Symbol,Any}(:x => X, :y => Y))
    built = build_kernel(fit)
    d = bind_data(fit, Dict{Symbol,Any}(:x => [[0.1], [0.2, 0.3]]))
    @test length(d.columns[:y]) == 2
    @test size(d.columns[:y][2]) == (2,)
    u = [0.3, -0.2]
    @test _ao_eval(prepare_query(built, d, :prior), u) ≈
        _ao_eval(prepare_query(built, fit, :prior), u) rtol = 1e-12
    @test _ao_refused(() -> prepare_query(built, d, :sampler), :y)
    # This program computes each cell's observation arguments inside the
    # plate over the response's entries, so `:observations` reads it.
    @test _ao_refused(() -> prepare_query(built, d, :observations), :y)

    # A per-index location computed by a function: the built graph says the
    # response holds one array per index.
    src = quote
        a ~ Normal(0, 1); s ~ Exponential(1)
        loc = _ao_shift_cells(t, a)
        @plate for i in eachindex(t)
            y[i] .~ Normal.(loc[i], s)
        end
    end
    data = Dict{Symbol,Any}(:t => X, :y => Y)
    fit = bind_data(lower_rkppl(src, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(fit)
    T = [[0.5], [0.7, 0.9]]
    d = bind_data(fit, Dict{Symbol,Any}(:t => T))
    a = constrain(built.layout, u).a
    @test _ao_eval(_ao_prepare(built, d, (:loc,)), u) ≈ [a .+ ti for ti in T] rtol = 1e-12
    @test _ao_eval(prepare_query(built, d, :prior), u) ≈
        _ao_eval(prepare_query(built, fit, :prior), u) rtol = 1e-12
    @test _ao_refused(() -> prepare_query(built, d, :likelihood), :y)
    # Not built yet: without per-index data or a built graph, a binding
    # cannot tell that the absent response holds one array per index, so a
    # graph built from it observes one value per index.
    @test_broken try
        _ao_eval(prepare_query(build_kernel(d), fit, :prior), u) ≈
            _ao_eval(prepare_query(built, fit, :prior), u)
    catch
        false
    end
end

@testset "absent observations: a conditioned declaration without its value" begin
    src = quote
        theta ~ Beta(2, 2)
        k ~ Binomial(n, theta)
    end
    fit = bind_data(lower_rkppl(src, (; n = 5, k = 2); conditioned = (:k,), mod = @__MODULE__),
        (; n = 5, k = 2))
    built = build_kernel(fit)
    d = bind_data(lower_rkppl(src, (; n = 5); conditioned = (:k,), mod = @__MODULE__), (; n = 5))
    @test ReactiveKernelsPPL._absent_observations(d) == [:k]
    u = [0.4]
    theta = constrain(built.layout, u).theta
    @test _ao_eval(prepare_query(built, d, :prior), u) ≈ logpdf(Beta(2, 2), theta) rtol = 1e-12
    @test _ao_refused(() -> prepare_query(built, d, :likelihood), :k)
    @test _ao_refused(() -> prepare_query(built, d, :sampler), :k)
end

@testset "absent observations: values computed from an absent observation" begin
    X = _ao_x(6)
    Y = 1.0 .+ X
    src = quote
        ybar = Statistics.mean(y)
        a ~ Normal(ybar, 1); s ~ Exponential(1)
        mu = a .+ x
        y .~ Normal.(mu, s)
    end
    data = Dict{Symbol,Any}(:x => X, :y => Y)
    fit = bind_data(lower_rkppl(src, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(fit)
    d = bind_data(fit, Dict{Symbol,Any}(:x => X[1:2]))
    # `ybar` reads `y`, so it is absent too; `mu` reads neither.
    @test sort(ReactiveKernelsPPL._absent_inputs(d)) == [:y, :ybar]
    u = unconstrain(built.layout, (; a = 0.7, s = 1.1))
    @test _ao_eval(_ao_prepare(built, d, (:mu,)), u) ≈ 0.7 .+ X[1:2] rtol = 1e-12
    @test _ao_refused(() -> prepare_query(built, d, :prior), :y)
    # Its type and shape come from the definition: a graph built without `y`
    # is the fit's program.
    @test _ao_eval(prepare_query(build_kernel(d), fit, :sampler), u) ≈
        _ao_eval(prepare_query(built, fit, :sampler), u) rtol = 1e-12

    # A program shape read from an absent observation cannot be known.
    ordinal = quote
        b ~ Normal(0, 1)
        cuts ~ Ordered(Normal(0, 1), length(levels(y)) - 1)
        y .~ OrderedLogistic.(b .* x, Ref(cuts))
    end
    plan = lower_rkppl(ordinal, (:x, :y); conditioned = (:y,), mod = @__MODULE__)
    err = try
        bind_data(plan, Dict{Symbol,Any}(:x => X))
        nothing
    catch e
        e
    end
    @test err !== nothing && occursin("`y` is absent", sprint(showerror, err))
    # Rebinding a binding that sized it keeps that size: the same program.
    data = Dict{Symbol,Any}(:x => X, :y => [1, 2, 3, 1, 2, 3])
    sized = bind_data(plan, data)
    d = bind_data(sized, Dict{Symbol,Any}(:x => X[1:3]))
    @test only(d.vector_parameters).size == 2
    built = build_kernel(sized)
    u = [0.3, -0.2, 0.4]
    @test _ao_eval(prepare_query(built, d, :prior), u) ≈
        _ao_eval(prepare_query(built, sized, :prior), u) rtol = 1e-12
end

@testset "absent observations: what stays refused" begin
    X = _ao_x(6)
    data = Dict{Symbol,Any}(:x => X, :x_fit => X, :y => X)
    fit = bind_data(lower_rkppl(_AO_REPORTER, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(fit)
    # Only observations may be absent: other data stay required.
    @test_throws ContractValidationError bind_data(fit, Dict{Symbol,Any}(:x_fit => X))
    # Another program (other factor levels) still needs its own build.
    src = quote
        a ~ Normal(0, 1); s ~ Exponential(1)
        c[levels(g)] .~ Normal.(0, 1)
        y .~ Normal.(a .+ c[g], s)
    end
    gfit = bind_data(lower_rkppl(src, Dict{Symbol,Any}(:g => [1, 2, 1, 2], :y => X[1:4]);
        conditioned = (:y,), mod = @__MODULE__), Dict{Symbol,Any}(:g => [1, 2, 1, 2], :y => X[1:4]))
    gbuilt = build_kernel(gfit)
    d = bind_data(gfit, Dict{Symbol,Any}(:g => [1, 2, 3]))
    @test_throws "different program" prepare_query(gbuilt, d, :prior)
    # The documented raw cut reads the binding's columns: available values
    # evaluate, and reading the absent response names it.
    d = bind_data(fit, Dict{Symbol,Any}(:x => X[1:2], :x_fit => X))
    ks = Tuple(sort!(collect(keys(d.columns))))
    bound = NamedTuple{ks}(Tuple(d.columns[k] for k in ks))
    u = unconstrain(built.layout, (; a = 0.5, b = 1.5, s = 1.0))
    mu = Base.invokelatest(ReactiveKernels.prepare(built.spec; have = (:unconstrained, ks...),
        want = (:mu,), bound), u)
    @test mu ≈ 0.5 .+ 1.5 .* (X[1:2] .- mean(X)) ./ std(X) rtol = 1e-12
    @test_throws ReactiveKernelsPPL.AbsentObservationError Base.invokelatest(
        ReactiveKernels.prepare(built.spec; have = (:unconstrained, ks...),
            want = (:likelihood,), bound), u)
end
