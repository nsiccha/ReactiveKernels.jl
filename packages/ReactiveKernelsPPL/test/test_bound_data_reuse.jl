module BoundDataReuseTests
using ReactiveKernels, ReactiveKernelsPPL, Test

const calls = Ref(0)
function support(x)
    calls[] += 1
    (n = length(x), shift = sum(x))
end
prediction(x, a, s) = sum(a) .* x .+ s.shift

@testset "data definitions are shared by one binding" begin
    for sized in (false, true)
        declaration = sized ? :(a[1:s.n] .~ Normal.(0, 1)) : :(a ~ Normal(0, 1))
        source = quote
            s = support(x_fit)
            $declaration
            mu = prediction(x, a, s)
            y .~ Normal.(mu, 1)
        end
        data = (x = [0.5, 1.0, 1.5], x_fit = [0.2, 0.4], y = [0.1, 0.3, 0.5])
        plan = lower_rkppl(source, keys(data); conditioned = (:y,), mod = @__MODULE__)
        calls[] = 0
        bound = bind_data(plan, data)
        built = build_kernel(bound)
        u = zeros(built.layout.total)
        prior = prepare_query(built, bound, :prior)
        @test calls[] == Int(sized)
        results = Dict{Symbol,Any}()
        for preset in (:sampler, :likelihood, :pointwise, :observations, :sampler)
            q = prepare_query(built, bound, preset)
            results[preset] = Base.invokelatest(q, u)
            @test calls[] == 1
        end
        results[:prior] = Base.invokelatest(prior, u)
        state = frozen_values(bound, :s)
        @test calls[] == 1
        frozen = bind_data(plan, data; frozen = state)
        @test build_kernel(frozen).program == built.program
        for preset in keys(results)
            q = prepare_query(built, frozen, preset)
            @test isequal(Base.invokelatest(q, u), results[preset])
        end
        @test calls[] == 1

        # Both rebinding the original structure and rebinding a binding get
        # new storage, even when the new values have the same shape.
        for base in (plan, bound)
            rebound = bind_data(base, merge(data, (; x_fit = [2.0, 4.0])))
            q = prepare_query(built, rebound, :observations)
            @test Base.invokelatest(q, u).y.location == fill(6.0, 3)
        end
        @test calls[] == 3
        old = prepare_query(built, bound, :observations)
        @test Base.invokelatest(old, u) == results[:observations]
        @test calls[] == 3
    end
end

const guarded_calls = Ref(0)
function guarded_support(x)
    guarded_calls[] += 1
    x < 0 && error("negative support")
    x + 1
end

@testset "cache follows demand and retains lazy branches" begin
    spec = @kernel begin
        x::Float64
        t::Float64
        unused = guarded_support(-1.0)
        shared = guarded_support(x)
        y = t + shared
        z = t * shared
        lazy = t > 0 ? guarded_support(-1.0) : x
    end
    cache = ReactiveKernels._DataValues(spec.graph)
    guarded_calls[] = 0
    for (want, expected) in ((:y, 5.0), (:z, 6.0), (:y, 5.0))
        q = prepare(spec; have = (:x, :t), want, bound = (; x = 2.0), _data_cache = cache)
        @test Base.invokelatest(q, 2.0) == expected
        @test guarded_calls[] == 1
    end
    q = prepare(spec; have = (:x, :t), want = :lazy, bound = (; x = 2.0),
        _data_cache = cache)
    @test guarded_calls[] == 1
    @test Base.invokelatest(q, -1.0) == 2.0
    @test guarded_calls[] == 1
    @test_throws "negative support" Base.invokelatest(q, 1.0)
    # A reused internal store must also notice a changed bound input.
    q = prepare(spec; have = (:x, :t), want = :y, bound = (; x = 4.0), _data_cache = cache)
    @test Base.invokelatest(q, 2.0) == 7.0
end

@testset "data subexpressions share across query cuts" begin
    spec = @kernel begin
        x::Float64
        t::Float64
        y = t + guarded_support(x)
        z = y * t
    end
    cache = ReactiveKernels._DataValues(spec.graph)
    guarded_calls[] = 0
    q = prepare(spec; have = (:x, :t), want = :y, bound = (; x = 2.0), _data_cache = cache)
    @test Base.invokelatest(q, 2.0) == 5.0
    r = prepare(spec; have = (:x, :t), want = :z, bound = (; x = 2.0), _data_cache = cache)
    @test Base.invokelatest(r, 2.0) == 10.0
    @test guarded_calls[] == 1
end

@testset "composed SMC queries share binding data" begin
    for inline in (false, true)
        rhs = inline ? :(a .* x .+ support(x_fit).shift) : :(a .* x .+ s.shift)
        source = quote
            s = support(x_fit)
            a ~ Normal(0, 1)
            mu = $rhs
            y .~ Normal.(mu, 1)
        end
        data = (x = [0.5, 1.0, 1.5], x_fit = [0.2, 0.4], y = [0.1, 0.3, 0.5])
        calls[] = 0
        bound = bind_data(lower_rkppl(source, keys(data);
            conditioned = (:y,), mod = @__MODULE__), data)
        built = build_kernel(bound)
        u = zeros(built.layout.total)
        sampler = Base.invokelatest(prepare_query(built, bound, :sampler), u)
        likelihood = Base.invokelatest(prepare_query(built, bound, :likelihood), u)
        for fresh in (Int[], collect(eachindex(u)))
            terms = ReactiveKernelsPPL._smc_window_terms(built, bound, fresh, u)
            kernel = ReactiveKernelsPPL._smc_selected_kernel(terms, collect(eachindex(u)), u)
            @test calls[] == 1
            @test Base.invokelatest(kernel, u) ≈ (isempty(fresh) ? likelihood : sampler)
        end
    end
end

@traceable function checked_shift(x)
    x > 0 || throw(ArgumentError("positive support required"))
    (shift = x + 1,)
end
checked_location(a, s) = fill(a + s.shift, 2)

@testset "ignored errors do not populate the ordinary binding cache" begin
    source = quote
        s = checked_shift(x_fit)
        a ~ Normal(0, 1)
        mu = checked_location(a, s)
        y .~ Normal.(mu, 1)
    end
    data = (x_fit = -2.0, y = [0.0, 0.0])
    bound = bind_data(lower_rkppl(source, keys(data);
        conditioned = (:y,), mod = @__MODULE__), data)
    spec = @kernel begin
        x_fit::Float64
        a::Float64
        s = checked_shift(x_fit)
        result = a + s.shift
    end
    q = ReactiveKernelsPPL._prepare_cached(bound, spec; have = (:x_fit, :a),
        want = :result, bound = (; x_fit = -2.0), on_error = :ignore)
    @test Base.invokelatest(q, 0.0) == -1.0
    @test_throws "positive support required" ReactiveKernelsPPL._prepare_cached(bound, spec;
        have = (:x_fit, :a), want = :result, bound = (; x_fit = -2.0))
end
end
