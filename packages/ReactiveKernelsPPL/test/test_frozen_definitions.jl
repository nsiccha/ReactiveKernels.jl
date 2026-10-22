using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Serialization, Statistics, Test

# Data-only definitions frozen by value (snag rkppl-bind-froze-da223711; user
# no-preference pick on design decision `07xnou7`: a `frozen` keyword on
# `bind_data` plus `frozen_values`, and the training binding's program).
# A model fits its preprocessing in the graph from a fit-role copy of a
# covariate; a design takes the fitted values from the fit's binding and
# binds without the training data, exactly as if it had evaluated them.

const _FZ_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
_fz_fit_mean(v) = mean(v)
_fz_fit_sd(v) = std(v)
_fz_interval(v) = [minimum(v), maximum(v)]
_fz_bounds(v) = (lo = minimum(v), hi = maximum(v))
_fz_shifted(x, b, iv) = b .* (x .- iv[1]) ./ (iv[2] - iv[1])
_fz_scaled(x, b, r) = b .* (x .- r.lo) ./ (r.hi - r.lo)
_fz_eval(q, u) = Base.invokelatest(q, u)
_fz_refused(f, pattern) = try
    f()
    false
catch e
    e isa ContractValidationError && occursin(pattern, sprint(showerror, e))
end

const _FZ_X = collect(1.0:12.0)
const _FZ_Y = 0.3 .* _FZ_X .+ 1 .+ 0.1 .* sin.(1:12)
const _FZ_XNEW = [2.5, 7.0, 13.0]
const _FZ_YNEW = [1.6, 3.0, 5.2]

# Bind-time fits (`bind_data` evaluates them and binds them as data).
const _FZ_STANDARDIZED = quote
    x_mean = _fz_fit_mean(x_fit)
    x_sd = _fz_fit_sd(x_fit)
    z = (x .- x_mean) ./ x_sd
    a ~ Normal(0, 5)
    b ~ Normal(0, 2)
    sigma ~ Exponential(1.0)
    mu = a .+ b .* z
    y .~ Normal.(mu, sigma)
end
# A whole value a parameter-dependent call reads (RK preparation folds it).
const _FZ_INTERVAL = quote
    iv = _fz_interval(x_fit)
    a ~ Normal(0, 5)
    b ~ Normal(0, 2)
    sigma ~ Exponential(1.0)
    mu = a .+ _fz_shifted(x, b, iv)
    y .~ Normal.(mu, sigma)
end
# The same whole value inside an affine location (lowering keeps it named
# beside the fused location) and passed to an effect submodel.
const _FZ_AFFINE = quote
    iv = _fz_interval(x_fit)
    a ~ Normal(0, 5)
    b ~ Normal(0, 2)
    sigma ~ Exponential(1.0)
    mu = a .+ b .* x .+ _fz_shifted(x, b, iv)
    y .~ Normal.(mu, sigma)
end
@rkppl _fz_effect(x, iv) = begin
    b ~ Normal(0, 2)
    f = _fz_shifted(x, b, iv)
    f
end
const _FZ_SUBMODEL = quote
    iv = _fz_interval(x_fit)
    a ~ Normal(0, 5)
    sigma ~ Exponential(1.0)
    e ~ _fz_effect(x, iv)
    mu = a .+ e
    y .~ Normal.(mu, sigma)
end
# Both, a non-vector whole value, and a definition reading a frozen one.
const _FZ_MIXED = quote
    x_mean = _fz_fit_mean(x_fit)
    x_sd2 = 2 * _fz_fit_sd(x_fit)
    r = _fz_bounds(x_fit)
    z = (x .- x_mean) ./ x_sd2
    a ~ Normal(0, 5)
    b ~ Normal(0, 2)
    c ~ Normal(0, 2)
    sigma ~ Exponential(1.0)
    mu = a .+ b .* z .+ _fz_scaled(x, c, r)
    y .~ Normal.(mu, sigma)
end

_fz_fit(src) = bind_data(lower_rkppl(src, (; x = _FZ_X, x_fit = _FZ_X, y = _FZ_Y);
    conditioned = (:y,), mod = @__MODULE__), (; x = _FZ_X, x_fit = _FZ_X, y = _FZ_Y))

# The frozen design against the design that evaluates from the training
# column: same program, and every value, density and gradient equal.
function _fz_check_equivalent(fit, state, points)
    # Each structural-cache hit of these binds is recomputed and compared.
    ReactiveKernelsPPL._STRUCTURAL_CACHE_CHECK[] = true
    reference, design = try
        (bind_data(fit, (; x = _FZ_XNEW, x_fit = _FZ_X, y = _FZ_YNEW)),
            bind_data(fit, (; x = _FZ_XNEW, y = _FZ_YNEW); frozen = state))
    finally
        ReactiveKernelsPPL._STRUCTURAL_CACHE_CHECK[] = false
    end
    @test design.frozen == state
    @test !ReactiveKernelsPPL._supplied(design.columns[:x_fit])
    @test sort!(collect(keys(design.columns))) == sort!(collect(keys(reference.columns)))
    for (k, v) in reference.columns
        k === :x_fit || @test isequal(design.columns[k], v)
    end
    built = build_kernel(reference)
    @test build_kernel(design).program == built.program
    sampler = prepare_sampler(built, design, points[1]; backend = _FZ_BACKEND)
    sampler_ref = prepare_sampler(built, reference, points[1]; backend = _FZ_BACKEND)
    for u in points
        for preset in (:observations, :sampler, :pointwise, :prior)
            @test isequal(_fz_eval(prepare_query(built, design, preset), u),
                _fz_eval(prepare_query(built, reference, preset), u))
        end
        g, g_ref = similar(u), similar(u)
        v, _ = sampler_value_and_gradient!(sampler, g, u)
        v_ref, _ = sampler_value_and_gradient!(sampler_ref, g_ref, u)
        @test isequal(v, v_ref)
        @test isequal(g, g_ref)
    end
    return design, reference
end

@testset "frozen definitions: bind-time fits" begin
    fit = _fz_fit(_FZ_STANDARDIZED)
    state = frozen_values(fit, (:x_sd, :x_mean))
    @test state.values == (x_mean = mean(_FZ_X), x_sd = std(_FZ_X))
    # The record types the training input; it holds none of its values.
    @test state.inputs == (x_fit = (type = Vector{Float64}, size = (12,)),)
    @test frozen_values(fit, :x_mean).values == (x_mean = mean(_FZ_X),)
    layout = build_kernel(fit).layout
    points = [unconstrain(layout, (; a = 0.3, b = -0.7, sigma = 0.8)),
        unconstrain(layout, (; a = -1.1, b = 0.4, sigma = 2.5))]
    design, _ = _fz_check_equivalent(fit, state, points)

    # A rebinding keeps the frozen values; the fitted anchors stay the
    # training ones, whatever the design's rows.
    other = bind_data(design, (; x = [0.5, 1.5]))
    @test other.frozen == state
    @test other.columns[:x_mean] == mean(_FZ_X)
    a, b = -0.3, 0.6
    u = unconstrain(layout, (; a, b, sigma = 1.0))
    obs = _fz_eval(prepare_query(build_kernel(fit), other, :observations), u)
    @test obs.y.location ≈ a .+ b .* ([0.5, 1.5] .- mean(_FZ_X)) ./ std(_FZ_X) rtol = 1e-14
    # Its record is the same: a frozen definition keeps its frozen value.
    @test frozen_values(other, (:x_mean, :x_sd)) == state

    # `frozen = nothing` evaluates again, so the training input is required.
    # refused: an unfrozen definition reads its input (bind_data contract)
    @test _fz_refused(() -> bind_data(design, (; x = [0.5]); frozen = nothing), "x_fit")
    @test bind_data(design, (; x = [0.5], x_fit = [1.0, 3.0]); frozen = nothing).columns[:x_mean] == 2.0

    # Supplying the input anyway binds it; the frozen values still apply.
    both = bind_data(fit, (; x = _FZ_XNEW, x_fit = [0.0, 1.0]); frozen = state)
    @test both.columns[:x_fit] == [0.0, 1.0]
    @test both.columns[:x_mean] == mean(_FZ_X)

    # The record survives Serialization (a fitted-model bundle).
    io = IOBuffer()
    serialize(io, state)
    seekstart(io)
    @test deserialize(io) == state
end

@testset "frozen definitions: preparation-folded whole values" begin
    fit = _fz_fit(_FZ_INTERVAL)
    @test !haskey(fit.columns, :iv)                  # preparation folds it
    state = frozen_values(fit, :iv)
    @test state.values == (iv = [1.0, 12.0],)
    @test state.inputs == (x_fit = (type = Vector{Float64}, size = (12,)),)
    layout = build_kernel(fit).layout
    points = [unconstrain(layout, (; a = 0.3, b = -0.7, sigma = 0.8)),
        unconstrain(layout, (; a = -1.1, b = 0.4, sigma = 2.5))]
    design, _ = _fz_check_equivalent(fit, state, points)
    @test !haskey(design.columns, :iv)
    @test ReactiveKernelsPPL._query_have(design)[end] === :iv
    for src in (_FZ_AFFINE, _FZ_SUBMODEL)
        fit = _fz_fit(src)
        @test any(d -> d.name === :iv, fit.assignments)   # one named value
        state = frozen_values(fit, :iv)
        @test state.values == (iv = [1.0, 12.0],)
        n = build_kernel(fit).layout.total
        _fz_check_equivalent(fit, state, [0.2 .* (1:n) .- 0.3, 0.5 .- 0.3 .* (1:n)])
    end
end

@testset "frozen definitions: both classes, any value, dependent definitions" begin
    fit = _fz_fit(_FZ_MIXED)
    state = frozen_values(fit, (:x_mean, :x_sd2, :r))
    @test state.values == (r = (lo = 1.0, hi = 12.0), x_mean = mean(_FZ_X),
        x_sd2 = 2 * std(_FZ_X))
    layout = build_kernel(fit).layout
    points = [unconstrain(layout, (; a = 0.3, b = -0.7, c = 0.2, sigma = 0.8)),
        unconstrain(layout, (; a = -1.1, b = 0.4, c = -0.9, sigma = 2.5))]
    _fz_check_equivalent(fit, state, points)
end

@testset "frozen definitions: a design lowered without the training data" begin
    # A data-free bundle lowers its program fresh from a design and binds it
    # with the exported record: the fit's program, on the fit's kernel.
    fit = _fz_fit(_FZ_MIXED)
    state = frozen_values(fit, (:x_mean, :x_sd2, :r))
    plan = lower_rkppl(_FZ_MIXED, (; x = _FZ_XNEW); conditioned = (:y,),
        mod = @__MODULE__, frozen = state)
    @test plan.frozen == state
    design = bind_data(plan, (; x = _FZ_XNEW, y = _FZ_YNEW))   # binds `plan.frozen`
    reference = bind_data(fit, (; x = _FZ_XNEW, x_fit = _FZ_X, y = _FZ_YNEW))
    built = build_kernel(fit)
    u = unconstrain(built.layout, (; a = 0.3, b = -0.7, c = 0.2, sigma = 0.8))
    for preset in (:observations, :sampler)
        @test isequal(_fz_eval(prepare_query(built, design, preset), u),
            _fz_eval(prepare_query(built, reference, preset), u))
    end
    # Lowered from names, the record names the frozen inputs too.
    named = lower_rkppl(_FZ_MIXED, (:x, :y); conditioned = (:y,),
        mod = @__MODULE__, frozen = state)
    design = bind_data(named, (; x = _FZ_XNEW, y = _FZ_YNEW))
    @test isequal(_fz_eval(prepare_query(built, design, :sampler), u),
        _fz_eval(prepare_query(built, reference, :sampler), u))
    # A lowering without the record cannot tell that the omitted name is
    # data, so it plans another (equally valid) program, with its own build.
    other = bind_data(lower_rkppl(_FZ_MIXED, (; x = _FZ_XNEW); conditioned = (:y,),
        mod = @__MODULE__), (; x = _FZ_XNEW, y = _FZ_YNEW); frozen = state)
    @test _fz_refused(() -> prepare_query(built, other, :sampler), "different program")
    own = build_kernel(other)
    @test _fz_eval(prepare_query(own, other, :sampler), u) ≈
        _fz_eval(prepare_query(built, reference, :sampler), u) rtol = 1e-12
end

@testset "frozen definitions: an input another value reads stays required" begin
    # `x` is read by the frozen fit and by the per-row transform.
    src = quote
        x_mean = _fz_fit_mean(x)
        z = x .- x_mean
        a ~ Normal(0, 5)
        sigma ~ Exponential(1.0)
        y .~ Normal.(a .+ z, sigma)
    end
    fit = bind_data(lower_rkppl(src, (; x = _FZ_X, y = _FZ_Y); conditioned = (:y,),
        mod = @__MODULE__), (; x = _FZ_X, y = _FZ_Y))
    state = frozen_values(fit, :x_mean)
    @test state.inputs == (x = (type = Vector{Float64}, size = (12,)),)
    design = bind_data(fit, (; x = _FZ_XNEW); frozen = state)
    @test design.columns[:x] == _FZ_XNEW
    @test design.columns[:x_mean] == mean(_FZ_X)   # not the design's mean
    # refused: data the program reads is required (bind_data contract)
    @test_throws ContractValidationError bind_data(fit, (; y = _FZ_YNEW); frozen = state)
end

@testset "frozen definitions: refusals" begin
    fit = _fz_fit(_FZ_STANDARDIZED)
    state = frozen_values(fit, (:x_mean, :x_sd))
    # refused: a computed definition supplied as plain data could be a stale
    # override; freezing is the explicit route (snag rkppl-bind-froze-da223711)
    @test _fz_refused(() -> bind_data(fit, (; x = _FZ_XNEW, x_mean = 6.5, x_sd = 3.6)),
        "collide with raw columns")
    # refused: frozen and supplied at once leaves two values for one name
    @test _fz_refused(() -> bind_data(fit, (; x = _FZ_XNEW, x_mean = 6.5); frozen = state),
        "frozen and also supplied as data")
    # refused: a definition reading a parameter has no value apart from it
    interval = _fz_fit(_FZ_INTERVAL)
    @test any(d -> d.name === :mu, interval.assignments)
    @test _fz_refused(() -> frozen_values(interval, :mu), "reads a parameter")
    @test _fz_refused(() -> bind_data(interval, (; x = _FZ_XNEW);
        frozen = FrozenValues((; mu = [1.0]), (;))), "reads a parameter")
    # refused: nothing to freeze
    @test _fz_refused(() -> frozen_values(fit, :nope), "defines no `nope`")
    @test _fz_refused(() -> bind_data(fit, (; x = _FZ_XNEW);
        frozen = FrozenValues((; nope = 1.0), (;))), "defines no `nope`")
    unbound = lower_rkppl(_FZ_STANDARDIZED, (:x, :x_fit, :y); conditioned = (:y,),
        mod = @__MODULE__)
    @test _fz_refused(() -> frozen_values(unbound, :x_mean), "bound plan")
end
