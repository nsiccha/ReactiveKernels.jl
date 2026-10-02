using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels,
    ReactiveKernelsPPL, Test

@rkppl _sc_inner(xx) = begin
    b ~ Normal(0, 1)
    return b .* xx
end
@rkppl _sc_outer(xx) = begin
    q ~ _sc_inner(xx)
    return q
end
@rkppl _sc_bc(xx) = begin
    b_c ~ Normal(0, 1)
    return b_c .* xx
end
@rkppl _sc_c(xx) = begin
    c ~ Normal(0, 1)
    return c .* xx
end
@rkppl _sc_record(xx) = begin
    b ~ Normal(0, 1)
    d = xx .+ 1
    return (b = 2 * b, extra = b + 1)
end
@rkppl _sc_cell(m) = begin
    b ~ Normal(0, 1)
    return m + b
end
@rkppl _sc_cell_outer(m) = begin
    w ~ _sc_cell(m)
    return w
end
@rkppl _sc_center(m) = begin
    b ~ Normal(m, 1)
    return b
end
@rkppl _sc_vector(X) = begin
    b[axes(X, 2)] .~ Normal.(0, 1)
    return X * b
end
@rkppl _sc_vector_record(X) = begin
    b[axes(X, 2)] .~ Normal.(0, 1)
    return (b = b, extra = sum(b))
end
_sc_scalar_only(v::Number) = v + 0.3
@rkppl _sc_free(xx) = begin
    b ~ Normal(0, 1)
    return b .* xx .+ z_b
end
@rkppl _sc_future_free(xx) = begin
    c ~ Normal(0, 1)
    return c .* xx .+ var"##rkppl_scope#00000001"
end
@rkppl _sc_nested_future_free(xx) = begin
    b ~ Normal(0, 1)
    w ~ _sc_future_free(xx)
    return w
end
@rkppl _sc_bad_argument(xx) = begin
    xx = 1
    return xx
end
@rkppl _sc_scan_drawnames(beta, sigma) = begin
    _ppl_scan_z_level ~ Normal(0, 1)
    _ppl_scan_z_level_1 ~ Normal(0, 1)
    @scan begin
        level[1] = 0.0
        increment[1] = 0.0
        for t in 2:T
            z ~ Normal(0, 1)
            increment[t] = beta * increment[t - 1] + sigma * z
            level[t] = level[t - 1] + increment[t]
        end
    end
    return level
end

const _SC_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
const _SC_COLLISION = quote
    z_b ~ Normal(0, 1)
    z ~ _sc_inner(x)
    mu = z .+ z_b .+ z.b
    y .~ Normal.(mu, 1)
end

const _SC_NESTED = quote
    a ~ _sc_bc(x)
    a_b ~ _sc_c(x)
    z ~ _sc_outer(x)
    mu = a .+ a_b .+ z .+ z.q.b
    y .~ Normal.(mu, 1)
end
const _SC_RECORD = quote
    z ~ _sc_record(x)
    q ~ Normal(_sc_scalar_only(getproperty(z, :b)), 1)
    sigma = exp(z.extra)
    y .~ Normal.(z.b .* x .+ z.d, sigma)
end
const _SC_CELL = quote
    @plate for i in eachindex(y)
        theta[i] ~ _sc_cell_outer(x[i])
        y[i] ~ Normal(theta[i], 1)
    end
    @plate for i in eachindex(y2)
        y2[i] ~ Normal(theta[i].w.b, 1)
    end
end
const _SC_ARRAY_RECORD = quote
    z ~ _sc_vector_record(X)
    q ~ Normal(_sc_scalar_only(getproperty(z, :extra)), 1)
    y .~ Normal.(X * z.b, 1)
end

function _sc_build(ast, data)
    plan = lower_rkppl(ast, data; mod = @__MODULE__, conditioned = data)
    bound = bind_data(plan, Dict{Symbol,Any}(pairs(data)))
    built = build_kernel(bound)
    u = [0.27 * sin(i) - 0.15 for i in 1:built.layout.total]
    return bound, built, u
end

_sc_prior(v) = sum(logpdf.(Normal(), v))
_sc_ll(mu, y; sigma = 1) = sum(logpdf.(Normal.(mu, sigma), y))
function _sc_oracle(kind, nt, data)
    x, y = data.x, data.y
    if kind === :collision
        return _sc_prior([nt.z_b, nt.z.b]) +
            _sc_ll(nt.z.b .* x .+ nt.z_b .+ nt.z.b, y)
    elseif kind === :nested
        b = nt.z.q.b
        return _sc_prior([nt.a.b_c, nt.a_b.c, b]) +
            _sc_ll((nt.a.b_c + nt.a_b.c + b) .* x .+ b, y)
    elseif kind === :record
        b = nt.z.b
        return logpdf(Normal(), b) + logpdf(Normal(2b + 0.3, 1), nt.q) +
            _sc_ll(b .* x .+ x .+ 1, y; sigma = exp(b + 1))
    elseif kind === :cell
        b = nt.theta.w.b
        return _sc_prior(b) + _sc_ll(b, data.y2) +
            _sc_ll(x .+ b, y)
    elseif kind === :array_record
        b = nt.z.b
        return _sc_prior(b) + logpdf(Normal(sum(b) + 0.3, 1), nt.q) +
            _sc_ll(data.X * b, y)
    end
    error("unknown scoped oracle $kind")
end

function _sc_fd(f, u; h = 1e-6)
    map(eachindex(u)) do i
        up, dn = copy(u), copy(u)
        up[i] += h; dn[i] -= h
        (f(up) - f(dn)) / (2h)
    end
end

@testset "scoped submodels: independent values, gradients and draws" begin
    data = (; x = [-0.5, 0.2, 1.0], y = [0.1, -0.2, 0.3],
        y2 = [0.3, -0.1, 0.2], X = [1.0 0.2; -0.3 1.1; 0.4 -0.8])
    for (kind, ast) in ((:collision, _SC_COLLISION), (:nested, _SC_NESTED),
            (:record, _SC_RECORD), (:cell, _SC_CELL),
            (:array_record, _SC_ARRAY_RECORD))
        @testset "$kind" begin
            bound, built, u = _sc_build(ast, data)
            nt = constrain(built.layout, u)
            @test unconstrain(built.layout, nt) ≈ u
            @test all(!occursin("##", string(n)) for n in coordinate_names(built.layout))
            reference(v) = _sc_oracle(kind, constrain(built.layout, v), data)
            q = prepare_sampler(built, bound, u; backend = _SC_BACKEND)
            g = similar(u)
            value, _ = sampler_value_and_gradient!(q, g, u)
            @test value ≈ reference(u) rtol = 1e-12
            @test g ≈ _sc_fd(reference, u) rtol = 1e-5 atol = 1e-7
            draws = restore_draws(built.layout, hcat(u, u))
            empty_draws = restore_draws(built.layout, zeros(length(u), 0))
            path = kind === :nested ? (:z, :q, :b) :
                kind === :cell ? (:theta, :w, :b) : (:z, :b)
            getpath(v) = foldl(getproperty, path; init = v)
            isvector = kind in (:cell, :array_record)
            @test getpath(draws) == (isvector ?
                hcat(getpath(nt), getpath(nt)) : fill(getpath(nt), 2))
            @test size(getpath(empty_draws)) == (isvector ?
                (length(getpath(nt)), 0) : (0,))
        end
    end
end

@testset "scoped submodels: arrays and direct cell returns" begin
    X = [1.0 0.2; -0.3 1.1; 0.4 -0.8]
    y = [0.1, -0.2, 0.3]
    bound, built, u = _sc_build(quote
        z ~ _sc_vector(X)
        y .~ Normal.(z, 1)
    end, (; X, y))
    nt = constrain(built.layout, u)
    @test coordinate_names(built.layout) == [Symbol("z.b.1"), Symbol("z.b.2")]
    @test Base.invokelatest(prepare_query(built, bound, :sampler), u) ≈
        _sc_prior(nt.z.b) + _sc_ll(X * nt.z.b, y)
    @test unconstrain(built.layout, nt) ≈ u
    @test size(restore_draws(built.layout, zeros(length(u), 0)).z.b) == (2, 0)
    bound, built, u = _sc_build(quote
        @plate for i in eachindex(y)
            theta[i] ~ _sc_center(x[i])
            y[i] ~ Normal(theta.b[i], 1)
        end
    end, (; x = [0.1, 0.2, 0.3], y))
    nt = constrain(built.layout, u)
    @test Base.invokelatest(prepare_query(built, bound, :sampler), u) ≈
        sum(logpdf.(Normal.([0.1, 0.2, 0.3], 1), nt.theta.b)) +
        _sc_ll(nt.theta.b, y)
    @test unconstrain(built.layout, nt) ≈ u
end

@testset "scoped submodels: scope records and genuine redefinitions" begin
    plan = lower_rkppl(_SC_NESTED, (:x, :y); mod = @__MODULE__, conditioned = (:x, :y))
    @test Set(scope.path for scope in plan.submodel_scopes) ==
        Set([(:a,), (:a_b,), (:z,), (:z, :q)])
    @test haskey(only(filter(s -> s.path == (:a,), plan.submodel_scopes)).locals, :b_c)
    @test haskey(only(filter(s -> s.path == (:a_b,), plan.submodel_scopes)).locals, :c)
    names = [s.locals[n] for s in plan.submodel_scopes for n in keys(s.locals)]
    @test allunique(names)
    # Neither a dot in a literal author name nor a private-looking spelling
    # can capture a local or collide with its displayed coordinate path.
    bound, built, u = _sc_build(quote
        var"z.b" ~ Normal(0, 1)
        var"##rkppl_scope#00000001" ~ Normal(0, 1)
        z ~ _sc_inner(x)
        y .~ Normal.(z, 1)
    end, (; x = [-0.5, 0.2, 1.0], y = [0.1, -0.2, 0.3]))
    @test Set(coordinate_names(built.layout)) == Set([
        Symbol("var\"z.b\""), Symbol("var\"##rkppl_scope#00000001\""), Symbol("z.b")])
    nt = constrain(built.layout, u)
    @test nt[Symbol("z.b")] isa Float64 && nt.z.b isa Float64
    @test unconstrain(built.layout, nt) ≈ u
    # refused: scoped declarations are read-only (user choice `1f0p0fx`).
    @test_throws "read-only" lower_rkppl(quote
        z ~ _sc_inner(x)
        z.b = 1
        y .~ Normal.(z, 1)
    end, (:x, :y); mod = @__MODULE__, conditioned = (:x, :y))
    # refused: duplicate call bindings are actual redefinitions (`1f0p0fx`).
    @test_throws "defined more than once" lower_rkppl(quote
        z ~ _sc_inner(x)
        z ~ _sc_inner(x)
        y .~ Normal.(z, 1)
    end, (:x, :y); mod = @__MODULE__, conditioned = (:x, :y))
    # refused: a local cannot redefine an argument (lexical binding contract).
    @test_throws "both an argument and a local" lower_rkppl(quote
        z ~ _sc_bad_argument(x)
        y .~ Normal.(z, 1)
    end, (:x, :y); mod = @__MODULE__, conditioned = (:x, :y))
    # refused: undeclared free names remain errors (`1f0p0fx`, strict `16yyy0t`).
    @test_throws "z_b" lower_rkppl(quote
        z ~ _sc_free(x)
        y .~ Normal.(z, 1)
    end, (:x, :y); mod = @__MODULE__, conditioned = (:x, :y))
    @test lower_rkppl(quote
        z_b ~ Normal(0, 1)
        z ~ _sc_free(x)
        y .~ Normal.(z, 1)
    end, (:x, :y); mod = @__MODULE__, conditioned = (:x, :y)) isa StructuralPlan
end

@testset "scoped submodels: future free names cannot capture prior locals" begin
    # refused: later callees cannot capture a generated local with an
    # undeclared free name (`1f0p0fx`, strict `16yyy0t`).
    @test_throws "##rkppl_scope#00000001" lower_rkppl(quote
        z ~ _sc_inner(x)
        w ~ _sc_future_free(x)
        y .~ Normal.(z .+ w, 1)
    end, (:x, :y); mod = @__MODULE__, conditioned = (:x, :y))
    # refused: the same rule holds for a nested callee (`1f0p0fx`).
    @test_throws "##rkppl_scope#00000001" lower_rkppl(quote
        z ~ _sc_nested_future_free(x)
        y .~ Normal.(z, 1)
    end, (:x, :y); mod = @__MODULE__, conditioned = (:x, :y))
    data = (; x = [-0.5, 0.2, 1.0], y = [0.1, -0.2, 0.3])
    bound, built, u = _sc_build(quote
        var"##rkppl_scope#00000001" ~ Normal(0, 1)
        z ~ _sc_inner(x)
        w ~ _sc_future_free(x)
        y .~ Normal.(z .+ w, 1)
    end, data)
    @test unconstrain(built.layout, constrain(built.layout, u)) ≈ u
    @test allunique(coordinate_names(built.layout))
    function reference(v)
        nt = constrain(built.layout, v)
        caller = nt[Symbol("##rkppl_scope#00000001")]
        return _sc_prior([caller, nt.z.b, nt.w.c]) +
            _sc_ll((nt.z.b + nt.w.c) .* data.x .+ caller, data.y)
    end
    sampler = prepare_sampler(built, bound, u; backend = _SC_BACKEND)
    g = similar(u)
    value, _ = sampler_value_and_gradient!(sampler, g, u)
    @test value ≈ reference(u) rtol = 1e-12
    @test g ≈ _sc_fd(reference, u) rtol = 1e-5 atol = 1e-7
end

@testset "scoped submodels: generated draw aliases preserve author names" begin
    data = (; y = [0.1, -0.2, 0.3])
    bound, built, u = _sc_build(quote
        x ~ _sc_scan_drawnames(0.4, 0.2)
        y .~ Normal.(x, 1)
    end, data)
    nt = constrain(built.layout, u)
    @test allunique(coordinate_names(built.layout))
    @test nt.x._ppl_scan_z_level isa Float64
    @test nt.x._ppl_scan_z_level_1 isa Float64
    @test length(nt.x._ppl_scan_z_level_2) == length(data.y) - 1
    @test unconstrain(built.layout, nt) ≈ u
    @test size(restore_draws(built.layout, zeros(length(u), 0)).x._ppl_scan_z_level_2) ==
        (length(data.y) - 1, 0)
    function reference(v)
        values = constrain(built.layout, v).x
        innovations = values._ppl_scan_z_level_2
        path = zeros(length(data.y))
        increment = 0.0
        for t in 2:length(path)
            increment = 0.4 * increment + 0.2 * innovations[t - 1]
            path[t] = path[t - 1] + increment
        end
        return logpdf(Normal(), values._ppl_scan_z_level) +
            logpdf(Normal(), values._ppl_scan_z_level_1) +
            _sc_prior(innovations) + _sc_ll(path, data.y)
    end
    sampler = prepare_sampler(built, bound, u; backend = _SC_BACKEND)
    g = similar(u)
    value, _ = sampler_value_and_gradient!(sampler, g, u)
    @test value ≈ reference(u) rtol = 1e-12
    @test g ≈ _sc_fd(reference, u) rtol = 1e-5 atol = 1e-7
end
