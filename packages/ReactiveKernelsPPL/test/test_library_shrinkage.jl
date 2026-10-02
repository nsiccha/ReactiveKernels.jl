_ls_plan_canon(plan) = sprint(_canon, _test_scope_math(plan))

# Shipped shrinkage submodels (`src/library.jl`): `r2d2_coefs` and
# `horseshoe_coefs` state every prior in their bodies, lower to their
# hand-inlined twins, match Distributions.jl oracles and the built-ins
# they replace, and take an `hcat` matrix read as a value or a bound data
# matrix. Needs `_canon` (test_corpus.jl), `_query` and `_check_gradient`
# (test_generator.jl).
using Distributions: Normal, Beta, Cauchy, Exponential, Dirichlet, truncated,
    logpdf
using Statistics: var
using Test

_ls_cols() = Dict{Symbol,Any}(
    :x1 => [0.5, -1.0, 1.5, 0.0, 1.0, -0.5],
    :x2 => [1.0, 0.5, -0.5, 2.0, 0.0, 1.0],
    :y => [1.0, 2.0, 1.5, 2.5, 1.0, 2.0])

_ls_names(cols) = Set{Symbol}(keys(cols))

function _ls_build(ast, cols)
    bound = bind_data(lower_rkppl(ast, _ls_names(cols)), cols)
    return bound, build_kernel(bound)
end

_ls_u(built) = collect(range(-0.45, 0.55; length = built.layout.total))

# Re-pack constrained values `vals` (by name) into `layout`'s coordinates.
function _ls_unconstrain(layout, vals::NamedTuple)
    nt = constrain(layout, zeros(layout.total))
    return unconstrain(layout, merge(nt, vals))
end

const _LS_R2D2 = quote
    a ~ Normal(0, 1)
    X = hcat(x1, x2)
    b ~ r2d2_coefs(X, [1.0, 1.0])
    mu = a .+ X * b
    sigma ~ Exponential(1.0)
    y .~ Normal.(mu, sigma)
end

const _LS_HORSESHOE = quote
    a ~ Normal(0, 1)
    X = hcat(x1, x2)
    b ~ horseshoe_coefs(X)
    mu = a .+ X * b
    sigma ~ Exponential(1.0)
    y .~ Normal.(mu, sigma)
end

_ls_halfnormal(s, x) = logpdf(truncated(Normal(0, s), 0, Inf), x)
_ls_halfcauchy(s, x) = logpdf(truncated(Cauchy(0, s), 0, Inf), x)

@testset "library r2d2_coefs: the hand-inlined body" begin
    twin = quote
        a ~ Normal(0, 1)
        X = hcat(x1, x2)
        b_R2 ~ Beta(1, 1)
        b_phi ~ Dirichlet([1.0, 1.0])
        b_tau ~ HalfNormal(1)
        b_varx = var.(eachcol(X))
        b_b[axes(X, 2)] .~ Normal.(0,
            sqrt.(b_phi .* b_R2 .* b_tau^2 ./ b_varx))
        b = b_b
        mu = a .+ X * b
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, sigma)
    end
    # The body's module calls (`eachcol`) resolve in the module that
    # defines the submodel, so the twin lowers there.
    names = Set([:x1, :x2, :y])
    @test _ls_plan_canon(lower_rkppl(_LS_R2D2, names)) ==
        _ls_plan_canon(lower_rkppl(twin, names; mod = ReactiveKernelsPPL))
end

@testset "library r2d2_coefs: Distributions oracle and gradient" begin
    cols = _ls_cols()
    bound, built = _ls_build(_LS_R2D2, cols)
    # The hcat matrix is read as a value (`var.(eachcol(X))`): bound data.
    @test bound.columns[:X] == hcat(cols[:x1], cols[:x2])
    @test Set(coordinate_names(built.layout)) == Set([:a,
        Symbol("b.R2"), Symbol("b.tau"), :sigma, Symbol("b.phi.1"),
        Symbol("b.b.1"), Symbol("b.b.2")])
    u = _ls_u(built)
    nt = constrain(built.layout, u)
    X = hcat(cols[:x1], cols[:x2])
    a = nt.a
    sd = sqrt.(nt.b.phi .* nt.b.R2 .* nt.b.tau^2 ./ var.(eachcol(X)))
    pr = logpdf(Normal(0, 1), a) + logpdf(Beta(1, 1), nt.b.R2) +
        logpdf(Dirichlet([1.0, 1.0]), nt.b.phi) +
        _ls_halfnormal(1, nt.b.tau) + sum(logpdf.(Normal.(0, sd), nt.b.b)) +
        logpdf(Exponential(1.0), nt.sigma)
    ll = sum(logpdf.(Normal.(a .+ X * nt.b.b, nt.sigma), cols[:y]))
    @test _query(built.spec, bound, :prior, u) ≈ pr
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    _check_gradient(built.spec, bound, u)
end

# Acceptance S2: the R2D2 result bound to a name and summed into the
# predictor, and the library called from inside another submodel.
@rkppl _ls_r2d2_effect(X, alpha) = begin
    b ~ r2d2_coefs(X, alpha)
    return X * b
end

@testset "library r2d2_coefs: named and nested uses (acceptance S2)" begin
    cols = _ls_cols()
    names = _ls_names(cols)
    named = quote
        a ~ Normal(0, 5)
        X = hcat(x1, x2)
        b ~ r2d2_coefs(X, [1.0, 1.0])
        eta = X * b
        mu = a .+ eta
        sigma ~ Exponential(1)
        y .~ Normal.(mu, sigma)
    end
    nested = quote
        a ~ Normal(0, 5)
        X = hcat(x1, x2)
        eta ~ _ls_r2d2_effect(X, [1.0, 1.0])
        mu = a .+ eta
        sigma ~ Exponential(1)
        y .~ Normal.(mu, sigma)
    end
    # Both levels inlined by hand; the library body's `eachcol` resolves in
    # the module that defines it, so the twin lowers there.
    twin = quote
        a ~ Normal(0, 5)
        X = hcat(x1, x2)
        eta_b_R2 ~ Beta(1, 1)
        eta_b_phi ~ Dirichlet([1.0, 1.0])
        eta_b_tau ~ HalfNormal(1)
        eta_b_varx = var.(eachcol(X))
        eta_b_b[axes(X, 2)] .~ Normal.(0,
            sqrt.(eta_b_phi .* eta_b_R2 .* eta_b_tau^2 ./ eta_b_varx))
        eta_b = eta_b_b
        eta = X * eta_b
        mu = a .+ eta
        sigma ~ Exponential(1)
        y .~ Normal.(mu, sigma)
    end
    @test _ls_plan_canon(lower_rkppl(nested, names; mod = @__MODULE__)) ==
        _ls_plan_canon(lower_rkppl(twin, names; mod = ReactiveKernelsPPL))
    X = hcat(cols[:x1], cols[:x2])
    for (ast, path) in ((named, (:b,)), (nested, (:eta, :b)))
        bound = bind_data(lower_rkppl(ast, names; mod = @__MODULE__), cols)
        built = build_kernel(bound)
        u = _ls_u(built)
        nt = constrain(built.layout, u)
        local_draws = foldl(getproperty, path; init = nt)
        R2, phi = local_draws.R2, local_draws.phi
        tau, b = local_draws.tau, local_draws.b
        a = nt.a
        sd = sqrt.(phi .* R2 .* tau^2 ./ var.(eachcol(X)))
        pr = logpdf(Normal(0, 5), a) + logpdf(Beta(1, 1), R2) +
            logpdf(Dirichlet([1.0, 1.0]), phi) + _ls_halfnormal(1, tau) +
            sum(logpdf.(Normal.(0, sd), b)) + logpdf(Exponential(1), nt.sigma)
        @test _query(built.spec, bound, :prior, u) ≈ pr
        @test _query(built.spec, bound, :likelihood, u) ≈
            sum(logpdf.(Normal.(a .+ X * b, nt.sigma), cols[:y]))
        _check_gradient(built.spec, bound, u)
    end
end

@testset "library r2d2_coefs over a bound data matrix" begin
    cols = _ls_cols()
    X = hcat(cols[:x1], cols[:x2], cols[:x1] .* cols[:x2])
    data = Dict{Symbol,Any}(:X => X, :y => cols[:y])
    bound, built = _ls_build(quote
            a ~ Normal(0, 1)
            b ~ r2d2_coefs(X, [1.0, 2.0, 3.0])
            mu = a .+ X * b
            sigma ~ Exponential(1.0)
            y .~ Normal.(mu, sigma)
        end, data)
    u = _ls_u(built)
    nt = constrain(built.layout, u)
    @test length(nt.b.b) == 3 && length(nt.b.phi) == 3
    a = nt.a
    sd = sqrt.(nt.b.phi .* nt.b.R2 .* nt.b.tau^2 ./ var.(eachcol(X)))
    pr = logpdf(Normal(0, 1), a) + logpdf(Beta(1, 1), nt.b.R2) +
        logpdf(Dirichlet([1.0, 2.0, 3.0]), nt.b.phi) +
        _ls_halfnormal(1, nt.b.tau) + sum(logpdf.(Normal.(0, sd), nt.b.b)) +
        logpdf(Exponential(1.0), nt.sigma)
    ll = sum(logpdf.(Normal.(a .+ X * nt.b.b, nt.sigma), cols[:y]))
    @test _query(built.spec, bound, :prior, u) ≈ pr
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    _check_gradient(built.spec, bound, u)
end

@testset "library r2d2_coefs matches the r2d2 built-in" begin
    # Corpus 42's built-in: the intercept takes the share-0 Normal(0, 1)
    # and tau a half-Normal(0, 1), which the library states explicitly.
    cols = _ls_cols()
    bb, builtb = _ls_build(quote
            R2 ~ Beta(1.0, 1.0)
            phi ~ Dirichlet([1.0, 1.0])
            mu = a .+ b1 .* x1 .+ b2 .* x2
            r2d2(mu, R2, phi)
            sigma ~ Exponential(1.0)
            y .~ Normal.(mu, sigma)
        end, cols)
    bl, builtl = _ls_build(_LS_R2D2, cols)
    ub = _ls_u(builtb)
    v = constrain(builtb.layout, ub)
    ul = _ls_unconstrain(builtl.layout, (; a = v.mu[1], sigma = v.sigma,
        b = (; R2 = v.R2, tau = v.r2d2_mu_tau_bsv, phi = v.phi,
            b = v.mu[2:3])))
    for want in (:prior, :likelihood, :posterior)
        @test _query(builtl.spec, bl, want, ul) ≈
            _query(builtb.spec, bb, want, ub)
    end
end

@testset "library horseshoe_coefs: the hand-inlined body" begin
    twin = quote
        a ~ Normal(0, 1)
        X = hcat(x1, x2)
        b_tau ~ HalfCauchy(1)
        b_lambda[axes(X, 2)] .~ HalfCauchy.(1)
        b_z[axes(X, 2)] .~ Normal.(0, 1)
        b = b_z .* b_lambda .* b_tau
        mu = a .+ X * b
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, sigma)
    end
    names = Set([:x1, :x2, :y])
    @test _ls_plan_canon(lower_rkppl(_LS_HORSESHOE, names)) ==
        _ls_plan_canon(lower_rkppl(twin, names))
end

@testset "library horseshoe_coefs: Distributions oracle and gradient" begin
    cols = _ls_cols()
    bound, built = _ls_build(_LS_HORSESHOE, cols)
    @test bound.columns[:X] == hcat(cols[:x1], cols[:x2])
    u = _ls_u(built)
    nt = constrain(built.layout, u)
    X = hcat(cols[:x1], cols[:x2])
    a = nt.a
    b = nt.b.z .* nt.b.lambda .* nt.b.tau
    pr = logpdf(Normal(0, 1), a) + _ls_halfcauchy(1, nt.b.tau) +
        sum(_ls_halfcauchy.(1, nt.b.lambda)) +
        sum(logpdf.(Normal(0, 1), nt.b.z)) +
        logpdf(Exponential(1.0), nt.sigma)
    ll = sum(logpdf.(Normal.(a .+ X * b, nt.sigma), cols[:y]))
    @test _query(built.spec, bound, :prior, u) ≈ pr
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    _check_gradient(built.spec, bound, u)
end

@testset "library horseshoe_coefs matches Horseshoe() at one coefficient" begin
    # The built-in mints one (raw, lambda, tau) triple per coefficient with
    # unnormalized Stan-kernel Cauchy halves; the library's two halves are
    # normalized, so its density is higher by 2 log 2. With one column the
    # global and local scales coincide with the built-in's.
    cols = _ls_cols()
    bb, builtb = _ls_build(quote
            a ~ Normal(0, 1)
            b1 ~ Horseshoe()
            mu = a .+ b1 .* x1
            sigma ~ Exponential(1.0)
            y .~ Normal.(mu, sigma)
        end, cols)
    bl, builtl = _ls_build(quote
            a ~ Normal(0, 1)
            X = hcat(x1)
            b ~ horseshoe_coefs(X)
            mu = a .+ X * b
            sigma ~ Exponential(1.0)
            y .~ Normal.(mu, sigma)
        end, cols)
    ub = _ls_u(builtb)
    v = constrain(builtb.layout, ub)
    ul = _ls_unconstrain(builtl.layout, (; a = v.horseshoe_mu_Intercept_normal,
        b = (; tau = v.horseshoe_mu_x1_tau, lambda = [v.horseshoe_mu_x1_lambda],
            z = [v.horseshoe_mu_x1_raw]), sigma = v.sigma))
    @test _query(builtl.spec, bl, :likelihood, ul) ≈
        _query(builtb.spec, bb, :likelihood, ub)
    for want in (:prior, :posterior)
        @test _query(builtl.spec, bl, want, ul) ≈
            _query(builtb.spec, bb, want, ub) + 2 * log(2)
    end
end

@testset "hcat matrices read as values" begin
    cols = _ls_cols()
    names = _ls_names(cols)
    # Read only as `X * b` with a coefficient prior: the design-matrix route.
    term = lower_rkppl(quote
            X = hcat(1, x1, x2)
            b[axes(X, 2)] .~ Normal.(0, 2)
            mu = X * b
            sigma ~ Exponential(1.0)
            y .~ Normal.(mu, sigma)
        end, names)
    @test any(t -> t.kind === MatrixTerm, only(term.predictors).terms)
    @test isempty(ReactiveKernelsPPL._value_design_matrix_names(term))
    # Read as a matrix elsewhere: a value, built at bind (`1` = a ones
    # column, as in the design matrix), with the same density as the same
    # program over a bound data matrix.
    valued = quote
        X = hcat(1, x1, x2)
        v = var.(eachcol(X))
        b[axes(X, 2)] .~ Normal.(0, 1 .+ v)
        mu = X * b
        sigma ~ Exponential(1.0)
        y .~ Normal.(mu, sigma)
    end
    plan = lower_rkppl(valued, names)
    @test !any(t -> t.kind === MatrixTerm, only(plan.predictors).terms)
    @test ReactiveKernelsPPL._value_design_matrix_names(plan) == Set([:X])
    bound, built = _ls_build(valued, cols)
    Xv = hcat(ones(6), cols[:x1], cols[:x2])
    @test bound.columns[:X] == Xv
    u = _ls_u(built)
    nt = constrain(built.layout, u)
    pr = sum(logpdf.(Normal.(0, 1 .+ var.(eachcol(Xv))), nt.b)) +
        logpdf(Exponential(1.0), nt.sigma)
    @test _query(built.spec, bound, :prior, u) ≈ pr
    @test _query(built.spec, bound, :likelihood, u) ≈
        sum(logpdf.(Normal.(Xv * nt.b, nt.sigma), cols[:y]))
    _check_gradient(built.spec, bound, u)
    datab, datak = _ls_build(quote
            v = var.(eachcol(X))
            b[axes(X, 2)] .~ Normal.(0, 1 .+ v)
            mu = X * b
            sigma ~ Exponential(1.0)
            y .~ Normal.(mu, sigma)
        end, Dict{Symbol,Any}(:X => Xv, :y => cols[:y]))
    @test coordinate_names(datak.layout) == coordinate_names(built.layout)
    @test _query(datak.spec, datab, :posterior, u) ≈
        _query(built.spec, bound, :posterior, u)
    # A computed coefficient vector (`X * (z .* s)`) reads `X` as a value.
    ncp = lower_rkppl(quote
            X = hcat(x1, x2)
            s ~ HalfNormal(1)
            z[axes(X, 2)] .~ Normal.(0, 1)
            mu = X * (z .* s)
            sigma ~ Exponential(1.0)
            y .~ Normal.(mu, sigma)
        end, names)
    @test ReactiveKernelsPPL._value_design_matrix_names(ncp) == Set([:X])
    # A value matrix is plain data: a repeated column is a valid matrix.
    rep, _ = _ls_build(quote
            X = hcat(x1, x1)
            v = var.(eachcol(X))
            b[axes(X, 2)] .~ Normal.(0, v)
            mu = X * b
            sigma ~ Exponential(1.0)
            y .~ Normal.(mu, sigma)
        end, cols)
    @test rep.columns[:X] == hcat(cols[:x1], cols[:x1])
    # refused: the model computes X, so a caller column X is a second
    # source for one value (the module-data precedent).
    # refused: caller X collides with the computed X definition (single assignment)
    @test_throws ContractValidationError bind_data(plan,
        merge(cols, Dict{Symbol,Any}(:X => Xv)))
    # Capability gap: a value matrix over a column the model derives
    # (`lx = log.(x2 .+ 3)`) is not built at bind yet.
    derived_ok = try
        _ls_build(quote
                a ~ Normal(0, 1)
                lx = log.(x2 .+ 3)
                X = hcat(x1, lx)
                b ~ r2d2_coefs(X, [1.0, 1.0])
                mu = a .+ X * b
                sigma ~ Exponential(1.0)
                y .~ Normal.(mu, sigma)
            end, cols)
        true
    catch e
        e isa SurfaceLoweringError || rethrow()
        false
    end
    @test_broken derived_ok
end

@testset "library re-spells of corpus 42, 47 and 56 bind" begin
    cols = _ls_cols()
    X3 = hcat(cols[:x1], cols[:x2], cols[:x1] .* cols[:x2])
    for (name, data) in (
            ("42_r2d2_library", cols),
            ("47_matrix_r2d2_library",
                Dict{Symbol,Any}(:X => X3, :y => cols[:y])),
            ("56_horseshoe_library", cols))
        ast, names = _load_corpus_case(joinpath(_CORPUS_DIR, name * ".jl"))
        @test Set(names) == Set(keys(data))
        bound = bind_data(lower_rkppl(ast, names), data)
        built = build_kernel(bound)
        u = _ls_u(built)
        @test isfinite(_query(built.spec, bound, :posterior, u))
    end
end
