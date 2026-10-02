using Distributions
using LinearAlgebra
using ReactiveKernels
using ReactiveKernelsPPL
using SpecialFunctions: besseli
using Test

# Smooths and HSGPs as data-side bases plus library submodels (todo
# 0xoiq57; user decisions 1cmodra prongs smooths / hyper-lp, 10ldrvz prong
# smooth_hypers, 0z5bsqi). The bases are plain functions over data
# (`tps_basis`, `t2_basis`, `hsgp_basis`, `hsgp_periodic_basis`), run once by
# `bind_data`; the library submodels (`penalized_smooth`, `t2_smooth`,
# `hsgp_effect`, `hsgp_periodic_effect`, `hsgp_grouped_effect`) state every
# prior in their bodies. Densities are checked against Distributions.jl and
# against the built-in `spline_basis` / `hsgp_basis` constructs at matched
# parameter values, gradients against central differences. Helpers from
# earlier includes: `_query`, `_check_gradient` (test_generator.jl), `_canon`,
# `_load_corpus_case`, `_CORPUS_DIR` (test_corpus.jl). Synthetic data only.

const _LS = @__MODULE__

_ls_n() = 40
_ls_x() = [2.0 * sin(1.7 * i) + 0.1 * i / _ls_n() for i in 1:_ls_n()]
_ls_z() = [1.5 * cos(0.9 * i + 0.3) for i in 1:_ls_n()]
_ls_x2() = [mod(0.37 * i, 3.0) - 1.4 for i in 1:_ls_n()]
_ls_y() = [sin(xi) + 0.2 * cos(3.1 * i) for (i, xi) in enumerate(_ls_x())]
_ls_g() = ["a", "b", "c"][[mod1(i * 7, 3) for i in 1:_ls_n()]]

function _ls_data()
    return Dict{Symbol,Any}(:y => _ls_y(), :x => _ls_x(), :z => _ls_z(),
        :x2 => _ls_x2(), :g => _ls_g())
end

_ls_pt(n) = [0.37 * sin(1.3 * i) - 0.2 for i in 1:n]

function _ls_bind(ast, names; mod = _LS)
    plan = lower_rkppl(ast, names; mod = mod)
    data = _ls_data()
    return bind_data(plan, Dict{Symbol,ColumnData}(k => data[k] for k in names))
end

_ls_node(built, bound, node, u) = _query(built.spec, bound, node, u)

# ── data-side bases ──────────────────────────────────────────────────

@testset "smooth bases: columns equal the built-in's bound columns" begin
    x, z = _ls_x(), _ls_z()
    b = _ls_bind(quote
        spline_basis(:s, x; k = 5)
        a ~ Normal(0, 1)
        mu = a .+ spline(:s)
        y .~ Normal.(mu, 1.0)
    end, (:y, :x))
    X, Z = tps_basis(x; k = 5)
    @test size(X) == (_ls_n(), 1) && size(Z) == (_ls_n(), 3)
    @test X[:, 1] == b.columns[:s_Xnull_1]
    @test all(j -> Z[:, j] == b.columns[Symbol("s_Zpen_", j)], 1:3)
    b2 = _ls_bind(quote
        spline_basis(:t, x, z; k = (4, 5))
        a ~ Normal(0, 1)
        mu = a .+ spline(:t)
        y .~ Normal.(mu, 1.0)
    end, (:y, :x, :z))
    blocks = t2_basis(x, z; k = (4, 5))
    for (M, tag) in zip(blocks, (:Xfixed, :Zrr, :Zrn, :Znr))
        @test all(j -> M[:, j] == b2.columns[Symbol("t_", tag, "_", j)],
            axes(M, 2))
    end
    @test size.(blocks) == ((40, 3), (40, 6), (40, 4), (40, 6))
    # HSGP floors from the eigenvalues equal the built-in's fit floors.
    x2 = _ls_x2()
    PHI, lambda = hsgp_basis(x, x2; k = (3, 4), c = (1.5, 2.0))
    @test size(PHI) == (40, 12) && size(lambda) == (12, 2)
    fits = [ReactiveKernelsPPL._hsgp_axis_fit(x, 1.5, :t, "x"),
        ReactiveKernelsPPL._hsgp_axis_fit(x2, 2.0, :t, "x2")]
    @test hsgp_rho_floors(lambda) ≈
        ReactiveKernelsPPL._hsgp_floors([3, 4], fits, false)
    @test hsgp_periodic_rho_floor([1, 2, 3, 1, 2, 3]) ==
        ReactiveKernelsPPL._hsgp_periodic_rho_lower(3)
    # The grouped basis splits each column across the groups.
    G, _ = hsgp_basis(x; k = 4, by = _ls_g())
    P1, _ = hsgp_basis(x; k = 4)
    @test size(G) == (40, 12)
    @test G * ones(12) ≈ P1 * ones(4)
end

# ── in-model data values ─────────────────────────────────────────────

_ls_mat(x) = hcat(x, x .^ 2)
_ls_rows(x) = hcat(vcat(x, 0.0), vcat(x, 0.0))
_ls_sq(x) = x .^ 2
_ls_mean(x) = sum(x) / length(x)
const _LS_CALLS = Ref(0)
_ls_pair(x) = (_LS_CALLS[] += 1; (reshape(x, :, 1), hcat(x .^ 2, x .^ 3)))

@testset "in-model data values read per observation" begin
    # A data-only module value read as the matrix of a product, or as a
    # predictor column, is that observation column (decision 0z5bsqi,
    # prong in_model); bind_data computes it once and checks its rows.
    y, x = _ls_y(), _ls_x()
    bound = _ls_bind(quote
        B = _ls_mat(x)
        w[axes(B, 2)] .~ Normal.(0, 1)
        sd ~ HalfNormal(1)
        a ~ Normal(0, 1)
        mu = a .+ B * (sd .* w)
        y .~ Normal.(mu, 1.0)
    end, (:y, :x))
    @test bound.columns[:B] == _ls_mat(x)
    built = build_kernel(bound)
    u = _ls_pt(built.layout.total)
    nt = constrain(built.layout, u)
    a = nt.a
    @test _ls_node(built, bound, :likelihood, u) ≈
        sum(logpdf.(Normal.(a .+ _ls_mat(x) * (nt.sd .* nt.w), 1.0), y))
    @test _ls_node(built, bound, :prior, u) ≈ logpdf(Normal(0, 1), a) +
        logpdf(truncated(Normal(0, 1), 0, Inf), nt.sd) +
        sum(logpdf.(Normal(0, 1), nt.w))
    _check_gradient(built.spec, bound, u)

    bound = _ls_bind(quote
        q = _ls_sq(x)
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* q
        y .~ Normal.(mu, 1.0)
    end, (:y, :x))
    built = build_kernel(bound)
    u = _ls_pt(built.layout.total)
    nt = constrain(built.layout, u)
    @test _ls_node(built, bound, :likelihood, u) ≈
        sum(logpdf.(Normal.(nt.a .+ nt.b .* x .^ 2, 1.0), y))

    # A data-only value elementwise with an array, in the vector of a
    # supplied data matrix's product, is read whole: it stays model-level.
    lam = [0.5, 1.0]
    plan = lower_rkppl(quote
        L = _ls_sq(lam)
        w[axes(B, 2)] .~ Normal.(0, 1)
        sd ~ HalfNormal(1)
        S = sd .* exp.(-0.25 .* L) .* w
        a ~ Normal(0, 1)
        mu = a .+ B * S
        y .~ Normal.(mu, 1.0)
    end, (:y, :B, :lam); mod = _LS)
    B = _ls_mat(x)
    bound = bind_data(plan, Dict{Symbol,ColumnData}(:y => y, :B => B,
        :lam => lam))
    @test bound.columns[:L] == lam .^ 2
    built = build_kernel(bound)
    u = _ls_pt(built.layout.total)
    nt = constrain(built.layout, u)
    @test _ls_node(built, bound, :likelihood, u) ≈ sum(logpdf.(Normal.(
        nt.a .+ B * (nt.sd .* exp.(-0.25 .* lam .^ 2) .* nt.w), 1.0),
        y))

    # Beside a per-observation operand a data-only value is not needed per
    # observation: a scalar broadcasts, as in Julia (`m .* x`, `x ./ m`,
    # nested `x .* (b .* m)`), and stays model-level.
    for (pred, want) in (
            (:(a .+ m .* x), (nt, m) -> nt.a .+ m .* x),
            (:(a .+ b .* (x ./ m)), (nt, m) -> nt.a .+ nt.b .* x ./ m),
            (:(a .+ x .* (b .* m)), (nt, m) -> nt.a .+ x .* (nt.b * m)))
        ast = Expr(:block, :(m = _ls_mean(x)), :(a ~ Normal(0, 1)),
            :(b ~ Normal(0, 1)), :(mu = $pred), :(y .~ Normal.(mu, 1.0)))
        pred === :(a .+ m .* x) && deleteat!(ast.args, 3)
        bound = _ls_bind(ast, (:y, :x))
        built = build_kernel(bound)
        u = _ls_pt(built.layout.total)
        nt = constrain(built.layout, u)
        @test _ls_node(built, bound, :likelihood, u) ≈
            sum(logpdf.(Normal.(want(nt, _ls_mean(x)), 1.0), y))
    end

    # A matrix with a row count other than the observations' fails at bind.
    plan = lower_rkppl(quote
        B = _ls_rows(x)
        w[axes(B, 2)] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ B * w
        y .~ Normal.(mu, 1.0)
    end, (:y, :x); mod = _LS)
    err = try
        bind_data(plan, Dict{Symbol,ColumnData}(:y => y, :x => x))
        nothing
    catch e
        e
    end
    # refused: B * w has rows that cannot broadcast with the observed y axis (Julia dimensions, P3).
    @test err isa ContractValidationError
    @test occursin("B", err.message) && occursin("rows", err.message)
end

@testset "destructuring a data-only call" begin
    # `(B1, B2) = f(x)` binds each element (Julia's tuple semantics); the
    # call runs once per bind.
    y, x = _ls_y(), _ls_x()
    _LS_CALLS[] = 0
    bound = _ls_bind(quote
        (B1, B2) = _ls_pair(x)
        w1[axes(B1, 2)] .~ Normal.(0, 1)
        w2[axes(B2, 2)] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ B1 * w1 .+ B2 * w2
        y .~ Normal.(mu, 1.0)
    end, (:y, :x))
    @test _LS_CALLS[] == 1
    @test bound.columns[:B2] == hcat(x .^ 2, x .^ 3)
    built = build_kernel(bound)
    u = _ls_pt(built.layout.total)
    nt = constrain(built.layout, u)
    @test _ls_node(built, bound, :likelihood, u) ≈
        sum(logpdf.(Normal.(nt.a .+ x .* nt.w1[1] .+
            hcat(x .^ 2, x .^ 3) * nt.w2, 1.0), y))
end

# ── the validity floor: a lower-truncated LogNormal ──────────────────

@testset "truncated(LogNormal(m, s), lo, Inf) matches Distributions" begin
    y, x = _ls_y(), _ls_x()
    # A literal bound, a model-level data value (a named data reduction,
    # the HSGP floor's spelling) and the untruncated zero bound.
    data_lo = maximum(hsgp_rho_floors(last(hsgp_basis(x; k = 4))))
    for (lo_src, lo) in ((0.5, 0.5), (:rho_floor, data_lo), (0.0, 0.0))
        defs = lo_src === :rho_floor ?
            [:(rho_floor = maximum(hsgp_rho_floors(last(hsgp_basis(x; k = 4)))))] :
            Expr[]
        ast = Expr(:block, defs...,
            :(rho ~ truncated(LogNormal(0.3, 0.7), $lo_src, Inf)),
            :(a ~ Normal(0, 1)),
            :(mu = a .+ rho .* x),
            :(y .~ Normal.(mu, 1.0)))
        bound = _ls_bind(ast, (:y, :x))
        built = build_kernel(bound)
        u = _ls_pt(built.layout.total)
        nt = constrain(built.layout, u)
        @test nt.rho > lo
        @test _ls_node(built, bound, :prior, u) ≈
            logpdf(truncated(LogNormal(0.3, 0.7), lo, Inf), nt.rho) +
            logpdf(Normal(0, 1), nt.a)
        @test _ls_node(built, bound, :log_jacobian, u) ≈ log(nt.rho - lo)
        @test unconstrain(built.layout, nt) ≈ u
        _check_gradient(built.spec, bound, u)
    end
end

# ── library submodels lower like their hand-inlined bodies ───────────

@rkppl _ls_outer_smooth(xx) = begin
    (Xf, Zp) = tps_basis(xx; k = 4)
    f ~ penalized_smooth(Xf, Zp)
    return f
end

_ls_canon(ast, names) = sprint(_canon, lower_rkppl(ast, names; mod = _LS))

@testset "library smooths: S5 shape and nesting are transparent" begin
    names = (:y, :x)
    # Probe S5: a smooth bound to a name, then added to the predictor.
    @test _ls_canon(quote
        a ~ Normal(0, 5)
        (Xf, Zp) = tps_basis(x; k = 4)
        f ~ penalized_smooth(Xf, Zp)
        sigma ~ Exponential(1)
        mu = a .+ f
        y .~ Normal.(mu, sigma)
    end, names) == _ls_canon(quote
        a ~ Normal(0, 5)
        (Xf, Zp) = tps_basis(x; k = 4)
        f_b[axes(Xf, 2)] .~ Flat.()
        f_sd ~ HalfNormal(1)
        f_z[axes(Zp, 2)] .~ Normal.(0, 1)
        f = Xf * f_b .+ Zp * (f_sd .* f_z)
        sigma ~ Exponential(1)
        mu = a .+ f
        y .~ Normal.(mu, sigma)
    end, names)
    # Called from inside another submodel (with a destructured basis).
    @test _ls_canon(quote
        a ~ Normal(0, 5)
        s ~ _ls_outer_smooth(x)
        mu = a .+ s
        y .~ Normal.(mu, 1.0)
    end, names) == _ls_canon(quote
        a ~ Normal(0, 5)
        (s_Xf, s_Zp) = tps_basis(x; k = 4)
        s_f_b[axes(s_Xf, 2)] .~ Flat.()
        s_f_sd ~ HalfNormal(1)
        s_f_z[axes(s_Zp, 2)] .~ Normal.(0, 1)
        s_f = s_Xf * s_f_b .+ s_Zp * (s_f_sd .* s_f_z)
        s = s_f
        mu = a .+ s
        y .~ Normal.(mu, 1.0)
    end, names)
    @test _ls_canon(quote
        a ~ Normal(0, 1)
        (PHI, lambda) = hsgp_basis(x; k = 6)
        f ~ hsgp_effect(PHI, lambda)
        mu = a .+ f
        y .~ Normal.(mu, 1.5)
    end, names) == _ls_canon(quote
        a ~ Normal(0, 1)
        (PHI, lambda) = hsgp_basis(x; k = 6)
        # The body's calls resolve in its defining module.
        f_rho_floor = maximum(ReactiveKernelsPPL.hsgp_rho_floors(lambda))
        f_rho ~ truncated(LogNormal(0, 1), f_rho_floor, Inf)
        f_sigma ~ LogNormal(0, 1)
        f_z[axes(PHI, 2)] .~ Normal.(0, 1)
        f = PHI * (ReactiveKernelsPPL.hsgp_sqrt_spd(lambda, f_sigma, f_rho) .*
            f_z)
        mu = a .+ f
        y .~ Normal.(mu, 1.5)
    end, names)
end

# ── library vs built-in and Distributions oracles ────────────────────

# The built-in twin of a corpus pair: the original corpus program, or (for
# the synthetic domain / grouped shapes, whose originals are not in the
# corpus) the built-in spelling here.
const _LS_BUILTIN_EXTRA = Dict(
    "30_hsgp_domain_library" => quote
        a ~ Normal(0, 1)
        mu = a .+ hsgp(:h_x)
        y .~ Normal.(mu, 1.5)
        hsgp_basis(:h_x, x; k = 6, domain = (-5.0, 5.0))
    end,
    "30_hsgp_by_library" => quote
        a ~ Normal(0, 1)
        mu = a .+ hsgp(:h_x)
        y .~ Normal.(mu, 1.5)
        hsgp_basis(:h_x, x; k = 5, by = g, length_scale = 1 + (1 | g),
            sd = 1 + (1 | g))
    end)

_ls_lncc(lo) = logccdf(LogNormal(0, 1), lo)

# (library corpus case, built-in case, library → built-in values, the
# library prior minus the built-in's at matched values)
function _ls_pairs()
    x, z, x2 = _ls_x(), _ls_z(), _ls_x2()
    fl(xs...; kw...) = maximum(hsgp_rho_floors(last(hsgp_basis(xs...; kw...))))
    pf = hsgp_periodic_rho_floor(collect(1:4))
    afl = hsgp_rho_floors(last(hsgp_basis(x, z; k = (4, 3), c = (1.5, 2.0))))
    return [
        ("24_spline_s_library", "24_spline_s",
            nt -> (a = nt.a, sigma = nt.sigma, b_s_x_fixed = nt.f_b,
                b_s_x_raw = nt.f_z, sd_s_x = [nt.f_sd]),
            _ -> log(2)),
        ("25_spline_t2_library", "25_spline_t2",
            nt -> (a = nt.a, sigma = nt.sigma, b_t2_xz_fixed = nt.f_b,
                b_t2_xz_rr_raw = nt.f_z_rr, b_t2_xz_rn_raw = nt.f_z_rn,
                b_t2_xz_nr_raw = nt.f_z_nr, sd_t2_xz = nt.f_sd),
            _ -> 3 * log(2)),
        ("30_hsgp_1d_library", "30_hsgp_1d",
            nt -> (a = nt.a, rho_h_x = nt.f_rho, sigma_h_x = nt.f_sigma,
                beta_raw_h_x = nt.f_z),
            _ -> -_ls_lncc(fl(x; k = 4))),
        ("30_hsgp_domain_library", "30_hsgp_domain_library",
            nt -> (a = nt.a, rho_h_x = nt.f_rho, sigma_h_x = nt.f_sigma,
                beta_raw_h_x = nt.f_z),
            _ -> -_ls_lncc(fl(x; k = 6, domain = (-5.0, 5.0)))),
        ("30_hsgp_by_library", "30_hsgp_by_library",
            nt -> (a = nt.a, beta0_rho_h_x = nt.f_rho_mu,
                sd_rho_h_x = nt.f_rho_sd, z_rho_h_x = nt.f_rho_z,
                beta0_sigma_h_x = nt.f_sigma_mu,
                sd_sigma_h_x = nt.f_sigma_sd, z_sigma_h_x = nt.f_sigma_z,
                beta_raw_h_x = nt.f_z),
            _ -> 2 * log(2)),
        ("31_hsgp_aniso_library", "31_hsgp_aniso",
            nt -> (a = nt.a, rho_h_xz_1 = nt.rho_1, rho_h_xz_2 = nt.rho_2,
                sigma_h_xz = nt.sigma_f, beta_raw_h_xz = nt.z_f),
            _ -> -_ls_lncc(afl[1]) - _ls_lncc(afl[2])),
        ("68_hsgp_periodic_library", "68_hsgp_periodic",
            nt -> (a = nt.a, rho_h_p = nt.f_rho, sigma_h_p = nt.f_sigma,
                beta_raw_h_p = nt.f_z),
            _ -> -_ls_lncc(pf)),
        ("89_hsgp_hyper_priors_library", "89_hsgp_hyper_priors",
            nt -> (b0 = nt.b0, s0 = nt.s0, rho_h_x = nt.h_rho,
                sigma_h_x = nt.h_sigma, beta_raw_h_x = nt.h_z,
                rho_h_x2 = nt.h2_rho, sigma_h_x2 = nt.h2_sigma,
                beta_raw_h_x2 = nt.h2_z),
            _ -> 2 * log(2)),
        ("90_hsgp_only_library", "90_hsgp_only",
            nt -> (rho_h_x = nt.h_rho,
                sigma_h_x = nt.h_sigma, beta_raw_h_x = nt.h_z,
                rho_h_x2 = nt.h2_rho, sigma_h_x2 = nt.h2_sigma,
                beta_raw_h_x2 = nt.h2_z),
            _ -> 0.0),
        ("91_spline_sd_prior_library", "91_spline_sd_prior",
            nt -> (b0 = nt.b0, s0 = nt.s0, b_s_x_fixed = nt.s_b,
                b_s_x_raw = nt.s_z, sd_s_x = [nt.s_sd],
                b_s_x2_fixed = nt.s2_b, b_s_x2_raw = nt.s2_z,
                sd_s_x2 = [nt.s2_sd]),
            _ -> 2 * log(2)),
    ]
end

@testset "library smooths match the built-ins at matched values" begin
    # Same likelihood and log-Jacobian at the same constrained values; the
    # priors differ only by stated constants: Distributions halves where
    # the built-in uses Stan's unnormalized halves (`log(2)` each), and the
    # normalized truncation of the floored length scale
    # (`-logccdf(LogNormal(0, 1), floor)`; the built-in's lower-bound kernel
    # drops it).
    for (lib, blt, tobuilt, offset) in _ls_pairs()
        @testset "$lib" begin
            last_ast, names = _load_corpus_case(joinpath(_CORPUS_DIR,
                lib * ".jl"))
            bl = _ls_bind(last_ast, names)
            bast = haskey(_LS_BUILTIN_EXTRA, blt) ? _LS_BUILTIN_EXTRA[blt] :
                first(_load_corpus_case(joinpath(_CORPUS_DIR, blt * ".jl")))
            bb = _ls_bind(bast, names)
            kl, kb = build_kernel(bl), build_kernel(bb)
            @test kl.layout.total == kb.layout.total
            ul = _ls_pt(kl.layout.total)
            ntl = constrain(kl.layout, ul)
            ub = unconstrain(kb.layout, tobuilt(ntl))
            for node in (:likelihood, :log_jacobian)
                @test _ls_node(kl, bl, node, ul) ≈ _ls_node(kb, bb, node, ub)
            end
            @test _ls_node(kl, bl, :prior, ul) - _ls_node(kb, bb, :prior, ub) ≈
                offset(ntl) atol = 1e-9
            _check_gradient(kl.spec, bl, ul)
        end
    end
end

# Independent references (Distributions densities, spectral densities
# written out per basis function).
_ls_spd_eq(lam, sigma, rho) = sigma * sqrt(sqrt(2pi) * rho) * exp(-rho^2 * lam / 4)

@testset "library HSGP effects match hand-written oracles" begin
    y, x, g = _ls_y(), _ls_x(), _ls_g()
    # Isotropic exp-quad.
    bound = _ls_bind(first(_load_corpus_case(joinpath(_CORPUS_DIR,
        "30_hsgp_1d_library.jl"))), (:y, :x))
    built = build_kernel(bound)
    u = _ls_pt(built.layout.total)
    nt = constrain(built.layout, u)
    PHI, lambda = hsgp_basis(x; k = 4)
    floor = maximum(hsgp_rho_floors(lambda))
    w = [_ls_spd_eq(lambda[m, 1], nt.f_sigma, nt.f_rho) for m in 1:4] .* nt.f_z
    a = nt.a
    @test _ls_node(built, bound, :likelihood, u) ≈
        sum(logpdf.(Normal.(a .+ PHI * w, 1.5), y))
    @test _ls_node(built, bound, :prior, u) ≈ logpdf(Normal(0, 1), a) +
        logpdf(truncated(LogNormal(0, 1), floor, Inf), nt.f_rho) +
        logpdf(LogNormal(0, 1), nt.f_sigma) + sum(logpdf.(Normal(), nt.f_z))
    # Periodic.
    bound = _ls_bind(first(_load_corpus_case(joinpath(_CORPUS_DIR,
        "68_hsgp_periodic_library.jl"))), (:y, :x))
    built = build_kernel(bound)
    u = _ls_pt(built.layout.total)
    nt = constrain(built.layout, u)
    PHI, h = hsgp_periodic_basis(x; k = 4, period = 2.0)
    q = [nt.f_sigma * sqrt(2 * exp(-1 / nt.f_rho^2) * besseli(j, 1 / nt.f_rho^2))
        for j in h]
    @test _ls_node(built, bound, :likelihood, u) ≈
        sum(logpdf.(Normal.(nt.a .+ PHI * (q .* nt.f_z), 1.5), y))
    # Grouped: one curve per level, per-group hypers.
    bound = _ls_bind(first(_load_corpus_case(joinpath(_CORPUS_DIR,
        "30_hsgp_by_library.jl"))), (:y, :x, :g))
    built = build_kernel(bound)
    u = _ls_pt(built.layout.total)
    nt = constrain(built.layout, u)
    P1, lambda = hsgp_basis(x; k = 5)
    floor = maximum(hsgp_rho_floors(lambda))
    lv = sort(unique(g))
    rho = max.(exp.(nt.f_rho_mu .+ nt.f_rho_sd .* nt.f_rho_z), floor)
    sig = exp.(nt.f_sigma_mu .+ nt.f_sigma_sd .* nt.f_sigma_z)
    Z = reshape(nt.f_z, 3, 5)
    f = [sum(P1[i, m] * _ls_spd_eq(lambda[m, 1], sig[gi], rho[gi]) * Z[gi, m]
        for m in 1:5) for (i, gi) in enumerate(indexin(g, lv))]
    @test _ls_node(built, bound, :likelihood, u) ≈
        sum(logpdf.(Normal.(nt.a .+ f, 1.5), y))
    @test _ls_node(built, bound, :prior, u) ≈ logpdf(Normal(0, 1),
        nt.a) + logpdf(Normal(0, 1), nt.f_rho_mu) +
        logpdf(truncated(Normal(0, 1), 0, Inf), nt.f_rho_sd) +
        logpdf(Normal(0, 1), nt.f_sigma_mu) +
        logpdf(truncated(Normal(0, 1), 0, Inf), nt.f_sigma_sd) +
        sum(logpdf.(Normal(), nt.f_rho_z)) +
        sum(logpdf.(Normal(), nt.f_sigma_z)) + sum(logpdf.(Normal(), nt.f_z))
end
