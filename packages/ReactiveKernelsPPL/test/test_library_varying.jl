using DifferentiationInterface
using Distributions
using Enzyme
using LinearAlgebra
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Varying effects as library submodels (decision 1cmodra, prong
# `varying`): `varying_coefs`, `varying_coefs_correlated`,
# `varying_coefs_centered` and `varying_stratified` (src/library.jl)
# state every prior in their bodies (hunt-priors 10ldrvz `varying_sd`:
# `sd ~ HalfNormal(1)` per margin, `LKJCholesky(K, 1.0)` for K ≥ 2) and
# return per-level coefficients the use site reads with plain indexing.
#
# Each re-spelled corpus program (`test/corpus/*_lib.jl`) is checked
# against its built-in original at matching values: the same likelihood
# and log-Jacobian, and a prior that differs only by the documented
# constant (the built-in sd prior is Stan's unnormalized half,
# `+log(2)` short per half-Normal or half-Cauchy margin). Library
# densities are checked against Distributions.jl, and gradients against
# central differences. Data are synthetic. Helpers: `_canon`,
# `_load_corpus_case`, `_CORPUS_DIR` (test_corpus.jl), `_query`,
# `_check_gradient` (test_generator.jl).

_lv_lower(ex, data) = lower_rkppl(ex, data; mod = @__MODULE__)
_lv_canon(ex, data) = sprint(_canon, _lv_lower(ex, data))

const _LV_N = 24

# Synthetic columns for a corpus program's data names.
function _lv_cols(data; bernoulli::Bool = false, string_groups::Bool = false)
    d = Dict{Symbol,AbstractVector}()
    t = range(-1.3, 1.7; length = _LV_N)
    for k in data
        d[k] = if k === :g && string_groups
            repeat(["a", "b", "c"], _LV_N ÷ 3)
        elseif k in (:g, :person, :item)
            repeat([1, 2, 3, 4], _LV_N ÷ 4)
        elseif k === :g1
            repeat([1, 2, 3], _LV_N ÷ 3)
        elseif k === :g2
            repeat([2, 3, 4, 1], _LV_N ÷ 4)
        elseif k in (:b, :c)
            repeat([1, 2], _LV_N ÷ 2)
        elseif k === :w1
            [0.5 + 0.4 * abs(sin(1.7 * i)) for i in 1:_LV_N]
        elseif k in (:w2, :sigma)
            [0.6 + 0.3 * abs(cos(0.9 * i)) for i in 1:_LV_N]
        elseif k === :y && bernoulli
            [isodd(i ÷ 2) ? 1 : 0 for i in 1:_LV_N]
        elseif k === :x
            collect(t)
        elseif k === :z
            [0.4 * cos(2.1 * i) for i in 1:_LV_N]
        else
            [0.8 * sin(0.7 * i) + 0.1 * i / _LV_N for i in 1:_LV_N]
        end
    end
    return d
end

function _lv_build(name::AbstractString; kw...)
    ast, data = _load_corpus_case(joinpath(_CORPUS_DIR, name * ".jl"))
    bound = bind_data(lower_rkppl(ast, data), _lv_cols(data; kw...))
    return bound, build_kernel(bound)
end

_lv_point(n; scale = 0.4) = [scale * sin(1.1 * i + 0.3) for i in 1:n]

# Coordinate-name map from a library draws block (`P_sd`, `P_L`, `P_z`)
# to a built-in draws block (`tau_S`, `L_S`, `z_flat_S`, the z block
# K × G column-major) over G levels and K margins.
function _lv_block_map(P, S, K, G; levels = 1:G)
    m = Dict{Symbol,Symbol}()
    if K == 1
        m[Symbol(P, "_sd")] = Symbol("tau_", S, ".1")
        for (j, jb) in zip(1:G, levels)
            m[Symbol(P, "_z.", j)] = Symbol("z_flat_", S, ".", jb)
        end
    else
        for k in 1:K
            m[Symbol(P, "_sd.", k)] = Symbol("tau_", S, ".", k)
        end
        for p in 1:(K * (K - 1) ÷ 2)
            m[Symbol(P, "_L.", p)] = Symbol("L_", S, ".", p)
        end
        for j in 1:G, q in 1:K
            m[Symbol(P, "_z.", j, ".", q)] =
                Symbol("z_flat_", S, ".", q + (j - 1) * K)
        end
    end
    return m
end

# The built-in point holding the library point's values: shared names map
# to themselves, `map` renames the rest, unmapped built-in coordinates
# (prior-only levels) are 0.
function _lv_translate(ulib, llib, lbi, map)
    names_lib = coordinate_names(llib)
    names_bi = coordinate_names(lbi)
    from = Dict{Symbol,Float64}()
    for (nm, v) in zip(names_lib, ulib)
        from[get(map, nm, nm)] = v
    end
    @test all(nm -> get(map, nm, nm) in names_bi, names_lib)
    return [get(from, nm, 0.0) for nm in names_bi]
end

# Library vs built-in at matching values: identical likelihood and
# log-Jacobian; prior difference `prior_shift(ubi)`.
function _lv_parity(orig, lib, map, prior_shift; kw...)
    bb, kb = _lv_build(orig; kw...)
    bl, kl = _lv_build(lib; kw...)
    for s in (0.3, 0.7)
        ul = _lv_point(kl.layout.total; scale = s)
        ub = _lv_translate(ul, kl.layout, kb.layout, map)
        @test _query(kl.spec, bl, :likelihood, ul) ≈
            _query(kb.spec, bb, :likelihood, ub) rtol = 1e-12 atol = 1e-10
        @test _query(kl.spec, bl, :log_jacobian, ul) ≈
            _query(kb.spec, bb, :log_jacobian, ub) atol = 1e-10
        @test _query(kl.spec, bl, :prior, ul) - _query(kb.spec, bb, :prior, ub) ≈
            prior_shift(ub, kb.layout) atol = 1e-10
    end
    return nothing
end

_lv_halves(n) = (_, _) -> n * log(2)

# `ex` with every occurrence of the expression `from` replaced by `to`.
postwalk_replace(ex, from, to) = ex == from ? to :
    ex isa Expr ? Expr(ex.head, (postwalk_replace(a, from, to)
        for a in ex.args)...) : ex

# Statements binding `d` or `d_…` (the written-out draws block).
function _lv_binds_prefix(st::Expr, p)
    lhs = st.args[st.head === :(=) ? 1 : 2]
    nm = lhs isa Expr && lhs.head === :ref ? lhs.args[1] : lhs
    nm isa Symbol || return false
    s = string(nm)
    return s == p || startswith(s, p * "_")
end

# `called` with its statements in `like`'s order of first binding, so the
# library call sits where the written-out block began.
function _lv_reorder(called::Expr, like::Expr)
    body = [a for a in like.args if !(a isa LineNumberNode)]
    i = findfirst(a -> a isa Expr && a.head in (:call, :(=)) &&
        _lv_binds_prefix(a, "d"), body)
    rest = [a for a in called.args[2:end] if !(a isa LineNumberNode)]
    return Expr(:block, rest[1:i-1]..., called.args[1], rest[i:end]...)
end

@testset "library varying submodels lower to their written-out bodies" begin
    data = (:y, :x, :g, :b)
    head = (:(a ~ Normal(0, 5)), :(sigma ~ Exponential(1)))
    tail = :(y .~ Normal.(mu, sigma))
    prog(stmts...) = Expr(:block, head..., stmts..., tail)
    @test _lv_canon(prog(:(r ~ varying_coefs(g)), :(mu = a .+ r[g])), data) ==
        _lv_canon(prog(:(r_sd ~ HalfNormal(1)),
            :(r_z[levels(g)] .~ Normal.(0, 1)), :(r = r_sd .* r_z),
            :(mu = a .+ r[g])), data)
    @test _lv_canon(prog(:(r ~ varying_coefs_correlated(g, 3)),
            :(mu = a .+ r[g, 1] .+ x .* r[g, 2] .- x .* r[g, 3])), data) ==
        _lv_canon(prog(:(r_sd[1:3] .~ HalfNormal.(1)),
            :(r_L ~ LKJCholesky(3, 1.0)),
            :(r_z[levels(g), 1:3] .~ Normal.(0, 1)),
            :(r = r_z * (r_sd .* r_L)'),
            :(mu = a .+ r[g, 1] .+ x .* r[g, 2] .- x .* r[g, 3])), data)
    @test _lv_canon(prog(:(c ~ varying_coefs_centered(g)),
            :(mu = a .+ c[g])), data) ==
        _lv_canon(prog(:(c_sd ~ HalfNormal(1)),
            :(c_c[levels(g)] .~ Normal.(0, c_sd)), :(c = c_c),
            :(mu = a .+ c[g])), data)
    @test _lv_canon(prog(:(u ~ varying_stratified(g, b)), :(mu = a .+ u)),
            data) ==
        _lv_canon(prog(:(u_sd[levels(b)] .~ HalfNormal.(1)),
            :(u_z[levels(g)] .~ Normal.(0, 1)), :(u = u_sd[b] .* u_z[g]),
            :(mu = a .+ u)), data)
    # The written-out corpus programs are the library expansion with one
    # statement changed: with the shipped default restored they are the
    # library call.
    lib28, data28 = _load_corpus_case(joinpath(_CORPUS_DIR,
        "28_varying_multislice_lib.jl"))
    restored = postwalk_replace(lib28, :(LKJCholesky(2, 2.0)),
        :(LKJCholesky(2, 1.0)))
    called = Expr(:block, (a for a in lib28.args
        if !(a isa Expr && a.head in (:call, :(=)) &&
            _lv_binds_prefix(a, "d"))  )...)
    pushfirst!(called.args, :(d ~ varying_coefs_correlated(g, 2)))
    @test _lv_canon(restored, data28) == _lv_canon(_lv_reorder(called,
        restored), data28)
end

@testset "library varying re-spellings: parity with the built-ins" begin
    G = 4
    # One margin, plain grouping.
    for (orig, lib) in (("15_varying_plain", "15_varying_plain_lib"),
            ("26_varying_intercept", "26_varying_intercept_lib"),
            ("27_varying_slope", "27_varying_slope_lib"))
        _lv_parity(orig, lib, _lv_block_map(:r, :g, 1, G), _lv_halves(1))
    end
    # Two correlated margins: intercept + slope, indicator, derived local.
    for (orig, lib) in (("16_varying_correlated", "16_varying_correlated_lib"),
            ("17_varying_dummy", "17_varying_dummy_lib"),
            ("44_derived_margin", "44_derived_margin_lib"))
        _lv_parity(orig, lib, _lv_block_map(:r, :g, 2, G), _lv_halves(2))
    end
    # Shared draws sliced into two predictors (LKJ shape 2.0, stated).
    _lv_parity("28_varying_multislice", "28_varying_multislice_lib",
        _lv_block_map(:d, :g, 2, G), _lv_halves(2))
    # Declared levels ["c", "a", "b", "d"]: the library reads the sorted
    # observed levels; the built-in's prior-only level "d" (coordinate 4,
    # held at 0) adds its standard-normal term.
    _lv_parity("45_varying_levels", "45_varying_levels_lib",
        _lv_block_map(:r, :g, 1, 3; levels = [2, 3, 1]),
        (ub, lb) -> log(2) - logpdf(Normal(0, 1),
            ub[findfirst(==(Symbol("z_flat_g.4")), coordinate_names(lb))]);
        string_groups = true)
    # Multi-membership over the union of the membership columns.
    _lv_parity("62_mm_intercept", "62_mm_intercept_lib",
        _lv_block_map(:r, :mm__g1__g2, 1, G), _lv_halves(1))
    _lv_parity("64_mm_slope", "64_mm_slope_lib",
        _lv_block_map(:r, :mm__g1__g2, 1, G), _lv_halves(1))
    _lv_parity("63_mm_correlated", "63_mm_correlated_lib",
        _lv_block_map(:r, :mm__g1__g2__w__w1__w2, 2, G), _lv_halves(2))
    # Stratified, one margin: one sd per stratum.
    strat = Dict{Symbol,Symbol}(Symbol("r_sd.", k) => Symbol("tau_g_s", k, ".1")
        for k in 1:2)
    for j in 1:G
        strat[Symbol("r_z.", j)] = Symbol("z_flat_g.", j)
    end
    _lv_parity("66_stratified_k1", "66_stratified_k1_lib", strat,
        _lv_halves(2))
    # Stated non-default sd priors: half-Cauchy (proper vs the built-in's
    # unnormalized half) and Exponential (identical).
    _lv_parity("75_varying_sd_cauchy", "75_varying_sd_cauchy_lib",
        _lv_block_map(:r, :g, 1, G), _lv_halves(1))
    # `b0` is a plain parameter in the library spelling (its `eta` lowers
    # as one computed column) and the intercept of the composed `b`
    # sub-predictor in the built-in.
    m85 = merge(_lv_block_map(:r_t, :person, 1, G),
        _lv_block_map(:r_a, :item, 1, G), _lv_block_map(:r_b, :item_r_b, 1, G),
        Dict(:b0 => Symbol("b.Intercept")))
    _lv_parity("85_composed_varying_exp", "85_composed_varying_exp_lib", m85,
        _lv_halves(3); bernoulli = true)
    _lv_parity("87_composed_correlated_slices",
        "87_composed_correlated_slices_lib",
        _lv_block_map(:dx, :item, 2, G), (_, _) -> 0.0; bernoulli = true)
    # Stratified correlated (corpus 65): per stratum k its sds
    # (`r_sd.k.j`) and LKJ factor (`r_L.p.k`); the levels of g share z.
    m65 = Dict{Symbol,Symbol}()
    for k in 1:2, j in 1:2
        m65[Symbol("r_sd.", k, ".", j)] = Symbol("tau_g_s", k, ".", j)
    end
    for k in 1:2
        m65[Symbol("r_L.1.", k)] = Symbol("L_g_s", k, ".1")
    end
    for j in 1:G, q in 1:2
        m65[Symbol("r_z.", j, ".", q)] = Symbol("z_flat_g.", q + (j - 1) * 2)
    end
    _lv_parity("65_stratified", "65_stratified_lib", m65, _lv_halves(4))
end

_lv_halfnormal(s) = logpdf(truncated(Normal(0, 1), 0, Inf), s)
_lv_lkj(L, eta) = logpdf(LKJCholesky(size(L, 1), eta),
    Cholesky(LowerTriangular(L)))

@testset "library varying densities: Distributions.jl oracles" begin
    # Correlated intercept + slope (corpus 16, library spelling).
    b16, k16 = _lv_build("16_varying_correlated_lib")
    u = _lv_point(k16.layout.total)
    nt = constrain(k16.layout, u)
    x, y, g = b16.columns[:x], b16.columns[:y], b16.columns[:g]
    B = nt.r_z * (nt.r_sd .* nt.r_L)'
    a, bx = nt.mu
    mu = a .+ bx .* x .+ B[g, 1] .+ x .* B[g, 2]
    @test _query(k16.spec, b16, :likelihood, u) ≈
        sum(logpdf.(Normal.(mu, nt.sigma), y))
    @test _query(k16.spec, b16, :prior, u) ≈ logpdf(Normal(0, 1), a) +
        logpdf(Normal(0, 2), bx) + logpdf(Exponential(1), nt.sigma) +
        sum(_lv_halfnormal, nt.r_sd) + _lv_lkj(nt.r_L, 1.0) +
        sum(logpdf.(Normal(0, 1), nt.r_z))
    _check_gradient(k16.spec, b16, u)
    # Multi-membership with normalized weights (corpus 63).
    b63, k63 = _lv_build("63_mm_correlated_lib")
    u = _lv_point(k63.layout.total)
    nt = constrain(k63.layout, u)
    c = b63.columns
    lv = sort(unique(vcat(c[:g1], c[:g2])))
    B = nt.r_z * (nt.r_sd .* nt.r_L)'
    row(gcol) = [findfirst(==(v), lv) for v in gcol]
    i1, i2 = row(c[:g1]), row(c[:g2])
    wt = c[:w1] .+ c[:w2]
    mu = only(nt.mu) .+ (c[:w1] ./ wt) .* (B[i1, 1] .+ c[:x] .* B[i1, 2]) .+
        (c[:w2] ./ wt) .* (B[i2, 1] .+ c[:x] .* B[i2, 2])
    @test _query(k63.spec, b63, :likelihood, u) ≈
        sum(logpdf.(Normal.(mu, nt.sigma), c[:y]))
    @test _query(k63.spec, b63, :prior, u) ≈ logpdf(Normal(0, 5),
        only(nt.mu)) + logpdf(Exponential(1), nt.sigma) +
        sum(_lv_halfnormal, nt.r_sd) + _lv_lkj(nt.r_L, 1.0) +
        sum(logpdf.(Normal(0, 1), nt.r_z))
    _check_gradient(k63.spec, b63, u)
    # Stratified, one margin (corpus 66).
    b66, k66 = _lv_build("66_stratified_k1_lib")
    u = _lv_point(k66.layout.total)
    nt = constrain(k66.layout, u)
    c = b66.columns
    mu = only(nt.mu) .+ nt.r_sd[c[:b]] .* nt.r_z[c[:g]]
    @test _query(k66.spec, b66, :likelihood, u) ≈
        sum(logpdf.(Normal.(mu, nt.sigma), c[:y]))
    @test _query(k66.spec, b66, :prior, u) ≈ logpdf(Normal(0, 5),
        only(nt.mu)) + logpdf(Exponential(1), nt.sigma) +
        sum(_lv_halfnormal, nt.r_sd) + sum(logpdf.(Normal(0, 1), nt.r_z))
    _check_gradient(k66.spec, b66, u)
    # Centered, one margin (corpus 72, library spelling).
    b72, k72 = _lv_build("72_centered_levels_lib")
    u = _lv_point(k72.layout.total)
    nt = constrain(k72.layout, u)
    c = b72.columns
    mu_alpha = only(nt.mu)
    mu = mu_alpha .+ nt.c_c[c[:g]]
    @test _query(k72.spec, b72, :likelihood, u) ≈
        sum(logpdf.(Normal.(mu, nt.s), c[:y]))
    @test _query(k72.spec, b72, :prior, u) ≈ logpdf(Normal(0, 10),
        mu_alpha) + logpdf(Exponential(1), nt.s) +
        _lv_halfnormal(nt.c_sd) + sum(logpdf.(Normal(0, nt.c_sd), nt.c_c))
    _check_gradient(k72.spec, b72, u)
    # Gradients of every other re-spelling.
    for (name, kw) in (("15_varying_plain_lib", ()),
            ("17_varying_dummy_lib", ()), ("26_varying_intercept_lib", ()),
            ("27_varying_slope_lib", ()), ("28_varying_multislice_lib", ()),
            ("44_derived_margin_lib", ()),
            ("45_varying_levels_lib", (; string_groups = true)),
            ("62_mm_intercept_lib", ()), ("64_mm_slope_lib", ()),
            ("75_varying_sd_cauchy_lib", ()),
            ("85_composed_varying_exp_lib", (; bernoulli = true)),
            ("87_composed_correlated_slices_lib", (; bernoulli = true)))
        bnd, blt = _lv_build(name; kw...)
        _check_gradient(blt.spec, bnd, _lv_point(blt.layout.total))
    end
end

@testset "gathers from array-valued definitions" begin
    data = (:y, :x, :g)
    cols = _lv_cols(data)
    head = (:(a ~ Normal(0, 5)), :(s ~ Exponential(1)),
        :(sd[1:2] .~ HalfNormal.(1)), :(z[levels(g), 1:2] .~ Normal.(0, 1)))
    # A level gather from a definition reads the definition's level axis:
    # inline in a predictor and through a named column give one density.
    inline = _lv_lower(Expr(:block, head..., :(b = z .* sd'),
        :(mu = a .+ b[g, 1] .+ x .* b[g, 2]), :(y .~ Normal.(mu, s))), data)
    named = _lv_lower(Expr(:block, head..., :(b = z .* sd'),
        :(r = b[g, 1] .+ x .* b[g, 2]), :(mu = a .+ r),
        :(y .~ Normal.(mu, s))), data)
    bi, bn = bind_data(inline, cols), bind_data(named, cols)
    ki, kn = build_kernel(bi), build_kernel(bn)
    u = _lv_point(ki.layout.total)
    @test _query(ki.spec, bi, :posterior, u) ≈
        _query(kn.spec, bn, :posterior, u)
    nt = constrain(ki.layout, u)
    B = nt.z .* nt.sd'
    g = cols[:g]
    @test _query(ki.spec, bi, :likelihood, u) ≈ sum(logpdf.(Normal.(
        only(nt.mu) .+ B[g, 1] .+ cols[:x] .* B[g, 2], nt.s), cols[:y]))
    # Not built yet (todo 11e81k8): valid Julia, so these are capability
    # gaps, not refusals. A single index on the 1×L row `(z * sd)'` is a
    # linear, positional read; a module function's result has axes RKPPL
    # cannot track, so `b[g, 1]` is positional too. Both lower today only
    # with known level axes and one index per axis.
    @test_broken (lower_rkppl(Expr(:block, head..., :(b = (z * sd)'),
        :(mu = a .+ b[g]), :(y .~ Normal.(mu, s))), data); true)
    @test_broken (lower_rkppl(Expr(:block, head..., :(b = identity(z)),
        :(mu = a .+ b[g, 1]), :(y .~ Normal.(mu, s))), data;
        mod = @__MODULE__); true)
end

@testset "centered correlated: library entry, oracle, built-in parity" begin
    data = (:y, :x, :g)
    head = (:(a ~ Normal(0, 5)), :(sigma ~ Exponential(1)))
    tail = :(y .~ Normal.(mu, sigma))
    prog(stmts...) = Expr(:block, head..., stmts..., tail)
    use = :(mu = a .+ b[g, 1] .+ x .* b[g, 2])
    lib = prog(:(b ~ varying_coefs_centered_correlated(g, 2)), use)
    # The library call is its written-out body over the §3 row statement.
    @test _lv_canon(lib, data) == _lv_canon(prog(
        :(b_sd[1:2] .~ HalfNormal.(1)), :(b_L ~ LKJCholesky(2, 1.0)),
        :(b_F = b_sd .* b_L),
        :(eachrow(b_c[levels(g), 1:2]) .~ MvNormalCholesky(zeros(2), b_F)),
        :(b = b_c), use), data)
    cols = _lv_cols(data)
    bl = bind_data(_lv_lower(lib, data), cols)
    kl = build_kernel(bl)
    u = _lv_point(kl.layout.total)
    nt = constrain(kl.layout, u)
    lv = sort(unique(cols[:g]))
    gi = [findfirst(==(v), lv) for v in cols[:g]]
    F = Diagonal(nt.b_sd) * nt.b_L
    C = nt.b_c
    mu = only(nt.mu) .+ C[gi, 1] .+ cols[:x] .* C[gi, 2]
    @test _query(kl.spec, bl, :likelihood, u) ≈
        sum(logpdf.(Normal.(mu, nt.sigma), cols[:y]))
    @test _query(kl.spec, bl, :prior, u) ≈ logpdf(Normal(0, 5),
        only(nt.mu)) + logpdf(Exponential(1), nt.sigma) +
        sum(_lv_halfnormal, nt.b_sd) + _lv_lkj(nt.b_L, 1.0) +
        sum(logpdf(MvNormal(zeros(2), F * F'), C[j, :]) for j in axes(C, 1))
    _check_gradient(kl.spec, bl, u)
    # Parity with the built-in centered geometry (`varying_draws(...;
    # centered = true)`) under the same priors, at matching values: the
    # written-out body with those priors gives the same likelihood, prior
    # and log-Jacobian.
    builtin = Expr(:block, head..., :(d ~ varying_draws(g, [1, x];
        centered = true, eta = 1.5, sd = Exponential(2.0))),
        :(r ~ varying_slice(d, 1:2)), :(mu = a .+ r), tail)
    written = prog(:(d_sd[1:2] .~ Exponential.(2.0)),
        :(d_L ~ LKJCholesky(2, 1.5)), :(d_F = d_sd .* d_L),
        :(eachrow(d_c[levels(g), 1:2]) .~ MvNormalCholesky(zeros(2), d_F)),
        :(mu = a .+ d_c[g, 1] .+ x .* d_c[g, 2]))
    bb = bind_data(_lv_lower(builtin, data), cols)
    kb = build_kernel(bb)
    bw = bind_data(_lv_lower(written, data), cols)
    kw = build_kernel(bw)
    uw = _lv_point(kw.layout.total)
    w = constrain(kw.layout, uw)
    base = constrain(kb.layout, zeros(kb.layout.total))
    # The built-in stores the centered rows as `b_flat_g` (K × G
    # column-major) and exposes `b_g` as their matrix view.
    ub = unconstrain(kb.layout, merge(base, (mu = w.mu, sigma = w.sigma,
        tau_g = w.d_sd, L_g = w.d_L, b_flat_g = vec(permutedims(w.d_c)),
        b_g = w.d_c)))
    for port in (:likelihood, :prior, :log_jacobian)
        @test _query(kw.spec, bw, port, uw) ≈ _query(kb.spec, bb, port, ub)
    end
end

@testset "stratified correlated: library entry and Distributions.jl oracle" begin
    b65, k65 = _lv_build("65_stratified_lib")
    u = _lv_point(k65.layout.total)
    nt = constrain(k65.layout, u)
    c = b65.columns
    mu = map(eachindex(c[:y])) do i
        si, gi = c[:b][i], c[:g][i]
        row = (Diagonal(nt.r_sd[si, :]) * nt.r_L[:, :, si]) * nt.r_z[gi, :]
        only(nt.mu) + row[1] + c[:x][i] * row[2]
    end
    @test _query(k65.spec, b65, :likelihood, u) ≈
        sum(logpdf.(Normal.(mu, nt.sigma), c[:y]))
    @test _query(k65.spec, b65, :prior, u) ≈ logpdf(Normal(0, 5),
        only(nt.mu)) + logpdf(Exponential(1), nt.sigma) +
        sum(_lv_halfnormal, nt.r_sd) +
        sum(_lv_lkj(nt.r_L[:, :, k], 1.0) for k in 1:2) +
        sum(logpdf.(Normal(0, 1), nt.r_z))
    _check_gradient(k65.spec, b65, u)
end
