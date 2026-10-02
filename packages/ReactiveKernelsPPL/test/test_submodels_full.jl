using Distributions
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Full submodels: nested calls, indexed / sized priors, `@plate` / `@scan` /
# construct statements and data / vector arguments inside bodies. Expansion
# is transparent: every submodel program here lowers to a plan whose
# canonical serialization (`_canon`, test_corpus.jl) is byte-identical to its
# hand-inlined twin, written out with the documented namespacing rule
# (`RKPPLSubmodel`). Helpers from earlier includes: `_canon`
# (test_corpus.jl), `_plans_equal` (test_surface.jl), `_gen_columns`,
# `_query`, `_check_gradient` (test_generator.jl). Toy data only.

const _SMF = @__MODULE__

_smf_canon(plan) = sprint(_canon, plan)

# The lowering outcome: the canonical plan, or the error text when lowering
# refuses. A twin pair is transparent when both outcomes are equal — the
# submodel neither adds nor removes a refusal the hand-inlined program has.
function _smf_outcome(ast, data; mod = _SMF)
    try
        return _smf_canon(lower_rkppl(ast, data; mod = mod))
    catch e
        e isa SurfaceLoweringError || rethrow()
        return "SurfaceLoweringError: " * e.message
    end
end

_smf_expand(ast, data = Set{Symbol}()) =
    first(ReactiveKernelsPPL._expand_submodels(ast, Set{Symbol}(data), _SMF))

_smf_stmts(ex) = Any[a for a in ex.args if !(a isa LineNumberNode)]

_smf_errmsg(f) = try
    f()
    ""
catch e
    e isa SurfaceLoweringError || rethrow()
    e.message
end

# ── Fixtures ─────────────────────────────────────────────────────────────
@rkppl smf_inner(xx) = begin
    b ~ Normal(0, 1)
    return b .* xx
end
@rkppl smf_outer(xx) = begin
    e ~ smf_inner(xx)
    return e
end
@rkppl smf_top(xx) = begin
    w ~ smf_outer(xx)
    v ~ smf_inner(xx)
    return w .+ v
end
@rkppl smf_cranef(gg) = begin
    sg ~ HalfNormal(1)
    c[levels(gg)] .~ Normal.(0, sg)
    return c[gg]
end
@rkppl smf_linear(X) = begin
    b[axes(X, 2)] .~ Normal.(0, 1)
    return X * b
end
@rkppl smf_ncp(z, gg, sd) = begin
    zg = z[gg]
    return sd .* zg
end
@rkppl smf_hier_obs(yy, xx, s) = begin
    tau ~ Exponential(1)
    @plate for i in eachindex(yy)
        theta[i] ~ Normal(xx[i], tau)
        yy[i] ~ Normal.(theta[i], s)
    end
    return tau
end
@rkppl smf_ar1(phi, s, n) = begin
    @scan begin
        h[1] ~ Normal(0, 1)
        for t in 2:n
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end
    return h
end
@rkppl smf_ranef(gg) = begin
    r ~ varying_effect(gg, [1])
    return r
end
@rkppl smf_spline(xx) = begin
    spline_basis(:s_in, xx; k = 4)
    return spline(:s_in)
end
@rkppl smf_r2d2(mu, R2, phi) = begin
    r2d2(mu, R2, phi)
    return mu
end
@rkppl smf_kw(xx, k) = begin
    spline_basis(:s, xx; k, kind = :tps)
    return spline(:s)
end
@rkppl smf_rec_a(x) = begin
    v ~ smf_rec_b(x)
    return v
end
@rkppl smf_rec_b(x) = begin
    v ~ smf_rec_a(x)
    return v
end
@rkppl smf_head(f) = begin
    v = f(1.0)
    return v
end
@rkppl smf_argbind(x) = begin
    x ~ Normal(0, 1)
    return x
end
@rkppl smf_argdef(x) = begin
    x = 1.0
    return x
end
@rkppl smf_free(xx) = begin
    b ~ Normal(0, 1)
    return b .* xx .+ z_b
end
@rkppl smf_stream(x) = begin
    a ~ Normal(0, 5)
    s ~ Exponential(1)
    eta = a .+ x
    slot .~ Normal.(eta, s)
    return slot
end
@rkppl smf_pinned(x) = begin
    yy ~ smf_stream(x; predictor = mu)
    return yy
end
@rkppl smf_bc(xx) = begin
    b_c ~ Normal(0, 1)
    return b_c .* xx
end
@rkppl smf_c(xx) = begin
    c ~ Normal(0, 1)
    return c .* xx
end
@rkppl smf_pcs_ncp(m, t) = begin
    z ~ Normal(0, 1)
    return m .+ t .* z
end
@rkppl smf_pcs_outer(m, t) = begin
    w ~ smf_pcs_ncp(m, t)
    return w
end
@rkppl smf_pcs_plate(m) = begin
    @plate for i in 1:3
        q[i] ~ Normal(m, 1)
    end
    return m
end

@testset "full submodels: namespacing rule (expanded AST)" begin
    # Nested: names compose under each use-site LHS, inner first substituted
    # by the enclosing body (`z ~ smf_top` → `w ~ smf_outer` → `e ~ smf_inner`
    # → `b` gives `z_w_e_b`); the same inner twice namespaces apart.
    got = _smf_stmts(_smf_expand(quote
        z ~ smf_top(x)
    end, (:x,)))
    @test got == _smf_stmts(quote
        z_w_e_b ~ Normal(0, 1)
        z_w_e = z_w_e_b .* x
        z_w = z_w_e
        z_v_b ~ Normal(0, 1)
        z_v = z_v_b .* x
        z = z_w .+ z_v
    end)

    # Indexed prior: the base name namespaces, the index keeps its shape and
    # takes the argument (the `_ns(::Symbol, ::Expr)` crash).
    got = _smf_stmts(_smf_expand(quote
        r ~ smf_cranef(g)
    end, (:g,)))
    @test got == _smf_stmts(quote
        r_sg ~ HalfNormal(1)
        r_c[levels(g)] .~ Normal.(0, r_sg)
        r = r_c[g]
    end)

    # `@plate` / `@scan`: cells, loop variables and carried states are body
    # binders; an observed data argument is substituted, not namespaced.
    got = _smf_stmts(_smf_expand(quote
        t ~ smf_hier_obs(y, x, sigma)
    end, (:y, :x)))
    @test length(got) == 3
    @test got[1] == :(t_tau ~ Exponential(1))
    plate = got[2]
    @test plate.head === :macrocall && plate.args[1] === Symbol("@plate")
    loop = plate.args[end]
    @test loop.args[1] == :(t_i = eachindex(y))
    @test _smf_stmts(loop.args[2]) == _smf_stmts(quote
        t_theta[t_i] ~ Normal(x[t_i], t_tau)
        y[t_i] ~ Normal.(t_theta[t_i], sigma)
    end)
    @test got[3] == :(t = t_tau)
    got = _smf_stmts(_smf_expand(quote
        a ~ smf_ar1(phi, s, T)
    end, (:y,)))
    scan = got[1]
    @test scan.args[1] === Symbol("@scan")
    blk = _smf_stmts(scan.args[end])
    @test blk[1] == :(a_h[1] ~ Normal(0, 1))
    @test blk[2].head === :for && blk[2].args[1] == :(a_t = 2:T)
    @test _smf_stmts(blk[2].args[2]) ==
        [:(a_h[a_t] ~ Normal(phi * a_h[a_t - 1], s))]
    @test got[2] == :(a = a_h)

    # A declared basis id is a body name (declaration and use); keyword names
    # and other quoted symbols are untouched; a bare keyword shorthand whose
    # name is an argument expands to `k = <argument>`.
    got = _smf_stmts(_smf_expand(quote
        f ~ smf_kw(x, 4)
    end, (:x,)))
    @test got == _smf_stmts(quote
        spline_basis(:f_s, x; k = 4, kind = :tps)
        f = spline(:f_s)
    end)

    # Per-cell nesting composes per cell: a direct-bound slot that is itself a
    # per-cell submodel call expands in turn.
    got = _smf_expand(Expr(:block,
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, :(i = eachindex(y)),
                Expr(:block, :(theta[i] ~ smf_pcs_outer(mu, tau)))))))
    cells = _smf_stmts(got.args[1].args[end].args[2])
    @test cells == [:(theta_z[i] ~ Normal(0, 1)),
        :(theta[i] = mu .+ tau .* theta_z[i])]
end

@testset "full submodels: plans equal their hand-inlined twins" begin
    D = (:y, :x, :g, :x2)
    # Probe S4 (brief f4o1yk): a nested submodel.
    sub = quote
        a ~ Normal(0, 5)
        ee ~ smf_outer(x)
        sigma ~ Exponential(1)
        mu = a .+ ee
        y .~ Normal.(mu, sigma)
    end
    twin = quote
        a ~ Normal(0, 5)
        ee_e_b ~ Normal(0, 1)
        ee_e = ee_e_b .* x
        ee = ee_e
        sigma ~ Exponential(1)
        mu = a .+ ee
        y .~ Normal.(mu, sigma)
    end
    @test _smf_canon(lower_rkppl(sub, D; mod = _SMF)) ==
        _smf_canon(lower_rkppl(twin, D))

    # Probe S7: a centered random intercept with an indexed prior in the body.
    sub7 = quote
        r ~ smf_cranef(g)
        sigma ~ Exponential(1)
        mu = r
        y .~ Normal.(mu, sigma)
    end
    twin7 = quote
        r_sg ~ HalfNormal(1)
        r_c[levels(g)] .~ Normal.(0, r_sg)
        r = r_c[g]
        sigma ~ Exponential(1)
        mu = r
        y .~ Normal.(mu, sigma)
    end
    p7 = lower_rkppl(sub7, D; mod = _SMF)
    @test _smf_canon(p7) == _smf_canon(lower_rkppl(twin7, D))
    @test _plans_equal(p7, lower_rkppl(twin7, D))

    # Three-level nesting, the same inner twice.
    sub3 = quote
        a ~ Normal(0, 5)
        z ~ smf_top(x)
        sigma ~ Exponential(1)
        mu = a .+ z
        y .~ Normal.(mu, sigma)
    end
    twin3 = quote
        a ~ Normal(0, 5)
        z_w_e_b ~ Normal(0, 1)
        z_w_e = z_w_e_b .* x
        z_w = z_w_e
        z_v_b ~ Normal(0, 1)
        z_v = z_v_b .* x
        z = z_w .+ z_v
        sigma ~ Exponential(1)
        mu = a .+ z
        y .~ Normal.(mu, sigma)
    end
    @test _smf_outcome(sub3, D) == _smf_outcome(twin3, D)

    # A sized prior over a design matrix argument.
    subx = quote
        X = hcat(x, x2)
        a ~ Normal(0, 5)
        eta ~ smf_linear(X)
        sigma ~ Exponential(1)
        mu = a .+ eta
        y .~ Normal.(mu, sigma)
    end
    twinx = quote
        X = hcat(x, x2)
        a ~ Normal(0, 5)
        eta_b[axes(X, 2)] .~ Normal.(0, 1)
        eta = X * eta_b
        sigma ~ Exponential(1)
        mu = a .+ eta
        y .~ Normal.(mu, sigma)
    end
    @test _smf_outcome(subx, D) == _smf_outcome(twinx, D)

    # A vector-parameter argument and a data column passed through.
    subv = quote
        a ~ Normal(0, 5)
        sg ~ HalfNormal(1)
        z[levels(g)] .~ Normal.(0, 1)
        r ~ smf_ncp(z, g, sg)
        sigma ~ Exponential(1)
        mu = a .+ r
        y .~ Normal.(mu, sigma)
    end
    twinv = quote
        a ~ Normal(0, 5)
        sg ~ HalfNormal(1)
        z[levels(g)] .~ Normal.(0, 1)
        r_zg = z[g]
        r = sg .* r_zg
        sigma ~ Exponential(1)
        mu = a .+ r
        y .~ Normal.(mu, sigma)
    end
    @test _smf_outcome(subv, D) == _smf_outcome(twinv, D)

    # `@plate` inside a body, observing a data argument.
    subp = quote
        sigma ~ Exponential(1)
        t ~ smf_hier_obs(y, x, sigma)
    end
    twinp = Expr(:block, :(sigma ~ Exponential(1)), :(t_tau ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, :(t_i = eachindex(y)), Expr(:block,
                :(t_theta[t_i] ~ Normal(x[t_i], t_tau)),
                :(y[t_i] ~ Normal.(t_theta[t_i], sigma))))),
        :(t = t_tau))
    pp = lower_rkppl(subp, D; mod = _SMF)
    @test _smf_canon(pp) == _smf_canon(lower_rkppl(twinp, D))

    # `@scan` inside a body.
    subs = quote
        phi ~ Normal(0, 1)
        s ~ Exponential(1)
        sigma ~ Exponential(1)
        a ~ smf_ar1(phi, s, T)
        y .~ Normal.(a, sigma)
    end
    twins = quote
        phi ~ Normal(0, 1)
        s ~ Exponential(1)
        sigma ~ Exponential(1)
        @scan begin
            a_h[1] ~ Normal(0, 1)
            for a_t in 2:T
                a_h[a_t] ~ Normal(phi * a_h[a_t - 1], s)
            end
        end
        a = a_h
        y .~ Normal.(a, sigma)
    end
    @test _smf_outcome(subs, (:y,)) == _smf_outcome(twins, (:y,))

    # Construct statements inside bodies (probes S3, S5, S2) and probe S6
    # (random intercept beside a population intercept): each expands to its
    # hand-inlined twin, so it lowers — or is refused — exactly as the twin.
    for (sub, twin) in (
        (quote
            a ~ Normal(0, 5)
            rr ~ smf_ranef(g)
            sigma ~ Exponential(1)
            mu = a .+ rr
            y .~ Normal.(mu, sigma)
        end, quote
            a ~ Normal(0, 5)
            rr_r ~ varying_effect(g, [1])
            rr = rr_r
            sigma ~ Exponential(1)
            mu = a .+ rr
            y .~ Normal.(mu, sigma)
        end),
        (quote
            a ~ Normal(0, 5)
            f ~ smf_spline(x)
            sigma ~ Exponential(1)
            mu = a .+ f
            y .~ Normal.(mu, sigma)
        end, quote
            a ~ Normal(0, 5)
            spline_basis(:f_s_in, x; k = 4)
            f = spline(:f_s_in)
            sigma ~ Exponential(1)
            mu = a .+ f
            y .~ Normal.(mu, sigma)
        end),
        (quote
            a ~ Normal(0, 5)
            b1 ~ Normal(0, 1)
            R2 ~ Beta(1.0, 1.0)
            phi ~ Dirichlet([1.0])
            m0 = a .+ b1 .* x
            m ~ smf_r2d2(m0, R2, phi)
            sigma ~ Exponential(1)
            y .~ Normal.(m, sigma)
        end, quote
            a ~ Normal(0, 5)
            b1 ~ Normal(0, 1)
            R2 ~ Beta(1.0, 1.0)
            phi ~ Dirichlet([1.0])
            m0 = a .+ b1 .* x
            r2d2(m0, R2, phi)
            m = m0
            sigma ~ Exponential(1)
            y .~ Normal.(m, sigma)
        end),
        (quote
            a ~ Normal(0, 5)
            r ~ smf_cranef(g)
            sigma ~ Exponential(1)
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, quote
            a ~ Normal(0, 5)
            r_sg ~ HalfNormal(1)
            r_c[levels(g)] .~ Normal.(0, r_sg)
            r = r_c[g]
            sigma ~ Exponential(1)
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end))
        out = _smf_outcome(sub, D)
        @test out == _smf_outcome(twin, D)
        # The refusal, if any, is the construct's own — never the expansion.
        @test !occursin("submodel", out)
    end
    # Probe S6 lowers: a hierarchical (parameter-scale) prior identifies an
    # intercept beside a full-cover factor (fallback lane; the density
    # oracle is in test_fallback.jl).
    s6 = _smf_outcome(quote
        a ~ Normal(0, 5)
        r ~ smf_cranef(g)
        sigma ~ Exponential(1)
        mu = a .+ r
        y .~ Normal.(mu, sigma)
    end, D)
    @test !startswith(s6, "SurfaceLoweringError")

    # Per-cell: a nested per-cell call equals the direct per-cell program.
    pc(cell) = Expr(:block,
        :(mu ~ Normal(0, 5)), :(sigma ~ Exponential(1)), :(tau ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(5),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block, cell, :(y[i] ~ Normal.(theta[i], sigma))))))
    @test _smf_canon(lower_rkppl(pc(:(theta[i] ~ smf_pcs_outer(mu, tau))),
            (:y, :x); mod = _SMF)) ==
        _smf_canon(lower_rkppl(pc(:(theta[i] ~ smf_pcs_ncp(mu, tau))),
            (:y, :x); mod = _SMF))
end

@testset "full submodels: hierarchical density vs Distributions.jl" begin
    # Probe S7 end to end: y ~ Normal(c[g], sigma), c ~ Normal(0, sg),
    # sg ~ HalfNormal(1), sigma ~ Exponential(1).
    cols, _ = _gen_columns()
    m = @rkppl begin
        r ~ smf_cranef(g)
        sigma ~ Exponential(1)
        mu = r
        y .~ Normal.(mu, sigma)
    end
    bound = m(; y = cols[:y], g = cols[:g])
    built = build_kernel(bound)
    ng = length(unique(cols[:g]))
    u = collect(range(-0.4, 0.5; length = built.layout.total))
    nt = constrain(built.layout, u)
    sg, sigma = nt.r_sg, nt.sigma
    c = Vector(getproperty(nt, only(k for k in keys(nt)
        if k ∉ (:r_sg, :sigma))))
    @test length(c) == ng
    lev = sort(unique(cols[:g]))
    cg = c[indexin(cols[:g], lev)]
    ll = sum(logpdf.(Normal.(cg, sigma), cols[:y]))
    pr = logpdf(truncated(Normal(0, 1), 0, Inf), sg) +
        sum(logpdf.(Normal(0, sg), c)) + logpdf(Exponential(1), sigma)
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :posterior, u) ≈
        ll + pr + log(sg) + log(sigma)
    _check_gradient(built.spec, bound, u)

    # The per-cell hierarchical body observing a data argument.
    mp = @rkppl begin
        sigma ~ Exponential(1)
        t ~ smf_hier_obs(y, x, sigma)
    end
    bp = mp(; y = cols[:y], x = cols[:x])
    builtp = build_kernel(bp)
    up = collect(range(-0.3, 0.6; length = builtp.layout.total))
    ntp = constrain(builtp.layout, up)
    theta = Vector(ntp.t_theta)
    llp = sum(logpdf.(Normal.(theta, ntp.sigma), cols[:y]))
    prp = logpdf(Exponential(1), ntp.sigma) + logpdf(Exponential(1), ntp.t_tau) +
        sum(logpdf.(Normal.(cols[:x], ntp.t_tau), theta))
    @test _query(builtp.spec, bp, :likelihood, up) ≈ llp
    @test _query(builtp.spec, bp, :posterior, up) ≈
        llp + prp + log(ntp.sigma) + log(ntp.t_tau)
    _check_gradient(builtp.spec, bp, up)
end

@testset "full submodels: hygiene fails closed" begin
    D = (:y, :x)
    # A namespaced local that the program already binds.
    msg = _smf_errmsg(() -> lower_rkppl(quote
        z_b ~ Normal(0, 1)
        z ~ smf_inner(x)
        y .~ Normal.(z .+ z_b, 1.0)
    end, D; mod = _SMF))
    @test occursin("`z_b`", msg) && occursin("already a name in the program", msg)
    # ... or that is a data column.
    msg = _smf_errmsg(() -> lower_rkppl(quote
        z ~ smf_inner(x)
        y .~ Normal.(z, 1.0)
    end, (:y, :x, :z_b); mod = _SMF))
    @test occursin("`z_b`", msg)
    # ... or that another expansion introduced (`a`'s local `b_c` and
    # `a_b`'s local `c` both namespace to `a_b_c`).
    msg = _smf_errmsg(() -> lower_rkppl(quote
        a ~ smf_bc(x)
        a_b ~ smf_c(x)
        y .~ Normal.(a .+ a_b, 1.0)
    end, D; mod = _SMF))
    @test occursin("`a_b_c`", msg) && occursin("another submodel expansion", msg)
    # ... or that appears in the call's arguments.
    msg = _smf_errmsg(() -> lower_rkppl(quote
        z ~ smf_inner(z_b)
        y .~ Normal.(z, 1.0)
    end, (:y, :z_b); mod = _SMF))
    @test occursin("`z_b`", msg)
    # ... or that is a free name of the body (it would be captured).
    msg = _smf_errmsg(() -> lower_rkppl(quote
        z ~ smf_free(x)
        y .~ Normal.(z, 1.0)
    end, D; mod = _SMF))
    @test occursin("free name", msg)
    # An argument called as a function.
    msg = _smf_errmsg(() -> lower_rkppl(quote
        v ~ smf_head(exp)
        y .~ Normal.(v, 1.0)
    end, D; mod = _SMF))
    @test occursin("called as a function", msg)
    # A body binding an argument name: a parameter declaration through an
    # argument, and a redefinition.
    msg = _smf_errmsg(() -> lower_rkppl(quote
        v ~ smf_argbind(w)
        y .~ Normal.(v, 1.0)
    end, D; mod = _SMF))
    @test occursin("both an argument and a local", msg)
    msg = _smf_errmsg(() -> lower_rkppl(quote
        v ~ smf_argdef(x)
        y .~ Normal.(v, 1.0)
    end, D; mod = _SMF))
    @test occursin("both an argument and a local", msg)
    # Recursion.
    msg = _smf_errmsg(() -> lower_rkppl(quote
        v ~ smf_rec_a(x)
        y .~ Normal.(v, 1.0)
    end, D; mod = _SMF))
    @test occursin("calls itself", msg)
    # A `predictor =` pin inside a body.
    msg = _smf_errmsg(() -> lower_rkppl(quote
        v ~ smf_pinned(x)
        y .~ Normal.(v, 1.0)
    end, D; mod = _SMF))
    @test occursin("top-level use site only", msg)
    # A per-cell body holds scalar statements only.
    msg = _smf_errmsg(() -> lower_rkppl(Expr(:block,
        :(sigma ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, :(i = eachindex(y)), Expr(:block,
                :(theta[i] ~ smf_pcs_plate(0.0)),
                :(y[i] ~ Normal.(theta[i], sigma)))))), D; mod = _SMF))
    @test occursin("called per cell", msg)
end
