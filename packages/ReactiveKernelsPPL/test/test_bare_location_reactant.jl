# Compiled (Reactant) checks for the cases in test_bare_location.jl.
using Reactant

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
# `ad_kw` reaches `compile_ad_value_and_gradient` (e.g. the §7n
# `optimize = :only_enzyme` pin).
function _bare_reactant(prog::Expr, cols::AbstractDict{Symbol};
        ad_kw...)
    plan = lower_rkppl(prog, cols; conditioned = cols)
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_bare_reactant_measure, built, bound, post_q, u;
        ad_kw...)
end

function _bare_reactant_measure(built, bound, post_q, u; ad_kw...)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u);
        ad_kw...)
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), native, primal, val, g,
        rval = Float64(rval), rgrad = Array(rgrad))
end

@testset "bare locations under Reactant" begin
    progs = [
        (Meta.parse("""begin
            theta ~ Beta(1.0, 1.0)
            k .~ Binomial.(n, theta)
        end"""), Dict{Symbol,AbstractVector}(:k => [3, 5], :n => [10, 12])),
        (Meta.parse("""begin
            lambda ~ Gamma(2.0, 1.0)
            y .~ Poisson.(lambda)
        end"""), Dict{Symbol,AbstractVector}(:y => [0, 1, 3, 5, 2])),
    ]
    for (prog, cols) in progs
        fx = _bare_reactant(prog, cols)
        @test fx.primal ≈ fx.native rtol = 1e-9
        @test fx.rval ≈ fx.val rtol = 1e-9
        @test fx.rgrad ≈ fx.g rtol = 1e-7 atol = 1e-9
    end
    # Data-length invariance (constraints.md): more rows must not
    # replicate the loop body.
    prog = Meta.parse("""begin
        theta ~ Beta(2.0, 2.0)
        y .~ Bernoulli.(theta)
    end""")
    small = _bare_reactant(prog, Dict{Symbol,AbstractVector}(:y => [0, 1]))
    large = _bare_reactant(prog,
        Dict{Symbol,AbstractVector}(:y => [0, 1, 1, 0, 1, 0]))
    @test small.lines == large.lines
end

@testset "value locations under Reactant" begin
    pois = _bare_reactant(Meta.parse("""begin
        a ~ Normal(0, 1)
        y .~ Poisson.(exp.(a))
    end"""), Dict{Symbol,AbstractVector}(:y => [1, 0, 3, 2]))
    @test pois.primal ≈ pois.native rtol = 1e-9
    @test pois.rval ≈ pois.val rtol = 1e-9
    @test pois.rgrad ≈ pois.g rtol = 1e-7 atol = 1e-9
    # A scalar Gaussian location with a sampled scale is the §7n trigger
    # shape (reactivekernels-use §7n, nsiccha/ReactiveKernels.jl#17): the
    # default pipeline's reverse counts each row's `-log(s)` adjoint once
    # (+2.0 on the log-scale coordinate at three rows), as it does for the
    # intercept-only `eta = mu` spelling. Correctness pins
    # `optimize = :only_enzyme`; the default pipeline rides `@test_broken`
    # (the test_prior_vocab.jl ladder) — drop both when upstream is fixed.
    cols = Dict{Symbol,AbstractVector}(:y => [0.3, -1.2, 2.1])
    gau = _bare_reactant(_VALUE_NORMAL, cols; optimize = :only_enzyme)
    @test gau.primal ≈ gau.native rtol = 1e-9
    @test gau.rval ≈ gau.val rtol = 1e-9
    @test gau.rgrad ≈ gau.g rtol = 1e-7 atol = 1e-9
    gdef = _bare_reactant(_VALUE_NORMAL, cols)
    @test_broken gdef.rgrad ≈ gdef.g rtol = 1e-7 atol = 1e-9
    # Data-length invariance (constraints.md): more rows must not
    # replicate the broadcast location.
    small = _bare_reactant(_VALUE_NORMAL,
        Dict{Symbol,AbstractVector}(:y => [0.3, -1.2]))
    large = _bare_reactant(_VALUE_NORMAL,
        Dict{Symbol,AbstractVector}(:y => [0.3, -1.2, 2.1, 0.7, 1.1, -0.4]))
    @test small.lines == large.lines
end
