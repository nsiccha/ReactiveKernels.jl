using DifferentiationInterface
using Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Ordinal latent scale + stage effects (thin-layer slice v12): a modeled
# (in-graph sampled) scale naming a LogLink predictor — positivity
# structural via `exp` — over the literal/column/stage machinery of the
# leveled slice. Admission, stage-effect shapes, default-absent
# equivalence, and value+gradient parity against the independent
# SB-math `_ref_ordinal` mirror (test_generator.jl). Helpers `_query`,
# `_check_gradient`, `_ref_ordinal`, `_none_evidence`, `_have`,
# `_bound_nt`, `_GEN_BACKEND` come from the earlier includes.

function _os_columns(n = 9)
    Dict{Symbol,AbstractVector}(
        :y => repeat([1, 2, 3], outer = cld(n, 3))[1:n],
        :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0, -0.25, 0.75, -1.25][1:n],
        :g => repeat([1, 2, 3], outer = cld(n, 3))[1:n],
        :z1 => collect(1.0:n) ./ n,
        :z2 => reverse(collect(1.0:n)) ./ n,
        :d => [0.5, 1.0, 1.5, 2.0, 1.0, 0.8, 1.2, 0.9, 1.1][1:n],
        :w => [1.0, 0.5, 2.0, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0][1:n],
    )
end

# Plan builder (binds internally): eta `:mu` over `:x`, optional modeled
# scale `:disc` (`scale_kind = :continuous` intercept + `:x`, or `:factor`
# intercept + grouping `:g` — the `log(disc) ~ group` recipe), optional
# weights and per-threshold design.
function _os_plan(; link = LogitLink, structure = :cumulative,
        discrimination = :disc, scale_link = LogLink,
        scale_kind = :continuous, weights = nothing,
        tcols = Symbol[], coefs = nothing, coefsize = nothing,
        scale_priors = true, cols = nothing)
    cols = cols === nothing ? _os_columns() : cols
    eta_terms = TermSpec[TermSpec(ContinuousTerm, [:x], NamedTuple(), :x,
        :x_term)]
    preds = PredictorSpec[PredictorSpec(:mu, IdentityLink, eta_terms, :mu)]
    prs = PopulationPrior[PopulationPrior(:mu, :x, 0.0, 2.0)]
    maps = LevelMap[]
    if discrimination === :disc
        if scale_kind === :continuous
            sterms = TermSpec[
                TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
                    :intercept),
                TermSpec(ContinuousTerm, [:x], NamedTuple(), :x, :x_term)]
            saddrs = [:Intercept, :x]
        else
            sterms = TermSpec[
                TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
                    :intercept),
                TermSpec(FactorTerm, [:g], NamedTuple(), :g, :g_term)]
            saddrs = [:Intercept, :g]
            levs = sort(unique(cols[:g]))
            maps = LevelMap[LevelMap(:disc, :g, levs[2:end], :levels,
                (2, :end))]
        end
        push!(preds, PredictorSpec(:disc, scale_link, sterms, :disc))
        scale_priors && append!(prs, PopulationPrior[
            PopulationPrior(:disc, a, 0.0, 1.0) for a in saddrs])
    end
    vfam = structure === :cumulative ? :ordered_normal : :vector_normal
    vecs = VectorParameter[VectorParameter(:y_thresholds, vfam,
        (arg1 = 0.0, arg2 = 1.0), nothing, :y_thresholds)]
    coefs === nothing || push!(vecs, VectorParameter(coefs, :vector_normal,
        (arg1 = 0.0, arg2 = 1.0), coefsize, coefs))
    r = LikelihoodSpec(OrdinalFam, link, :y, :mu, nothing, weights,
        _none_evidence(), :y_resp, nothing, nothing;
        thresholds = :y_thresholds, ordinal_structure = structure,
        discrimination = discrimination, threshold_columns = tcols,
        threshold_coefs = coefs)
    unbound = StructuralPlan([r], preds, prs, SampledParameter[],
        AssignmentSpec[], Dict{Symbol,AbstractVector}(), 0;
        vector_parameters = vecs, levelmaps = maps)
    return bind_data(unbound, cols)
end

# Enzyme posterior gradient (the `_check_gradient` kernel without the FD
# comparison — for equivalence checks between spellings).
function _os_gradient(spec, plan, u)
    kern = prepare(spec; have = _have(plan), want = :posterior,
        bound = _bound_nt(plan))
    prep = prepare_ad(kern, _GEN_BACKEND, u; active = :unconstrained)
    return ReactiveKernels.ad_value_and_gradient!(prep, similar(u), u)[2]
end

@testset "ordinal modeled-scale admission" begin
    # A log-link scale predictor admits on both structures × all links;
    # the scale predictor counts as used (no unused-predictor failure).
    for link in (LogitLink, ProbitLink, CloglogLink),
            structure in (:cumulative, :stopping)
        @test (validate_plan(_os_plan(; link = link,
            structure = structure)); true)
    end
    # Grouping scale (factor terms) admits.
    @test (validate_plan(_os_plan(; scale_kind = :factor)); true)
    # A modeled scale under any other link fails closed (the Identity
    # case keeps the leveled slice's predictor rejection).
    @test_throws ContractValidationError _os_plan(; scale_link = IdentityLink)
    @test_throws ContractValidationError _os_plan(; scale_link = LogitLink)
    # Unknown names still fail at bind.
    @test_throws ContractValidationError _os_plan(; discrimination = :nope)
    # A predictor that is also a data column is ambiguous, never silently
    # resolved either way.
    cols = _os_columns()
    cols[:disc] = fill(1.5, 9)
    @test_throws ContractValidationError _os_plan(; cols = cols)
    # Scale coefficients need priors like any predictor's.
    @test_throws ContractValidationError _os_plan(; scale_priors = false)
    # Modeled scale composes with per-threshold design (stopping).
    good = _os_plan(; structure = :stopping, tcols = [:z1, :z2],
        coefs = :y_beta)
    @test (validate_plan(good); true)
    # ... but cumulative + stage effects still refuses with a modeled scale.
    @test_throws ContractValidationError _os_plan(; structure = :cumulative,
        tcols = [:z1], coefs = :y_beta)
end

@testset "ordinal stage-effect shapes" begin
    # Coef size infers to (K−1)×p across shapes.
    for (K, tcols) in ((2, [:z1]), (3, [:z1, :z2]), (4, [:z1]),
            (4, [:z1, :z2]))
        cols = _os_columns()
        cols[:y] = repeat(1:K, outer = cld(9, K))[1:9]
        plan = _os_plan(; structure = :stopping, discrimination = nothing,
            tcols = tcols, coefs = :y_beta, cols = cols)
        @test (validate_plan(plan); true)
        @test plan.responses[1].n_levels == K
        @test plan.vector_parameters[2].size == (K - 1) * length(tcols)
    end
    # Explicit sizes assert both ways at bind.
    cols4 = _os_columns()
    cols4[:y] = repeat(1:4, outer = 3)[1:9]
    kw = (; structure = :stopping, discrimination = nothing,
        tcols = [:z1, :z2], coefs = :y_beta, cols = cols4)
    @test (validate_plan(_os_plan(; kw..., coefsize = 6)); true)
    @test_throws ContractValidationError _os_plan(; kw..., coefsize = 5)
    # Stage-major orientation pins at a second shape (p=1, K=4): stage j
    # reads beta[j].
    plan = _os_plan(; kw..., tcols = [:z1], coefsize = 3)
    built = build_kernel(plan)
    @test built.layout.total == 1 + 3 + 3 # eta + thresholds + coefs
    u = [0.4, 0.1, -0.2, 0.3, 0.05, -0.1, 0.15]
    nt = constrain(built.layout, u)
    b = only(Vector(nt.mu))
    eta = b .* plan.columns[:x]
    t = Vector(nt.y_thresholds)
    beta = Vector(nt.y_beta)
    E = [plan.columns[:z1][i] * beta[j] for i in 1:9, j in 1:3]
    ll = sum(_ref_ordinal(y, e, t, 1.0, :stopping, LogitLink, E[i, :])
        for (i, (y, e)) in enumerate(zip(plan.columns[:y], eta)))
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
end

@testset "ordinal default-absent equivalence" begin
    cols = _os_columns()
    base = _os_plan(; discrimination = nothing, cols = cols)
    lit = _os_plan(; discrimination = 1.0, cols = cols)
    onescols = copy(cols)
    onescols[:o] = ones(9)
    col = _os_plan(; discrimination = :o, cols = onescols)
    b0, b1, b2 = build_kernel(base), build_kernel(lit), build_kernel(col)
    u = [0.4, -0.2, 0.25]
    for want in (:likelihood, :prior, :posterior)
        v0 = _query(b0.spec, base, want, u)
        @test isapprox(_query(b1.spec, lit, want, u), v0;
            rtol = 1e-12, atol = 1e-12)
        @test isapprox(_query(b2.spec, col, want, u), v0;
            rtol = 1e-12, atol = 1e-12)
    end
    g0 = _os_gradient(b0.spec, base, u)
    @test isapprox(_os_gradient(b1.spec, lit, u), g0; rtol = 1e-9, atol = 1e-9)
    @test isapprox(_os_gradient(b2.spec, col, u), g0; rtol = 1e-9, atol = 1e-9)
    # Absent emits no scale precompute; modeled emits one explicit dotted
    # `exp.` over the scale predictor's lp node (human-readable, Enzyme-safe).
    src0 = sprint(show, kernel_expr(base, b0.layout))
    @test !occursin("_ppl_disc_", src0)
    scaled = _os_plan()
    srcs = sprint(show, kernel_expr(scaled, build_kernel(scaled).layout))
    @test occursin("exp.(_ppl_lp_disc)", srcs)
end

@testset "ordinal modeled-scale parity" begin
    # Cumulative + stopping × all links with an intercept + continuous
    # modeled scale, against the SB-math mirror at 1e-12.
    for link in (LogitLink, ProbitLink, CloglogLink),
            structure in (:cumulative, :stopping)
        plan = _os_plan(; link = link, structure = structure)
        built = build_kernel(plan)
        @test built.layout.total == 5 # eta + 2 scale + 2 thresholds
        u = [0.4, 0.1, -0.2, 0.25, -0.15]
        nt = constrain(built.layout, u)
        b = only(Vector(nt.mu))
        eta = b .* plan.columns[:x]
        a = Vector(nt.disc)
        d = exp.(a[1] .+ a[2] .* plan.columns[:x])
        t = Vector(nt.y_thresholds)
        ll = sum(_ref_ordinal(y, e, t, di, structure, link)
            for (y, e, di) in zip(plan.columns[:y], eta, d))
        got = _query(built.spec, plan, :likelihood, u)
        @test isapprox(got, ll; rtol = 1e-12, atol = 1e-12)
        _check_gradient(built.spec, plan, u)
    end
    # Full combination: stopping + modeled scale + per-threshold design +
    # weights.
    plan = _os_plan(; link = LogitLink, structure = :stopping, weights = :w,
        tcols = [:z1, :z2], coefs = :y_beta)
    built = build_kernel(plan)
    @test built.layout.total == 9 # eta + 2 scale + 2 thresholds + 4 coefs
    u = [0.4, 0.1, -0.2, 0.25, -0.15, 0.05, -0.1, 0.15, 0.2]
    nt = constrain(built.layout, u)
    b = only(Vector(nt.mu))
    eta = b .* plan.columns[:x]
    a = Vector(nt.disc)
    d = exp.(a[1] .+ a[2] .* plan.columns[:x])
    t = Vector(nt.y_thresholds)
    beta = Vector(nt.y_beta)
    X = hcat(plan.columns[:z1], plan.columns[:z2])
    E = [sum(X[i, c] * beta[(j-1)*2+c] for c in 1:2)
        for i in 1:9, j in 1:2]
    ll = sum(plan.columns[:w][i] *
        _ref_ordinal(y, eta[i], t, d[i], :stopping, LogitLink, E[i, :])
        for (i, y) in enumerate(plan.columns[:y]))
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
    # Grouping scale (the `log(disc) ~ group` recipe): intercept + factor.
    plan = _os_plan(; link = LogitLink, structure = :cumulative,
        scale_kind = :factor)
    built = build_kernel(plan)
    @test built.layout.total == 6 # eta + 3 scale + 2 thresholds
    u = [0.4, 0.1, -0.2, 0.3, 0.25, -0.15]
    nt = constrain(built.layout, u)
    b = only(Vector(nt.mu))
    eta = b .* plan.columns[:x]
    levs = sort(unique(plan.columns[:g]))
    C = Float64.(plan.columns[:g] .== permutedims(levs[2:end]))
    logd = hcat(ones(9), C) * Vector(nt.disc)
    d = exp.(logd)
    t = Vector(nt.y_thresholds)
    ll = sum(_ref_ordinal(y, e, t, di, :cumulative, LogitLink)
        for (y, e, di) in zip(plan.columns[:y], eta, d))
    @test isapprox(_query(built.spec, plan, :likelihood, u), ll;
        rtol = 1e-12, atol = 1e-12)
    _check_gradient(built.spec, plan, u)
    # K=1 with a modeled scale: zero-information likelihood, live gradient.
    cols1 = _os_columns(6)
    cols1[:y] = ones(Int, 6)
    plan = _os_plan(; cols = cols1)
    built = build_kernel(plan)
    @test built.layout.total == 3 # eta + 2 scale, zero thresholds
    u = [0.4, 0.1, -0.2]
    @test _query(built.spec, plan, :likelihood, u) == 0.0
    _check_gradient(built.spec, plan, u)
end
