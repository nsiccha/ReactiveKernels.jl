# Ranef Stage A: bucket IR + surface lowering + validation (acceptance item 1
# of the ranef todo). Codegen is NOT Stage A — `build_kernel` fails closed
# (tested below); K=1 geometry lands in Stage B, LKJ-correlated in Stage C.

function _rbucket(;
        id::Union{Nothing,Symbol} = :ID, group::Symbol = :g,
        kind::Symbol = :correlated,
        margins::Vector{RanefMargin} = RanefMargin[
            RanefMargin(:mu, :Intercept, RanefZRecipe(:ones, :none, nothing)),
            RanefMargin(:mu, :x, RanefZRecipe(:column, :x, nothing))],
        slices::Vector{Tuple{Symbol,UnitRange{Int}}} = [(:mu, 1:2)],
        lkj_eta::Float64 = 1.0,
        label::Symbol = id === nothing ? Symbol("bucket_" * string(group)) :
            Symbol("bucket_" * string(id) * "_" * string(group)))
    return RanefBucket(id, group, kind, margins, slices, lkj_eta, label)
end

function _rgather(predictor::Symbol, id::Union{Nothing,Symbol}, group::Symbol)
    suffix = id === nothing ? string(group) : string(id) * "_" * string(group)
    label = Symbol("r_$(predictor)_" * suffix)
    return TermSpec(RanefGatherTerm, [group],
        (bucket_id = id, bucket_group = group), label, label)
end

# Base GLM plan to inject buckets into for contract-validator tests.
function _rbase(; with_gather::Bool = false)
    plan = lower_rkppl(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 2)
            sigma ~ Exponential(1)
            mu = a .+ b .* x
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :g))
    if with_gather
        pred = plan.predictors[1]
        terms = vcat(pred.terms, _rgather(:mu, :ID, :g))
        preds = PredictorSpec[pred for pred in plan.predictors]
        preds[1] = PredictorSpec(pred.name, pred.link, terms, pred.label)
        buckets = RanefBucket[_rbucket()]
        return StructuralPlan(plan.responses, preds, plan.population_priors,
            plan.parameters, plan.assignments, plan.columns, plan.n_obs;
            roles = plan.roles, derived = plan.derived,
            levelmaps = plan.levelmaps,
            plate_parameters = plan.plate_parameters, scans = plan.scans,
            ranef_buckets = buckets)
    end
    return plan
end

@testset "ranef bucket lowering ID" begin
    got = lower_rkppl(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 2)
            sigma ~ Exponential(1)
            mu = a .+ b .* x .+ ranef(:ID, g)
            y .~ Normal.(mu, sigma)
            ranef_bucket(:ID, g; eta = 1.0) do
                mu => [1, x]
            end
        end, (:y, :x, :g))
    @test _buckets_equal(got.ranef_buckets, RanefBucket[_rbucket()])
    t = last(got.predictors[1].terms)
    @test t.kind === RanefGatherTerm
    @test t.columns == [:g]
    @test t.options == (bucket_id = :ID, bucket_group = :g)
    @test t.addressee === t.label === :r_mu_ID_g
    # Multi-target slices partition 1:K in body order.
    multi = lower_rkppl(quote
            mu = a .+ ranef(:ID, g)
            sg = c .+ ranef(:ID, g)
            y .~ Normal.(mu, 1.5)
            y2 .~ Normal.(sg, 1.5)
            ranef_bucket(:ID, g) do
                mu => [1, x]
                sg => [1]
            end
        end, (:y, :y2, :g, :x))
    b = only(multi.ranef_buckets)
    @test b.slices == [(:mu, 1:2), (:sg, 3:3)]
    @test [m.predictor for m in b.margins] == [:mu, :mu, :sg]
    @test b.lkj_eta == 1.0
end

@testset "ranef bucket lowering plain K=1" begin
    one = lower_rkppl(quote
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1]
            end
        end, (:y, :g))
    b = only(one.ranef_buckets)
    @test b.kind === :intercept1
    @test isnan(b.lkj_eta)
    @test b.id === nothing
    @test b.label === :bucket_g
    slope = lower_rkppl(quote
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [x]
            end
        end, (:y, :x, :g))
    bs = only(slope.ranef_buckets)
    @test bs.kind === :slope1
    @test isnan(bs.lkj_eta)
    # K=1 ID bucket stays correlated (SB's ID path has no K=1 fast path).
    kid = lower_rkppl(quote
            mu = a .+ ranef(:ID, g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(:ID, g; eta = 2.0) do
                mu => [1]
            end
        end, (:y, :g))
    bk = only(kid.ranef_buckets)
    @test bk.kind === :correlated
    @test bk.lkj_eta == 2.0
end

@testset "ranef dummy margins" begin
    got = lower_rkppl(quote
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1, dummy(c, 2), dummy(s, "a")]
            end
        end, (:y, :c, :s, :g))
    b = only(got.ranef_buckets)
    @test b.kind === :correlated
    @test _margins_equal([b.margins[2]],
        [RanefMargin(:mu, :c_dummy_2, RanefZRecipe(:dummy, :c, 2))])
    @test b.margins[3].coefficient === :s_dummy_a
    @test b.margins[3].z.level == "a"
end

@testset "ranef surface failures" begin
    D = (:y, :x, :g)
    # Unquoted id reads as a data column.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .+ ranef(ID, g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(ID, g) do
                mu => [1]
            end
        end, D)
    # Non-data group.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .+ ranef(h)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(h) do
                mu => [1]
            end
        end, D)
    # Eta on K=1 plain.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g; eta = 1.0) do
                mu => [1]
            end
        end, D)
    # Non-positive eta.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g; eta = 0.0) do
                mu => [1, x]
            end
        end, D)
    # Duplicate bucket.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1]
            end
            ranef_bucket(g) do
                mu => [1]
            end
        end, D)
    # Duplicate target in one body.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1]
                mu => [x]
            end
        end, D)
    # Bad margin integer / unknown margin / tuple-not-vect / bare margin.
    for bad in (:(mu => [2]), :(mu => [zz]), :(mu => (1, x)), :(mu => 1))
        @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
                :(mu = a .+ ranef(g)), :(y .~ Normal.(mu, 1.5)),
                Expr(:do, :(ranef_bucket(g)), Expr(:(->), Expr(:tuple),
                        Expr(:block, bad)))), D)
    end
    # Unknown bucket gather / gather without slice.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .+ ranef(:ZZ, g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(:ID, g) do
                mu => [1, x]
            end
        end, D)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .+ b .* x
            sg = c .+ ranef(:ID, g)
            y .~ Normal.(mu, 1.5)
            y2 .~ Normal.(sg, 1.5)
            ranef_bucket(:ID, g) do
                mu => [1, x]
            end
        end, (:y, :y2, :x, :g))
    # Negated / nested gathers.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .- ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1]
            end
        end, D)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .+ ranef(g) .* x
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1]
            end
        end, D)
    # Gather inside a non-predictor definition.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            w = ranef(g) .+ 1.0
            mu = a .+ b .* x
            y .~ Normal.(mu, sigma)
            sigma ~ Exponential(1)
            ranef_bucket(g) do
                mu => [1]
            end
        end, (:y, :x, :g))
    # Gather inside a STRUCTURAL definition (coefficient reference inlines
    # even vector defs — the statement-level screen must catch it before
    # absorption silently turns the alias into a direct summand).
    @test_throws SurfaceLoweringError lower_rkppl(quote
            w = ranef(g) .+ b .* x
            mu = a .+ w
            y .~ Normal.(mu, sigma)
            sigma ~ Exponential(1)
            ranef_bucket(g) do
                mu => [1]
            end
        end, (:y, :x, :g))
    # Reserved names.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            ranef = 1.0
            y .~ Normal.(mu, 1.5)
        end, (:y,))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            dummy ~ Normal(0, 1)
            y .~ Normal.(mu, 1.5)
        end, (:y,))
end

@testset "ranef contract validation" begin
    base = _rbase()
    # Happy path validates.
    @test validate_structure(_rbase(; with_gather = true)) === nothing
    # Duplicate keys / bad kind / non-partition slices / margin mismatch.
    dup = _rbase(; with_gather = true)
    push!(dup.ranef_buckets, _rbucket())
    @test_throws ContractValidationError validate_structure(dup)
    badkind = _rbase(; with_gather = true)
    badkind.ranef_buckets[1] = RanefBucket(:ID, :g, :slope1,
        _rbucket().margins, [(:mu, 1:2)], NaN, :bucket_ID_g)
    @test_throws ContractValidationError validate_structure(badkind)
    badslice = _rbase(; with_gather = true)
    badslice.ranef_buckets[1] = RanefBucket(:ID, :g, :correlated,
        _rbucket().margins, [(:mu, 1:1)], 1.0, :bucket_ID_g)
    @test_throws ContractValidationError validate_structure(badslice)
    # Gather of unknown bucket / duplicate gather / dangling slice.
    nogather = _rbase()
    pred = nogather.predictors[1]
    preds = PredictorSpec[pred for pred in nogather.predictors]
    preds[1] = PredictorSpec(pred.name, pred.link,
        vcat(pred.terms, _rgather(:mu, :ZZ, :g)), pred.label)
    orphan = StructuralPlan(nogather.responses, preds,
        nogather.population_priors, nogather.parameters, nogather.assignments,
        nogather.columns, nogather.n_obs; roles = nogather.roles,
        derived = nogather.derived, levelmaps = nogather.levelmaps,
        plate_parameters = nogather.plate_parameters, scans = nogather.scans,
        ranef_buckets = RanefBucket[_rbucket()])
    @test_throws ContractValidationError validate_structure(orphan)
    dangling = _rbase()
    dangling2 = StructuralPlan(dangling.responses, dangling.predictors,
        dangling.population_priors, dangling.parameters, dangling.assignments,
        dangling.columns, dangling.n_obs; roles = dangling.roles,
        derived = dangling.derived, levelmaps = dangling.levelmaps,
        plate_parameters = dangling.plate_parameters, scans = dangling.scans,
        ranef_buckets = RanefBucket[_rbucket()])
    @test_throws ContractValidationError validate_structure(dangling2)
end

@testset "ranef bind validation" begin
    cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5],
        :x => [0.5, -1.0, 1.5, 0.0], :g => [1, 2, 1, 2])
    plan = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1]
            end
        end, (:y, :x, :g))
    bound = bind_data(plan, cols)
    @test bound.roles[:g] === :group
    # Group column missing.
    @test_throws ContractValidationError bind_data(plan,
        Dict{Symbol,AbstractVector}(:y => cols[:y], :x => cols[:x]))
    # Non-numeric continuous Z.
    strz = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [s]
            end
        end, (:y, :s, :g))
    @test_throws ContractValidationError bind_data(strz,
        Dict{Symbol,AbstractVector}(:y => cols[:y],
            :s => ["a", "b", "a", "b"], :g => cols[:g]))
    # Dummy membership ok + fail.
    dym = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [dummy(s, "a")]
            end
        end, (:y, :s, :g))
    bound2 = bind_data(dym, Dict{Symbol,AbstractVector}(:y => cols[:y],
        :s => ["a", "b", "a", "b"], :g => cols[:g]))
    @test isbound(bound2)
    dymbad = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [dummy(s, "z")]
            end
        end, (:y, :s, :g))
    @test_throws ContractValidationError bind_data(dymbad,
        Dict{Symbol,AbstractVector}(:y => cols[:y],
            :s => ["a", "b", "a", "b"], :g => cols[:g]))
end

@testset "ranef Stage-B gate" begin
    cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5],
        :x => [0.5, -1.0, 1.5, 0.0], :g => [1, 2, 1, 2])
    # K=1 buckets build (intercept + slope); LKJ-correlated still refuses.
    iplan = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1]
            end
        end, (:y, :x, :g))
    ibuilt = build_kernel(bind_data(iplan, cols))
    # a + log_scale_g + 2 xi cells.
    @test ibuilt.layout.total == 4
    splan = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [x]
            end
        end, (:y, :x, :g))
    sbuilt = build_kernel(bind_data(splan, cols))
    @test sbuilt.layout.total == 4
    cplan = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1, x]
            end
        end, (:y, :x, :g))
    @test_throws ContractValidationError build_kernel(bind_data(cplan, cols))
    idplan = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(:ID, g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(:ID, g) do
                mu => [1]
            end
        end, (:y, :x, :g))
    # K=1 with |ID| is :correlated (Stage C owns it).
    @test_throws ContractValidationError build_kernel(bind_data(idplan, cols))
end

@testset "ranef K=1 names" begin
    iplan = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1]
            end
        end, (:y, :x, :g))
    ib = only(iplan.ranef_buckets)
    @test ib.kind === :intercept1
    @test ReactiveKernelsPPL._ranef_k1_names(ib) ===
        (:log_scale_g, :xi_g)
    splan = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [x]
            end
        end, (:y, :x, :g))
    sb = only(splan.ranef_buckets)
    @test sb.kind === :slope1
    @test ReactiveKernelsPPL._ranef_k1_names(sb) === (:tau_g, :xi_g)
    cplan = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1, x]
            end
        end, (:y, :x, :g))
    @test_throws ContractValidationError ReactiveKernelsPPL._ranef_k1_names(
        only(cplan.ranef_buckets))
    # Claims: user definitions cannot collide with K=1 names.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 1)
            tau_g = 1.0
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [x]
            end
        end, (:y, :x, :g))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1]
            end
            xi_g ~ Normal(0, 1)
        end, (:y, :x, :g))
    # Name tables: a hand-built parameter under a K=1 name fails.
    clash = _rbase(; with_gather = true)
    push!(clash.parameters, SampledParameter(:log_scale_g, :normal,
        (arg1 = 0, arg2 = 1), nothing, :log_scale_g))
    push!(clash.ranef_buckets, RanefBucket(nothing, :g, :intercept1,
        RanefMargin[RanefMargin(:mu, :Intercept,
            RanefZRecipe(:ones, :none, nothing))],
        [(:mu, 1:1)], NaN, :bucket_g))
    @test_throws ContractValidationError validate_structure(clash)
end

function _rk1_cols()
    g = ["b", "a", "c", "a", "b", "c", "a", "b"]
    y = [0.5, -0.2, 0.8, 0.1, -0.5, 0.3, 0.0, 0.2]
    return Dict{Symbol,AbstractVector}(:g => g, :y => y), g, y
end

@testset "ranef K=1 layout" begin
    cols, _, _ = _rk1_cols()
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, sigma)
            ranef_bucket(g) do
                mu => [1]
            end
        end, (:y, :g))
    bound = bind_data(plan, cols)
    layout = assign_layout(bound)
    @test [e.kind for e in layout.entries] ==
        [:coefficient, :sampled, :sampled, :ranef]
    @test [e.size for e in layout.entries] == [1, 1, 1, 3]
    @test [e.transform for e in layout.entries] ==
        [:identity, :exp, :identity, :identity]
    @test layout.total == 6
    @test coordinate_names(layout)[2:3] ==
        [:sigma, :log_scale_g]
    u = [0.2, 0.1, -0.3, 0.4, 0.0, -0.1]
    nt = constrain(layout, u)
    @test nt.log_scale_g == -0.3 && nt.xi_g == [0.4, 0.0, -0.1]
    @test unconstrain(layout, nt) ≈ u
    @test logjac(layout, u) ≈ u[2]
    # Slope bucket: tau rides :exp (Stan lower-bound Jacobian, no renorm).
    splan = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.0)
            ranef_bucket(g) do
                mu => [x]
            end
        end, (:y, :x, :g))
    scols = Dict{Symbol,AbstractVector}(:g => cols[:g], :y => cols[:y],
        :x => collect(1.0:8.0))
    slayout = assign_layout(bind_data(splan, scols))
    @test [e.kind for e in slayout.entries] ==
        [:coefficient, :sampled, :ranef]
    @test [e.transform for e in slayout.entries] ==
        [:identity, :exp, :identity]
    @test coordinate_names(slayout)[2] === :tau_g
    us = [0.5, 0.3, 0.1, 0.2, 0.0]
    @test logjac(slayout, us) ≈ us[2]
end

# Independent intercept reference: SB `exp(log_scale) * xi[idx]` shape
# with explicit names/order (no contract helpers — this pins them).
function _ref_ranef_intercept(bound, nt)
    idx = [findfirst(==(v), ["a", "b", "c"]) for v in bound.columns[:g]]
    r = exp(nt.log_scale_g) .* nt.xi_g[idx]
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), bound.columns[:y]))
    pr = logpdf(Normal(0, 5), nt.mu[1]) + logpdf(Exponential(1), nt.sigma) +
        logpdf(Normal(0, 1), nt.log_scale_g) +
        sum(logpdf.(Normal(0, 1), nt.xi_g))
    return (; ll, pr)
end

@testset "ranef intercept e2e values and gradient" begin
    cols, _, _ = _rk1_cols()
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, sigma)
            ranef_bucket(g) do
                mu => [1]
            end
        end, (:y, :g))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = [0.2, 0.1, -0.3, 0.4, 0.0, -0.1]
    nt = constrain(built.layout, u)
    ref = _ref_ranef_intercept(bound, nt)
    @test _query(built.spec, bound, :likelihood, u) ≈ ref.ll
    @test _query(built.spec, bound, :prior, u) ≈ ref.pr
    @test _query(built.spec, bound, :posterior, u) ≈ ref.ll + ref.pr + u[2]
    _check_gradient(built.spec, bound, u)
end

# Independent slope reference: SB `tau * (xi[idx] .* Z)` association.
function _ref_ranef_slope(bound, nt, Z)
    idx = [findfirst(==(v), [1, 2, 3]) for v in bound.columns[:g]]
    r = nt.tau_g .* (nt.xi_g[idx] .* Z)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), bound.columns[:y]))
    pr = logpdf(Normal(0, 5), nt.mu[1]) + logpdf(Exponential(1), nt.sigma) +
        logpdf(Normal(0, 1), nt.tau_g) +
        sum(logpdf.(Normal(0, 1), nt.xi_g))
    return (; ll, pr)
end

@testset "ranef slope e2e values and gradient" begin
    _, _, y = _rk1_cols()
    g2 = [2, 1, 3, 1, 2, 3, 1, 2]
    x2 = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0, 2.0, -1.5]
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, sigma)
            ranef_bucket(g) do
                mu => [x]
            end
        end, (:y, :x, :g))
    bound = bind_data(plan,
        Dict{Symbol,AbstractVector}(:g => g2, :y => y, :x => x2))
    built = build_kernel(bound)
    u = [0.2, 0.1, -0.2, 0.3, 0.0, -0.1]
    nt = constrain(built.layout, u)
    ref = _ref_ranef_slope(bound, nt, x2)
    @test _query(built.spec, bound, :likelihood, u) ≈ ref.ll
    # No +log(2): SB Stan-convention tau (see the generator comment).
    @test _query(built.spec, bound, :prior, u) ≈ ref.pr
    @test _query(built.spec, bound, :posterior, u) ≈
        ref.ll + ref.pr + u[2] + u[3]
    _check_gradient(built.spec, bound, u)
    # Dummy-Z slope: same shape, indicator Z.
    c3 = [1, 2, 2, 1, 2, 1, 2, 1]
    dplan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, sigma)
            ranef_bucket(g) do
                mu => [dummy(c, 2)]
            end
        end, (:y, :c, :g))
    dbound = bind_data(dplan,
        Dict{Symbol,AbstractVector}(:g => g2, :y => y, :c => c3))
    dbuilt = build_kernel(dbound)
    dnt = constrain(dbuilt.layout, u)
    dref = _ref_ranef_slope(dbound, dnt, Float64.([v == 2 for v in c3]))
    @test _query(dbuilt.spec, dbound, :likelihood, u) ≈ dref.ll
    @test _query(dbuilt.spec, dbound, :prior, u) ≈ dref.pr
    @test _query(dbuilt.spec, dbound, :posterior, u) ≈
        dref.ll + dref.pr + u[2] + u[3]
end
