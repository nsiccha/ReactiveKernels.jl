# Ranef stages A–C: bucket IR + surface lowering + validation (Stage A),
# K=1 codegen (Stage B), LKJ-correlated codegen (Stage C). All three
# geometries build end to end (tested below).

using LinearAlgebra: det
using SpecialFunctions: loggamma

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

@testset "ranef Stage-C builds" begin
    cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5],
        :x => [0.5, -1.0, 1.5, 0.0], :g => [1, 2, 1, 2])
    # K=1 buckets build (intercept + slope); LKJ-correlated builds too.
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
    cbuilt = build_kernel(bind_data(cplan, cols))
    # a + 1 theta + 2 tau + 2*2 z cells.
    @test cbuilt.layout.total == 8
    idplan = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(:ID, g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(:ID, g) do
                mu => [1]
            end
        end, (:y, :x, :g))
    # K=1 with |ID| is :correlated (L packs zero coords).
    idbuilt = build_kernel(bind_data(idplan, cols))
    # a + 0 thetas + 1 tau + 2 z cells.
    @test idbuilt.layout.total == 4
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

@testset "ranef correlated names" begin
    cplan = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1, x]
            end
        end, (:y, :x, :g))
    cb = only(cplan.ranef_buckets)
    @test cb.kind === :correlated
    @test ReactiveKernelsPPL._ranef_corr_names(cb) ===
        (:L_g, :tau_g, :z_flat_g)
    idplan = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(:ID, g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(:ID, g) do
                mu => [1, x]
            end
        end, (:y, :x, :g))
    @test ReactiveKernelsPPL._ranef_corr_names(only(idplan.ranef_buckets)) ===
        (:L_ID_g, :tau_ID_g, :z_flat_ID_g)
    # K=1 kinds own no correlated names.
    k1plan = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1]
            end
        end, (:y, :x, :g))
    @test_throws ContractValidationError ReactiveKernelsPPL._ranef_corr_names(
        only(k1plan.ranef_buckets))
    # Claims: user definitions cannot collide with correlated names.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 1)
            L_g = 1.0
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1, x]
            end
        end, (:y, :x, :g))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1, x]
            end
            z_flat_g ~ Normal(0, 1)
        end, (:y, :x, :g))
    # The derived draws are claimed too (constrain-output key).
    @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 1)
            b_g = 1.0
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1, x]
            end
        end, (:y, :x, :g))
    # Name tables: a hand-built parameter under a correlated name fails.
    clash = _rbase(; with_gather = true)
    push!(clash.parameters, SampledParameter(:tau_ID_g, :normal,
        (arg1 = 0, arg2 = 1), nothing, :tau_ID_g))
    @test_throws ContractValidationError validate_structure(clash)
    # Empty slice ranges fail closed (a vacuous gather).
    empty = _rbase(; with_gather = true)
    b = only(empty.ranef_buckets)
    empty.ranef_buckets[1] = RanefBucket(b.id, b.group, b.kind, b.margins,
        [(:mu, 2:1)], b.lkj_eta, b.label)
    @test_throws ContractValidationError validate_structure(empty)
end

# NOTE (F1, 2026-09-19): this testset used to pin `lkj_chol_logjac`
# against a central-difference 1/2 logdet(J'J) of u -> vec(L) — the
# hyperspherical sphere-volume Gram factor. That expectation WAS the
# bug: the Stan-verbatim `lkj_corr_cholesky_logpdf` it pairs with is
# a density w.r.t. the correlation-matrix (Omega-pullback) volume
# element, whose Gram exponent is (i-j), one log-sin per angle more
# (posterior parity vs SBBRMI failed on L.2.1/L.2.2 until the fix).
# The Jacobian oracle now lives in test_lkj_jacobian.jl (Omega-volume
# identity + K=2 closed form); this testset keeps roundtrip/shape/
# fail-closed coverage only.

@testset "ranef LKJ transform" begin
    # Packed-dim inversion + fail-closed on non-triangular lengths.
    @test ReactiveKernelsPPL._lkj_dim(0) == 1
    @test ReactiveKernelsPPL._lkj_dim(1) == 2
    @test ReactiveKernelsPPL._lkj_dim(3) == 3
    @test ReactiveKernelsPPL._lkj_dim(6) == 4
    @test_throws ContractValidationError ReactiveKernelsPPL._lkj_dim(2)
    @test_throws ContractValidationError ReactiveKernelsPPL._lkj_dim(5)
    # Roundtrip + Cholesky validity, K = 1..4.
    for (K, u) in ((1, Float64[]), (2, [0.3]), (3, [0.3, -0.5, 0.7]),
            (4, [0.3, -0.5, 0.7, 0.1, -0.2, 0.4]))
        L = lkj_chol_constrain(u, K)
        @test size(L) == (K, K)
        @test L[1, 1] == 1.0
        for i in 1:K, j in i+1:K
            @test L[i, j] == 0.0
        end
        for i in 1:K
            @test sum(L[i, 1:i] .^ 2) ≈ 1.0
            @test L[i, i] > 0.0
        end
        @test lkj_chol_unconstrain(L, K) ≈ u
    end
    # Log-Jacobian oracle: test_lkj_jacobian.jl (see NOTE above).
    @test lkj_chol_logjac(Float64[], 1) == 0.0
    # Fail-closed: length mismatch, non-square, off-hemisphere.
    @test_throws ContractValidationError lkj_chol_constrain([0.1], 3)
    @test_throws ContractValidationError lkj_chol_unconstrain([1.0 0.0], 2)
    @test_throws ContractValidationError lkj_chol_unconstrain(
        [1.0 0.0; 0.5 -0.5], 2)
end

@testset "ranef LKJ constant" begin
    # K=1 is ±0.0 (Stan's K=1 term: no diagonal, zero constant).
    @test lkj_logconst(1, 1.0) == 0.0
    @test lkj_logconst(1, 2.5) == 0.0
    # K=2 closed form from the Beta integral ∫(1-r²)^{η-1}dr
    # (independent of the LKJ09-theorem-5 port): hand-evaluated at
    # half-integer/integer etas where the gammas telescope.
    @test lkj_logconst(2, 0.5) ≈ -log(pi)
    @test lkj_logconst(2, 1.0) ≈ -log(2.0)
    @test lkj_logconst(2, 2.0) ≈ log(0.75)
    @test lkj_logconst(2, 3.0) ≈ log(0.9375)
    # K=3/K=4 eta==1.0 branches, hand-evaluated from Stan's formula.
    @test lkj_logconst(3, 1.0) ≈ log(2.0) - 2 * log(pi)
    @test lkj_logconst(4, 1.0) ≈ 3 * log(6.0) - 2 * log(pi) - 8 * log(2.0)
    # lpdf: K=1 exactly zero; K=2 eta==1.0 is the bare constant.
    @test lkj_corr_cholesky_logpdf(reshape([1.0], 1, 1), 2.0) == 0.0
    L = lkj_chol_constrain([0.3], 2)
    @test lkj_corr_cholesky_logpdf(L, 1.0) ≈ -log(2.0)
    # K=2 general branch: const + (2η-2) log L22.
    @test lkj_corr_cholesky_logpdf(L, 2.0) ≈ log(0.75) + 2 * log(L[2, 2])
    @test_throws ContractValidationError lkj_corr_cholesky_logpdf(
        [1.0 0.0], 1.0)
end

@testset "ranef correlated layout" begin
    cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
        :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        :g => [1, 2, 1, 3, 2, 3])
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, sigma)
            ranef_bucket(g) do
                mu => [1, x]
            end
        end, (:y, :x, :g))
    bound = bind_data(plan, cols)
    layout = assign_layout(bound)
    # `a` rides the intercept coefficient; sampled = [sigma, bucket triple].
    @test [e.kind for e in layout.entries] ==
        [:coefficient, :sampled, :ranef_corr, :ranef, :ranef]
    @test [e.name for e in layout.entries] ==
        [:mu_coef, :sigma, :L_g, :tau_g, :z_flat_g]
    @test [e.size for e in layout.entries] == [1, 1, 1, 2, 6]
    @test [e.transform for e in layout.entries] ==
        [:identity, :exp, :lkj, :exp, :identity]
    @test layout.total == 11
    @test coordinate_names(layout)[2] === :sigma
    @test coordinate_names(layout)[3] == Symbol("L_g.1")
    @test coordinate_names(layout)[4:5] ==
        [Symbol("tau_g.1"), Symbol("tau_g.2")]
    @test coordinate_names(layout)[6] == Symbol("z_flat_g.1")
    u = collect(range(-0.5, 0.5; length = layout.total))
    nt = constrain(layout, u)
    @test size(nt.L_g) == (2, 2)
    @test nt.tau_g ≈ exp.(u[4:5])
    @test nt.z_flat_g == u[6:11]
    # Derived draws: shape + the trivially hand-checkable b[1,1].
    @test size(nt.b_g) == (3, 2)
    @test nt.b_g[1, 1] ≈ nt.tau_g[1] * nt.z_flat_g[1]
    # Roundtrip ignores the derived b (constrain output feeds unconstrain).
    @test unconstrain(layout, nt) ≈ u
    # Jacobian: sigma + tau exps + the K=2 theta term, hand-summed
    # (the (i-j) = 1 exponent keeps one log-sin term — F1 fix; the old
    # pure-logistic pin asserted the sphere-volume bug).
    s = 1 / (1 + exp(-u[3]))
    lkj = log(sin(pi * s)) + log(pi) + log(s) + log1p(-s)
    @test logjac(layout, u) ≈ u[2] + u[4] + u[5] + lkj
end

# Independent correlated-gather reference: SB `(diag(tau)*L*z)'` shape
# with explicit per-margin/per-group loops (never the fused forms),
# over the global margin subset `js` with Z columns `Zs` (Zs[j] is the
# j-th GLOBAL margin's column; `:ones` margins pass `ones(n)`).
# `levels` overrides the numbering order (declared-order tests); default
# is sort order (the bind fill for plain vectors).
function _ref_corr_r(bound, groupcol, L, tau, zflat, Zs, js; levels = nothing)
    K = length(Zs)
    lv = levels === nothing ? sort!(unique(bound.columns[groupcol])) : levels
    idx = [findfirst(==(v), lv) for v in bound.columns[groupcol]]
    r = zeros(Float64, length(idx))
    for m in eachindex(idx)
        g = idx[m]
        for j in js
            acc = 0.0
            for s in 1:j
                acc += tau[j] * L[j, s] * zflat[s + (g - 1) * K]
            end
            r[m] += Zs[j][m] * acc
        end
    end
    return r
end

# K=2 LKJ from the Beta-integral closed form (independent of the
# LKJ09-theorem-5 `lkj_logconst` port the emitter inlines).
function _ref_lkj_k2(L, eta)
    c = loggamma(eta + 0.5) - loggamma(eta) - 0.5 * log(pi)
    return c + (2 * eta - 2) * log(L[2, 2])
end

@testset "ranef correlated e2e values and gradient" begin
    gv = [1, 2, 1, 3, 2, 3]
    xv = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    yv = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    cols = Dict{Symbol,AbstractVector}(:g => gv, :x => xv, :y => yv)
    # Both LKJ emission branches: eta == 1.0 fast path + general.
    for eta in (1.0, 2.0)
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                sigma ~ Exponential(1)
                mu = a .+ ranef(g)
                y .~ Normal.(mu, sigma)
                ranef_bucket(g; eta = $eta) do
                    mu => [1, x]
                end
            end, (:y, :x, :g))
        bound = bind_data(plan, cols)
        built = build_kernel(bound)
        u = collect(range(-0.4, 0.4; length = built.layout.total))
        nt = constrain(built.layout, u)
        r = _ref_corr_r(bound, :g, nt.L_g, nt.tau_g, nt.z_flat_g,
            [ones(6), xv], 1:2)
        ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), yv))
        pr = logpdf(Normal(0, 5), nt.mu[1]) +
            logpdf(Exponential(1), nt.sigma) +
            _ref_lkj_k2(nt.L_g, eta) +
            sum(logpdf.(Normal(0, 1), nt.tau_g)) +
            sum(logpdf.(Normal(0, 1), nt.z_flat_g))
        @test _query(built.spec, bound, :likelihood, u) ≈ ll
        # No +log(2): SB Stan-convention tau (see the generator comment).
        @test _query(built.spec, bound, :prior, u) ≈ pr
        s = 1 / (1 + exp(-u[3]))
        jac = u[2] + u[4] + u[5] + log(sin(pi * s)) + log(pi) + log(s) +
            log1p(-s)
        @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
        _check_gradient(built.spec, bound, u)
    end
end

@testset "ranef K=3 e2e values and gradient" begin
    gv = [1, 2, 1, 3, 2, 3]
    x1v = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    x2v = [1.0, 0.5, -0.5, 2.0, -1.5, 0.0]
    yv = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    cols = Dict{Symbol,AbstractVector}(:g => gv, :x1 => x1v, :x2 => x2v,
        :y => yv)
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, sigma)
            ranef_bucket(g) do
                mu => [1, x1, x2]
            end
        end, (:y, :x1, :x2, :g))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    # coef + sigma + 3 thetas + 3 tau + 3*3 z cells.
    @test built.layout.total == 17
    u = collect(range(-0.4, 0.4; length = built.layout.total))
    nt = constrain(built.layout, u)
    r = _ref_corr_r(bound, :g, nt.L_g, nt.tau_g, nt.z_flat_g,
        [ones(6), x1v, x2v], 1:3)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), yv))
    # K=3 eta==1.0 LKJ, hand-derived from Stan's do_lkj_constant
    # (const = log2 - 2logpi; diag (3-2)logL22 + 0*logL33).
    lkj = log(2.0) - 2 * log(pi) + log(nt.L_g[2, 2])
    pr = logpdf(Normal(0, 5), nt.mu[1]) +
        logpdf(Exponential(1), nt.sigma) + lkj +
        sum(logpdf.(Normal(0, 1), nt.tau_g)) +
        sum(logpdf.(Normal(0, 1), nt.z_flat_g))
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    # Jacobian: sigma + tau exps + row-2/row-3 theta terms (the
    # (i-j) Gram exponents: 1, 2, 1 — F1 fix).
    s3 = 1 / (1 + exp(-u[3]))
    s4 = 1 / (1 + exp(-u[4]))
    s5 = 1 / (1 + exp(-u[5]))
    lj = log(sin(pi * s3)) + log(pi) + log(s3) + log1p(-s3) +
        2 * log(sin(pi * s4)) + log(pi) + log(s4) + log1p(-s4) +
        log(sin(pi * s5)) + log(pi) + log(s5) + log1p(-s5)
    jac = u[2] + u[6] + u[7] + u[8] + lj
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
    _check_gradient(built.spec, bound, u)
end

@testset "ranef K=1 ID e2e values and gradient" begin
    gv = [1, 2, 1, 2]
    xv = [0.5, -1.0, 1.5, 0.0]
    yv = [1.0, 2.0, 1.5, 2.5]
    cols = Dict{Symbol,AbstractVector}(:g => gv, :x => xv, :y => yv)
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ ranef(:ID, g)
            y .~ Normal.(mu, sigma)
            ranef_bucket(:ID, g) do
                mu => [x]
            end
        end, (:y, :x, :g))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    # coef + sigma + 0 thetas + 1 tau + 2 z cells.
    @test built.layout.total == 5
    u = [0.2, 0.1, -0.3, 0.4, -0.1]
    nt = constrain(built.layout, u)
    @test nt.L_ID_g == [1.0;;]
    idx = [findfirst(==(v), [1, 2]) for v in gv]
    r = nt.tau_ID_g[1] .* (nt.z_flat_ID_g[idx] .* xv)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), yv))
    # No LKJ term: Stan's K=1 LKJ contributes exactly 0.0.
    pr = logpdf(Normal(0, 5), nt.mu[1]) +
        logpdf(Exponential(1), nt.sigma) +
        logpdf(Normal(0, 1), nt.tau_ID_g[1]) +
        sum(logpdf.(Normal(0, 1), nt.z_flat_ID_g))
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + u[2] + u[3]
    _check_gradient(built.spec, bound, u)
end

@testset "ranef multislice ID e2e values and gradient" begin
    gv = [1, 2, 1, 3, 2, 3]
    xv = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    y1v = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    y2v = [0.5, 1.5, 1.0, 2.0, 2.5, 1.5]
    cols = Dict{Symbol,AbstractVector}(:g => gv, :x => xv, :y1 => y1v,
        :y2 => y2v)
    plan = lower_rkppl(quote
            a1 ~ Normal(0, 5)
            a2 ~ Normal(0, 5)
            s ~ Exponential(1)
            mu1 = a1 .+ ranef(:ID, g)
            mu2 = a2 .+ ranef(:ID, g)
            y1 .~ Normal.(mu1, s)
            y2 .~ Normal.(mu2, s)
            ranef_bucket(:ID, g) do
                mu1 => [1]
                mu2 => [x]
            end
        end, (:y1, :y2, :x, :g))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    @test [e.kind for e in built.layout.entries] ==
        [:coefficient, :coefficient, :sampled, :ranef_corr, :ranef, :ranef]
    # 2 coefs + s + 1 theta + 2 tau + 2*3 z cells.
    @test built.layout.total == 12
    u = collect(range(-0.4, 0.4; length = built.layout.total))
    nt = constrain(built.layout, u)
    Zs = [ones(6), xv]
    r1 = _ref_corr_r(bound, :g, nt.L_ID_g, nt.tau_ID_g, nt.z_flat_ID_g,
        Zs, 1:1)
    r2 = _ref_corr_r(bound, :g, nt.L_ID_g, nt.tau_ID_g, nt.z_flat_ID_g,
        Zs, 2:2)
    ll = sum(logpdf.(Normal.(nt.mu1[1] .+ r1, nt.s), y1v)) +
        sum(logpdf.(Normal.(nt.mu2[1] .+ r2, nt.s), y2v))
    pr = logpdf(Normal(0, 5), nt.mu1[1]) + logpdf(Normal(0, 5), nt.mu2[1]) +
        logpdf(Exponential(1), nt.s) + _ref_lkj_k2(nt.L_ID_g, 1.0) +
        sum(logpdf.(Normal(0, 1), nt.tau_ID_g)) +
        sum(logpdf.(Normal(0, 1), nt.z_flat_ID_g))
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    s4 = 1 / (1 + exp(-u[4]))
    jac = u[3] + u[5] + u[6] + log(sin(pi * s4)) + log(pi) + log(s4) +
        log1p(-s4)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
    _check_gradient(built.spec, bound, u)
end

@testset "ranef correlated restore_draws" begin
    cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5],
        :x => [0.5, -1.0, 1.5, 0.0], :g => [1, 2, 1, 2])
    plan = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1, x]
            end
        end, (:y, :x, :g))
    layout = assign_layout(bind_data(plan, cols))
    U = hcat(collect(range(-0.4, 0.4; length = layout.total)),
        collect(range(0.4, -0.4; length = layout.total)))
    draws = restore_draws(layout, U)
    # LKJ factors + derived draws restore as vectors of matrices.
    @test draws.L_g isa Vector{Matrix{Float64}}
    @test draws.b_g isa Vector{Matrix{Float64}}
    @test size(draws.L_g[1]) == (2, 2)
    @test size(draws.b_g[2]) == (2, 2)
    @test draws.L_g[1] ≈ constrain(layout, U[:, 1]).L_g
    @test draws.b_g[2] ≈ constrain(layout, U[:, 2]).b_g
    # Empty draws keep the keys, with empty matrix vectors.
    none = restore_draws(layout, Matrix{Float64}(undef, layout.total, 0))
    @test Tuple(keys(none)) == Tuple(keys(draws))
    @test isempty(none.L_g) && isempty(none.b_g)
end

# Swap declared levels onto one bucket of a lowered (unbound) plan.
function _with_levels(plan::StructuralPlan, levels::Vector, which::Int = 1)
    b = plan.ranef_buckets[which]
    plan.ranef_buckets[which] = RanefBucket(b.id, b.group, b.kind,
        b.margins, b.slices, b.lkj_eta, b.label, levels)
    return plan
end

@testset "declared codes helper" begin
    dc = ReactiveKernelsPPL._declared_codes
    # Declared order, never sorted: levels [3, 1, 2].
    @test dc([1, 3, 2, 1], [3, 1, 2]) == [2, 1, 3, 2]
    # Strings in declared order.
    @test dc(["b", "a", "c", "b"], ["c", "b", "a"]) == [2, 3, 1, 2]
    # Single level; repeated values.
    @test dc([7, 7], [7]) == [1, 1]
    # Unobserved declared levels keep their positions.
    @test dc(["a", "c"], ["c", "b", "a"]) == [3, 1]
    # Symbols.
    @test dc([:x, :y], [:y, :x]) == [2, 1]
end

@testset "ranef declared levels structure validation" begin
    b = _rbucket()
    mklevels(lv) = RanefBucket(b.id, b.group, b.kind, b.margins,
        b.slices, b.lkj_eta, b.label, lv)
    # Empty levels fail.
    bad = _rbase(; with_gather = true)
    bad.ranef_buckets[1] = mklevels([])
    @test_throws ContractValidationError validate_structure(bad)
    # Duplicate levels fail.
    dup = _rbase(; with_gather = true)
    dup.ranef_buckets[1] = mklevels([1, 2, 1])
    @test_throws ContractValidationError validate_structure(dup)
    # Non-literal-embeddable levels fail.
    nonemb = _rbase(; with_gather = true)
    nonemb.ranef_buckets[1] = mklevels([1, missing])
    @test_throws ContractValidationError validate_structure(nonemb)
    # Valid declared levels (order ≠ sorted) pass.
    ok = _rbase(; with_gather = true)
    ok.ranef_buckets[1] = mklevels(["b", "a"])
    @test validate_structure(ok) === nothing
end

@testset "ranef declared levels bind" begin
    cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5],
        :g => [2, 1, 3, 1])
    mkplan() = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(g) do
                mu => [1]
            end
        end, (:y, :g))
    # `nothing` fills sort-ordered observed levels (SB numbering for
    # plain vectors).
    bound = bind_data(mkplan(), cols)
    @test only(bound.ranef_buckets).levels == [1, 2, 3]
    # Provided levels pass through; G counts unobserved declared levels.
    bound2 = bind_data(_with_levels(mkplan(), [3, 1, 2, 4]), cols)
    @test only(bound2.ranef_buckets).levels == [3, 1, 2, 4]
    @test assign_layout(bound2).total == 6 # a + log_scale + 4 xi cells
    # Observed-but-undeclared values fail closed (they would encode 0).
    bad = _with_levels(mkplan(), [1, 2])
    @test_throws ContractValidationError bind_data(bad, cols)
    # Hand-built bound plans with `levels === nothing` fail loud.
    hand = bind_data(mkplan(), cols)
    hb = only(hand.ranef_buckets)
    hand.ranef_buckets[1] = RanefBucket(hb.id, hb.group, hb.kind,
        hb.margins, hb.slices, hb.lkj_eta, hb.label, nothing)
    @test_throws ContractValidationError validate_data(hand)
    # Same-group buckets must agree on levels (order included).
    two = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(:A, g) .+ ranef(:B, g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(:A, g) do
                mu => [1]
            end
            ranef_bucket(:B, g) do
                mu => [1]
            end
        end, (:y, :g))
    agree = bind_data(_with_levels(_with_levels(two, [2, 1, 3], 1),
        [2, 1, 3], 2), cols)
    @test agree.ranef_buckets[1].levels == agree.ranef_buckets[2].levels
    disagree = lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ ranef(:A, g) .+ ranef(:B, g)
            y .~ Normal.(mu, 1.5)
            ranef_bucket(:A, g) do
                mu => [1]
            end
            ranef_bucket(:B, g) do
                mu => [1]
            end
        end, (:y, :g))
    _with_levels(_with_levels(disagree, [2, 1, 3], 1), [1, 2, 3], 2)
    @test_throws ContractValidationError bind_data(disagree, cols)
end

# Declared-order K=1 intercept reference: SB `exp(log_scale) * xi[idx]`
# with `idx` in DECLARED position order (never sorted) and `xi` sized
# by the declared count (unobserved levels are prior-only).
function _ref_declared_intercept(bound, nt, levels)
    idx = [findfirst(==(v), levels) for v in bound.columns[:g]]
    r = exp(nt.log_scale_g) .* nt.xi_g[idx]
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), bound.columns[:y]))
    pr = logpdf(Normal(0, 5), nt.mu[1]) + logpdf(Exponential(1), nt.sigma) +
        logpdf(Normal(0, 1), nt.log_scale_g) +
        sum(logpdf.(Normal(0, 1), nt.xi_g))
    return (; ll, pr)
end

@testset "ranef declared-order intercept e2e values and gradient" begin
    _, _, y = _rk1_cols()
    g = ["a", "c", "b", "a", "c", "b", "a", "c"]
    levels = ["c", "a", "b", "d"] # declared ≠ sorted; "d" unobserved
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ ranef(g)
            y .~ Normal.(mu, sigma)
            ranef_bucket(g) do
                mu => [1]
            end
        end, (:y, :g))
    bound = bind_data(_with_levels(plan, levels),
        Dict{Symbol,AbstractVector}(:g => g, :y => y))
    built = build_kernel(bound)
    # a + sigma + log_scale + 4 xi cells (d is prior-only).
    @test built.layout.total == 7
    u = [0.2, 0.1, -0.3, 0.4, 0.0, -0.1, 0.25]
    nt = constrain(built.layout, u)
    ref = _ref_declared_intercept(bound, nt, levels)
    @test _query(built.spec, bound, :likelihood, u) ≈ ref.ll
    @test _query(built.spec, bound, :prior, u) ≈ ref.pr
    @test _query(built.spec, bound, :posterior, u) ≈ ref.ll + ref.pr + u[2]
    _check_gradient(built.spec, bound, u)
end

@testset "ranef declared-order correlated e2e values and gradient" begin
    gv = ["a", "c", "b", "c", "a", "b"]
    xv = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    yv = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    levels = ["c", "b", "a", "d"] # declared ≠ sorted; "d" unobserved
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ ranef(:ID, g)
            y .~ Normal.(mu, sigma)
            ranef_bucket(:ID, g) do
                mu => [1, x]
            end
        end, (:y, :x, :g))
    bound = bind_data(_with_levels(plan, levels),
        Dict{Symbol,AbstractVector}(:g => gv, :x => xv, :y => yv))
    built = build_kernel(bound)
    # coef + sigma + 1 theta + 2 tau + 2*4 z cells.
    @test built.layout.total == 13
    u = collect(range(-0.4, 0.4; length = built.layout.total))
    nt = constrain(built.layout, u)
    r = _ref_corr_r(bound, :g, nt.L_ID_g, nt.tau_ID_g, nt.z_flat_ID_g,
        [ones(6), xv], 1:2; levels)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), yv))
    pr = logpdf(Normal(0, 5), nt.mu[1]) +
        logpdf(Exponential(1), nt.sigma) +
        _ref_lkj_k2(nt.L_ID_g, 1.0) +
        sum(logpdf.(Normal(0, 1), nt.tau_ID_g)) +
        sum(logpdf.(Normal(0, 1), nt.z_flat_ID_g))
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    s = 1 / (1 + exp(-u[3]))
    jac = u[2] + u[4] + u[5] + log(sin(pi * s)) + log(pi) + log(s) +
        log1p(-s)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
    _check_gradient(built.spec, bound, u)
end
