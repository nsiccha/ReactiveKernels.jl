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

@testset "ranef generator arm" begin
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
    @test_throws ContractValidationError build_kernel(bound)
end
