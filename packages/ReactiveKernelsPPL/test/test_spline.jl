# Spline contract: in-graph s/t2 bases + vector priors + smoothing (SB
# `_sb_s_generic`/`_sb_t2_generic` mirror). Surface declarations, IR
# validation, bind-time materialization, layout, end-to-end
# values/gradients vs independent Distributions.jl references, and a
# Reactant/XLA native-vs-compiled leg. The port itself is verified
# against BRM by a /tmp differential (BRM is not a test dep); committed
# here are structural properties + hand references (+ SB-parity pins once
# the BRM peer publishes its BridgeStan literals).
using Reactant

using Reactant

function _svalid_plan(; t2::Bool = false)
    if t2
        return lower_rkppl(quote
                spline_basis(:t2_xz, x, z; k = (3, 4))
                a ~ Normal(0, 5)
                sigma ~ Exponential(1)
                mu = a .+ spline(:t2_xz)
                y .~ Normal.(mu, sigma)
            end, (:y, :x, :z); conditioned = (:y, :x, :z))
    end
    return lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ spline(:s_x)
            y .~ Normal.(mu, sigma)
        end, (:y, :x); conditioned = (:y, :x))
end

function _swith(plan::StructuralPlan; predictors = nothing, bases = nothing,
        vectors = nothing, parameters = nothing)
    return StructuralPlan(plan.responses,
        predictors === nothing ? plan.predictors : predictors,
        plan.population_priors,
        parameters === nothing ? plan.parameters : parameters,
        plan.assignments, plan.columns, plan.n_obs; roles = plan.roles,
        derived = plan.derived, levelmaps = plan.levelmaps,
        plate_parameters = plan.plate_parameters, scans = plan.scans,
        varying_draws = plan.varying_draws,
        varying_slices = plan.varying_slices,
        spline_bases = bases === nothing ? plan.spline_bases : bases,
        spline_vectors = vectors === nothing ? plan.spline_vectors : vectors)
end

_summand(pname::Symbol, id::Symbol) = TermSpec(SplineSummandTerm,
    ColumnRef[], (spline_id = id,), Symbol("spline_", pname, "_", id),
    Symbol("spline_", pname, "_", id))

function _spline_cols(; n::Int = 12, t2::Bool = false)
    x = collect(range(-2.0, 3.0; length = n))
    y = sin.(x) .+ 0.1 .* cos.(2.0 .* x)
    t2 || return Dict{Symbol,AbstractVector}(:x => x, :y => y)
    z = collect(range(0.0, 1.0; length = n)) .^ 1.5 .* 4 .- 1.0
    return Dict{Symbol,AbstractVector}(:x => x, :y => y, :z => z)
end

@testset "spline tps lowering" begin
    plan = _svalid_plan()
    @test length(plan.spline_bases) == 1
    sb = only(plan.spline_bases)
    @test sb.id === :s_x && sb.kind === :tps
    @test sb.axes == [:x] && sb.k == 4
    # No constant fixed column: the author's intercept `a` owns it.
    @test [(b.name, b.width) for b in sb.blocks] == [(:fixed, 1), (:pen, 2)]
    @test all(isempty(b.columns) for b in sb.blocks)
    @test sb.label === :spline_s_x
    got = [(v.name, v.family, v.args, v.support_override, v.width, v.basis)
        for v in plan.spline_vectors]
    @test got == [(:b_s_x_fixed, :flat, NamedTuple(), nothing, 1, :s_x),
        (:b_s_x_raw, :normal, (arg1 = 0, arg2 = 1), nothing, 2, :s_x),
        (:sd_s_x, :normal, (arg1 = 0, arg2 = 1), :positive_stan, 1, :s_x)]
    terms = only(plan.predictors).terms
    @test length(terms) == 2
    t = terms[2]
    @test t.kind === SplineSummandTerm && isempty(t.columns)
    @test t.options == (spline_id = :s_x,)
    @test t.addressee === t.label === :spline_mu_s_x
    # Self-addressed: no population prior for the summand.
    @test all(p -> p.addressee !== :spline_mu_s_x, plan.population_priors)
end

@testset "spline t2 lowering and defaults" begin
    plan = lower_rkppl(quote
            spline_basis(:t2_xz, x, z)
            a ~ Normal(0, 5)
            mu = a .+ spline(:t2_xz)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x, :z); conditioned = (:y, :x, :z))
    sb = only(plan.spline_bases)
    @test sb.kind === :t2 && sb.k == (5, 5)
    @test [(b.name, b.width) for b in sb.blocks] ==
        [(:fixed, 3), (:rr, 9), (:rn, 6), (:nr, 6)]
    got = [v.name for v in plan.spline_vectors]
    @test got == [:b_t2_xz_fixed, :b_t2_xz_rr_raw, :b_t2_xz_rn_raw,
        :b_t2_xz_nr_raw, :sd_t2_xz]
    sd = only(v for v in plan.spline_vectors if v.name === :sd_t2_xz)
    @test (sd.family, sd.support_override, sd.width) ===
        (:normal, :positive_stan, 3)
    # Explicit kind + default k on `s`.
    plain = lower_rkppl(quote
            spline_basis(:s_x, x; kind = :tps)
            a ~ Normal(0, 5)
            mu = a .+ spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    @test only(plain.spline_bases).k == 10
end

@testset "spline surface fail-closed" begin
    # Unquoted id.
    # refused: unquoted id `s_x` is an undeclared name (P6, 05oe96l)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(s_x, x; k = 4)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # Non-data axis.
    # refused: undeclared name `w` as axis (P6, 05oe96l)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, w; k = 4)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # Three axes.
    # capability: 3-axis spline basis (tensor smooth over three margins) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            spline_basis(:s_x, x, z, w; k = 4)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x, :z, :w); conditioned = (:y, :x, :z, :w)); true)
    # Bad kind value.
    # capability: cubic-regression spline basis (kind = :cr) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            spline_basis(:s_x, x; kind = :cr)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x)); true)
    # Kind/arity mismatch both ways.
    # capability: single-margin t2 spline (kind = :t2 on one axis; mgcv admits t2(x)) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            spline_basis(:s_x, x; kind = :t2)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x)); true)
    # capability: two-axis thin-plate spline (kind = :tps over (x, z), isotropic s(x, z)) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            spline_basis(:t2_xz, x, z; kind = :tps)
            mu = spline(:t2_xz)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x, :z); conditioned = (:y, :x, :z)); true)
    # k too small / non-literal / wrong shape.
    # refused: k = 2 leaves no penalized block (k must exceed the TPS null-space dimension 2)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 2)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # capability: data-derived basis size k (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            kk = length(x)
            spline_basis(:s_x, x; k = kk)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x)); true)
    # refused: k tuple length differs from axis count (malformed)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = (4, 4))
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # capability: scalar k broadcast per margin for t2 (hsgp_basis already broadcasts scalars) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            spline_basis(:t2_xz, x, z; k = 5)
            mu = spline(:t2_xz)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x, :z); conditioned = (:y, :x, :z)); true)
    # refused: margin k = 2 leaves no penalized block (degenerate)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:t2_xz, x, z; k = (5, 2))
            mu = spline(:t2_xz)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x, :z); conditioned = (:y, :x, :z))
    # Deferred options fail closed.
    # refused: `bs` is mgcv vocabulary duplicating `kind =` (P10, P2)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4, bs = :cr)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # Duplicate declaration.
    # refused: single assignment, basis :s_x declared twice
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            spline_basis(:s_x, x; k = 5)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # Unknown / reused / negated / nested uses.
    # refused: undeclared basis id :nope (P6, 05oe96l)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            mu = spline(:nope)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # capability: spline summand value reuse: same basis twice in one predictor (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            mu = spline(:s_x) .+ spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x)); true)
    # capability: one spline basis shared by two predictors (value reuse) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            a ~ Normal(0, 1)
            c ~ Normal(0, 1)
            spline_basis(:s_x, x; k = 4)
            mu = a .+ spline(:s_x)
            nu = c .+ spline(:s_x)
            y .~ Normal.(mu, 1.0)
            z .~ Normal.(nu, 1.0)
        end, (:y, :z, :x); conditioned = (:y, :z, :x)); true)
    # capability: negated spline summand (`a .- spline(:s)`) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            a ~ Normal(0, 1)
            spline_basis(:s_x, x; k = 4)
            mu = a .- spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x)); true)
    # refused: a coefficient times a flat-prior spline block is an unidentified ridge
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            mu = a .+ b .* spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # spline() inside definitions (scalar + derived).
    # capability: spline value bound in a definition (`w = spline(:s)`) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            a ~ Normal(0, 1)
            spline_basis(:s_x, x; k = 4)
            w = spline(:s_x)
            mu = a .+ w
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x)); true)
    # capability: spline value combined with data in a definition (`w = spline(:s) .+ x`) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            a ~ Normal(0, 1)
            spline_basis(:s_x, x; k = 4)
            w = spline(:s_x) .+ x
            mu = a .+ w
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x)); true)
    # capability: spline value combined with coefficient terms in a definition (`w = spline(:s) .+ b .* x`) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            spline_basis(:s_x, x; k = 4)
            w = spline(:s_x) .+ b .* x
            mu = a .+ w
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x)); true)
    # Reserved names.
    # refused: reserved-name collision `spline`
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline = 1.0
            y .~ Normal.(mu, 1.0)
        end, (:y,); conditioned = (:y,))
    # refused: reserved-name collision `spline_basis`
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis ~ Normal(0, 1)
            y .~ Normal.(mu, 1.0)
        end, (:y,); conditioned = (:y,))
    # Generated-name claims: user definitions cannot collide.
    # refused: single assignment, collides with basis-claimed `b_s_x_fixed`
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            b_s_x_fixed = 1.0
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # refused: single assignment, collides with materialized basis column `s_x_Xnull_1`
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            s_x_Xnull_1 = 1.0
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # refused: single assignment, collides with basis-claimed `sd_s_x`
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            sd_s_x ~ Normal(0, 1)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
end

# Rebuild the valid plan's basis with mutated fields (hand-built-plan
# defense in depth: the surface can never produce these).
function _smutant(; blocks = nothing, vectors = nothing, kind = nothing,
        k = nothing, axes = nothing, id = nothing)
    plan = _svalid_plan()
    sb = only(plan.spline_bases)
    nb = SplineBasis(id === nothing ? sb.id : id,
        kind === nothing ? sb.kind : kind,
        axes === nothing ? sb.axes : axes, k === nothing ? sb.k : k,
        blocks === nothing ? sb.blocks : blocks, sb.label)
    return _swith(plan; bases = SplineBasis[nb],
        vectors = vectors === nothing ? plan.spline_vectors : vectors)
end

@testset "spline contract validation" begin
    @test validate_structure(_svalid_plan()) === nothing
    @test validate_structure(_svalid_plan(; t2 = true)) === nothing
    # Block structure must match (kind, k) exactly.
    badblocks = SplineBasisBlock[SplineBasisBlock(:fixed, 2, Symbol[]),
        SplineBasisBlock(:pen, 3, Symbol[])]
    # refused: block structure does not match (kind, k) (IR contract)
    @test_throws ContractValidationError validate_structure(
        _smutant(; blocks = badblocks))
    # Vector set: dropped / extra / orphan.
    plan = _svalid_plan()
    # refused: dropped spline vector (IR contract)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = plan.spline_vectors[1:2]))
    extra = vcat(plan.spline_vectors,
        SplineVector[SplineVector(:bogus, :normal, (arg1 = 0, arg2 = 1),
            nothing, 1, :s_x, :bogus)])
    # refused: extra spline vector (IR contract)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = extra))
    orphan = vcat(plan.spline_vectors,
        SplineVector[SplineVector(:b_nope_fixed, :flat, NamedTuple(), nothing,
            2, :nope, :b_nope_fixed)])
    # refused: orphan spline vector (IR contract)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = orphan))
    # Vector shape: family / args / support / width pinned.
    vs = copy(plan.spline_vectors)
    vs[1] = SplineVector(vs[1].name, :normal, (arg1 = 0, arg2 = 1), nothing,
        vs[1].width, vs[1].basis, vs[1].label)
    # refused: fixed-vector family pinned (IR contract)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = vs))
    vs = copy(plan.spline_vectors)
    vs[2] = SplineVector(vs[2].name, vs[2].family, (arg1 = 0, arg2 = 2),
        nothing, vs[2].width, vs[2].basis, vs[2].label)
    # refused: raw-vector args pinned (IR contract)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = vs))
    vs = copy(plan.spline_vectors)
    vs[3] = SplineVector(vs[3].name, vs[3].family, vs[3].args, nothing,
        vs[3].width, vs[3].basis, vs[3].label)
    # refused: sd support override pinned (IR contract)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = vs))
    vs = copy(plan.spline_vectors)
    vs[2] = SplineVector(vs[2].name, vs[2].family, vs[2].args,
        vs[2].support_override, 3, vs[2].basis, vs[2].label)
    # refused: vector width pinned (IR contract)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = vs))
    # Linkage: dangling basis / double use / unknown summand target.
    nopred = PredictorSpec(plan.predictors[1].name,
        plan.predictors[1].link, plan.predictors[1].terms[1:1],
        plan.predictors[1].label)
    # refused: dangling basis with no summand (IR contract)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; predictors = PredictorSpec[nopred]))
    two = PredictorSpec(plan.predictors[1].name, plan.predictors[1].link,
        vcat(plan.predictors[1].terms, _summand(:mu, :s_x)),
        plan.predictors[1].label)
    # refused: basis used twice (IR contract; mirrors C at :192)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; predictors = PredictorSpec[two]))
    bad = PredictorSpec(plan.predictors[1].name, plan.predictors[1].link,
        vcat(plan.predictors[1].terms[1:1], _summand(:mu, :nope)),
        plan.predictors[1].label)
    # refused: summand names an unknown basis (IR contract)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; predictors = PredictorSpec[bad]))
    # kind/k/axes shape.
    # refused: kind :cr (IR contract; mirrors C at :131)
    @test_throws ContractValidationError validate_structure(
        _smutant(; kind = :cr))
    # refused: tuple k on a tps basis (IR contract)
    @test_throws ContractValidationError validate_structure(
        _smutant(; k = (4, 4)))
    # refused: k = 2 (IR contract)
    @test_throws ContractValidationError validate_structure(
        _smutant(; k = 2))
    # refused: tps basis with two axes (IR contract; mirrors C at :142)
    @test_throws ContractValidationError validate_structure(
        _smutant(; axes = [:x, :z]))
    # Materialized-name clash with a parameter: rename the id.
    clash = vcat(plan.parameters,
        SampledParameter[SampledParameter(:s_x_Xnull_1, :normal,
            (arg1 = 0, arg2 = 1), nothing, :s_x_Xnull_1)])
    # refused: materialized-name clash with a parameter (IR contract)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; parameters = clash))
    # Spline-vector names join the name tables.
    dup = copy(plan.spline_vectors)
    dup[1] = SplineVector(:a, dup[1].family, dup[1].args,
        dup[1].support_override, dup[1].width, dup[1].basis, dup[1].label)
    # refused: spline-vector name clashes with a name table (IR contract)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = dup))
end

@testset "spline bind materialization" begin
    plan = _svalid_plan()
    bound = bind_data(plan, _spline_cols())
    sb = only(bound.spline_bases)
    @test sb.blocks[1].columns == [:s_x_Xnull_1]
    @test sb.blocks[2].columns == [:s_x_Zpen_1, :s_x_Zpen_2]
    for c in [:s_x_Xnull_1, :s_x_Zpen_1, :s_x_Zpen_2]
        @test haskey(bound.columns, c)
        @test length(bound.columns[c]) == 12
        @test bound.roles[c] === :predictor
    end
    @test bound.roles[:x] === :data
    # Materialized columns equal the port's fit/apply on the same axis.
    fit = ReactiveKernelsPPL._rk_fit_spline(_spline_cols()[:x]; k = 4)
    X, Z = ReactiveKernelsPPL._rk_apply_spline(fit, _spline_cols()[:x])
    @test size(X, 2) == 1
    @test bound.columns[:s_x_Xnull_1] == X[:, 1]
    @test bound.columns[:s_x_Zpen_1] == Z[:, 1]
    @test bound.columns[:s_x_Zpen_2] == Z[:, 2]
    # Reserved-name exclusivity: caller columns cannot squat basis names.
    cols = _spline_cols()
    cols[:s_x_Xnull_1] = ones(12)
    # refused: caller column squats a generated basis-column name (name collision)
    @test_throws ContractValidationError bind_data(plan, cols)
    # Missing / non-numeric axes.
    # refused: missing axis data column
    @test_throws ContractValidationError bind_data(plan,
        Dict{Symbol,AbstractVector}(:y => _spline_cols()[:y]))
    cols = _spline_cols()
    cols[:x] = fill("a", 12)
    # refused: wrong eltype, non-numeric axis column
    @test_throws ContractValidationError bind_data(plan, cols)
    # Fit errors surface as bind errors (k=4 needs 4 unique x).
    few = Dict{Symbol,AbstractVector}(:x => [1.0, 1.0, 2.0],
        :y => [1.0, 2.0, 3.0])
    # refused: fewer unique axis values than k (basis fit impossible)
    @test_throws ContractValidationError bind_data(plan, few)
    # t2 materializes four blocks in SB order.
    t2 = bind_data(_svalid_plan(; t2 = true), _spline_cols(; t2 = true))
    tb = only(t2.spline_bases)
    @test [(b.name, length(b.columns)) for b in tb.blocks] ==
        [(:fixed, 3), (:rr, 2), (:rn, 2), (:nr, 4)]
    @test tb.blocks[2].columns[1] === :t2_xz_Zrr_1
    @test t2.roles[:t2_xz_Znr_4] === :predictor
end

@testset "spline basis properties" begin
    rk = ReactiveKernelsPPL
    x = collect(range(-2.0, 3.0; length = 25))
    z = collect(range(0.0, 1.0; length = 25)) .^ 1.5 .* 4 .- 1.0
    # TPS: the unpenalized block is the centered linear column alone (no
    # constant column), static widths, determinism.
    for k in (3, 6)
        X, Z = rk._rk_spline_basis_tps(x; k = k)
        @test size(X) == (25, 1) && size(Z) == (25, k - 2)
        @test X[:, 1] ≈ x .- sum(x) / length(x)
        @test abs(sum(X)) < 1e-12
    end
    X1, Z1 = rk._rk_spline_basis_tps(x; k = 5)
    X2, Z2 = rk._rk_spline_basis_tps(x; k = 5)
    @test X1 == X2 && Z1 == Z2
    # t2: centered fixed block, static widths from the tuple.
    for (k, w) in (((3, 3), (3, 1, 2, 2)), ((5, 5), (3, 9, 6, 6)))
        fit = rk._rk_fit_t2(x, z; k = k)
        F, RR, RN, NR = rk._rk_apply_t2(fit, x, z)
        @test (size(F, 2), size(RR, 2), size(RN, 2), size(NR, 2)) == w
        @test all(abs.(sum(F; dims = 1)) .< 1e-9)
    end
end

@testset "spline layout" begin
    bound = bind_data(_svalid_plan(), _spline_cols())
    layout = assign_layout(bound)
    @test [e.kind for e in layout.entries] ==
        [:sampled, :sampled, :spline, :spline, :spline]
    @test [e.size for e in layout.entries] == [1, 1, 1, 2, 1]
    @test [e.transform for e in layout.entries] ==
        [:identity, :exp, :identity, :identity, :exp]
    @test layout.total == 6
    names = coordinate_names(layout)
    @test names[3] == Symbol("b_s_x_fixed.1")
    @test names[end] === Symbol("sd_s_x.1")
    u = [0.3, 0.2, -0.1, 0.4, 0.0, -0.2]
    nt = constrain(layout, u)
    @test nt.sd_s_x == [exp(-0.2)]
    @test unconstrain(layout, nt) ≈ u
    @test logjac(layout, u) ≈ u[2] + u[6]
end

# Independent tps reference: `a + X*b + Z*(sd*b_raw)` with explicit
# names/indices (no contract helpers — this pins them). `X` is the
# centered raw axis, built here from `x` (SB's extra constant column is
# absent by design).
function _ref_spline_tps(bound, nt)
    x = bound.columns[:x]
    X = reshape(x .- sum(x) / length(x), :, 1)
    Z = hcat(bound.columns[:s_x_Zpen_1], bound.columns[:s_x_Zpen_2])
    eta = nt.a .+ X * nt.b_s_x_fixed .+
        Z * (nt.sd_s_x[1] .* nt.b_s_x_raw)
    ll = sum(logpdf.(Normal.(eta, nt.sigma), bound.columns[:y]))
    pr = logpdf(Normal(0, 5), nt.a) + logpdf(Exponential(1), nt.sigma) +
        sum(logpdf.(Normal(0, 1), nt.b_s_x_raw)) +
        logpdf(Normal(0, 1), nt.sd_s_x[1])  # `:positive_stan`: no +log(2)
    return (; ll, pr)
end

@testset "spline tps e2e values and gradient" begin
    bound = bind_data(_svalid_plan(), _spline_cols())
    built = build_kernel(bound)
    u = [0.3, 0.2, -0.1, 0.4, 0.0, -0.2]
    nt = constrain(built.layout, u)
    ref = _ref_spline_tps(bound, nt)
    @test _query(built.spec, bound, :likelihood, u) ≈ ref.ll
    @test _query(built.spec, bound, :prior, u) ≈ ref.pr
    @test _query(built.spec, bound, :posterior, u) ≈ ref.ll + ref.pr + u[2] + u[6]
    _check_gradient(built.spec, bound, u)
    # No ridge with the intercept: the constant vector is outside the span
    # of the spline columns (SB's basis contained it exactly).
    B = hcat(bound.columns[:s_x_Xnull_1], bound.columns[:s_x_Zpen_1],
        bound.columns[:s_x_Zpen_2])
    n = size(B, 1)
    @test sqrt(sum(abs2, ones(n) .- B * (B \ ones(n)))) > 0.1 * sqrt(n)
end

# A stated smoothing-sd prior's support follows its family: real-support
# families ride the Stan-kernel half (`:positive_stan`, plain `_lpdf`, no
# `+log(2)`); positive-support families and the bounding `Uniform` keep
# their own support and density. Positive families used to pass lowering
# and then die at `build_kernel` ("positive_stan override needs a
# real-support family").
@testset "spline stated sd prior support follows its family" begin
    u = [0.3, 0.2, -0.1, 0.4, 0.0, -0.2]
    cases = (
        (:(Normal(0, 2)), Normal(0, 2), :positive_stan),
        (:(StudentT(3, 0, 2)), nothing, :positive_stan),
        (:(LogNormal(0, 1)), LogNormal(0, 1), nothing),
        (:(Gamma(2, 0.5)), Gamma(2, 0.5), nothing),
        (:(Exponential(2)), Exponential(2), nothing),
        (:(InverseGamma(3, 2)), InverseGamma(3, 2), nothing),
        (:(Uniform(0, 10)), Uniform(0, 10), nothing),
    )
    for (spell, dist, support) in cases
        plan = lower_rkppl(quote
                spline_basis(:s_x, x; k = 4, sd = $spell)
                a ~ Normal(0, 5)
                sigma ~ Exponential(1)
                mu = a .+ spline(:s_x)
                y .~ Normal.(mu, sigma)
            end, (:y, :x); conditioned = (:y, :x))
        sd = only(v for v in plan.spline_vectors if v.name === :sd_s_x)
        @test sd.support_override === support
        @test validate_structure(plan) === nothing
        bound = bind_data(plan, _spline_cols())
        built = build_kernel(bound)
        dist === nothing && continue
        nt = constrain(built.layout, u)
        ref = _ref_spline_tps(bound, nt)
        s = nt.sd_s_x[1]
        pr = ref.pr - logpdf(Normal(0, 1), s) + logpdf(dist, s)
        jac = dist isa Uniform ? log(s) + log(10 - s) - log(10) : u[6]
        @test _query(built.spec, bound, :likelihood, u) ≈ ref.ll
        @test _query(built.spec, bound, :prior, u) ≈ pr
        @test _query(built.spec, bound, :posterior, u) ≈
            ref.ll + pr + u[2] + jac
        _check_gradient(built.spec, bound, u)
    end
    # A hand-built plan whose positive-family sd still carries the
    # Stan-kernel override is refused by the contract.
    plan = lower_rkppl(quote
            spline_basis(:s_x, x; k = 4, sd = LogNormal(0, 1))
            a ~ Normal(0, 5)
            mu = a .+ spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    vs = copy(plan.spline_vectors)
    i = findfirst(v -> v.name === :sd_s_x, vs)
    vs[i] = SplineVector(vs[i].name, vs[i].family, vs[i].args,
        :positive_stan, vs[i].width, vs[i].basis, vs[i].label)
    # refused: positive-support sd family carrying the Stan-kernel override (IR contract)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = vs))
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _spline_reactant(plan, cols)
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_spline_reactant_measure, built, bound, post_q, u)
end

function _spline_reactant_measure(built, bound, post_q, u)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), native, primal, val, g,
        rval = Float64(rval), rgrad = Array(rgrad))
end

# Independent t2 reference: SB three-block shape, sd in (rr,rn,nr) order.
function _ref_spline_t2(bound, nt)
    X = hcat(bound.columns[:t2_xz_Xfixed_1], bound.columns[:t2_xz_Xfixed_2],
        bound.columns[:t2_xz_Xfixed_3])
    RR = hcat(bound.columns[:t2_xz_Zrr_1])
    RN = hcat(bound.columns[:t2_xz_Zrn_1], bound.columns[:t2_xz_Zrn_2])
    NR = hcat(bound.columns[:t2_xz_Znr_1], bound.columns[:t2_xz_Znr_2])
    sd = nt.sd_t2_xz
    eta = nt.a .+ X * nt.b_t2_xz_fixed .+
        RR * (sd[1] .* nt.b_t2_xz_rr_raw) .+
        RN * (sd[2] .* nt.b_t2_xz_rn_raw) .+
        NR * (sd[3] .* nt.b_t2_xz_nr_raw)
    ll = sum(logpdf.(Normal.(eta, nt.sigma), bound.columns[:y]))
    pr = logpdf(Normal(0, 5), nt.a) + logpdf(Exponential(1), nt.sigma) +
        sum(logpdf.(Normal(0, 1), nt.b_t2_xz_rr_raw)) +
        sum(logpdf.(Normal(0, 1), nt.b_t2_xz_rn_raw)) +
        sum(logpdf.(Normal(0, 1), nt.b_t2_xz_nr_raw)) +
        sum(logpdf.(Normal(0, 1), sd))  # `:positive_stan`: no +3log(2)
    return (; ll, pr)
end

@testset "spline t2 e2e values and gradient" begin
    plan = lower_rkppl(quote
            spline_basis(:t2_xz, x, z; k = (3, 3))
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ spline(:t2_xz)
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :z); conditioned = (:y, :x, :z))
    bound = bind_data(plan, _spline_cols(; t2 = true))
    built = build_kernel(bound)
    @test built.layout.total == 13
    u = collect(range(-0.3, 0.3; length = 13))
    nt = constrain(built.layout, u)
    ref = _ref_spline_t2(bound, nt)
    @test _query(built.spec, bound, :likelihood, u) ≈ ref.ll
    @test _query(built.spec, bound, :prior, u) ≈ ref.pr
    jac = u[2] + u[end-2] + u[end-1] + u[end]
    @test _query(built.spec, bound, :posterior, u) ≈ ref.ll + ref.pr + jac
    _check_gradient(built.spec, bound, u)
end

# Build (evaluates a new generated model), then trace/compile in a call
# made through `Base.invokelatest`: the generated recipe closures are
# newer than the world of the enclosing top-level expression (the
# leveled-Reactant call-shape precedent).
function _spline_reactant(bound)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_spline_reactant_measure, built, bound, post_q,
        u)
end

function _spline_reactant_measure(built, bound, post_q, u)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; native, primal, val, g, rval = Float64(rval),
        rgrad = Array(rgrad), lines = count(==('\n'), hlo))
end

@testset "spline under Reactant" begin
    progs = [
        ("tps", () -> _svalid_plan(), () -> _spline_cols()),
        ("t2", () -> _svalid_plan(; t2 = true),
            () -> _spline_cols(; t2 = true)),
    ]
    for (name, plan_fn, cols_fn) in progs
        @testset "$name" begin
            bound = bind_data(plan_fn(), cols_fn())
            fx = _spline_reactant(bound)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
        end
    end
end

# M1/M2 SB parity at the u probes: RK value vs the independent oracle,
# the RK value pin, the peer lane's BridgeStan full-posterior literal
# (propto=false, jacobian=true), and the SB grads in RK u-order (brief
# 2026-09-27T13-01-42-742-16egzpo on
# BayesianRegressionModels:rk:parity-term-splines-stan, BRM 97bb5388,
# StanBlocks 24578c3, BridgeStan 2.9.0, Julia 1.10.11). The layout is all
# identity/exp, matching Stan's lower-bound kernels — no map gap, so the
# full posterior compares bit-exact (M1, d=0.0) / 1 ulp (M2, d=-7.11e-15).
# The peer verified its SB basis bit-exact vs the RK materialized columns
# (M1 4/4, M2 8/8) before comparing values.
@testset "spline M1/M2 SB-parity pins" begin
    @testset "M1 s k=4" begin
        # Intended divergence: SB's tps keeps a constant null column under
        # a flat prior, RK drops it (decision `1cmodra`, prong
        # `tps-intercept`; it duplicated the intercept, an exact ridge).
        # Folding SB's constant coefficient into the intercept makes the
        # two models identical, so the SB literals still pin RK exactly:
        # SB probe u_sb = [0.3, 0.2, 0.1, -0.1, 0.4, 0.0, -0.2] in the
        # pre-divergence RK order [mu.Intercept, sigma, b_s_x_fixed.1
        # (constant), b_s_x_fixed.2 (linear), b_s_x_raw.1, b_s_x_raw.2,
        # sd_s_x.1] (SB order [pop_mu_beta_pop.1, s_x_b_fixed.1,
        # s_x_b_fixed.2, s_x_sd_pen.1, s_x_b_pen_raw.1, s_x_b_pen_raw.2,
        # sigma]) is RK's u with intercept 0.3 + 0.1, up to the intercept
        # prior `Normal(0, 5)` at 0.4 instead of 0.3.
        bound = bind_data(_svalid_plan(), _spline_cols())
        built = build_kernel(bound)
        lay = built.layout
        @test coordinate_names(lay) == [:a, :sigma,
            Symbol("b_s_x_fixed.1"), Symbol("b_s_x_raw.1"),
            Symbol("b_s_x_raw.2"), Symbol("sd_s_x.1")]
        u = [0.4, 0.2, -0.1, 0.4, 0.0, -0.2]
        nt = constrain(lay, u)
        ref = _ref_spline_tps(bound, nt)
        post = _query(built.spec, bound, :posterior, u)
        @test post ≈ ref.ll + ref.pr + u[2] + u[6]
        # SB full posterior -22.244895346304787 at u_sb, bridged.
        bridged = -22.244895346304787 - logpdf(Normal(0, 5), 0.3) +
            logpdf(Normal(0, 5), 0.4)
        @test abs(post - bridged) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        sb = [-2.7161630767907803, -8.4332465153243046, -2.7041630767907803,
            5.9857744480450616, 2.7109163710758355, 4.5566881239299892,
            1.574046502394695]
        # SB ran on strato2's raw-LAPACK basis; RK's canonical basis
        # (decision 1vts6mb) has penalized column 2 flipped against it.
        # b_s_x_raw.2 is 0 at this u, so the value is unchanged and only
        # that gradient component changes sign. Drop once BRM adopts the
        # canonical signs (BRM todo 1wo7z27).
        sb[6] = -sb[6]
        # SB's constant-coefficient gradient is the likelihood's slope in
        # the intercept; add the moved intercept's prior slope -0.4/25.
        @test maximum(abs.(g .- [sb[3] - 0.4 / 25; sb[[2, 4, 5, 6, 7]]])) <
            1e-10
        _check_gradient(built.spec, bound, u)
    end
    @testset "M2 t2 k=(3,3)" begin
        # u = -0.3:0.05:0.3 (13). RK order [mu.Intercept, sigma,
        # b_t2_xz_fixed.1..3, b_t2_xz_rr_raw.1, b_t2_xz_rn_raw.1..2,
        # b_t2_xz_nr_raw.1..2, sd_t2_xz.1..3]; SB order
        # [pop_mu_beta_pop.1, t2_mu_x_z_b_fixed.1..3,
        # t2_mu_x_z_sd_pen.1..3, t2_mu_x_z_b_rr_raw.1,
        # t2_mu_x_z_b_rn_raw.1..2, t2_mu_x_z_b_nr_raw.1..2, sigma].
        plan = lower_rkppl(quote
                spline_basis(:t2_xz, x, z; k = (3, 3))
                a ~ Normal(0, 5)
                sigma ~ Exponential(1)
                mu = a .+ spline(:t2_xz)
                y .~ Normal.(mu, sigma)
            end, (:y, :x, :z); conditioned = (:y, :x, :z))
        bound = bind_data(plan, _spline_cols(; t2 = true))
        built = build_kernel(bound)
        lay = built.layout
        @test coordinate_names(lay) ==
            [:a, :sigma, Symbol("b_t2_xz_fixed.1"),
                Symbol("b_t2_xz_fixed.2"), Symbol("b_t2_xz_fixed.3"),
                Symbol("b_t2_xz_rr_raw.1"), Symbol("b_t2_xz_rn_raw.1"),
                Symbol("b_t2_xz_rn_raw.2"), Symbol("b_t2_xz_nr_raw.1"),
                Symbol("b_t2_xz_nr_raw.2"), Symbol("sd_t2_xz.1"),
                Symbol("sd_t2_xz.2"), Symbol("sd_t2_xz.3")]
        u = collect(range(-0.3, 0.3; length = 13))
        nt = constrain(lay, u)
        ref = _ref_spline_t2(bound, nt)
        post = _query(built.spec, bound, :posterior, u)
        jac = u[2] + u[end-2] + u[end-1] + u[end]
        @test post ≈ ref.ll + ref.pr + jac
        @test abs(post - (-27.739270897924577)) < 1e-12
        # SB full posterior -27.739270897924584, 1 ulp from RK.
        @test abs(post - (-27.739270897924584)) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        sb = [7.2585406630028375, 2.3240804584385835, 0.98991856030834768,
            1.0727353584400312, -0.33308773248555568, 0.1091189056022403,
            -0.204465212786443, -0.045932924265366423, -0.42583579743795452,
            -0.19444326886339713, -0.49478064292138235, -0.64851791691339633,
            -0.86136887046381405]
        @test maximum(abs.(g .- sb)) < 1e-10
        _check_gradient(built.spec, bound, u)
    end
end
@testset "spline tps HLO length invariance" begin
    # Data-length invariance (constraints.md): more rows must not
    # replicate the loop body. (Value+grad Reactant parity lives in
    # "spline under Reactant" above; this leg adds only the HLO check
    # against the shared `_spline_reactant` helper.)
    small = _spline_reactant(bind_data(_svalid_plan(), _spline_cols()))
    large = _spline_reactant(bind_data(_svalid_plan(),
        _spline_cols(; n = 24)))
    @test small.lines == large.lines
end
