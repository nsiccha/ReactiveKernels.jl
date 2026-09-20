# Spline contract: in-graph s/t2 bases + vector priors + smoothing (SB
# `_sb_s_generic`/`_sb_t2_generic` mirror). Surface declarations, IR
# validation, bind-time materialization, layout, and end-to-end
# values/gradients vs independent Distributions.jl references. The port
# itself is verified against BRM by a /tmp differential (BRM is not a test
# dep); committed here are structural properties + hand references.

function _svalid_plan(; t2::Bool = false)
    if t2
        return lower_rkppl(quote
                spline_basis(:t2_xz, x, z; k = (3, 4))
                a ~ Normal(0, 5)
                sigma ~ Exponential(1)
                mu = a .+ spline(:t2_xz)
                y .~ Normal.(mu, sigma)
            end, (:y, :x, :z))
    end
    return lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ spline(:s_x)
            y .~ Normal.(mu, sigma)
        end, (:y, :x))
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
    @test [(b.name, b.width) for b in sb.blocks] == [(:fixed, 2), (:pen, 2)]
    @test all(isempty(b.columns) for b in sb.blocks)
    @test sb.label === :spline_s_x
    got = [(v.name, v.family, v.args, v.support_override, v.width, v.basis)
        for v in plan.spline_vectors]
    @test got == [(:b_s_x_fixed, :flat, NamedTuple(), nothing, 2, :s_x),
        (:b_s_x_raw, :normal, (arg1 = 0, arg2 = 1), nothing, 2, :s_x),
        (:sd_s_x, :normal, (arg1 = 0, arg2 = 1), :positive, 1, :s_x)]
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
        end, (:y, :x, :z))
    sb = only(plan.spline_bases)
    @test sb.kind === :t2 && sb.k == (5, 5)
    @test [(b.name, b.width) for b in sb.blocks] ==
        [(:fixed, 3), (:rr, 9), (:rn, 6), (:nr, 6)]
    got = [v.name for v in plan.spline_vectors]
    @test got == [:b_t2_xz_fixed, :b_t2_xz_rr_raw, :b_t2_xz_rn_raw,
        :b_t2_xz_nr_raw, :sd_t2_xz]
    sd = only(v for v in plan.spline_vectors if v.name === :sd_t2_xz)
    @test (sd.family, sd.support_override, sd.width) ===
        (:normal, :positive, 3)
    # Explicit kind + default k on `s`.
    plain = lower_rkppl(quote
            spline_basis(:s_x, x; kind = :tps)
            a ~ Normal(0, 5)
            mu = a .+ spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test only(plain.spline_bases).k == 10
end

@testset "spline surface fail-closed" begin
    # Unquoted id.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(s_x, x; k = 4)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    # Non-data axis.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, w; k = 4)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    # Three axes.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x, z, w; k = 4)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x, :z, :w))
    # Bad kind value.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; kind = :cr)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    # Kind/arity mismatch both ways.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; kind = :t2)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:t2_xz, x, z; kind = :tps)
            mu = spline(:t2_xz)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x, :z))
    # k too small / non-literal / wrong shape.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 2)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = kk)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = (4, 4))
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:t2_xz, x, z; k = 5)
            mu = spline(:t2_xz)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x, :z))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:t2_xz, x, z; k = (5, 2))
            mu = spline(:t2_xz)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x, :z))
    # Deferred options fail closed.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4, bs = :cr)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    # Duplicate declaration.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            spline_basis(:s_x, x; k = 5)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    # Unknown / reused / negated / nested uses.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            mu = spline(:nope)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            mu = spline(:s_x) .+ spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            mu = a .+ spline(:s_x)
            nu = c .+ spline(:s_x)
            y .~ Normal.(mu, 1.0)
            z .~ Normal.(nu, 1.0)
        end, (:y, :z, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            mu = a .- spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            mu = a .+ b .* spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    # spline() inside definitions (scalar + derived).
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            w = spline(:s_x)
            mu = a .+ w
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            w = spline(:s_x) .+ x
            mu = a .+ w
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            w = spline(:s_x) .+ b .* x
            mu = a .+ w
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    # Reserved names.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline = 1.0
            y .~ Normal.(mu, 1.0)
        end, (:y,))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis ~ Normal(0, 1)
            y .~ Normal.(mu, 1.0)
        end, (:y,))
    # Generated-name claims: user definitions cannot collide.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            b_s_x_fixed = 1.0
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            s_x_Xnull_1 = 1.0
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            spline_basis(:s_x, x; k = 4)
            sd_s_x ~ Normal(0, 1)
            mu = spline(:s_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
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
    @test_throws ContractValidationError validate_structure(
        _smutant(; blocks = badblocks))
    # Vector set: dropped / extra / orphan.
    plan = _svalid_plan()
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = plan.spline_vectors[1:2]))
    extra = vcat(plan.spline_vectors,
        SplineVector[SplineVector(:bogus, :normal, (arg1 = 0, arg2 = 1),
            nothing, 1, :s_x, :bogus)])
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = extra))
    orphan = vcat(plan.spline_vectors,
        SplineVector[SplineVector(:b_nope_fixed, :flat, NamedTuple(), nothing,
            2, :nope, :b_nope_fixed)])
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = orphan))
    # Vector shape: family / args / support / width pinned.
    vs = copy(plan.spline_vectors)
    vs[1] = SplineVector(vs[1].name, :normal, (arg1 = 0, arg2 = 1), nothing,
        vs[1].width, vs[1].basis, vs[1].label)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = vs))
    vs = copy(plan.spline_vectors)
    vs[2] = SplineVector(vs[2].name, vs[2].family, (arg1 = 0, arg2 = 2),
        nothing, vs[2].width, vs[2].basis, vs[2].label)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = vs))
    vs = copy(plan.spline_vectors)
    vs[3] = SplineVector(vs[3].name, vs[3].family, vs[3].args, nothing,
        vs[3].width, vs[3].basis, vs[3].label)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = vs))
    vs = copy(plan.spline_vectors)
    vs[2] = SplineVector(vs[2].name, vs[2].family, vs[2].args,
        vs[2].support_override, 3, vs[2].basis, vs[2].label)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = vs))
    # Linkage: dangling basis / double use / unknown summand target.
    nopred = PredictorSpec(plan.predictors[1].name,
        plan.predictors[1].link, plan.predictors[1].terms[1:1],
        plan.predictors[1].label)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; predictors = PredictorSpec[nopred]))
    two = PredictorSpec(plan.predictors[1].name, plan.predictors[1].link,
        vcat(plan.predictors[1].terms, _summand(:mu, :s_x)),
        plan.predictors[1].label)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; predictors = PredictorSpec[two]))
    bad = PredictorSpec(plan.predictors[1].name, plan.predictors[1].link,
        vcat(plan.predictors[1].terms[1:1], _summand(:mu, :nope)),
        plan.predictors[1].label)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; predictors = PredictorSpec[bad]))
    # kind/k/axes shape.
    @test_throws ContractValidationError validate_structure(
        _smutant(; kind = :cr))
    @test_throws ContractValidationError validate_structure(
        _smutant(; k = (4, 4)))
    @test_throws ContractValidationError validate_structure(
        _smutant(; k = 2))
    @test_throws ContractValidationError validate_structure(
        _smutant(; axes = [:x, :z]))
    # Materialized-name clash with a parameter: rename the id.
    clash = vcat(plan.parameters,
        SampledParameter[SampledParameter(:s_x_Xnull_1, :normal,
            (arg1 = 0, arg2 = 1), nothing, :s_x_Xnull_1)])
    @test_throws ContractValidationError validate_structure(
        _swith(plan; parameters = clash))
    # Spline-vector names join the name tables.
    dup = copy(plan.spline_vectors)
    dup[1] = SplineVector(:a, dup[1].family, dup[1].args,
        dup[1].support_override, dup[1].width, dup[1].basis, dup[1].label)
    @test_throws ContractValidationError validate_structure(
        _swith(plan; vectors = dup))
end

@testset "spline bind materialization" begin
    plan = _svalid_plan()
    bound = bind_data(plan, _spline_cols())
    sb = only(bound.spline_bases)
    @test sb.blocks[1].columns == [:s_x_Xnull_1, :s_x_Xnull_2]
    @test sb.blocks[2].columns == [:s_x_Zpen_1, :s_x_Zpen_2]
    for c in [:s_x_Xnull_1, :s_x_Xnull_2, :s_x_Zpen_1, :s_x_Zpen_2]
        @test haskey(bound.columns, c)
        @test length(bound.columns[c]) == 12
        @test bound.roles[c] === :predictor
    end
    @test bound.roles[:x] === :data
    # Materialized columns equal the port's fit/apply on the same axis.
    fit = ReactiveKernelsPPL._rk_fit_spline(_spline_cols()[:x]; k = 4)
    X, Z = ReactiveKernelsPPL._rk_apply_spline(fit, _spline_cols()[:x])
    @test bound.columns[:s_x_Xnull_1] == X[:, 1]
    @test bound.columns[:s_x_Xnull_2] == X[:, 2]
    @test bound.columns[:s_x_Zpen_1] == Z[:, 1]
    @test bound.columns[:s_x_Zpen_2] == Z[:, 2]
    # Reserved-name exclusivity: caller columns cannot squat basis names.
    cols = _spline_cols()
    cols[:s_x_Xnull_1] = ones(12)
    @test_throws ContractValidationError bind_data(plan, cols)
    # Missing / non-numeric axes.
    @test_throws ContractValidationError bind_data(plan,
        Dict{Symbol,AbstractVector}(:y => _spline_cols()[:y]))
    cols = _spline_cols()
    cols[:x] = fill("a", 12)
    @test_throws ContractValidationError bind_data(plan, cols)
    # Fit errors surface as bind errors (k=4 needs 4 unique x).
    few = Dict{Symbol,AbstractVector}(:x => [1.0, 1.0, 2.0],
        :y => [1.0, 2.0, 3.0])
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
    # TPS: constant-first null column, static widths, determinism.
    for k in (3, 6)
        X, Z = rk._rk_spline_basis_tps(x; k = k)
        @test size(X) == (25, 2) && size(Z) == (25, k - 2)
        @test X[:, 1] == ones(25)
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
        [:coefficient, :sampled, :spline, :spline, :spline]
    @test [e.size for e in layout.entries] == [1, 1, 2, 2, 1]
    @test [e.transform for e in layout.entries] ==
        [:identity, :exp, :identity, :identity, :exp]
    @test layout.total == 7
    names = coordinate_names(layout)
    @test names[3:4] == [Symbol("b_s_x_fixed.1"), Symbol("b_s_x_fixed.2")]
    @test names[end] === Symbol("sd_s_x.1")
    u = [0.3, 0.2, 0.1, -0.1, 0.4, 0.0, -0.2]
    nt = constrain(layout, u)
    @test nt.sd_s_x == [exp(-0.2)]
    @test unconstrain(layout, nt) ≈ u
    @test logjac(layout, u) ≈ u[2] + u[7]
end

# Independent tps reference: SB `a + X*b + Z*(sd*b_raw)` shape with
# explicit names/indices (no contract helpers — this pins them).
function _ref_spline_tps(bound, nt)
    X = hcat(bound.columns[:s_x_Xnull_1], bound.columns[:s_x_Xnull_2])
    Z = hcat(bound.columns[:s_x_Zpen_1], bound.columns[:s_x_Zpen_2])
    eta = nt.mu[1] .+ X * nt.b_s_x_fixed .+
        Z * (nt.sd_s_x[1] .* nt.b_s_x_raw)
    ll = sum(logpdf.(Normal.(eta, nt.sigma), bound.columns[:y]))
    pr = logpdf(Normal(0, 5), nt.mu[1]) + logpdf(Exponential(1), nt.sigma) +
        sum(logpdf.(Normal(0, 1), nt.b_s_x_raw)) +
        logpdf(Normal(0, 1), nt.sd_s_x[1]) + log(2)
    return (; ll, pr)
end

@testset "spline tps e2e values and gradient" begin
    bound = bind_data(_svalid_plan(), _spline_cols())
    built = build_kernel(bound)
    u = [0.3, 0.2, 0.1, -0.1, 0.4, 0.0, -0.2]
    nt = constrain(built.layout, u)
    ref = _ref_spline_tps(bound, nt)
    @test _query(built.spec, bound, :likelihood, u) ≈ ref.ll
    @test _query(built.spec, bound, :prior, u) ≈ ref.pr
    @test _query(built.spec, bound, :posterior, u) ≈ ref.ll + ref.pr + u[2] + u[7]
    _check_gradient(built.spec, bound, u)
end

# Independent t2 reference: SB three-block shape, sd in (rr,rn,nr) order.
function _ref_spline_t2(bound, nt)
    X = hcat(bound.columns[:t2_xz_Xfixed_1], bound.columns[:t2_xz_Xfixed_2],
        bound.columns[:t2_xz_Xfixed_3])
    RR = hcat(bound.columns[:t2_xz_Zrr_1])
    RN = hcat(bound.columns[:t2_xz_Zrn_1], bound.columns[:t2_xz_Zrn_2])
    NR = hcat(bound.columns[:t2_xz_Znr_1], bound.columns[:t2_xz_Znr_2])
    sd = nt.sd_t2_xz
    eta = nt.mu[1] .+ X * nt.b_t2_xz_fixed .+
        RR * (sd[1] .* nt.b_t2_xz_rr_raw) .+
        RN * (sd[2] .* nt.b_t2_xz_rn_raw) .+
        NR * (sd[3] .* nt.b_t2_xz_nr_raw)
    ll = sum(logpdf.(Normal.(eta, nt.sigma), bound.columns[:y]))
    pr = logpdf(Normal(0, 5), nt.mu[1]) + logpdf(Exponential(1), nt.sigma) +
        sum(logpdf.(Normal(0, 1), nt.b_t2_xz_rr_raw)) +
        sum(logpdf.(Normal(0, 1), nt.b_t2_xz_rn_raw)) +
        sum(logpdf.(Normal(0, 1), nt.b_t2_xz_nr_raw)) +
        sum(logpdf.(Normal(0, 1), sd)) + 3 * log(2)
    return (; ll, pr)
end

@testset "spline t2 e2e values and gradient" begin
    plan = lower_rkppl(quote
            spline_basis(:t2_xz, x, z; k = (3, 3))
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ spline(:t2_xz)
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :z))
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
