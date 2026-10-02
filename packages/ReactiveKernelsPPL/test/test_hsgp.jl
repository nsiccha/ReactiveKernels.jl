# HSGP contract: basis IR + surface lowering + validation + bind fit
# (Stage A) and layout + in-graph basis evaluation + matmul summand
# (Stage B, SB `_sb_hsgp` mirror). End-to-end values vs independent
# SB-shape hand references (per-row loops + Distributions oracles, never
# the emitted expressions) plus Enzyme-vs-findiff gradients. (`_query` /
# `_check_gradient` come from test_generator.jl, included first.)
using DifferentiationInterface
using Distributions: LogNormal, Normal, Exponential, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using SpecialFunctions
using Test

function _hvalid_plan(; aniso::Bool = false)
    if aniso
        return lower_rkppl(quote
                hsgp_basis(:h_xz, x, z; k = (4, 3), c = (1.5, 2.0),
                    iso = false)
                a ~ Normal(0, 5)
                sigma ~ Exponential(1)
                mu = a .+ hsgp(:h_xz)
                y .~ Normal.(mu, sigma)
            end, (:y, :x, :z); conditioned = (:y, :x, :z))
    end
    return lower_rkppl(quote
            hsgp_basis(:h_x, x; k = 4)
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, sigma)
        end, (:y, :x); conditioned = (:y, :x))
end

function _hwith(plan::StructuralPlan; predictors = nothing, bases = nothing)
    return StructuralPlan(plan.responses,
        predictors === nothing ? plan.predictors : predictors,
        plan.population_priors, plan.parameters, plan.assignments,
        plan.columns, plan.n_obs; roles = plan.roles,
        derived = plan.derived, levelmaps = plan.levelmaps,
        plate_parameters = plan.plate_parameters, scans = plan.scans,
        varying_draws = plan.varying_draws,
        varying_slices = plan.varying_slices,
        spline_bases = plan.spline_bases,
        spline_vectors = plan.spline_vectors,
        hsgp_bases = bases === nothing ? plan.hsgp_bases : bases)
end

_hsummand(pname::Symbol, id::Symbol) = TermSpec(HSGPSummandTerm,
    ColumnRef[], (hsgp_id = id,), Symbol("hsgp_", pname, "_", id),
    Symbol("hsgp_", pname, "_", id))

function _hsgp_cols()
    return Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5],
        :x => [0.5, -1.0, 1.5, 0.0], :z => [1.0, 0.5, -0.5, 2.0])
end

@testset "hsgp 1D lowering" begin
    plan = _hvalid_plan()
    @test length(plan.hsgp_bases) == 1
    hb = only(plan.hsgp_bases)
    @test hb.id === :h_x
    @test hb.axes == [:x] && hb.K == [4] && hb.c == [1.5] && hb.iso
    @test isempty(hb.fits)
    @test hb.label === :hsgp_h_x
    @test ReactiveKernelsPPL._hsgp_n_basis(hb) == 4
    @test ReactiveKernelsPPL._hsgp_all_names(hb) ==
        [:beta_raw_h_x, :sigma_h_x, :rho_h_x]
    t = only(plan.predictors).terms[2]
    @test t.kind === HSGPSummandTerm && isempty(t.columns)
    @test t.options.hsgp_id === :h_x
    # Defaults: k=20, c=1.5, iso=true (SB `_brm_axis_option` defaults).
    dflt = lower_rkppl(quote
            a ~ Normal(0, 1)
            hsgp_basis(:h_d, x)
            mu = a .+ hsgp(:h_d)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    hb = only(dflt.hsgp_bases)
    @test hb.K == [20] && hb.c == [1.5] && hb.iso
end

@testset "hsgp aniso lowering" begin
    plan = _hvalid_plan(; aniso = true)
    hb = only(plan.hsgp_bases)
    @test hb.axes == [:x, :z] && hb.K == [4, 3]
    @test hb.c == [1.5, 2.0] && !hb.iso
    @test ReactiveKernelsPPL._hsgp_n_basis(hb) == 12
    @test ReactiveKernelsPPL._hsgp_all_names(hb) ==
        [:beta_raw_h_xz, :sigma_h_xz, :rho_h_xz_1, :rho_h_xz_2]
end

@testset "hsgp surface fail-closed" begin
    # Declaration shape.
    # refused: malformed hsgp_basis call, basis id missing (arity; Julia MethodError analogue, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(x)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # refused: degenerate constant axis (literal 1.0 has no spread, L = 0)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, 1.0)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # refused: undeclared name `q` as axis (P6, 05oe96l)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, q)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # refused: unidentified, two axes on one column (x, x)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x, x; k = (2, 2))
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # refused: single assignment, basis :h_x declared twice
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x)
            hsgp_basis(:h_x, x)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # k/c/iso literals.
    # refused: invalid basis count k = 0
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x; k = 0)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # refused: non-integer basis count k = 2.5
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x; k = 2.5)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # refused: k tuple length differs from axis count (malformed)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x, z; k = (2, 3, 4))
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x, :z); conditioned = (:y, :x, :z))
    # refused: expansion factor c must exceed 1 (c = 1 puts the boundary on the data)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x; c = 1.0)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # refused: non-finite expansion factor c = Inf
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x; c = Inf)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # refused: non-Bool `iso = 1` (Julia non-boolean-context TypeError, P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x; iso = 1)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # `by=` names a bound grouping column (grouped bases themselves:
    # test_smooth_sb.jl).
    # refused: undeclared name `g` in by= (P6, 05oe96l)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x; by = g)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # Use-site discipline.
    # refused: undeclared basis id :h_nope (P6, 05oe96l)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .+ hsgp(:h_nope)
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_x, x)
        end, (:y, :x); conditioned = (:y, :x))
    # capability: hsgp summand value reuse: same basis twice in one predictor (values compose, P3/P8) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ hsgp(:h_x) .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_x, x)
        end, (:y, :x); conditioned = (:y, :x)); true)
    # capability: negated hsgp summand (`a .- hsgp(:h)`) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .- hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_x, x)
        end, (:y, :x); conditioned = (:y, :x)); true)
    # capability: literal-scaled hsgp summand (`2.0 .* hsgp(:h)`) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            a ~ Normal(0, 1)
            mu = a .+ 2.0 .* hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_x, x)
        end, (:y, :x); conditioned = (:y, :x)); true)
    # Never inside definitions; never redefined or sampled.
    # capability: hsgp value bound in a definition (`h = hsgp(:h)`; values compose, P3/P8) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            a ~ Normal(0, 1)
            h = hsgp(:h_x)
            mu = a .+ h
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_x, x)
        end, (:y, :x); conditioned = (:y, :x)); true)
    # refused: reserved-name collision `hsgp` (then calls a Float64)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp = 1.0
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_x, x)
        end, (:y, :x); conditioned = (:y, :x))
    # refused: reserved-name collision `hsgp_basis`
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis = 1.0
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # Claims: user definitions cannot collide with sampled names.
    # refused: single assignment, user definition collides with basis-claimed `rho_h_x`
    @test_throws SurfaceLoweringError lower_rkppl(quote
            rho_h_x = 1.0
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_x, x)
        end, (:y, :x); conditioned = (:y, :x))
    # refused: single assignment, `~` collides with basis-claimed `beta_raw_h_x`
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_x, x)
            beta_raw_h_x ~ Normal(0, 1)
        end, (:y, :x); conditioned = (:y, :x))
end

@testset "hsgp contract validation" begin
    good = _hvalid_plan()
    hb = only(good.hsgp_bases)
    # Duplicate ids / labels.
    dup = HSGPBasis(:h_x, [:x], [2], [1.5], true,
        Tuple{Float64,Float64}[], :hsgp_h_x)
    # refused: duplicate basis id/label (IR contract)
    @test_throws ContractValidationError validate_structure(
        _hwith(good; bases = [hb, dup]))
    # K/c shape and values.
    # refused: K length differs from axes (IR contract)
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [HSGPBasis(:h_x, [:x], [2, 3], [1.5], true,
            Tuple{Float64,Float64}[], :hsgp_h_x)]))
    # refused: K = 0 (IR contract)
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [HSGPBasis(:h_x, [:x], [0], [1.5], true,
            Tuple{Float64,Float64}[], :hsgp_h_x)]))
    # refused: c <= 1 (IR contract)
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [HSGPBasis(:h_x, [:x], [2], [1.0], true,
            Tuple{Float64,Float64}[], :hsgp_h_x)]))
    # refused: empty axes (IR contract)
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [HSGPBasis(:h_x, Symbol[], Int[], Float64[], true,
            Tuple{Float64,Float64}[], :hsgp_h_x)]))
    # Fits: wrong count / non-positive L.
    # refused: fits count differs from axes (IR contract)
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [HSGPBasis(:h_x, [:x], [2], [1.5], true,
            [(0.0, 1.0), (0.0, 1.0)], :hsgp_h_x)]))
    # refused: non-positive fit L (IR contract)
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [HSGPBasis(:h_x, [:x], [2], [1.5], true,
            [(0.0, 0.0)], :hsgp_h_x)]))
    # Linkage: dangling / double-use / unknown summand.
    nopred = _hwith(good; predictors = PredictorSpec[
        PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                :Intercept, :mu_intercept)], :mu)])
    # refused: dangling basis with no summand (IR contract)
    @test_throws ContractValidationError validate_structure(nopred)
    twopred = _hwith(good; predictors = vcat(good.predictors,
        [PredictorSpec(:sg, IdentityLink, TermSpec[_hsummand(:sg, :h_x)],
            :sg)]))
    # refused: one basis feeding two predictors (IR contract; mirrors capability C at :166)
    @test_throws ContractValidationError validate_structure(twopred)
    baduse = _hwith(good; predictors = PredictorSpec[
        PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                :Intercept, :mu_intercept), _hsummand(:mu, :h_nope)],
            :mu)])
    # refused: summand names an unknown basis (IR contract)
    @test_throws ContractValidationError validate_structure(baduse)
    # Summand shape: options / columns / addressee.
    for (opts, cols, addr) in (((foo = 1,), Symbol[], :hsgp_mu_h_x),
            ((hsgp_id = :h_x,), Symbol[:x], :hsgp_mu_h_x),
            ((hsgp_id = :h_x,), Symbol[], :nope))
        t = TermSpec(HSGPSummandTerm, cols, opts, addr, :hsgp_mu_h_x)
        badpred = _hwith(good; predictors = PredictorSpec[
            PredictorSpec(:mu, IdentityLink,
                TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                    :Intercept, :mu_intercept), t], :mu)])
        # refused: malformed HSGP summand options/columns/addressee (IR contract)
        @test_throws ContractValidationError validate_structure(badpred)
    end
    # Name tables: a hand-built parameter under an hsgp name fails.
    clash = _hwith(good)
    push!(clash.parameters, SampledParameter(:rho_h_x, :normal,
        (arg1 = 0, arg2 = 1), nothing, :rho_h_x))
    # refused: hand-built parameter under an hsgp-claimed name (IR contract)
    @test_throws ContractValidationError validate_structure(clash)
end

@testset "hsgp bind fit" begin
    cols = _hsgp_cols()
    bound = bind_data(_hvalid_plan(), cols)
    hb = only(bound.hsgp_bases)
    # mu = mean(x), L = c*max|x-mu| (SB `_brm_fit_hsgp` verbatim).
    @test hb.fits == [(0.25, 1.875)]
    @test bound.roles[:x] === :predictor
    abound = bind_data(_hvalid_plan(; aniso = true), cols)
    ahb = only(abound.hsgp_bases)
    @test ahb.fits == [(0.25, 1.875), (0.75, 2.5)]
    # Floors (SB `_brm_hsgp_rho_lower[_s]` verbatim).
    fl = ReactiveKernelsPPL._hsgp_floors([4], hb.fits, true)
    @test fl ≈ [(4 * 1.875 / pi) * sqrt(log(100.0) / 15)]
    afl = ReactiveKernelsPPL._hsgp_floors([4, 3], ahb.fits, false)
    @test afl ≈ [(4 * 1.875 / pi) * sqrt(log(100.0) / 15),
        (4 * 2.5 / pi) * sqrt(log(100.0) / 8)]
    isofl = ReactiveKernelsPPL._hsgp_floors([4, 3], ahb.fits, true)
    @test isofl == [maximum(afl)]
    # K=1 stays unbounded (SB degenerate-basis rule).
    @test ReactiveKernelsPPL._hsgp_floors([1], [(0.0, 1.0)], true) == [0.0]
    # Bind fail-closed: degenerate / non-numeric / unbound axes.
    constcols = Dict{Symbol,AbstractVector}(:y => cols[:y],
        :x => fill(2.0, 4))
    # refused: degenerate constant axis column at bind (L = 0)
    @test_throws ContractValidationError bind_data(_hvalid_plan(), constcols)
    strcols = Dict{Symbol,AbstractVector}(:y => cols[:y],
        :x => ["a", "b", "c", "d"])
    # refused: wrong eltype, non-numeric axis column
    @test_throws ContractValidationError bind_data(_hvalid_plan(), strcols)
    nancols = Dict{Symbol,AbstractVector}(:y => cols[:y],
        :x => [0.5, NaN, 1.5, 0.0])
    # refused: non-finite (NaN) axis data
    @test_throws ContractValidationError bind_data(_hvalid_plan(), nancols)
end

@testset "hsgp layout" begin
    bound = bind_data(_hvalid_plan(), _hsgp_cols())
    layout = assign_layout(bound)
    # SB `_sb_hsgp` declaration order per basis (rho, sigma, beta),
    # appended after the slice-1 entries so peer offsets never move.
    kinds = [(e.kind, e.name, e.size, e.transform) for e in layout.entries]
    @test kinds == [(:sampled, :a, 1, :identity),
        (:sampled, :sigma, 1, :exp),
        (:sampled, :rho_h_x, 1, :floored),
        (:sampled, :sigma_h_x, 1, :exp),
        (:hsgp, :beta_raw_h_x, 4, :identity)]
    rho_e = layout.entries[3]
    @test rho_e.lo ≈ (4 * 1.875 / pi) * sqrt(log(100.0) / 15)
    @test isnan(rho_e.hi)
    @test layout.total == 8
    names = coordinate_names(layout)
    @test names[3:8] == [:rho_h_x, :sigma_h_x,
        Symbol("beta_raw_h_x.1"), Symbol("beta_raw_h_x.2"),
        Symbol("beta_raw_h_x.3"), Symbol("beta_raw_h_x.4")]
    # Jacobian: the exp/floored coords only (beta is identity).
    u = collect(0.1:0.1:0.8)
    @test logjac(layout, u) ≈ u[2] + u[3] + u[4]
    nt = constrain(layout, u)
    @test nt.rho_h_x ≈ rho_e.lo + exp(u[3])
    @test nt.sigma_h_x ≈ exp(u[4])
    @test Vector(nt.beta_raw_h_x) ≈ u[5:8]
    @test unconstrain(layout, nt) ≈ u
    # Aniso: per-axis floors on d rho scalars, then sigma, then beta.
    abound = bind_data(_hvalid_plan(; aniso = true), _hsgp_cols())
    alayout = assign_layout(abound)
    akinds = [(e.kind, e.name, e.size, e.transform) for e in alayout.entries]
    @test akinds == [(:sampled, :a, 1, :identity),
        (:sampled, :sigma, 1, :exp),
        (:sampled, :rho_h_xz_1, 1, :floored),
        (:sampled, :rho_h_xz_2, 1, :floored),
        (:sampled, :sigma_h_xz, 1, :exp),
        (:hsgp, :beta_raw_h_xz, 12, :identity)]
    @test alayout.entries[3].lo ≈
        (4 * 1.875 / pi) * sqrt(log(100.0) / 15)
    @test alayout.entries[4].lo ≈ (4 * 2.5 / pi) * sqrt(log(100.0) / 8)
    @test alayout.total == 17
    # K=1: the zero floor routes rho to plain :exp (bit-identical to a
    # zero-lo :floored edge).
    k1 = lower_rkppl(quote
            hsgp_basis(:h_1, x; k = 1)
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ hsgp(:h_1)
            y .~ Normal.(mu, sigma)
        end, (:y, :x); conditioned = (:y, :x))
    k1layout = assign_layout(bind_data(k1, _hsgp_cols()))
    k1rho = only(e for e in k1layout.entries if e.name === :rho_h_1)
    @test (k1rho.kind, k1rho.transform) === (:sampled, :exp)
    # Two bases: per-basis triples in plan order (offsets compose).
    two = lower_rkppl(quote
            hsgp_basis(:h_a, x; k = 2)
            hsgp_basis(:h_b, x; k = 3)
            a ~ Normal(0, 1)
            mu = a .+ hsgp(:h_a) .+ hsgp(:h_b)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    twolayout = assign_layout(bind_data(two, _hsgp_cols()))
    twokinds =
        [(e.kind, e.name, e.size, e.transform) for e in twolayout.entries]
    @test twokinds == [(:sampled, :a, 1, :identity),
        (:sampled, :rho_h_a, 1, :floored),
        (:sampled, :sigma_h_a, 1, :exp),
        (:hsgp, :beta_raw_h_a, 2, :identity),
        (:sampled, :rho_h_b, 1, :floored),
        (:sampled, :sigma_h_b, 1, :exp),
        (:hsgp, :beta_raw_h_b, 3, :identity)]
    @test twolayout.total == 10
    @test build_kernel(bind_data(two, _hsgp_cols())).layout.total == 10
end

@testset "hsgp floored bijector" begin
    # The parameterized library entry (the `interval_bijector` mirror):
    # offset-exp with the bare-`u` Stan kernel Jacobian.
    lo = 1.322782995409751
    u = 0.35
    x = lo + exp(u)
    EP = ReactiveKernelsPPL._prepared_floored_endpoint
    @test EP(lo, :constrain)(u) ≈ x
    @test EP(lo, :logjac)(u) ≈ u
    @test EP(lo, :unconstrain)(x) ≈ u
    @test EP(lo, :constrain)(u) > lo
end

# Single-expression wrappers: `@test_throws T (f(x))` parses as one macro
# argument, so the multi-arg private calls route through these.
_hsgp_basis_statements_bad(bad) =
    ReactiveKernelsPPL._hsgp_basis_statements(bad)
_hsgp_summand_expr_bad(bound) =
    ReactiveKernelsPPL._hsgp_summand_expr(bound, :h_nope)

@testset "hsgp codegen fail-closed" begin
    bound = bind_data(_hvalid_plan(), _hsgp_cols())
    hb = only(bound.hsgp_bases)
    emptyfits = HSGPBasis(hb.id, hb.axes, hb.K, hb.c, hb.iso,
        Tuple{Float64,Float64}[], hb.label)
    bad = _hwith(bound; bases = [emptyfits])
    # The public path fails at validation; the layout/emitter guards are
    # loud defense in depth behind it.
    # refused: bound basis with empty fits (IR contract)
    @test_throws ContractValidationError build_kernel(bad)
    # refused: bound basis with empty fits, layout guard (IR contract)
    @test_throws ContractValidationError assign_layout(bad)
    # refused: bound basis with empty fits, emitter guard (IR contract)
    @test_throws ContractValidationError _hsgp_basis_statements_bad(bad)
    # refused: summand for unknown basis id, emitter guard (IR contract)
    @test_throws ContractValidationError _hsgp_summand_expr_bad(bound)
end

# Independent posterior reference for the `_hvalid_plan` shapes (one
# basis, intercept-only `mu`, `sigma ~ Exponential(1)` likelihood
# scale): SB `_brm_apply_hsgp` loop nests + `brm_hsgp_sqrt_spd` op order
# + Distributions oracles. Constrained values come from `constrain`
# (locked absolutely by "hsgp layout"); the Jacobian is hand-summed
# from coordinates, never `logjac`.
function _hsgp_ref_posterior(bound::StructuralPlan, layout::LayoutTable,
        u::AbstractVector{<:Real})
    hb = only(bound.hsgp_bases)
    n = bound.n_obs
    y = Vector{Float64}(bound.columns[:y])
    nt = constrain(layout, u)
    hsgp = ReactiveKernelsPPL._hsgp_names(hb)
    a = nt.a
    sig = Float64(nt.sigma)
    rhos = [Float64(getproperty(nt, r)) for r in hsgp.rhos]
    sigh = Float64(getproperty(nt, hsgp.sigma))
    beta = Vector{Float64}(getproperty(nt, hsgp.beta))
    axisPHI = map(enumerate(hb.axes)) do (j, ax)
        x = Vector{Float64}(bound.columns[ax])
        mu, L = hb.fits[j]
        lam = [(k * pi / (2 * L))^2 for k in 1:hb.K[j]]
        P = zeros(n, hb.K[j])
        for k in 1:hb.K[j], i in 1:n
            P[i, k] = (1 / sqrt(L)) * sin(sqrt(lam[k]) * (x[i] - mu + L))
        end
        (P, lam)
    end
    d = length(hb.axes)
    K = Tuple(hb.K)
    M = prod(K)
    PHI = zeros(n, M)
    o2 = zeros(M, d)
    for (b, I) in enumerate(CartesianIndices(K))
        for i in 1:n
            v = 1.0
            for j in 1:d
                v *= axisPHI[j][1][i, I[j]]
            end
            PHI[i, b] = v
        end
        for j in 1:d
            o2[b, j] = axisPHI[j][2][I[j]]
        end
    end
    rr = hb.iso ? fill(rhos[1], d) : rhos
    scale = sigh
    for j in 1:d
        scale *= sqrt(rr[j] * 2.5066282746310002)
    end
    sspd = Vector{Float64}(undef, M)
    for b in 1:M
        ex = 0.0
        for j in 1:d
            ex += rr[j] * rr[j] * o2[b, j]
        end
        sspd[b] = scale * exp(-0.25 * ex)
    end
    muv = a .+ PHI * (sspd .* beta)
    ll = sum(logpdf.(Normal.(muv, sig), y))
    pr = logpdf(Normal(0, 5), a) + logpdf(Exponential(1), sig) +
        sum(logpdf(LogNormal(0, 1), r) for r in rhos) +
        logpdf(LogNormal(0, 1), sigh) + sum(logpdf.(Normal(0, 1), beta))
    cnames = coordinate_names(layout)
    jac = sum(u[findfirst(==(s), cnames)]
        for s in [:sigma, hsgp.rhos..., hsgp.sigma])
    return ll + pr + jac
end

@testset "hsgp 1d end to end" begin
    bound = bind_data(_hvalid_plan(), _hsgp_cols())
    built = build_kernel(bound)
    u = [0.2, -0.1, 0.15, 0.05, 0.3, -0.2, 0.1, 0.0]
    ref = _hsgp_ref_posterior(bound, built.layout, u)
    @test isapprox(_query(built.spec, bound, :posterior, u), ref; rtol = 1e-12)
    _check_gradient(built.spec, bound, u)
end

@testset "hsgp aniso end to end" begin
    bound = bind_data(_hvalid_plan(; aniso = true), _hsgp_cols())
    built = build_kernel(bound)
    u = [0.2, -0.1, 0.15, -0.05, 0.05, 0.3, -0.2, 0.1, 0.0, 0.25, -0.15,
        0.05, 0.1, -0.3, 0.2, -0.1, 0.12]
    ref = _hsgp_ref_posterior(bound, built.layout, u)
    @test isapprox(_query(built.spec, bound, :posterior, u), ref; rtol = 1e-12)
    _check_gradient(built.spec, bound, u)
end

@testset "hsgp k1 end to end" begin
    k1 = lower_rkppl(quote
            hsgp_basis(:h_1, x; k = 1)
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ hsgp(:h_1)
            y .~ Normal.(mu, sigma)
        end, (:y, :x); conditioned = (:y, :x))
    bound = bind_data(k1, _hsgp_cols())
    built = build_kernel(bound)
    u = [0.2, -0.1, 0.15, 0.05, 0.3]
    ref = _hsgp_ref_posterior(bound, built.layout, u)
    @test isapprox(_query(built.spec, bound, :posterior, u), ref; rtol = 1e-12)
    _check_gradient(built.spec, bound, u)
end

function _hperiodic_plan(; k::Int = 4, period::Float64 = 2.0)
    return lower_rkppl(quote
            hsgp_basis(:h_p, x; k = $k, cov = :periodic, period = $period)
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ hsgp(:h_p)
            y .~ Normal.(mu, sigma)
        end, (:y, :x); conditioned = (:y, :x))
end

@testset "hsgp periodic lowering" begin
    plan = _hperiodic_plan()
    @test length(plan.hsgp_bases) == 1
    hb = only(plan.hsgp_bases)
    @test hb.id === :h_p
    @test hb.axes == [:x] && hb.K == [4] && hb.c == [1.5] && hb.iso
    @test hb.cov === :periodic && hb.period == 2.0
    @test isempty(hb.fits)
    @test hb.label === :hsgp_h_p
    @test ReactiveKernelsPPL._hsgp_n_basis(hb) == 8
    @test ReactiveKernelsPPL._hsgp_all_names(hb) ==
        [:beta_raw_h_p, :sigma_h_p, :rho_h_p]
    t = only(plan.predictors).terms[2]
    @test t.kind === HSGPSummandTerm && isempty(t.columns)
    @test t.options.hsgp_id === :h_p
    # Defaults: k=20 like exp-quad; explicit cov=:exp_quad keeps NaN period.
    dflt = lower_rkppl(quote
            a ~ Normal(0, 1)
            hsgp_basis(:h_d, x; cov = :periodic, period = 1.0)
            mu = a .+ hsgp(:h_d)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    dhb = only(dflt.hsgp_bases)
    @test dhb.K == [20] && dhb.cov === :periodic && dhb.period == 1.0
    @test ReactiveKernelsPPL._hsgp_n_basis(dhb) == 40
    eq = lower_rkppl(quote
            a ~ Normal(0, 1)
            hsgp_basis(:h_e, x; k = 3, cov = :exp_quad)
            mu = a .+ hsgp(:h_e)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    ehb = only(eq.hsgp_bases)
    @test ehb.cov === :exp_quad && isnan(ehb.period)
    # `c` is accepted with periodic (SB validates its form, ignores its
    # value — no domain).
    cacc = lower_rkppl(quote
            a ~ Normal(0, 1)
            hsgp_basis(:h_c, x; k = 3, c = 2.5, cov = :periodic, period = 1.0)
            mu = a .+ hsgp(:h_c)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    @test only(cacc.hsgp_bases).c == [2.5]
end

@testset "hsgp periodic surface fail-closed" begin
    # cov spelling.
    # capability: Matern-covariance HSGP (cov = :matern; nu should be explicit, e.g. :matern32/:matern52) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            a ~ Normal(0, 1)
            hsgp_basis(:h_p, x; k = 4, cov = :matern, period = 2.0)
            mu = a .+ hsgp(:h_p)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x)); true)
    # refused: bare `periodic` is an undeclared name (P6, 05oe96l)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_p, x; k = 4, cov = periodic, period = 2.0)
            mu = a .+ hsgp(:h_p)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # period required iff periodic (SB `_brm_gp_period`).
    # refused: periodic kernel without its period; no defaulted/minted period (P2; P7, 0d5a67r)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_p, x; k = 4, cov = :periodic)
            mu = a .+ hsgp(:h_p)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # refused: `period=` means nothing without cov=:periodic (P2)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x; k = 4, period = 2.0)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    # refused: `period=` means nothing for cov=:exp_quad (P2)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x; k = 4, cov = :exp_quad, period = 2.0)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x))
    for bad in (0.0, -1.0, Inf, NaN, "2.0")
        # refused: invalid period literal (non-positive / non-finite / non-numeric)
        @test_throws SurfaceLoweringError lower_rkppl(quote
                hsgp_basis(:h_p, x; k = 4, cov = :periodic, period = $bad)
                mu = a .+ hsgp(:h_p)
                y .~ Normal.(mu, 1.0)
            end, (:y, :x); conditioned = (:y, :x))
    end
    # One isotropic axis (SB "periodic hsgp requires one isotropic axis").
    # capability: multi-axis periodic HSGP (the SB one-axis limit is not a principle, P10) (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            a ~ Normal(0, 1)
            hsgp_basis(:h_p, x, z; k = (4, 3), cov = :periodic, period = 2.0)
            mu = a .+ hsgp(:h_p)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x, :z); conditioned = (:y, :x, :z)); true)
    # capability: anisotropic (iso=false) periodic HSGP (todo `0bfiemp`)
    @test_broken (lower_rkppl(quote
            a ~ Normal(0, 1)
            hsgp_basis(:h_p, x; k = 4, cov = :periodic, period = 2.0,
                iso = false)
            mu = a .+ hsgp(:h_p)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x); conditioned = (:y, :x)); true)
    # Periodic claims the same names (collision still loud).
    # refused: single assignment, user definition collides with basis-claimed `rho_h_p`
    @test_throws SurfaceLoweringError lower_rkppl(quote
            rho_h_p = 1.0
            mu = a .+ hsgp(:h_p)
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_p, x; k = 4, cov = :periodic, period = 2.0)
        end, (:y, :x); conditioned = (:y, :x))
end

@testset "hsgp periodic contract validation" begin
    good = _hperiodic_plan()
    hb = only(good.hsgp_bases)
    per(args...) = HSGPBasis(args..., :periodic, 2.0)
    # cov membership / periodic shape / period / fits.
    # refused: unknown cov :matern (IR contract; mirrors C at :577)
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [HSGPBasis(hb.id, hb.axes, hb.K, hb.c, hb.iso, hb.fits,
            hb.label, :matern, 2.0)]))
    # refused: periodic with two axes (IR contract; mirrors C at :611)
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [per(hb.id, [:x, :z], [4, 3], [1.5, 2.0], true,
            Tuple{Float64,Float64}[], hb.label)]))
    # refused: periodic with iso=false (IR contract; mirrors C at :616)
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [per(hb.id, [:x], [4], [1.5], false,
            Tuple{Float64,Float64}[], hb.label)]))
    for badperiod in (NaN, 0.0, -2.0, Inf)
        # refused: invalid period (IR contract)
        @test_throws ContractValidationError validate_structure(_hwith(good;
            bases = [HSGPBasis(hb.id, hb.axes, hb.K, hb.c, hb.iso, hb.fits,
                hb.label, :periodic, badperiod)]))
    end
    # refused: fits on a periodic basis (IR contract)
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [per(hb.id, hb.axes, hb.K, hb.c, hb.iso, [(0.0, 1.0)],
            hb.label)]))
    # exp_quad with a period set is inconsistent (period iff periodic).
    # refused: exp_quad basis carrying a period (IR contract)
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [HSGPBasis(hb.id, hb.axes, hb.K, hb.c, hb.iso, hb.fits,
            hb.label, :exp_quad, 2.0)]))
end

@testset "hsgp periodic bind" begin
    cols = _hsgp_cols()
    bound = bind_data(_hperiodic_plan(), cols)
    hb = only(bound.hsgp_bases)
    @test isempty(hb.fits)
    @test hb.period == 2.0 && hb.cov === :periodic
    @test bound.roles[:x] === :predictor
    # A constant axis is a usable periodic domain (no degeneracy gate —
    # the exp-quad L == 0 rejection does not apply).
    constcols = Dict{Symbol,AbstractVector}(:y => cols[:y],
        :x => fill(2.0, 4))
    cbound = bind_data(_hperiodic_plan(), constcols)
    @test isempty(only(cbound.hsgp_bases).fits)
    # Bind fail-closed: non-numeric / non-finite axes.
    strcols = Dict{Symbol,AbstractVector}(:y => cols[:y],
        :x => ["a", "b", "c", "d"])
    # refused: wrong eltype, non-numeric axis column
    @test_throws ContractValidationError bind_data(_hperiodic_plan(), strcols)
    nancols = Dict{Symbol,AbstractVector}(:y => cols[:y],
        :x => [0.5, NaN, 1.5, 0.0])
    # refused: non-finite (NaN) axis data
    @test_throws ContractValidationError bind_data(_hperiodic_plan(), nancols)
    # Codegen/layout guards behind validation: fits on a periodic basis.
    withfits = HSGPBasis(hb.id, hb.axes, hb.K, hb.c, hb.iso, [(0.0, 1.0)],
        hb.label, :periodic, 2.0)
    bad = _hwith(bound; bases = [withfits])
    # refused: fits on a periodic basis, build guard (IR contract)
    @test_throws ContractValidationError build_kernel(bad)
    # refused: fits on a periodic basis, layout guard (IR contract)
    @test_throws ContractValidationError assign_layout(bad)
end

@testset "hsgp periodic floor" begin
    fl = ReactiveKernelsPPL._hsgp_periodic_rho_lower
    # K=1 stays unbounded (the exp-quad degenerate-basis rule).
    @test fl(1) == 0.0
    # Defining property: at a = 1/floor^2 the K-th harmonic's spectral
    # amplitude ratio is 1e-4 (SB `_brm_hsgp_periodic_rho_lower`
    # rule — checked through the scaled Bessel ratio, the bisection's
    # own residual).
    for K in (2, 3, 4, 8, 20)
        a = 1 / fl(K)^2
        @test besselix(K, a) / besselix(1, a) ≈ 1e-4
    end
    # Floors tighten with K (more harmonics resolve shorter scales).
    fs = [fl(K) for K in 2:8]
    @test all(>(0), fs) && issorted(fs; rev = true)
end

@testset "hsgp periodic layout" begin
    bound = bind_data(_hperiodic_plan(), _hsgp_cols())
    layout = assign_layout(bound)
    # SB `_sb_hsgp_periodic` declaration order (rho, sigma, beta),
    # appended after the slice-1 entries like the exp-quad triple.
    kinds = [(e.kind, e.name, e.size, e.transform) for e in layout.entries]
    @test kinds == [(:sampled, :a, 1, :identity),
        (:sampled, :sigma, 1, :exp),
        (:sampled, :rho_h_p, 1, :floored),
        (:sampled, :sigma_h_p, 1, :exp),
        (:hsgp, :beta_raw_h_p, 8, :identity)]
    rho_e = layout.entries[3]
    @test rho_e.lo ≈ ReactiveKernelsPPL._hsgp_periodic_rho_lower(4)
    @test isnan(rho_e.hi)
    @test layout.total == 12
    names = coordinate_names(layout)
    @test names[3:5] == [:rho_h_p, :sigma_h_p, Symbol("beta_raw_h_p.1")]
    @test names[end] == Symbol("beta_raw_h_p.8")
    # Jacobian: the exp/floored coords only (beta is identity).
    u = collect(0.1:0.1:1.2)
    @test logjac(layout, u) ≈ u[2] + u[3] + u[4]
    nt = constrain(layout, u)
    @test nt.rho_h_p ≈ rho_e.lo + exp(u[3])
    @test nt.sigma_h_p ≈ exp(u[4])
    @test Vector(nt.beta_raw_h_p) ≈ u[5:12]
    @test unconstrain(layout, nt) ≈ u
    # K=1: the zero floor routes rho to plain :exp.
    k1 = lower_rkppl(quote
            hsgp_basis(:h_1p, x; k = 1, cov = :periodic, period = 2.0)
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ hsgp(:h_1p)
            y .~ Normal.(mu, sigma)
        end, (:y, :x); conditioned = (:y, :x))
    k1layout = assign_layout(bind_data(k1, _hsgp_cols()))
    k1rho = only(e for e in k1layout.entries if e.name === :rho_h_1p)
    @test (k1rho.kind, k1rho.transform) === (:sampled, :exp)
    k1beta = only(e for e in k1layout.entries if e.name === :beta_raw_h_1p)
    @test k1beta.size == 2
end

# Independent posterior reference for the `_hperiodic_plan` shape (one
# periodic basis, intercept-only `mu`, `sigma ~ Exponential(1)`
# likelihood scale): SB `_brm_apply_hsgp_periodic` loop nests + the
# direct-form spectral weights `sigma*sqrt(2*exp(-a)*I_j(a))` (the
# unscaled-AMOS path — the emission uses scaled-log space, so the two
# agree only if both Bessels are right) + Distributions oracles.
# Constrained values come from `constrain` (locked absolutely by
# "hsgp periodic layout"); the Jacobian is hand-summed from
# coordinates, never `logjac`.
function _hsgp_periodic_ref_posterior(bound::StructuralPlan,
        layout::LayoutTable, u::AbstractVector{<:Real})
    hb = only(bound.hsgp_bases)
    n = bound.n_obs
    y = Vector{Float64}(bound.columns[:y])
    nt = constrain(layout, u)
    hsgp = ReactiveKernelsPPL._hsgp_names(hb)
    a = nt.a
    sig = Float64(nt.sigma)
    rho = Float64(only(getproperty(nt, r) for r in hsgp.rhos))
    sigh = Float64(getproperty(nt, hsgp.sigma))
    beta = Vector{Float64}(getproperty(nt, hsgp.beta))
    x = Vector{Float64}(bound.columns[only(hb.axes)])
    k = only(hb.K)
    w0 = 2pi / hb.period
    PHI = zeros(n, 2k)
    for j in 1:k, i in 1:n
        angle = w0 * j * x[i]
        PHI[i, j] = cos(angle)
        PHI[i, k + j] = sin(angle)
    end
    aa = 1 / (rho * rho)
    sspd = Vector{Float64}(undef, 2k)
    for (b, h) in enumerate(vcat(1:k, 1:k))
        sspd[b] = sigh * sqrt(2 * exp(-aa) * besseli(h, aa))
    end
    muv = a .+ PHI * (sspd .* beta)
    ll = sum(logpdf.(Normal.(muv, sig), y))
    pr = logpdf(Normal(0, 5), a) + logpdf(Exponential(1), sig) +
        logpdf(LogNormal(0, 1), rho) +
        logpdf(LogNormal(0, 1), sigh) + sum(logpdf.(Normal(0, 1), beta))
    cnames = coordinate_names(layout)
    jac = sum(u[findfirst(==(s), cnames)]
        for s in [:sigma, hsgp.rhos..., hsgp.sigma])
    return ll + pr + jac
end

@testset "hsgp periodic end to end" begin
    bound = bind_data(_hperiodic_plan(), _hsgp_cols())
    built = build_kernel(bound)
    u = [0.2, -0.1, 0.15, 0.05, 0.3, -0.2, 0.1, 0.0, 0.25, -0.15, 0.05, 0.1]
    ref = _hsgp_periodic_ref_posterior(bound, built.layout, u)
    @test isapprox(_query(built.spec, bound, :posterior, u), ref; rtol = 1e-12)
    _check_gradient(built.spec, bound, u)
end

@testset "hsgp periodic k1 end to end" begin
    k1 = lower_rkppl(quote
            hsgp_basis(:h_1p, x; k = 1, cov = :periodic, period = 2.0)
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ hsgp(:h_1p)
            y .~ Normal.(mu, sigma)
        end, (:y, :x); conditioned = (:y, :x))
    bound = bind_data(k1, _hsgp_cols())
    built = build_kernel(bound)
    u = [0.2, -0.1, 0.15, 0.05, 0.3, -0.2]
    ref = _hsgp_periodic_ref_posterior(bound, built.layout, u)
    @test isapprox(_query(built.spec, bound, :posterior, u), ref; rtol = 1e-12)
    _check_gradient(built.spec, bound, u)
end

# Build (evaluates a new generated model), then trace/compile in a call
# made through `Base.invokelatest`: the generated recipe closures are
# newer than the world of the enclosing top-level expression (the
# leveled-Reactant call-shape precedent).
function _hsgp_reactant(bound)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_hsgp_reactant_measure, built, bound, post_q, u)
end

function _hsgp_reactant_measure(built, bound, post_q, u)
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; native, primal, val, g, rval = Float64(rval),
        rgrad = Array(rgrad))
end

function _hsgp_xla_cols()
    return Dict{Symbol,AbstractVector}(
        :y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
        :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        :z => [1.0, 0.5, -0.5, 2.0, 0.0, 1.5])
end

# Upstream XLA gap (the von-Mises pin precedent): `besselix` has no
# method for a traced scalar, so every periodic program fails at trace
# time with `MethodError: no method matching besselix(::Int64,
# ::TracedRNumber{Float64})` (measured on Reactant 0.2.288: the
# primal trace throws before any gradient is staged). The signature
# below is exactly that gap; anything else rethrows loudly.
_hsgp_is_upstream_gap(e) =
    e isa MethodError && e.f === SpecialFunctions.besselix &&
    length(e.args) == 2 && e.args[2] isa Reactant.TracedRNumber

# Programs blocked by the gap above. The `@test_broken true` at the
# end of a pinned prog's body FIRES (Unexpected Pass) once upstream
# wires besselix — then drop the name here and the try/catch.
const _HSGP_UPSTREAM_PINNED = ("periodic",)

@testset "hsgp under Reactant" begin
    progs = [
        ("1d", quote
            hsgp_basis(:h_x, x; k = 4)
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, sigma)
        end),
        ("aniso", quote
            hsgp_basis(:h_xz, x, z; k = (4, 3), c = (1.5, 2.0), iso = false)
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ hsgp(:h_xz)
            y .~ Normal.(mu, sigma)
        end),
        ("periodic", quote
            hsgp_basis(:h_p, x; k = 4, cov = :periodic, period = 2.0)
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ hsgp(:h_p)
            y .~ Normal.(mu, sigma)
        end),
    ]
    for (name, prog) in progs
        @testset "$name" begin
            cols = _hsgp_xla_cols()
            keep = name == "aniso" ? (:y, :x, :z) : (:y, :x)
            bound = bind_data(lower_rkppl(prog, keep; conditioned = keep), cols)
            try
                fx = _hsgp_reactant(bound)
                @test fx.primal ≈ fx.native rtol = 1e-9
                @test fx.val ≈ fx.native rtol = 1e-12
                @test fx.rval ≈ fx.native rtol = 1e-9
                @test fx.rgrad ≈ fx.g rtol = 1e-8
                if name in _HSGP_UPSTREAM_PINNED
                    # Self-firing pin: errors (Unexpected Pass) once
                    # upstream wires besselix, forcing removal of the
                    # try/catch.
                    @test_broken true
                end
            catch e
                _hsgp_is_upstream_gap(e) || rethrow()
                name in _HSGP_UPSTREAM_PINNED || rethrow()
                # Known upstream besselix gap (above): pinned, not passing.
                @test_broken false
            end
        end
    end
end
