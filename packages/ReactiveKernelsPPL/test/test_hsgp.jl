# HSGP contract: basis IR + surface lowering + validation + bind fit
# (Stage A) and layout + in-graph basis evaluation + matmul summand
# (Stage B, SB `_sb_hsgp` mirror). End-to-end values vs independent
# SB-shape hand references (per-row loops + Distributions oracles, never
# the emitted expressions) plus Enzyme-vs-findiff gradients. (`_query` /
# `_check_gradient` come from test_generator.jl, included first.)
using Distributions: LogNormal, Normal, Exponential, logpdf

function _hvalid_plan(; aniso::Bool = false)
    if aniso
        return lower_rkppl(quote
                hsgp_basis(:h_xz, x, z; k = (4, 3), c = (1.5, 2.0),
                    iso = false)
                a ~ Normal(0, 5)
                sigma ~ Exponential(1)
                mu = a .+ hsgp(:h_xz)
                y .~ Normal.(mu, sigma)
            end, (:y, :x, :z))
    end
    return lower_rkppl(quote
            hsgp_basis(:h_x, x; k = 4)
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, sigma)
        end, (:y, :x))
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
            hsgp_basis(:h_d, x)
            mu = a .+ hsgp(:h_d)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
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
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(x)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, 1.0)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, q)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x, x; k = (2, 2))
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x)
            hsgp_basis(:h_x, x)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    # k/c/iso literals.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x; k = 0)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x; k = 2.5)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x, z; k = (2, 3, 4))
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x, :z))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x; c = 1.0)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x; c = Inf)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x; iso = 1)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis(:h_x, x; by = g)
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x, :g))
    # Use-site discipline.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .+ hsgp(:h_nope)
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_x, x)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .+ hsgp(:h_x) .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_x, x)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .- hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_x, x)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .+ 2.0 .* hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_x, x)
        end, (:y, :x))
    # Never inside definitions; never redefined or sampled.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            h = hsgp(:h_x)
            mu = a .+ h
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_x, x)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp = 1.0
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_x, x)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            hsgp_basis = 1.0
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
        end, (:y, :x))
    # Claims: user definitions cannot collide with sampled names.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            rho_h_x = 1.0
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_x, x)
        end, (:y, :x))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu = a .+ hsgp(:h_x)
            y .~ Normal.(mu, 1.0)
            hsgp_basis(:h_x, x)
            beta_raw_h_x ~ Normal(0, 1)
        end, (:y, :x))
end

@testset "hsgp contract validation" begin
    good = _hvalid_plan()
    hb = only(good.hsgp_bases)
    # Duplicate ids / labels.
    dup = HSGPBasis(:h_x, [:x], [2], [1.5], true,
        Tuple{Float64,Float64}[], :hsgp_h_x)
    @test_throws ContractValidationError validate_structure(
        _hwith(good; bases = [hb, dup]))
    # K/c shape and values.
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [HSGPBasis(:h_x, [:x], [2, 3], [1.5], true,
            Tuple{Float64,Float64}[], :hsgp_h_x)]))
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [HSGPBasis(:h_x, [:x], [0], [1.5], true,
            Tuple{Float64,Float64}[], :hsgp_h_x)]))
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [HSGPBasis(:h_x, [:x], [2], [1.0], true,
            Tuple{Float64,Float64}[], :hsgp_h_x)]))
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [HSGPBasis(:h_x, Symbol[], Int[], Float64[], true,
            Tuple{Float64,Float64}[], :hsgp_h_x)]))
    # Fits: wrong count / non-positive L.
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [HSGPBasis(:h_x, [:x], [2], [1.5], true,
            [(0.0, 1.0), (0.0, 1.0)], :hsgp_h_x)]))
    @test_throws ContractValidationError validate_structure(_hwith(good;
        bases = [HSGPBasis(:h_x, [:x], [2], [1.5], true,
            [(0.0, 0.0)], :hsgp_h_x)]))
    # Linkage: dangling / double-use / unknown summand.
    nopred = _hwith(good; predictors = PredictorSpec[
        PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                :Intercept, :mu_intercept)], :mu)])
    @test_throws ContractValidationError validate_structure(nopred)
    twopred = _hwith(good; predictors = vcat(good.predictors,
        [PredictorSpec(:sg, IdentityLink, TermSpec[_hsummand(:sg, :h_x)],
            :sg)]))
    @test_throws ContractValidationError validate_structure(twopred)
    baduse = _hwith(good; predictors = PredictorSpec[
        PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                :Intercept, :mu_intercept), _hsummand(:mu, :h_nope)],
            :mu)])
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
        @test_throws ContractValidationError validate_structure(badpred)
    end
    # Name tables: a hand-built parameter under an hsgp name fails.
    clash = _hwith(good)
    push!(clash.parameters, SampledParameter(:rho_h_x, :normal,
        (arg1 = 0, arg2 = 1), nothing, :rho_h_x))
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
    @test_throws ContractValidationError bind_data(_hvalid_plan(), constcols)
    strcols = Dict{Symbol,AbstractVector}(:y => cols[:y],
        :x => ["a", "b", "c", "d"])
    @test_throws ContractValidationError bind_data(_hvalid_plan(), strcols)
    nancols = Dict{Symbol,AbstractVector}(:y => cols[:y],
        :x => [0.5, NaN, 1.5, 0.0])
    @test_throws ContractValidationError bind_data(_hvalid_plan(), nancols)
end

@testset "hsgp layout" begin
    bound = bind_data(_hvalid_plan(), _hsgp_cols())
    layout = assign_layout(bound)
    # SB `_sb_hsgp` declaration order per basis (rho, sigma, beta),
    # appended after the slice-1 entries so peer offsets never move.
    kinds = [(e.kind, e.name, e.size, e.transform) for e in layout.entries]
    @test kinds == [(:coefficient, :mu_coef, 1, :identity),
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
    @test akinds == [(:coefficient, :mu_coef, 1, :identity),
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
        end, (:y, :x))
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
        end, (:y, :x))
    twolayout = assign_layout(bind_data(two, _hsgp_cols()))
    twokinds =
        [(e.kind, e.name, e.size, e.transform) for e in twolayout.entries]
    @test twokinds == [(:coefficient, :mu_coef, 1, :identity),
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
    @test_throws ContractValidationError build_kernel(bad)
    @test_throws ContractValidationError assign_layout(bad)
    @test_throws ContractValidationError _hsgp_basis_statements_bad(bad)
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
    a = only(nt.mu)
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
        end, (:y, :x))
    bound = bind_data(k1, _hsgp_cols())
    built = build_kernel(bound)
    u = [0.2, -0.1, 0.15, 0.05, 0.3]
    ref = _hsgp_ref_posterior(bound, built.layout, u)
    @test isapprox(_query(built.spec, bound, :posterior, u), ref; rtol = 1e-12)
    _check_gradient(built.spec, bound, u)
end
