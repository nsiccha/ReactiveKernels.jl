# HSGP Stage A: basis IR + surface lowering + validation + bind fit
# (acceptance item 1 of the HSGP todo). Codegen is NOT Stage A —
# `build_kernel` fails closed (tested below); layout + in-graph basis
# evaluation land in Stage B.

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
        ranef_buckets = plan.ranef_buckets,
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

@testset "hsgp Stage-B gate" begin
    bound = bind_data(_hvalid_plan(), _hsgp_cols())
    @test_throws ContractValidationError build_kernel(bound)
end
