using Distributions
using ReactiveKernelsPPL
using Test

# Differenced-AR(1) trajectory (dar) contract tests: SB `_sb_dar1`'s path —
# `beta ~ truncated(Normal(0.5, 0.2), 0, 1)` (`(:truncated, 0, 1)`) and
# `sigma ~ HalfNormal(0.2)` (`:positive`), each keeping the meaning of its
# statement as written (Distributions semantics, normalizers included: user
# decision `0m1j3iz`, prong `dar-kernel`), `z[T-1] ~ std_normal`, and the
# zero-started integrated path `x[t+1] = x[t] + d[t]` spliced beta-free into
# the linear predictor (the formula intercept is the initial level). The
# library spelling `x ~ differenced_ar1(beta, sigma)` is pinned at the end.
# Corpus drift coverage lives in 48_dar and 48_dar_library.
#
# Helpers `_query`, `_check_gradient` (test_generator.jl) and
# `_none_evidence` (test_contract.jl) are included first in runtests.jl.
# References are independent per-row loops / Distributions calls, never
# the emitted forms.

_dar_beta() = SampledParameter(:beta, :normal, (arg1 = 0.5, arg2 = 0.2),
    (:interval, 0.0, 1.0), :beta)
_dar_sigma() = SampledParameter(:sigmad, :normal, (arg1 = 0.0, arg2 = 0.2),
    :positive, :sigmad)

function _dar_spec(; state = :dar_mu, beta = :beta, sigma = :sigmad)
    return DarSpec(state, beta, sigma, state)
end

# A minimal bound Gaussian plan (intercept-only `mu`, scalar `sigma`)
# carrying the given dar trajectories — used by the IR/layout testsets.
function _dar_min_plan(dars; n = 4)
    return StructuralPlan(
        [LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:mu, IdentityLink,
            [TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
                :intercept)],
            :mu)],
        [PopulationPrior(:mu, :Intercept, 0.0, 1.0)],
        [SampledParameter(:sigma, :exponential, (arg1 = 1.0,), nothing,
            :sigma),
            _dar_beta(), _dar_sigma()],
        AssignmentSpec[],
        Dict{Symbol,AbstractVector}(:y => zeros(n)),
        n;
        dar_paths = dars,
    )
end

# Hand-built bound plan: `y ~ Normal(a + x, sigma)` with the dar state
# spliced beta-free (summand options overridable for rejection tests).
function _dar_plan(spec; dar_id = spec.state, addressee = spec.state,
        columns = ColumnRef[], options = nothing)
    opts = options === nothing ? (dar_id = dar_id,) : options
    terms = TermSpec[
        TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
            :intercept),
        TermSpec(DarSummandTerm, columns, opts, addressee, spec.state)]
    return StructuralPlan(
        [LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:mu, IdentityLink, terms, :mu)],
        [PopulationPrior(:mu, :Intercept, 0.0, 1.0)],
        [SampledParameter(:sigma, :exponential, (arg1 = 1.0,), nothing,
            :sigma),
            _dar_beta(), _dar_sigma()],
        AssignmentSpec[],
        Dict{Symbol,AbstractVector}(:y => zeros(4)),
        4;
        dar_paths = [spec],
    )
end

@testset "dar IR: integrates into StructuralPlan + validation" begin
    spec = _dar_spec()

    # happy path: an (as-yet unreferenced) dar trajectory validates
    @test (validate_structure(_dar_min_plan([spec])); true)
    @test _dar_min_plan([spec]).dar_paths[1].state === :dar_mu
    @test isempty(_dar_min_plan(DarSpec[]).dar_paths)

    # General truncation and legacy normalized overrides admit the same
    # authored priors. The surface now emits the general representation.
    for beta_support in ((:interval, 0.0, 1.0), (:truncated, 0.0, 1.0)),
            sigma_support in (:positive, (:truncated, 0.0, Inf))
        p = _dar_min_plan([spec])
        i = findfirst(q -> q.name === :beta, p.parameters)
        j = findfirst(q -> q.name === :sigmad, p.parameters)
        p.parameters[i] = ReactiveKernelsPPL._with(p.parameters[i];
            support_override = beta_support)
        p.parameters[j] = ReactiveKernelsPPL._with(p.parameters[j];
            support_override = sigma_support)
        @test (validate_structure(p); true)
    end

    # dar-state name colliding with a parameter is rejected by the name table
    bad_state = DarSpec(:sigma, :beta, :sigmad, :sigma)
    # refused: dar state name collides with a parameter (IR name-table contract)
    @test_throws ContractValidationError validate_structure(
        _dar_min_plan([bad_state]))

    # dar-state name colliding with a scan state is rejected too
    mk_scan = parse_scan_block(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end)
    clash = _dar_min_plan([_dar_spec(; state = :h)])
    push!(clash.scans, mk_scan)
    # refused: dar state name collides with a scan state (IR name-table contract)
    @test_throws ContractValidationError validate_structure(clash)

    # unknown persistence / scale names
    # refused: DarSpec names an unknown persistence parameter (IR contract)
    @test_throws ContractValidationError validate_structure(
        _dar_min_plan([_dar_spec(; beta = :nope)]))
    # refused: DarSpec names an unknown scale parameter (IR contract)
    @test_throws ContractValidationError validate_structure(
        _dar_min_plan([_dar_spec(; sigma = :nope)]))

    # General persistence/scale priors and one parameter serving both roles
    # are exercised through explicit differenced_ar1 declarations in
    # test_dar_capabilities.jl, with density/gradient oracles. DarSpec is the
    # historical built-in IR; extending it would mint undeclared latents.

    # override-shaped args still admit (location/scale ride free)
    p = _dar_min_plan([spec])
    i = findfirst(q -> q.name === :beta, p.parameters)
    p.parameters[i] = SampledParameter(:beta, :normal,
        (arg1 = 0.7, arg2 = 0.1), (:interval, 0.0, 1.0), :beta)
    @test (validate_structure(p); true)
end

@testset "dar summand: IR validation + design" begin
    spec = _dar_spec()
    good = _dar_plan(spec)
    @test (validate_structure(good); true)
    # the summand is self-addressed: no PopulationPrior for it (only the
    # intercept prior is present, and validation passes)
    @test length(good.population_priors) == 1

    shape = design_shape(only(good.predictors), good.columns;
        levelmaps = good.levelmaps)
    @test shape.width == 1             # intercept only; the summand adds none
    dblock = only(b for b in shape.blocks if b.kind === DarSummandTerm)
    @test dblock.width == 0 && isempty(dblock.labels)
    @test dblock.column === :dar_mu

    # hand-built IR emits (contract/generator agreement smoke):
    # mu_coef(1) + beta/sigmad/sigma(3) + z[1..3]
    @test build_kernel(good).layout.total == 1 + 3 + 3

    bad_opts = [
        # refused: dar-summand options `unknown dar` (IR contract)
        ("unknown dar", (dar_id = :nope,)),
        # refused: dar-summand options `wrong keys` (IR contract)
        ("wrong keys", (scan_id = :dar_mu,)),
        # refused: dar-summand options `extra coef` (IR contract)
        ("extra coef", (dar_id = :dar_mu, coef = :b)),
        # refused: dar-summand options `empty` (IR contract)
        ("empty", NamedTuple()),
    ]
    for (what, opts) in bad_opts
        # refused: malformed dar-summand options (IR contract)
        @test_throws ContractValidationError validate_structure(
            _dar_plan(spec; options = opts))
    end
    # summands carry no columns and are self-addressed
    # refused: dar summand carrying columns (IR contract)
    @test_throws ContractValidationError validate_structure(
        _dar_plan(spec; columns = [:dar_mu]))
    # refused: dar summand not self-addressed (IR contract)
    @test_throws ContractValidationError validate_structure(
        _dar_plan(spec; addressee = :mu))
end

@testset "dar layout: T-1 innovation slice" begin
    spec = _dar_spec()
    lt = assign_layout(_dar_min_plan([spec]; n = 4))
    z = only(e for e in lt.entries if e.kind === :scan)
    @test z.name === :_ppl_dar_z_dar_mu
    @test z.size == 3 && z.transform === :identity
    @test lt.total == 1 + 3 + 3   # mu_coef + beta/sigmad/sigma + z[1..3]

    u = collect(1.0:7.0) ./ 10
    nt = constrain(lt, u)
    @test nt._ppl_dar_z_dar_mu == u[z.offset:(z.offset + 2)]
    @test unconstrain(lt, nt) ≈ u
    # beta (interval) + sigmad/sigma (exp) contribute; z (identity) adds 0
    names = coordinate_names(lt)
    isd = findfirst(==(:sigmad), names)
    isig = findfirst(==(:sigma), names)
    @test logjac(lt, u) ≈
        (log(nt.beta) + log(1 - nt.beta)) + u[isd] + u[isig]
    @test length(names) == lt.total
    @test Symbol("_ppl_dar_z_dar_mu.1") in names
    @test Symbol("_ppl_dar_z_dar_mu.3") in names

    # The zero-innovation case uses the explicit library declaration; its
    # positive layout and density/gradient tests are in test_dar_capabilities.jl.

    # dar and scan innovation slices coexist under distinct names
    ar1 = parse_scan_block(quote
        u[1] ~ Normal(0, 1)
        for t in 2:T
            eps ~ Normal(0, 1)
            u[t] = phi * u[t - 1] + eps
        end
    end)
    mixed = _dar_min_plan([spec]; n = 4)
    push!(mixed.scans, ar1)
    push!(mixed.parameters, SampledParameter(:phi, :normal,
        (arg1 = 0.0, arg2 = 1.0), nothing, :phi))
    lt2 = assign_layout(mixed)
    kinds = Dict(e.name => e.size for e in lt2.entries if e.kind === :scan)
    @test kinds == Dict(:_ppl_dar_z_dar_mu => 3, :_ppl_scan_z_u => 4)
end

@testset "dar surface: spelling + fail-closed" begin
    plan = lower_rkppl(quote
        a ~ Normal(0, 1)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        mu = a .+ dar(beta, sigmad)
        y .~ Normal.(mu, sigma)
    end, (:y,); conditioned = (:y,))
    @test length(plan.dar_paths) == 1
    ds = only(plan.dar_paths)
    @test (ds.state, ds.beta, ds.sigma) === (:dar_mu, :beta, :sigmad)
    terms = only(plan.predictors).terms
    @test length(terms) == 2
    st = terms[2]
    @test st.kind === DarSummandTerm
    @test st.options == (dar_id = :dar_mu,)
    @test st.addressee === st.label === :dar_mu
    @test isempty(st.columns)
    # both parameters are sampled scalars, never population priors, with
    # the meaning of their statements as written (no Stan-kernel re-key)
    byname = Dict(p.name => p for p in plan.parameters)
    @test byname[:beta].family === :normal
    @test byname[:beta].support_override == (:truncated, 0.0, 1.0)
    @test byname[:sigmad].family === :normal
    @test byname[:sigmad].support_override === :positive
    @test all(pr -> pr.addressee !== :beta && pr.addressee !== :sigmad,
        plan.population_priors)

    # The truncated-half sigma spelling has the same normalized density.
    plan2 = lower_rkppl(quote
        a ~ Normal(0, 1)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ truncated(Normal(0, 0.2), 0, Inf)
        sigma ~ Exponential(1)
        mu = a .+ dar(beta, sigmad)
        y .~ Normal.(mu, sigma)
    end, (:y,); conditioned = (:y,))
    @test only(plan2.dar_paths).sigma === :sigmad
    byname2 = Dict(p.name => p for p in plan2.parameters)
    @test byname2[:sigmad].support_override == (:truncated, 0.0, Inf)

    # a dar-shaped `~` never consumed by `dar()` lowers identically:
    # `dar()` no longer changes what its parameters' statements mean
    plan3 = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        mu = a .+ b .* x
        y .~ Normal.(mu, sigma)
    end, (:y, :x); conditioned = (:y, :x))
    @test isempty(plan3.dar_paths)
    byname3 = Dict(p.name => p for p in plan3.parameters)
    @test byname3[:beta].support_override == (:truncated, 0.0, 1.0)
    @test byname3[:sigmad].support_override === :positive

    # refused: remaining calls violate arity or strict declarations (P3/P6, 05oe96l).
    reject(loc) = @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        $(loc)
        y .~ Normal.(mu, sigma)
    end, (:x, :y); conditioned = (:x, :y))
    # Ordinary scaled/nested/data/subtracted DAR values use an explicitly
    # declared library trajectory; test_dar_capabilities.jl checks each against
    # an independent sequential oracle, rather than minting latents in a call.
    # wrong arity / non-name args
    # refused: `dar(beta)` arity (Julia MethodError, P3)
    reject(:(mu = a .+ dar(beta)))
    # refused: `dar(beta, sigmad, sigma)` arity (Julia MethodError, P3)
    reject(:(mu = a .+ dar(beta, sigmad, sigma)))
    # Literal arguments, shared arguments, standalone locations and two
    # independent declarations are covered by the explicit library fixtures.
    # A dar parameter also retains its ordinary affine reader.
    dplan = lower_rkppl(quote
        a ~ Normal(0, 1)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        mu = a .+ beta .* x .+ dar(beta, sigmad)
        y .~ Normal.(mu, sigma)
    end, (:x, :y); conditioned = (:x, :y))
    @test :beta in Set(p.name for p in dplan.parameters)
    @test only(t for t in only(dplan.predictors).terms
        if t.kind === ContinuousTerm).options.parameter === :beta

    # dar() inside a definition (referenced or not) fails closed
    # refused: `w = dar(...)` mints latent innovations through `=`; parameters come from `~` (P2, P8 1cmodra; 10gzbm9)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        w = dar(beta, sigmad)
        mu = a .+ w
        y .~ Normal.(mu, sigma)
    end, (:y,); conditioned = (:y,))
    # refused: `w = dar(...)` mints latent innovations through `=`; parameters come from `~` (P2, P8 1cmodra; 10gzbm9)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        w = dar(beta, sigmad)
        mu = a
        y .~ Normal.(mu, sigma)
    end, (:y,); conditioned = (:y,))

    # Authored persistence and scale priors retain their stated densities in
    # the explicit library path, including Normal, Exponential, HalfNormal,
    # HalfCauchy and truncated-Normal priors (test_dar_capabilities.jl).
    # parameters with no `~` statement
    # refused: undeclared dar argument `q` (P6, 05oe96l)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        mu = a .+ dar(q, sigmad)
        y .~ Normal.(mu, sigma)
    end, (:y,); conditioned = (:y,))
    # Library scopes keep an authored dar_mu declaration distinct from the
    # trajectory; the collision fixture checks both density and gradient.

end

# Independent oracle for the zero-started differenced-AR(1) model
# `y ~ Normal(a + x, sigma)`: SB `differenced_ar1_path` ported line-for-line,
# with the author's priors exactly as written, evaluated by Distributions.jl
# (never the emitted forms). Layout order: a, beta, sigmad, sigma, z[1..T-1].
function _dar_oracle(u, y; beta_prior = truncated(Normal(0.5, 0.2), 0, 1),
        sigmad_prior = truncated(Normal(0, 0.2), 0, Inf))
    T = length(y)
    aa = u[1]
    beta = 1 / (1 + exp(-u[2]))
    sigmad = exp(u[3])
    sigma = exp(u[4])
    z = u[5:(5 + T - 2)]
    x = zeros(T)
    d = 0.0
    for t in 1:(T - 1)
        d = beta * d + sigmad * z[t]
        x[t + 1] = x[t] + d
    end
    @test x[1] == 0.0   # the zero start is load-bearing, not incidental
    mu = aa .+ x
    ll = sum(logpdf(Normal(mu[t], sigma), y[t]) for t in 1:T)
    pr = logpdf(Normal(0, 1), aa) +
         logpdf(beta_prior, beta) +
         logpdf(sigmad_prior, sigmad) +
         logpdf(Exponential(1), sigma) +
         sum(logpdf(Normal(0, 1), zt) for zt in z)
    jac = log(beta) + log(1 - beta) + u[3] + u[4]
    return ll + pr + jac
end

@testset "dar: trajectory end to end (oracle + gradient)" begin
    # `beta ~ truncated(Normal(0.5, 0.2), 0, 1)` and `sigmad ~
    # HalfNormal(0.2)` mean what they say (normalized), `z ~
    # std_normal(; n=T-1)`, and the zero-started path spliced beta-free.
    m = @rkppl begin
        a ~ Normal(0, 1)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        mu = a .+ dar(beta, sigmad)
        y .~ Normal.(mu, sigma)
    end
    ydata = [0.3, -0.1, 0.5, 0.2, -0.4]
    plan = (m() | (; y = ydata))
    @test only(plan.predictors).terms[2].kind === DarSummandTerm
    built = build_kernel(plan)
    # mu_coef(a) + beta + sigmad + sigma + z[1..4]
    @test built.layout.total == 4 + (length(ydata) - 1)
    @test coordinate_names(built.layout)[1:4] ==
        [:a, :beta, :sigmad, :sigma]
    for u in ([0.1, 0.3, -0.5, -0.4, 0.2, -0.1, 0.4, 0.0],
              [-0.2, 0.6, 0.1, 0.3, -0.4, 0.2, -0.3, 0.1])
        @test _query(built.spec, plan, :posterior, u) ≈ _dar_oracle(u, ydata)
        _check_gradient(built.spec, plan, u)
    end
end

# D1/D2 SB-parity pins (the `test/rk_parity.jl` dar corpus cases, mirrored
# RK-natively). SB's literals (BridgeStan full posterior, propto=false,
# jacobian=true) are Stan-kernel: its bounded `beta ~ normal(m, s;
# lower=0, upper=1)` and `sigma ~ normal(0, s; lower=0)` drop the
# truncation normalizers. RK keeps the author's statements as written, so
# its posterior is SB's plus the two normalizers — a constant in `u`, so
# the gradients equal SB's exactly (brief
# 2026-09-27T12-50-22-591-v7kukl on
# BayesianRegressionModels:rk:parity-term-ar-dar, BRM 97bb538,
# StanBlocks 24578c3, BridgeStan 2.9.0, Julia 1.10.11). Shared columns
# `y = [0.5, -0.2, 0.1, 0.9, 1.4, 1.1]` (the `t` axis is length-only on
# the RK side); `u = range(-0.4, 0.4; length = 9)` matches the BRM
# layout order (mu_coef, beta, sigmad, s, z.1..5), so the SAME `u`
# compares directly against the SB literal (SB unpacks a permutation
# of it peer-side). SB grad order
# [pop_mu_beta_pop.1, dar_mu_t_beta, dar_mu_t_sigma, dar_mu_t_z.1..5, s]
# permutes to RK u-order as [1, 2, 3, 9, 4, 5, 6, 7, 8].
_dar_normalizers(bloc, bsca, ssca) =
    -log(cdf(Normal(bloc, bsca), 1.0) - cdf(Normal(bloc, bsca), 0.0)) + log(2)

@testset "dar D1/D2 SB-parity pins" begin
    ydata = [0.5, -0.2, 0.1, 0.9, 1.4, 1.1]
    u = collect(range(-0.4, 0.4; length = 9))
    @testset "D1 dar default" begin
        # SB: mu ~ 1 + dar(t); beta ~ Normal(0.5, 0.2); sigma ~
        # Normal(0, 0.2); s ~ Exponential(1). SB full -22.57116529873909.
        m = @rkppl begin
            a ~ Normal(0, 1)
            beta ~ truncated(Normal(0.5, 0.2), 0, 1)
            sigmad ~ HalfNormal(0.2)
            s ~ Exponential(1)
            mu = a .+ dar(beta, sigmad)
            y .~ Normal.(mu, s)
        end
        plan = (m() | (; y = ydata))
        built = build_kernel(plan)
        @test built.layout.total == 9
        @test coordinate_names(built.layout)[1:4] ==
            [:a, :beta, :sigmad, :s]
        post = _query(built.spec, plan, :posterior, u)
        @test post ≈ _dar_oracle(u, ydata)
        @test abs(post - (-22.57116529873909 +
            _dar_normalizers(0.5, 0.2, 0.2))) < 1e-12
        sb = [5.466993138042674, 0.8344512151440091, -13.924728401033837,
            -1.4386732731205805, 5.160141062738464, 4.392079285424707,
            3.192200195514296, 1.4901788654922776, 0.021427806410332872]
        prep = prepare_sampler(built, plan, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test maximum(abs.(g .- sb)) < 1e-10
        _check_gradient(built.spec, plan, u)
    end
    @testset "D2 dar prior overrides" begin
        # SB: D1 + ar(mu, dar(t)) ~ Normal(0.6, 0.1) + sd(mu, dar(t)) ~
        # Normal(0, 0.3). SB full -19.08072138630899.
        m = @rkppl begin
            a ~ Normal(0, 1)
            beta ~ truncated(Normal(0.6, 0.1), 0, 1)
            sigmad ~ HalfNormal(0.3)
            s ~ Exponential(1)
            mu = a .+ dar(beta, sigmad)
            y .~ Normal.(mu, s)
        end
        plan = (m() | (; y = ydata))
        built = build_kernel(plan)
        @test built.layout.total == 9
        @test coordinate_names(built.layout)[1:4] ==
            [:a, :beta, :sigmad, :s]
        post = _query(built.spec, plan, :posterior, u)
        @test post ≈ _dar_oracle(u, ydata;
            beta_prior = truncated(Normal(0.6, 0.1), 0, 1),
            sigmad_prior = truncated(Normal(0, 0.3), 0, Inf))
        @test abs(post - (-19.08072138630899 +
            _dar_normalizers(0.6, 0.1, 0.3))) < 1e-12
        sb = [5.466993138042674, 4.643891230385576, -4.614727761649957,
            -1.4386732731205805, 5.160141062738464, 4.392079285424707,
            3.192200195514296, 1.4901788654922776, 0.021427806410332872]
        prep = prepare_sampler(built, plan, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test maximum(abs.(g .- sb)) < 1e-10
        _check_gradient(built.spec, plan, u)
    end
end

# The library spelling: `x ~ differenced_ar1(beta, sigma)` is an ordinary
# `@scan` submodel (tuple carry `(level, increment)`, deterministic zero
# seeds) whose body states only the innovation prior; the author's
# `beta`/`sigma` statements keep their meaning. Same layout order and the
# same density as the built-in `dar()`, and as the oracle.
@testset "dar library: differenced_ar1 matches the built-in and the oracle" begin
    lib = @rkppl begin
        a ~ Normal(0, 1)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        x ~ differenced_ar1(beta, sigmad)
        mu = a .+ x
        y .~ Normal.(mu, sigma)
    end
    builtin = @rkppl begin
        a ~ Normal(0, 1)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        mu = a .+ dar(beta, sigmad)
        y .~ Normal.(mu, sigma)
    end
    ydata = [0.3, -0.1, 0.5, 0.2, -0.4]
    pl = (lib() | (; y = ydata))
    pb = (builtin() | (; y = ydata))
    sc = only(pl.scans)
    @test [_test_scope_name(pl, s) for s in sc.states] == [:x_level, :x_increment]
    @test [f.kind for f in sc.setup] == [:assign, :assign]
    @test isempty(pl.dar_paths)
    @test any(t -> t.kind === ScanSummandTerm &&
        _test_scope_name(pl, t.options.scan_id) === :x_level &&
        t.options.coef === nothing,
        only(pl.predictors).terms)
    byname = Dict(p.name => p for p in pl.parameters)
    @test byname[:beta].support_override == (:truncated, 0.0, 1.0)
    @test byname[:sigmad].support_override === :positive
    bl = build_kernel(pl)
    bb = build_kernel(pb)
    @test coordinate_names(bl.layout)[1:4] ==
        coordinate_names(bb.layout)[1:4]
    @test bl.layout.total == bb.layout.total == 4 + (length(ydata) - 1)
    @test all(!occursin("##", string(n)) for n in coordinate_names(bl.layout))
    u0 = zeros(bl.layout.total)
    draws = constrain(bl.layout, u0)
    @test length(draws.x._ppl_scan_z_level) == length(ydata) - 1
    @test unconstrain(bl.layout, draws) ≈ u0
    @test size(restore_draws(bl.layout, zeros(bl.layout.total, 0)).x._ppl_scan_z_level) ==
        (length(ydata) - 1, 0)
    for u in ([0.1, 0.3, -0.5, -0.4, 0.2, -0.1, 0.4, 0.0],
              [-0.2, 0.6, 0.1, 0.3, -0.4, 0.2, -0.3, 0.1])
        vl = _query(bl.spec, pl, :posterior, u)
        @test vl ≈ _query(bb.spec, pb, :posterior, u)
        @test vl ≈ _dar_oracle(u, ydata)
        _check_gradient(bl.spec, pl, u)
    end

    # The library value is a plain value: an unscaled location works too.
    direct = @rkppl begin
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        x ~ differenced_ar1(beta, sigmad)
        y .~ Normal.(x, sigma)
    end
    pd = (direct() | (; y = ydata))
    @test _test_scope_name(pd, only(pd.responses).predictor) === :x_level
    bd = build_kernel(pd)
    ud = [0.3, -0.5, -0.4, 0.2, -0.1, 0.4, 0.0]
    vd = _query(bd.spec, pd, :posterior, ud)
    @test vd ≈ _dar_oracle([0.0; ud], ydata) - logpdf(Normal(0, 1), 0.0)
    _check_gradient(bd.spec, pd, ud)
end
