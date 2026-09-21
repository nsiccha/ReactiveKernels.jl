using Distributions
using ReactiveKernelsPPL
using Test

# Differenced-AR(1) trajectory (dar) contract tests: SB `_sb_dar1` mirror —
# `beta ~ Normal(0.5, 0.2)` on `[0, 1]`, positive `sigma ~ Normal(0, 0.2)`,
# `z[T-1] ~ std_normal`, and the zero-started integrated path
# `x[t+1] = x[t] + d[t]` spliced beta-free into the linear predictor (the
# formula intercept is the initial level). Corpus drift coverage lives in
# 48_dar.
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

    # dar-state name colliding with a parameter is rejected by the name table
    bad_state = DarSpec(:sigma, :beta, :sigmad, :sigma)
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
    @test_throws ContractValidationError validate_structure(clash)

    # unknown persistence / scale names
    @test_throws ContractValidationError validate_structure(
        _dar_min_plan([_dar_spec(; beta = :nope)]))
    @test_throws ContractValidationError validate_structure(
        _dar_min_plan([_dar_spec(; sigma = :nope)]))

    # persistence and scale must be distinct parameters
    @test_throws ContractValidationError validate_structure(
        _dar_min_plan([DarSpec(:dar_mu, :beta, :beta, :dar_mu)]))

    # persistence geometry: Normal on exactly (:interval, 0, 1)
    for (fam, sup) in ((:normal, nothing), (:normal, :positive),
            (:normal, (:interval, 0.0, 2.0)), (:exponential, nothing))
        p = _dar_min_plan([spec])
        i = findfirst(q -> q.name === :beta, p.parameters)
        p.parameters[i] = SampledParameter(:beta, fam,
            (arg1 = 0.5, arg2 = 0.2), sup, :beta)
        @test_throws ContractValidationError validate_structure(p)
    end

    # scale geometry: Normal on :positive
    for (fam, sup) in ((:normal, nothing), (:exponential, nothing),
            (:normal, (:interval, 0.0, 1.0)), (:cauchy, :positive))
        p = _dar_min_plan([spec])
        i = findfirst(q -> q.name === :sigmad, p.parameters)
        p.parameters[i] = SampledParameter(:sigmad, fam,
            (arg1 = 0.0, arg2 = 0.2), sup, :sigmad)
        @test_throws ContractValidationError validate_structure(p)
    end

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
        ("unknown dar", (dar_id = :nope,)),
        ("wrong keys", (scan_id = :dar_mu,)),
        ("extra coef", (dar_id = :dar_mu, coef = :b)),
        ("empty", NamedTuple()),
    ]
    for (what, opts) in bad_opts
        @test_throws ContractValidationError validate_structure(
            _dar_plan(spec; options = opts))
    end
    # summands carry no columns and are self-addressed
    @test_throws ContractValidationError validate_structure(
        _dar_plan(spec; columns = [:dar_mu]))
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

    # a single observation leaves no innovation — fail closed
    @test_throws ContractValidationError assign_layout(
        _dar_min_plan([spec]; n = 1))

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
    end, (:y,))
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
    # both parameters are sampled scalars, never population priors
    byname = Dict(p.name => p for p in plan.parameters)
    @test byname[:beta].family === :normal
    @test byname[:beta].support_override == (:interval, 0.0, 1.0)
    @test byname[:sigmad].family === :normal
    @test byname[:sigmad].support_override === :positive
    @test all(pr -> pr.addressee !== :beta && pr.addressee !== :sigmad,
        plan.population_priors)

    # the truncated-half sigma spelling lowers identically
    plan2 = lower_rkppl(quote
        a ~ Normal(0, 1)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ truncated(Normal(0, 0.2), 0, Inf)
        sigma ~ Exponential(1)
        mu = a .+ dar(beta, sigmad)
        y .~ Normal.(mu, sigma)
    end, (:y,))
    @test only(plan2.dar_paths).sigma === :sigmad

    reject(loc) = @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        $(loc)
        y .~ Normal.(mu, sigma)
    end, (:x, :y))
    # scaled dar (dar is beta-free — scaling is the ar shape, not dar)
    reject(:(mu = a .+ b .* dar(beta, sigmad)))
    # nested dar read (direct `dar(beta, sigma)` only)
    reject(:(mu = a .+ dar(beta, sigmad) .* 2.0))
    reject(:(mu = a .+ dar(beta, sigmad) .* x))
    # subtracted summand (additive only)
    reject(:(mu = a .- dar(beta, sigmad)))
    # wrong arity / non-name args
    reject(:(mu = a .+ dar(beta)))
    reject(:(mu = a .+ dar(beta, sigmad, sigma)))
    reject(:(mu = a .+ dar(0.5, sigmad)))
    # same parameter twice
    reject(:(mu = a .+ dar(beta, beta)))
    # dar-only predictor (a summand needs a sibling coefficient)
    reject(:(mu = dar(beta, sigmad)))
    # two dar calls in one predictor (one per predictor in v1)
    reject(:(mu = a .+ dar(beta, sigmad) .+ dar(beta, sigmad)))
    # a dar parameter as a population coefficient too
    reject(:(mu = a .+ beta .* x .+ dar(beta, sigmad)))
    # bare dar-state name (splices via its call, not as a coefficient)
    reject(:(mu = a .+ dar(beta, sigmad) .+ dar_mu))

    # dar() inside a definition (referenced or not) fails closed
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        w = dar(beta, sigmad)
        mu = a .+ w
        y .~ Normal.(mu, sigma)
    end, (:y,))
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        w = dar(beta, sigmad)
        mu = a
        y .~ Normal.(mu, sigma)
    end, (:y,))

    # persistence geometry rejections (use-site spelling)
    for beta_rhs in (:(Normal(0.5, 0.2)),
            :(truncated(Normal(0.5, 0.2), 0, 2)),
            :(truncated(Normal(0.5, 0.2), 0, Inf)),
            :(Exponential(1)))
        @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 1)
            beta ~ $(beta_rhs)
            sigmad ~ HalfNormal(0.2)
            sigma ~ Exponential(1)
            mu = a .+ dar(beta, sigmad)
            y .~ Normal.(mu, sigma)
        end, (:y,))
    end
    # scale geometry rejections (use-site spelling)
    for sigma_rhs in (:(Normal(0, 0.2)), :(Exponential(1)),
            :(truncated(Normal(0, 0.2), 0, 1)))
        @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 1)
            beta ~ truncated(Normal(0.5, 0.2), 0, 1)
            sigmad ~ $(sigma_rhs)
            sigma ~ Exponential(1)
            mu = a .+ dar(beta, sigmad)
            y .~ Normal.(mu, sigma)
        end, (:y,))
    end
    # parameters with no `~` statement
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        mu = a .+ dar(q, sigmad)
        y .~ Normal.(mu, sigma)
    end, (:y,))
    # synthesized state colliding with a user definition
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        dar_mu ~ Normal(0, 1)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        mu = a .+ dar(beta, sigmad)
        y .~ Normal.(mu, sigma)
    end, (:y,))
end

@testset "dar: SB trajectory end to end (oracle + gradient)" begin
    # SB `_sb_dar1` mirror: `beta ~ normal(0.5, 0.2; lower=0, upper=1)`,
    # `sigma ~ normal(0, 0.2; lower=0)`, `z ~ std_normal(; n=T-1)`, and
    # the zero-started `differenced_ar1_path` spliced beta-free.
    m = @rkppl begin
        a ~ Normal(0, 1)
        beta ~ truncated(Normal(0.5, 0.2), 0, 1)
        sigmad ~ HalfNormal(0.2)
        sigma ~ Exponential(1)
        mu = a .+ dar(beta, sigmad)
        y .~ Normal.(mu, sigma)
    end
    ydata = [0.3, -0.1, 0.5, 0.2, -0.4]
    plan = m(; y = ydata)
    @test only(plan.predictors).terms[2].kind === DarSummandTerm
    built = build_kernel(plan)
    # mu_coef(a) + beta + sigmad + sigma + z[1..4]
    @test built.layout.total == 4 + (length(ydata) - 1)
    @test coordinate_names(built.layout)[1:4] ==
        [Symbol("mu.Intercept"), :beta, :sigmad, :sigma]

    # independent oracle: SB `differenced_ar1_path` ported line-for-line,
    # priors and likelihood via Distributions.jl (never the emitted forms)
    function dar_oracle(u, y)
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
        zn = Normal(0.5, 0.2)
        pr = logpdf(Normal(0, 1), aa) +
             logpdf(zn, beta) - log(cdf(zn, 1) - cdf(zn, 0)) +
             logpdf(Normal(0, 0.2), sigmad) + log(2) +
             logpdf(Exponential(1), sigma) +
             sum(logpdf(Normal(0, 1), zt) for zt in z)
        jac = log(beta) + log(1 - beta) + u[3] + u[4]
        return ll + pr + jac
    end

    for u in ([0.1, 0.3, -0.5, -0.4, 0.2, -0.1, 0.4, 0.0],
              [-0.2, 0.6, 0.1, 0.3, -0.4, 0.2, -0.3, 0.1])
        @test _query(built.spec, plan, :posterior, u) ≈ dar_oracle(u, ydata)
        _check_gradient(built.spec, plan, u)
    end
end
