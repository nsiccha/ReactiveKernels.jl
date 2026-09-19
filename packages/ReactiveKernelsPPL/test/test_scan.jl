using ReactiveKernelsPPL
using Test

# A minimal bound Gaussian plan (intercept-only `mu`, scalar `sigma`) carrying
# the given scans — used by the integration/layout testsets.
_scan_min_plan(scans; n = 3) = StructuralPlan(
    [LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
        ResponseEvidence(:none, nothing, nothing), :y_resp)],
    [PredictorSpec(:mu, IdentityLink,
        [TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept, :Intercept)],
        :mu)],
    [PopulationPrior(:mu, :Intercept, 0.0, 1.0)],
    [SampledParameter(:sigma, :exponential, (arg1 = 1.0,), nothing, :sigma)],
    AssignmentSpec[],
    Dict{Symbol,AbstractVector}(:y => zeros(n)),
    n;
    scans = scans,
)

# Front-end parse tests for `@scan` (sequential recurrence). `parse_scan_block`
# is purely syntactic — it consumes the quoted inner `begin … end` block, so
# distribution constructors here are unevaluated AST symbols (no Distributions).

@testset "scan front-end: centered AR(1)" begin
    sp = parse_scan_block(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end)
    @test sp.state === :h
    @test sp.loopvar === :t
    @test sp.lo == 2
    @test sp.hi === :T
    @test length(sp.setup) == 1
    @test sp.setup[1].index == 1
    @test sp.setup[1].family === :normal
    @test sp.setup[1].args == [0, 1]
    @test length(sp.step) == 1
    @test sp.step[1].kind === :sample
    @test sp.step[1].target === :h
    @test sp.step[1].indexed
    @test sp.step[1].family === :normal
    @test length(sp.step[1].args) == 2
    @test sp.maxlag == 1
end

@testset "scan front-end: non-centered AR(1)" begin
    sp = parse_scan_block(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            eps ~ Normal(0, 1)
            h[t] = phi * h[t - 1] + s * eps
        end
    end)
    @test length(sp.step) == 2
    @test sp.step[1].kind === :sample
    @test sp.step[1].target === :eps
    @test !sp.step[1].indexed              # fresh per-step innovation
    @test sp.step[2].kind === :assign
    @test sp.step[2].target === :h
    @test sp.step[2].indexed               # deterministic carry write
    @test sp.maxlag == 1
end

@testset "scan front-end: per-step deterministic local" begin
    sp = parse_scan_block(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            mu = phi * h[t - 1]
            h[t] ~ Normal(mu, s)
        end
    end)
    @test length(sp.step) == 2
    @test sp.step[1].kind === :assign && !sp.step[1].indexed   # `mu` local
    @test sp.step[2].kind === :sample && sp.step[2].indexed
    @test sp.maxlag == 1
end

@testset "scan front-end: lag 2 needs two seeds" begin
    sp = parse_scan_block(quote
        h[1] ~ Normal(0, 1)
        h[2] ~ Normal(0, 1)
        for t in 3:T
            h[t] ~ Normal(a * h[t - 1] + b * h[t - 2], s)
        end
    end)
    @test length(sp.setup) == 2
    @test sp.lo == 3
    @test sp.maxlag == 2
end

@testset "scan front-end: literal loop bound" begin
    sp = parse_scan_block(quote
        h[1] ~ Normal(0, 1)
        for t in 2:10
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end)
    @test sp.hi == 10
    @test sp.hi isa Int
end

@testset "scan front-end: rejections" begin
    reject(blk) = @test_throws SurfaceLoweringError parse_scan_block(blk)

    # forward reference
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(phi * h[t + 1], s)
        end
    end)
    # current-index self-read on the RHS
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(phi * h[t], s)
        end
    end)
    # loop start not one past the seeds
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 3:T
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end)
    # non-contiguous seeds
    reject(quote
        h[1] ~ Normal(0, 1)
        h[3] ~ Normal(0, 1)
        for t in 3:T
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end)
    # lag deeper than the seed depth
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(a * h[t - 2], s)
        end
    end)
    # no lag at all → that is `@plate`, not `@scan`
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(mu, s)
        end
    end)
    # bare read of the carried array inside its own loop
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            z = sum(h)
            h[t] ~ Normal(z, s)
        end
    end)
    # no trailing `for`
    reject(quote
        h[1] ~ Normal(0, 1)
    end)
    # rebinding the whole carried array
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h = phi
        end
    end)
    # Stan-style family spelling
    reject(quote
        h[1] ~ normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end)
    # write at a non-loop index
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t - 1] ~ Normal(phi * h[t - 1], s)
        end
    end)
    # a second carried array (v1 threads exactly one)
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            g[t] ~ Normal(phi * h[t - 1], s)
            h[t] ~ Normal(g[t], s)
        end
    end)
    # symbolic (data-int) lag depth is not supported in v1
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(phi * h[t - k], s)
        end
    end)
end

@testset "scan IR: integrates into StructuralPlan + validation" begin
    ar1 = parse_scan_block(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end)

    # minimal bound Gaussian plan (intercept-only), carrying `scans`
    plan(scans) = StructuralPlan(
        [LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
            ResponseEvidence(:none, nothing, nothing), :y_resp)],
        [PredictorSpec(:mu, IdentityLink,
            [TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept, :Intercept)],
            :mu)],
        [PopulationPrior(:mu, :Intercept, 0.0, 1.0)],
        [SampledParameter(:sigma, :exponential, (arg1 = 1.0,), nothing, :sigma)],
        AssignmentSpec[],
        Dict{Symbol,AbstractVector}(:y => Float64[0, 0, 0]),
        3;
        scans = scans,
    )

    # happy path: an (as-yet unreferenced) scan latent validates structurally
    @test (validate_structure(plan([ar1])); true)
    @test plan([ar1]).scans[1].state === :h
    @test isempty(plan(ScanSpec[]).scans)      # pre-scan compat: no scans field needed

    # scan-state name colliding with a parameter is rejected by the name table
    bad_state = ScanSpec(:sigma, ar1.loopvar, ar1.lo, ar1.hi, ar1.setup,
        ar1.step, ar1.maxlag, :sigma)
    @test_throws ContractValidationError validate_structure(plan([bad_state]))

    # malformed ScanSpec (loop start not one past the seeds) caught by _validate_scans
    malformed = ScanSpec(:g, :t, 3, :T,
        [ScanSetup(1, :normal, Any[0, 1])],
        [ScanStep(:sample, :g, true, :normal, Any[:mu], nothing)],
        1, :g)
    @test_throws ContractValidationError validate_structure(plan([malformed]))
end

@testset "scan surface: @scan lowers into the plan" begin
    # A valid population-GLM model with an (as-yet unreferenced) @scan block.
    plan = lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        sigma ~ Exponential(1)
        phi ~ Normal(0, 1)
        s ~ Exponential(1)
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:T
                h[t] ~ Normal(phi * h[t - 1], s)
            end
        end
        mu = a .+ b .* x
        y .~ Normal.(mu, sigma)
    end, (:x, :y))
    @test length(plan.scans) == 1
    @test plan.scans[1].state === :h
    @test plan.scans[1].maxlag == 1
    @test plan.scans[1].hi === :T
    @test plan.scans[1].label === :h          # label defaults to the state name
    @test length(plan.responses) == 1         # the GLM response still lowered
    @test :sigma in [p.name for p in plan.parameters]

    # scan-state name colliding with a declared parameter is a double-definition
    @test_throws SurfaceLoweringError lower_rkppl(quote
        h ~ Normal(0, 1)
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:T
                h[t] ~ Normal(phi * h[t - 1], s)
            end
        end
        y .~ Normal.(mu, 1.0)
    end, (:y,))

    # an invalid @scan block (no trailing recurrence `for`) still errors loudly
    @test_throws SurfaceLoweringError lower_rkppl(quote
        @scan begin
            h[1] ~ Normal(0, 1)
        end
        y .~ Normal.(mu, 1.0)
    end, (:y,))
end

@testset "scan layout: array-sampled entry" begin
    ar1 = parse_scan_block(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end)
    p = _scan_min_plan([ar1]; n = 3)      # data length name `T` resolves to n_obs = 3
    lt = assign_layout(p)

    scan_entry = only(e for e in lt.entries if e.kind === :scan)
    @test scan_entry.name === :h
    @test scan_entry.size == 3            # T resolved to n_obs
    @test scan_entry.transform === :identity
    @test lt.total == 5                   # mu_coef(1) + sigma(1) + h(3)

    # identity slice: constrain/unconstrain roundtrips over the scan coords
    u = collect(1.0:5.0)
    nt = constrain(lt, u)
    @test nt.h == u[scan_entry.offset:(scan_entry.offset + 2)]
    @test unconstrain(lt, nt) ≈ u
    @test logjac(lt, u) == u[2]           # only sigma (:exp) contributes; scan (identity) adds 0

    # per-coordinate readout names include the scan coords
    names = coordinate_names(lt)
    @test length(names) == lt.total
    @test Symbol("h.1") in names && Symbol("h.3") in names

    # a literal loop bound sets the exact latent length
    ar_lit = parse_scan_block(quote
        h[1] ~ Normal(0, 1)
        for t in 2:4
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end)
    lt2 = assign_layout(_scan_min_plan([ar_lit]; n = 3))
    @test only(e for e in lt2.entries if e.kind === :scan).size == 4

    # a bound that leaves the recurrence with no steps is rejected
    ar_long = parse_scan_block(quote
        h[1] ~ Normal(0, 1)
        h[2] ~ Normal(0, 1)
        h[3] ~ Normal(0, 1)
        for t in 4:T
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end)
    @test_throws ContractValidationError assign_layout(_scan_min_plan([ar_long]; n = 3))
end

@testset "scan surface: scan state as a response location (4a)" begin
    plan = lower_rkppl(quote
        phi ~ Normal(0, 1)
        s ~ Exponential(1)
        sigma ~ Exponential(1)
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:T
                h[t] ~ Normal(phi * h[t - 1], s)
            end
        end
        y .~ Normal.(h, sigma)
    end, (:y,))
    r = only(plan.responses)
    @test r.predictor === :h            # the location IS the scan state
    @test r.family === GaussianFam
    @test r.scale === :sigma
    @test isempty(plan.predictors)      # no linear predictor synthesized for h
    @test plan.scans[1].state === :h

    # a non-Gaussian response over a scan state is rejected in slice 1
    ar1 = parse_scan_block(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end)
    bad = StructuralPlan(
        [LikelihoodSpec(BernoulliLogitFam, LogitLink, :y, :h, nothing, nothing,
            ResponseEvidence(:none, nothing, nothing), :y_resp)],
        PredictorSpec[], PopulationPrior[], SampledParameter[], AssignmentSpec[],
        Dict{Symbol,AbstractVector}(:y => zeros(3)), 3; scans = [ar1])
    @test_throws ContractValidationError validate_structure(bad)
end

# A non-centered AR(1) scan (SB-`ar` shape): `Normal(0, 1)` seed, one
# `Normal(0, 1)` innovation sample, one deterministic carry write.
_scan_ar_spec() = parse_scan_block(quote
    u[1] ~ Normal(0, 1)
    for t in 2:T
        eps ~ Normal(0, 1)
        u[t] = phi * u[t - 1] + eps
    end
end)

# Hand-built bound plan: `y ~ Normal(a + beta_ar * u, sigma)` with a free
# Normal `beta_ar` (summand options overridable for rejection tests).
function _scan_ar_plan(scan; coef = :beta_ar, coef_family = :normal,
        scan_id = :u, addressee = :scan_mu_u, columns = ColumnRef[],
        options = nothing)
    opts = options === nothing ? (scan_id = scan_id, coef = coef) : options
    terms = TermSpec[
        TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
            :intercept),
        TermSpec(ScanSummandTerm, columns, opts, addressee, :scan_mu_u)]
    StructuralPlan(
        [LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma, nothing,
            ResponseEvidence(:none, nothing, nothing), :y_resp)],
        [PredictorSpec(:mu, IdentityLink, terms, :mu)],
        [PopulationPrior(:mu, :Intercept, 0.0, 1.0)],
        [SampledParameter(:sigma, :exponential, (arg1 = 1.0,), nothing, :sigma),
            SampledParameter(:phi, :normal, (arg1 = 0.0, arg2 = 1.0), nothing,
                :phi),
            SampledParameter(coef, coef_family, (arg1 = 0.0, arg2 = 2.0),
                nothing, coef)],
        AssignmentSpec[],
        Dict{Symbol,AbstractVector}(:y => zeros(3)),
        3;
        scans = [scan],
    )
end

@testset "scan summand: IR validation + design" begin
    scan = _scan_ar_spec()
    good = _scan_ar_plan(scan)
    @test (validate_structure(good); true)
    # the summand is self-addressed: no PopulationPrior for it (only the
    # intercept prior is present, and validation passes)
    @test length(good.population_priors) == 1

    shape = design_shape(only(good.predictors), good.columns;
        levelmaps = good.levelmaps)
    @test shape.width == 1             # intercept only; the summand adds none
    sblock = only(b for b in shape.blocks if b.kind === ScanSummandTerm)
    @test sblock.width == 0 && isempty(sblock.labels)
    @test sblock.column === :u

    # hand-built IR emits (contract/generator agreement smoke)
    @test build_kernel(good).layout.total == 1 + 3 + 3

    bad_opts = [
        ("unknown scan", (scan_id = :nope, coef = :beta_ar)),
        ("unknown coef", (scan_id = :u, coef = :nope)),
        ("wrong keys", (scan_id = :u,)) ,
        ("swapped keys", (coef = :beta_ar, scan_id = :u)),
    ]
    for (what, opts) in bad_opts
        @test_throws ContractValidationError validate_structure(
            _scan_ar_plan(scan; options = opts))
    end
    # non-Normal coefficient (v1 admits SB's Normal `ar` beta only)
    @test_throws ContractValidationError validate_structure(
        _scan_ar_plan(scan; coef_family = :exponential))
    # a computed scalar is not a sampled coefficient
    p0 = _scan_ar_plan(scan)
    params = filter(p -> p.name !== :beta_ar, p0.parameters)
    p = StructuralPlan(p0.responses, p0.predictors, p0.population_priors,
        params, [AssignmentSpec(:beta_ar, :(phi * phi), :beta_ar)],
        p0.columns, p0.n_obs; scans = p0.scans)
    @test_throws ContractValidationError validate_structure(p)
    # summands carry no columns and are self-addressed
    @test_throws ContractValidationError validate_structure(
        _scan_ar_plan(scan; columns = [:u]))
    @test_throws ContractValidationError validate_structure(
        _scan_ar_plan(scan; addressee = :u))
end

@testset "non-centered layout: innovation slice" begin
    scan = _scan_ar_spec()
    lt = assign_layout(_scan_ar_plan(scan))
    z = only(e for e in lt.entries if e.kind === :scan)
    @test z.name === :_ppl_scan_z_u
    @test z.size == 3 && z.transform === :identity
    @test lt.total == 1 + 3 + 3   # mu_coef + phi/beta_ar/sigma + z[1..3]

    u = collect(1.0:7.0)
    nt = constrain(lt, u)
    @test nt._ppl_scan_z_u == u[z.offset:(z.offset + 2)]
    @test unconstrain(lt, nt) ≈ u
    @test logjac(lt, u) == u[2]   # only sigma (:exp) contributes
    names = coordinate_names(lt)
    @test length(names) == lt.total
    @test Symbol("_ppl_scan_z_u.1") in names

    # centered and non-centered scans coexist: the centered slice keeps the
    # state name, the non-centered one takes the innovation name
    centered = parse_scan_block(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end)
    mixed = _scan_ar_plan(scan)
    push!(mixed.scans, centered)
    push!(mixed.parameters,
        SampledParameter(:s, :exponential, (arg1 = 1.0,), nothing, :s))
    lt2 = assign_layout(mixed)
    kinds = Dict(e.name => e.size for e in lt2.entries if e.kind === :scan)
    @test kinds == Dict(:_ppl_scan_z_u => 3, :h => 3)
end

@testset "scan summand: surface spelling + fail-closed" begin
    plan = lower_rkppl(quote
        phi_raw ~ Normal(0, 1)
        beta_ar ~ Normal(0, 2)
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @scan begin
            u[1] ~ Normal(0, 1)
            for t in 2:T
                eps ~ Normal(0, 1)
                u[t] = phi * u[t - 1] + eps
            end
        end
        phi = tanh(phi_raw)
        mu = a .+ beta_ar .* u
        y .~ Normal.(mu, sigma)
    end, (:y,))
    terms = only(plan.predictors).terms
    @test length(terms) == 2
    st = terms[2]
    @test st.kind === ScanSummandTerm
    @test st.options == (scan_id = :u, coef = :beta_ar)
    @test st.addressee === st.label === :scan_mu_u
    @test isempty(st.columns)
    # the coefficient is a sampled scalar, never a population prior
    @test :beta_ar in [p.name for p in plan.parameters]
    @test all(pr -> pr.addressee !== :beta_ar, plan.population_priors)
    @test only(a for a in plan.assignments if a.name === :phi).expr ==
        :(tanh(phi_raw))

    reject(loc) = @test_throws SurfaceLoweringError lower_rkppl(quote
        phi_raw ~ Normal(0, 1)
        beta_ar ~ Normal(0, 2)
        b ~ Normal(0, 1)
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @scan begin
            u[1] ~ Normal(0, 1)
            for t in 2:T
                eps ~ Normal(0, 1)
                u[t] = phi * u[t - 1] + eps
            end
        end
        phi = tanh(phi_raw)
        $(loc)
        y .~ Normal.(mu, sigma)
    end, (:x, :y))
    # bare scan state in a location (LP use needs `coef .* state`)
    reject(:(mu = a .+ u))
    # literal scaling (coefficient-free is the dar shape, not ar)
    reject(:(mu = a .+ 2.0 .* u))
    # data scaling (interactions are planned)
    reject(:(mu = a .+ x .* u))
    # computed-scalar scaling (computed coefficients are out of slice)
    reject(:(mu = a .+ phi .* u))
    # coefficient with no `~` statement
    reject(:(mu = a .+ q .* u))
    # one name as both a population coefficient and a scan coefficient
    reject(:(mu = a .+ b .* x .+ b .* u))
    # subtracted summand (additive only)
    reject(:(mu = a .- beta_ar .* u))
    # nested scan read (direct `coef .* state` only)
    reject(:(mu = a .+ beta_ar .* (u .+ x)))
    # scan-only predictor (a summand needs a sibling coefficient)
    reject(:(mu = beta_ar .* u))
end

@testset "tanh assignment: scalar admitted, dotted rejected" begin
    plan = lower_rkppl(quote
        phi_raw ~ Normal(0, 1)
        a ~ Normal(0, 1)
        s ~ Normal(phi, 1.0)
        sigma ~ Exponential(1)
        phi = tanh(phi_raw)
        mu = a
        y .~ Normal.(mu, sigma)
    end, (:y,))
    @test only(a for a in plan.assignments if a.name === :phi).expr ==
        :(tanh(phi_raw))
    @test (validate_structure(plan); true)
    # dotted `tanh.` stays fail-closed (scalar vocabulary only in v1)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        w = tanh.(x)
        mu = a .+ w
        y .~ Normal.(mu, sigma)
    end, (:x, :y))
end

@testset "non-centered emission: fail-closed shapes" begin
    # each model parses (the parser admits any step shape) but refuses to emit
    scan_block(stmts...) = Expr(:macrocall, Symbol("@scan"),
        LineNumberNode(1), Expr(:block, stmts...))
    build_block(stmts...) = build_kernel(bind_data(
        lower_rkppl(Expr(:block,
            :(phi ~ Normal(0, 1)),
            :(a ~ Normal(0, 1)),
            :(b ~ Normal(0, 1)),
            :(s ~ Exponential(1)),
            :(sigma ~ Exponential(1)),
            scan_block(stmts...),
            :(y .~ Normal.(h, sigma))), (:y,)),
        Dict{Symbol,AbstractVector}(:y => [0.1, 0.2, 0.3])))
    # three steps (v1: one innovation sample + one carry write)
    @test_throws ContractValidationError build_block(
        :(h[1] ~ Normal(0, 1)),
        :(for t in 2:T
            eps ~ Normal(0, 1)
            eps2 ~ Normal(0, 1)
            h[t] = phi * h[t - 1] + eps + eps2
        end))
    # non-Normal innovation
    @test_throws ContractValidationError build_block(
        :(h[1] ~ Normal(0, 1)),
        :(for t in 2:T
            eps ~ Exponential(1)
            h[t] = phi * h[t - 1] + eps
        end))
    # non-Normal seed
    @test_throws ContractValidationError build_block(
        :(h[1] ~ Exponential(1)),
        :(for t in 2:T
            eps ~ Normal(0, 1)
            h[t] = phi * h[t - 1] + eps
        end))
    # AR(2) lag (tuple carry planned)
    @test_throws ContractValidationError build_block(
        :(h[1] ~ Normal(0, 1)),
        :(h[2] ~ Normal(0, 1)),
        :(for t in 3:T
            eps ~ Normal(0, 1)
            h[t] = a * h[t - 1] + b * h[t - 2] + eps
        end))
    # carry write before the innovation sample
    @test_throws ContractValidationError build_block(
        :(h[1] ~ Normal(0, 1)),
        :(for t in 2:T
            h[t] = phi * h[t - 1] + eps
            eps ~ Normal(0, 1)
        end))
    # unknown leaf in the carry write
    @test_throws ContractValidationError build_block(
        :(h[1] ~ Normal(0, 1)),
        :(for t in 2:T
            eps ~ Normal(0, 1)
            h[t] = nosuch * h[t - 1] + eps
        end))
    # loop-index leaf in the carry write
    @test_throws ContractValidationError build_block(
        :(h[1] ~ Normal(0, 1)),
        :(for t in 2:T
            eps ~ Normal(0, 1)
            h[t] = phi * h[t - 1] + eps * t
        end))
end
