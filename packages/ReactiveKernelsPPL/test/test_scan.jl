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
