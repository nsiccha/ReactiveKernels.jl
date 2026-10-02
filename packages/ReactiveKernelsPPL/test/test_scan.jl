using Distributions
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

# A shape this file pins as not built yet: `f()` fails with an error whose
# message contains `needle` (`@test_broken`), and a different failure is a
# real error. When the shape starts working, `@test_broken` reports an
# unexpected pass, so the pin turns into a positive test.
function _scan_gap(f, needle)
    ok = try
        f()
        true
    catch e
        occursin(needle, sprint(showerror, e)) || rethrow()
        false
    end
    # capability: generalized scan values and recurrence shapes (P8 1cmodra; todo `0yc2qgp`).
    @test_broken ok
end

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
    @test sp.states == [:h]
    @test sp.loopvar === :t
    @test sp.lo == 2
    @test sp.hi === :T
    @test length(sp.setup) == 1
    @test sp.setup[1].target === :h
    @test sp.setup[1].index == 1
    @test sp.setup[1].kind === :sample
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
    # refused: remaining entries violate Julia statement/index order or
    # the explicit carry/seed declaration (P3; IR contract).
    reject(blk) = @test_throws SurfaceLoweringError parse_scan_block(blk)
    # capability: lag-free and data-integer-lag scans (10gzbm9 degenerate;
    # P8 1cmodra; todo `0yc2qgp`).
    capable(blk) = @test_broken (parse_scan_block(blk); true)

    # refused: forward read `h[t + 1]` before it is written (P3)
    # forward reference
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(phi * h[t + 1], s)
        end
    end)
    # refused: reads `h[t]` before it is defined (P3)
    # current-index self-read on the RHS
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(phi * h[t], s)
        end
    end)
    # refused: `h[2]` never defined nor given a prior (P7, P3)
    # loop start not one past the seeds
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 3:T
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end)
    # refused: `h[3]` seeded then rewritten by the loop (single assignment); `h[2]` undefined
    # non-contiguous seeds
    reject(quote
        h[1] ~ Normal(0, 1)
        h[3] ~ Normal(0, 1)
        for t in 3:T
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end)
    # refused: reads `h[0]` at t = 2 (Julia BoundsError, P3)
    # lag deeper than the seed depth
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(a * h[t - 2], s)
        end
    end)
    # A scan may degenerate to independent draws.
    capable(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(mu, s)
        end
    end)
    # refused: `sum(h)` reads unwritten elements of the carried array (P3)
    # bare read of the carried array inside its own loop
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            z = sum(h)
            h[t] ~ Normal(z, s)
        end
    end)
    # refused: @scan with no recurrence `for`; `h` has no declared extent (P6)
    # no trailing `for`
    reject(quote
        h[1] ~ Normal(0, 1)
    end)
    # refused: rebinds the carried array `h` (single assignment)
    # rebinding the whole carried array
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h = phi
        end
    end)
    # refused: Stan-style `normal` is not a Julia distribution (P3, P10)
    # Stan-style family spelling
    reject(quote
        h[1] ~ normal(0, 1)
        for t in 2:T
            h[t] ~ Normal(phi * h[t - 1], s)
        end
    end)
    # refused: writes `h[t - 1]`, rewriting `h[1]` (single assignment)
    # write at a non-loop index
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            h[t - 1] ~ Normal(phi * h[t - 1], s)
        end
    end)
    # a carried array written in the loop but never seeded (`g[1]` would
    # be undefined)
    reject(quote
        h[1] ~ Normal(0, 1)
        for t in 2:T
            g[t] ~ Normal(phi * h[t - 1], s)
            h[t] ~ Normal(g[t], s)
        end
    end)
    # A data-integer lag is a retained runtime index.
    capable(quote
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
    @test plan([ar1]).scans[1].states == [:h]
    @test isempty(plan(ScanSpec[]).scans)      # pre-scan compat: no scans field needed

    # scan-state name colliding with a parameter is rejected by the name table
    bad_state = parse_scan_block(quote
        sigma[1] ~ Normal(0, 1)
        for t in 2:T
            sigma[t] ~ Normal(phi * sigma[t - 1], s)
        end
    end)
    # refused: scan state name collides with a parameter (IR name-table contract)
    @test_throws ContractValidationError validate_structure(plan([bad_state]))
    # ... including a second carried array of a tuple carry
    bad_tuple = parse_scan_block(quote
        h[1] ~ Normal(0, 1)
        sigma[1] = 0.0
        for t in 2:T
            e ~ Normal(0, 1)
            h[t] = phi * h[t - 1] + e
            sigma[t] = sigma[t - 1] + h[t]
        end
    end)
    # refused: carry states, seeds and step writes must agree with the ScanSpec declaration (IR contract)
    @test_throws ContractValidationError validate_structure(plan([bad_tuple]))

    # malformed ScanSpecs caught by _validate_scans: loop start not one past
    # the seeds; a carried array seeded short; a seed of a non-carried
    # array; a carried array written twice per step
    seed(a, k) = ScanSetup(a, k, :sample, :normal, Any[0, 1], nothing)
    write(a) = ScanStep(:assign, a, true, nothing, nothing, :($a[t - 1]))
    malformed = [
        ScanSpec([:g], :t, 3, :T, [seed(:g, 1)],
            [ScanStep(:sample, :g, true, :normal, Any[:mu], nothing)], 1, :g),
        ScanSpec([:g, :q], :t, 3, :T, [seed(:g, 1), seed(:g, 2), seed(:q, 1)],
            [write(:g), write(:q)], 1, :g),
        ScanSpec([:g], :t, 2, :T, [seed(:g, 1), seed(:q, 1)],
            [write(:g)], 1, :g),
        ScanSpec([:g], :t, 2, :T, [seed(:g, 1)], [write(:g), write(:g)],
            1, :g),
    ]
    for bad in malformed
        # refused: carry states, seeds and step writes must agree with the ScanSpec declaration (IR contract)
        @test_throws ContractValidationError validate_structure(plan([bad]))
    end
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
    end, (:x, :y); conditioned = (:x, :y))
    @test length(plan.scans) == 1
    @test plan.scans[1].states == [:h]
    @test plan.scans[1].maxlag == 1
    @test plan.scans[1].hi === :T
    @test plan.scans[1].label === :h          # label defaults to the state name
    @test length(plan.responses) == 1         # the GLM response still lowered
    @test :sigma in [p.name for p in plan.parameters]

    # scan-state name colliding with a declared parameter is a double-definition
    # refused: `h` declared by `~` and as the scan state (single assignment)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        h ~ Normal(0, 1)
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:T
                h[t] ~ Normal(phi * h[t - 1], s)
            end
        end
        y .~ Normal.(mu, 1.0)
    end, (:y,); conditioned = (:y,))

    # an invalid @scan block (no trailing recurrence `for`) still errors loudly
    # refused: @scan with no recurrence `for`; `h` has no declared extent (P6)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        @scan begin
            h[1] ~ Normal(0, 1)
        end
        y .~ Normal.(mu, 1.0)
    end, (:y,); conditioned = (:y,))
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
    # capability: zero-step @scan (T equals the seed count) (todo `0yc2qgp`)
    @test_broken (assign_layout(_scan_min_plan([ar_long]; n = 3)); true)
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
    end, (:y,); conditioned = (:y,))
    r = only(plan.responses)
    @test r.predictor === :h            # the location IS the scan state
    @test r.family === GaussianFam
    @test r.scale === :sigma
    @test isempty(plan.predictors)      # no linear predictor synthesized for h
    @test plan.scans[1].states == [:h]

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
    # capability: non-Gaussian response over a scan state (slice 1; hand-built plan) (todo `0yc2qgp`)
    @test_broken (validate_structure(bad); true)
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
    # a beta-free summand (`coef = nothing`) splices the bare state
    free = _scan_ar_plan(scan; options = (scan_id = :u, coef = nothing))
    @test (validate_structure(free); true)
    @test build_kernel(free).layout.total == 1 + 3 + 3

    bad_opts = [
        # refused: scan-summand options `unknown scan` (IR contract)
        ("unknown scan", (scan_id = :nope, coef = :beta_ar)),
        # refused: scan-summand options `unknown coef` (IR contract)
        ("unknown coef", (scan_id = :u, coef = :nope)),
        # refused: scan-summand options `wrong keys` (IR contract)
        ("wrong keys", (scan_id = :u,)) ,
        # refused: scan-summand options `swapped keys` (IR contract)
        ("swapped keys", (coef = :beta_ar, scan_id = :u)),
    ]
    for (what, opts) in bad_opts
        # refused: malformed scan-summand options (IR contract)
        @test_throws ContractValidationError validate_structure(
            _scan_ar_plan(scan; options = opts))
    end
    # non-Normal coefficient (v1 admits SB's Normal `ar` beta only)
    # capability: non-Normal scan-summand coefficient prior (v1 admits Normal only) (todo `0yc2qgp`)
    @test_broken (validate_structure(
        _scan_ar_plan(scan; coef_family = :exponential)); true)
    # a computed scalar is not a sampled coefficient
    p0 = _scan_ar_plan(scan)
    params = filter(p -> p.name !== :beta_ar, p0.parameters)
    p = StructuralPlan(p0.responses, p0.predictors, p0.population_priors,
        params, [AssignmentSpec(:beta_ar, :(phi * phi), :beta_ar)],
        p0.columns, p0.n_obs; scans = p0.scans)
    # capability: computed (assignment) scan-summand coefficient (P8 admits computed coefficients) (todo `0yc2qgp`)
    @test_broken (validate_structure(p); true)
    # summands carry no columns and are self-addressed
    # refused: scan summand carrying columns (IR contract)
    @test_throws ContractValidationError validate_structure(
        _scan_ar_plan(scan; columns = [:u]))
    # refused: scan summand not self-addressed (IR contract)
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
    end, (:y,); conditioned = (:y,))
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

    prog(loc) = quote
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
    end
    # A bare scan state is a beta-free summand (the dar/`mo1` shape).
    bare = lower_rkppl(prog(:(mu = a .+ u)), (:x, :y); conditioned = (:x, :y))
    bt = only(bare.predictors).terms[2]
    @test bt.kind === ScanSummandTerm
    @test bt.options == (scan_id = :u, coef = nothing)
    # coefficient with no `~` statement (refused: undeclared names never
    # become parameters, decision `05oe96l`)
    # refused: q has no declaration (P6, 05oe96l)
    @test_throws SurfaceLoweringError lower_rkppl(prog(:(mu = a .+ q .* u)),
        (:x, :y); conditioned = (:x, :y))
    # Valid Julia the emitter does not build yet (a scan state is spliced
    # bare or scaled by one sampled scalar, additively): pinned as gaps so
    # building one fails this file instead of a silent refusal surviving.
    # capability: scan-state arithmetic composes as ordinary values (P8 1cmodra; todo `0yc2qgp`).
    gap(loc, needle) = _scan_gap(needle) do
        lower_rkppl(prog(loc), (:x, :y); conditioned = (:x, :y))
    end
    gap(:(mu = a .+ 2.0 .* u), "scan coefficients are bare sampled scalars")
    gap(:(mu = a .+ x .* u), "scales scan state")
    gap(:(mu = a .+ phi .* u), "scan coefficients are bare sampled scalars")
    gap(:(mu = a .- beta_ar .* u), "additive")
    gap(:(mu = a .+ beta_ar .* (u .+ x)), "scan states lower only as direct")
    gap(:(mu = beta_ar .* u), "no estimated coefficients")
    # One name as both a population coefficient and a scan coefficient is
    # one ordinary parameter read by both summands (test_fallback.jl).
    both = lower_rkppl(quote
        phi_raw ~ Normal(0, 1)
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
        mu = a .+ b .* x .+ b .* u
        y .~ Normal.(mu, sigma)
    end, (:x, :y); conditioned = (:x, :y))
    @test any(p -> p.name === :b, both.parameters)
end

@testset "tanh assignment: scalar and dotted admitted" begin
    plan = lower_rkppl(quote
        phi_raw ~ Normal(0, 1)
        a ~ Normal(0, 1)
        s ~ Normal(phi, 1.0)
        sigma ~ Exponential(1)
        phi = tanh(phi_raw)
        mu = a
        y .~ Normal.(mu, sigma)
    end, (:y,); conditioned = (:y,))
    @test only(a for a in plan.assignments if a.name === :phi).expr ==
        :(tanh(phi_raw))
    @test (validate_structure(plan); true)
    # Dotted `tanh.` broadcasts the built-in itself (functions as values):
    # an elementwise column over `x` (density: test_functions_as_values.jl).
    dotted = lower_rkppl(quote
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        w = tanh.(x)
        mu = a .+ w
        y .~ Normal.(mu, sigma)
    end, (:x, :y); conditioned = (:x, :y))
    w = only(d for d in dotted.derived if d.name === :w).expr
    @test w.head === :. && w.args[1] isa GlobalRef &&
        w.args[1].name === :tanh && w.args[2] == Expr(:tuple, :x)
    @test (validate_structure(dotted); true)
end

@testset "non-centered emission: gaps and refusals" begin
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
            :(y .~ Normal.(h, sigma))), (:y,); conditioned = (:y,)),
        Dict{Symbol,AbstractVector}(:y => [0.1, 0.2, 0.3])))
    # a carry write that reads its innovation before the step drawing it
    # (refused: in a Julia loop body `eps` is not defined yet)
    # refused: the carry write reads its innovation before the step drawing it (Julia statement order, P3)
    @test_throws SurfaceLoweringError build_block(
        :(h[1] ~ Normal(0, 1)),
        :(for t in 2:T
            h[t] = phi * h[t - 1] + eps
            eps ~ Normal(0, 1)
        end))
    # an unknown leaf in the carry write (refused: undeclared names never
    # become parameters, decision `05oe96l`)
    # refused: undeclared name `nosuch` (P6, 05oe96l)
    @test_throws ContractValidationError build_block(
        :(h[1] ~ Normal(0, 1)),
        :(for t in 2:T
            eps ~ Normal(0, 1)
            h[t] = nosuch * h[t - 1] + eps
        end))
    # Valid recurrences the emitter does not build yet.
    # capability: arbitrary-support innovations and retained deterministic/data-dependent scans (P8 1cmodra; todo `0yc2qgp`).
    gap(needle, stmts...) = _scan_gap(() -> build_block(stmts...), needle)
    # a positive-support innovation or seed (the latent slice is identity)
    gap("must have real support",
        :(h[1] ~ Normal(0, 1)),
        :(for t in 2:T
            eps ~ Exponential(1)
            h[t] = phi * h[t - 1] + eps
        end))
    gap("must have real support",
        :(h[1] ~ Exponential(1)),
        :(for t in 2:T
            eps ~ Normal(0, 1)
            h[t] = phi * h[t - 1] + eps
        end))
    # the loop index read directly
    gap("uses the loop index",
        :(h[1] ~ Normal(0, 1)),
        :(for t in 2:T
            eps ~ Normal(0, 1)
            h[t] = phi * h[t - 1] + eps * t
        end))
    # a fully deterministic recurrence (no per-step innovation)
    gap("needs a per-step innovation",
        :(h[1] ~ Normal(0, 1)),
        :(for t in 2:T
            h[t] = phi * h[t - 1]
        end))
    # centered and non-centered carried writes mixed in one scan
    gap("mixing centered and non-centered",
        :(h[1] ~ Normal(0, 1)),
        :(g[1] = 0.0),
        :(for t in 2:T
            h[t] ~ Normal(phi * h[t - 1], s)
            g[t] = g[t - 1] + h[t]
        end))
    # an innovation scale that reads a carried array (stochastic volatility)
    gap("innovation scales",
        :(h[1] ~ Normal(0, 1)),
        :(for t in 2:T
            eps ~ Normal(0, exp(h[t - 1]))
            h[t] = phi * h[t - 1] + eps
        end))
    # a data column read inside the recurrence
    gap("data-varying",
        :(h[1] = 0.0),
        :(for t in 2:T
            eps ~ Normal(0, 1)
            h[t] = phi * h[t - 1] + y[t - 1] + eps
        end))
end

@testset "scan front-end: tuple carry and deterministic seeds" begin
    sp = parse_scan_block(quote
        x[1] = 0.0
        d[1] = x0
        for t in 2:T
            z ~ Normal(0, 1)
            d[t] = beta * d[t - 1] + sigma * z
            x[t] = x[t - 1] + d[t]
        end
    end)
    @test sp.states == [:x, :d]
    @test sp.label === :x
    @test [(f.target, f.index, f.kind) for f in sp.setup] ==
        [(:x, 1, :assign), (:d, 1, :assign)]
    @test sp.setup[2].expr === :x0
    @test [(st.kind, st.target, st.indexed) for st in sp.step] ==
        [(:sample, :z, false), (:assign, :d, true), (:assign, :x, true)]
    @test sp.maxlag == 1

    # interleaved fills, a seed reading an earlier seed, and lag 2
    sp2 = parse_scan_block(quote
        h[1] ~ Normal(0, 1)
        lvl[1] = 0.0
        h[2] ~ Normal(0, 1)
        lvl[2] = lvl[1] + h[2]
        for t in 3:T
            e ~ Normal(0, 1)
            h[t] = a * h[t - 1] + b * h[t - 2] + e
            lvl[t] = lvl[t - 1] + h[t]
        end
    end)
    @test sp2.states == [:h, :lvl]
    @test sp2.maxlag == 2
    @test sp2.lo == 3

    # refused: each remaining entry violates statement order, single assignment or the declared carry/seed contract (P3; IR contract)
    reject(blk) = @test_throws SurfaceLoweringError parse_scan_block(blk)
    # a current value read before the step that writes it
    reject(quote
        x[1] = 0.0
        d[1] = 0.0
        for t in 2:T
            z ~ Normal(0, 1)
            x[t] = x[t - 1] + d[t]
            d[t] = beta * d[t - 1] + z
        end
    end)
    # a carried array written twice in one step
    reject(quote
        x[1] = 0.0
        for t in 2:T
            z ~ Normal(0, 1)
            x[t] = x[t - 1] + z
            x[t] = x[t - 1] - z
        end
    end)
    # a local read before its definition
    reject(quote
        x[1] = 0.0
        for t in 2:T
            x[t] = x[t - 1] + w
            w ~ Normal(0, 1)
        end
    end)
    # a local defined twice
    reject(quote
        x[1] = 0.0
        for t in 2:T
            w ~ Normal(0, 1)
            w = 2 * w
            x[t] = x[t - 1] + w
        end
    end)
    # carried arrays seeded to different depths
    reject(quote
        x[1] = 0.0
        x[2] = 0.0
        d[1] = 0.0
        for t in 3:T
            z ~ Normal(0, 1)
            d[t] = d[t - 1] + z
            x[t] = x[t - 1] + d[t]
        end
    end)
    # a seed reading a value not seeded above it
    reject(quote
        x[1] = d[1]
        d[1] = 0.0
        for t in 2:T
            z ~ Normal(0, 1)
            d[t] = d[t - 1] + z
            x[t] = x[t - 1] + d[t]
        end
    end)
    # a seeded array the loop never writes
    reject(quote
        x[1] = 0.0
        d[1] = 0.0
        for t in 2:T
            z ~ Normal(0, 1)
            x[t] = x[t - 1] + z
        end
    end)
end

# Oracle for the tuple-carry program of corpus `38_scan_tuple_carry`: an
# AR(2) `h` with sampled seeds h[1], h[2] and one innovation per step, and
# its running level `lvl[t] = lvl[t-1] + h[t]` from `lvl[1] = 0`,
# `lvl[2] = h[2]`; `y ~ Normal(a + lvl, sigma)`. Layout order: a, phi1,
# phi2, s, sigma, then the scan slice [h1, h2, eps_3..T].
function _tuple_carry_oracle(u, y)
    T = length(y)
    a, phi1, phi2 = u[1], u[2], u[3]
    s, sigma = exp(u[4]), exp(u[5])
    z = u[6:end]
    h = zeros(T)
    lvl = zeros(T)
    h[1], h[2] = z[1], z[2]
    lvl[2] = h[2]
    for t in 3:T
        h[t] = phi1 * h[t - 1] + phi2 * h[t - 2] + s * z[t]
        lvl[t] = lvl[t - 1] + h[t]
    end
    ll = sum(logpdf(Normal(a + lvl[t], sigma), y[t]) for t in 1:T)
    pr = logpdf(Normal(0, 1), a) + logpdf(Normal(0, 0.5), phi1) +
         logpdf(Normal(0, 0.5), phi2) + logpdf(Exponential(1), s) +
         logpdf(Exponential(1), sigma) + sum(logpdf.(Normal(0, 1), z))
    return ll + pr + u[4] + u[5]
end

@testset "scan: tuple carry end to end (lag 2, seeds, oracle + gradient)" begin
    path = joinpath(_CORPUS_DIR, "38_scan_tuple_carry.jl")
    ast, data = _load_corpus_case(path)
    ydata = [0.4, -0.2, 0.9, 0.3, 1.2, 0.8]
    plan = bind_data(lower_rkppl(ast, data; conditioned = data),
        Dict{Symbol,AbstractVector}(:y => ydata))
    sc = only(plan.scans)
    @test sc.states == [:h, :level]
    @test only(plan.predictors).terms[2].options ==
        (scan_id = :level, coef = nothing)
    built = build_kernel(plan)
    z = only(e for e in built.layout.entries if e.kind === :scan)
    @test z.name === :_ppl_scan_z_h
    @test z.size == length(ydata)          # 2 seeds + (T - 2) innovations
    @test built.layout.total == 5 + length(ydata)
    for u in (collect(range(-0.5, 0.6; length = 11)),
              [0.2, 0.4, -0.3, -0.6, 0.1, 0.5, -0.2, 0.3, 0.0, -0.4, 0.7])
        @test _query(built.spec, plan, :posterior, u) ≈
            _tuple_carry_oracle(u, ydata)
        _check_gradient(built.spec, plan, u)
    end
end

@testset "scan: two innovations, a step local, a parameter seed" begin
    m = @rkppl begin
        h0 ~ Normal(0, 1)
        phi ~ Normal(0, 0.5)
        s1 ~ Exponential(1)
        s2 ~ Exponential(1)
        sigma ~ Exponential(1)
        @scan begin
            h[1] = h0
            for t in 2:T
                e1 ~ Normal(0, 1)
                e2 ~ Cauchy(0, 1)
                drift = phi * h[t - 1]
                h[t] = drift + s1 * e1 + s2 * e2
            end
        end
        y .~ Normal.(h, sigma)
    end
    ydata = [0.1, 0.5, -0.2, 0.3]
    plan = (m() | (; y = ydata))
    built = build_kernel(plan)
    T = length(ydata)
    z = only(e for e in built.layout.entries if e.kind === :scan)
    @test z.size == 2 * (T - 1)           # no sampled seed; e1 block, e2 block
    function oracle(u)
        nt = constrain(built.layout, u)
        e1 = nt._ppl_scan_z_h[1:(T - 1)]
        e2 = nt._ppl_scan_z_h[T:end]
        h = zeros(T)
        h[1] = nt.h0
        for t in 2:T
            h[t] = nt.phi * h[t - 1] + nt.s1 * e1[t - 1] + nt.s2 * e2[t - 1]
        end
        ll = sum(logpdf(Normal(h[t], nt.sigma), ydata[t]) for t in 1:T)
        pr = logpdf(Normal(0, 1), nt.h0) + logpdf(Normal(0, 0.5), nt.phi) +
             logpdf(Exponential(1), nt.s1) + logpdf(Exponential(1), nt.s2) +
             logpdf(Exponential(1), nt.sigma) +
             sum(logpdf.(Normal(0, 1), e1)) + sum(logpdf.(Cauchy(0, 1), e2))
        return ll + pr + logjac(built.layout, u)
    end
    n = built.layout.total
    for u in (collect(range(-0.4, 0.5; length = n)),
              collect(range(0.6, -0.3; length = n)))
        @test _query(built.spec, plan, :posterior, u) ≈ oracle(u)
        _check_gradient(built.spec, plan, u)
    end
end
