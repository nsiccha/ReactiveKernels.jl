using Distributions
using DifferentiationInterface
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# One `@plate` (decision 1cmodra, prong `plates`): a cell means one
# iteration of the Julia `for` loop it is written as — every value in it is
# a scalar — and shapes come from named data (the range, a data index
# column, a schedule's subject column), never from dims keys. Uses `_canon`
# from test_corpus.jl (included right after it). Data and models are
# synthetic; densities are checked against Distributions.jl.

_pl_canon(plan) = sprint(_canon, plan)
_pl_q(built, bound, preset, u) =
    Base.invokelatest(prepare_query(built, bound, preset), u)

function _pl_bind(ast, data, cols; dims = Dict{Symbol,Int}())
    plan = lower_rkppl(ast, data; conditioned = data)
    bound = bind_data(plan, Dict{Symbol,AbstractVector}(cols); dims)
    return bound, build_kernel(bound)
end

_pl_plate(R, cells...) = Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
    Expr(:for, Expr(:(=), :i, R), Expr(:block, cells...)))

@testset "plate cells mean one loop iteration" begin
    data = (:y, :x)
    head = (:(a ~ Normal(0, 1)), :(b ~ Normal(0, 2)), :(s ~ Exponential(1)))
    top = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
        :(y .~ Normal.(mu, s))), data; conditioned = data)
    # A scalar observation object and its broadcast spelling are the same
    # statement in a loop body (broadcasting over scalars returns the
    # scalar); both are the vectorized observation.
    for obj in (:(Normal(mu[i], s)), :(Normal.(mu[i], s)))
        got = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
            _pl_plate(:(eachindex(y)), :(y[i] ~ $obj))), data; conditioned = data)
        @test _pl_canon(got) == _pl_canon(top)
    end
    # Scalar arithmetic in a cell local: `a + b * x[i]` is the cell value;
    # the dotted spelling of the same line lowers identically.
    for rhs in (:(a + b * x[i]), :(a .+ b .* x[i]))
        got = lower_rkppl(Expr(:block, head...,
            _pl_plate(:(eachindex(y)), :(mu = $rhs), :(y[i] ~ Normal(mu, s)))),
            data; conditioned = data)
        @test _pl_canon(got) == _pl_canon(top)
    end
    # Response wrappers draw once per index too.
    wraptop = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
        :(y .~ truncated.(Normal.(mu, s), 0, 10))), data; conditioned = data)
    wrap = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
        _pl_plate(:(eachindex(y)), :(y[i] ~ truncated(Normal(mu[i], s), 0, 10)))),
        data; conditioned = data)
    @test _pl_canon(wrap) == _pl_canon(wraptop)
    # Operators over scalar-only operands stay scalar (`2 * s` is the same
    # value in every iteration): the cell local is a scalar definition.
    sc = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
        _pl_plate(:(eachindex(y)), :(sd = 2 * s), :(y[i] ~ Normal(mu[i], sd)))),
        data; conditioned = data)
    sctop = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
        :(sd = 2 * s), :(y .~ Normal.(mu, sd))), data; conditioned = data)
    @test _pl_canon(sc) == _pl_canon(sctop)
end

@testset "plate gathers through a data index column" begin
    data = (:y, :g)
    head = (:(c[levels(g)] .~ Normal.(0, 2)), :(s ~ Exponential(1)))
    top = lower_rkppl(Expr(:block, head..., :(y .~ Normal.(c[g], s))), data; conditioned = data)
    got = lower_rkppl(Expr(:block, head...,
        _pl_plate(:(eachindex(y)), :(y[i] ~ Normal(c[g[i]], s)))), data; conditioned = data)
    # capability: a data-index alias is an ordinary value (P8 1cmodra; todo `15lq8iu`).
    aliased = lower_rkppl(Expr(:block, head..., :(h = g),
        _pl_plate(:(eachindex(y)), :(y[i] ~ Normal(c[h[i]], s)))), data; conditioned = data)
    # A retained plate and a vectorized gather may have different plans;
    # both, including the index alias, must preserve the authored density.
    cols = Dict(:y => [-0.1, 0.8, 0.2], :g => [1, 2, 1])
    for plan in (top, got, aliased)
        bound = bind_data(plan, cols)
        built = build_kernel(bound)
        u = unconstrain(built.layout, (; s = 1.1, c = [-0.2, 0.4]))
        oracle = v -> begin
            q = constrain(built.layout, v)
            sum(logpdf.(Normal(0, 2), q.c)) + logpdf(Exponential(), q.s) +
                log(q.s) + sum(logpdf.(Normal.(q.c[cols[:g]], q.s), cols[:y]))
        end
        _check_model_math(built, bound, u, oracle)
    end
    # Other cross-index reads stay refused.
    # refused: reads `g[i - 1]`, out of bounds at i = 1 in the Julia loop (P3)
    @test_throws "cross-index reads" lower_rkppl(Expr(:block, head...,
        _pl_plate(:(eachindex(y)), :(y[i] ~ Normal(c[g[i - 1]], s)))), data; conditioned = data)
end

# --- Schedule chains: today's grouped PK cell, written as statements -------

# Two subjects; subject 1 has a repeated-dose segment, subject 2 a single
# dose (the synthetic grouped-PK data of test_ppl_pk_nonallocating.jl).
_pl_pk_cols() = Dict{Symbol,AbstractVector}(
    :subj => [1, 1, 2, 2], :time => [96.0, 120.0, 0.0, 5.0],
    :dsubj => [1, 1, 1, 1, 2], :dtime => [0.0, 24.0, 48.0, 72.0, 0.0],
    :damt => [100.0, 100.0, 100.0, 100.0, 50.0],
    :dv => [10.0, 8.0, 0.5, 7.0], :cc => [3, 1, 0, 2], :age_s => [35.0, 52.0])
const _PL_PK_DATA = (:subj, :time, :dsubj, :dtime, :damt, :dv, :cc, :age_s)

_pl_pk_head() = quote
    sigma ~ Exponential(1.0)
    b0_vc ~ Normal(0.0, 1.0)
    b1_vc ~ Normal(0.0, 1.0)
    b0_k10 ~ Normal(0.0, 1.0)
    b0_k12 ~ Normal(0.0, 1.0)
    b0_k21 ~ Normal(0.0, 1.0)
    b0_ka ~ Normal(0.0, 1.0)
    log_Vc = b0_vc .+ b1_vc .* age_s
    log_k10 = b0_k10
    log_k12 = b0_k12
    log_k21 = b0_k21
    log_ka = b0_ka
    pk_sched = linear_pk_schedule(obs = (:subj, :time),
        dose = (:dsubj, :dtime, :damt))
end

# The same model in the `@plate <result> for s in 1:N` form (dims key).
_pl_pk_legacy(cell...) = Expr(:block, _pl_pk_head().args...,
    Expr(:macrocall, Symbol("@plate"), LineNumberNode(1), :conc,
        Expr(:for, :(s = 1:kernel_nsub_conc), Expr(:block,
            :(read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_k10,
                log_k12, log_k21, log_ka)),
            :(mu = read_locs[pk_sched.obs_map]), cell..., :mu))))

_pl_pk_chain(obs...) = Expr(:block, _pl_pk_head().args...,
    :(read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_k10, log_k12,
        log_k21, log_ka)),
    :(conc = read_locs[pk_sched.obs_map]), obs...)

@testset "schedule chain lowers to the grouped kernel without a dims key" begin
    cols = _pl_pk_cols()
    legacy, lb = _pl_bind(_pl_pk_legacy(:(dv .~ Normal.(mu, sigma))),
        _PL_PK_DATA, cols; dims = Dict{Symbol,Int}(:kernel_nsub_conc => 2))
    for obs in (:(dv .~ Normal.(conc, sigma)),
            _pl_plate(:(eachindex(dv)), :(dv[i] ~ Normal(conc[i], sigma))))
        plan = lower_rkppl(_pl_pk_chain(obs), _PL_PK_DATA; conditioned = _PL_PK_DATA)
        kp = only(plan.kernel_plates)
        @test isempty(plan.responses)
        @test kp.subjects === nothing
        @test kp.result === :conc && kp.collected === :conc
        @test [nm for (nm, _) in kp.assignments] == [:read_locs, :conc]
        @test [s.name for s in kp.schedules] == [:pk_sched]
        bound, built = _pl_bind(_pl_pk_chain(obs), _PL_PK_DATA, cols)
        # Subjects from the schedule's subject column.
        @test only(bound.kernel_plates).subjects == 2
        names = coordinate_names(built.layout)
        @test names == coordinate_names(lb.layout)
        u = 0.05 .* (1:length(names)) .- 0.1
        for preset in (:likelihood, :prior, :sampler)
            @test _pl_q(built, bound, preset, u) ≈
                _pl_q(lb, legacy, preset, u) rtol = 1e-12
        end
        # No dims key is consumed: any key is refused, typo'd or not.
        # refused: legacy `kernel_nsub_conc` key not consumed (wrong data)
        for dims in (Dict(:kernel_nsub_conc => 2), Dict(:nsub => 2))
            # refused: dims key not consumed by a dims-free schedule chain (wrong data)
            @test_throws "not consumed" bind_data(plan, cols;
                dims = Dict{Symbol,Int}(dims))
        end
    end
end

@testset "schedule chain with two observations is one kernel" begin
    cols = _pl_pk_cols()
    two = _pl_pk_chain(
        _pl_plate(:(eachindex(dv)), :(dv[i] ~ Normal(conc[i], sigma))),
        _pl_plate(:(eachindex(cc)), :(lam = exp(conc[i])),
            :(cc[i] ~ Poisson(lam))))
    plan = lower_rkppl(two, _PL_PK_DATA; conditioned = _PL_PK_DATA)
    kp = only(plan.kernel_plates)
    @test length(kp.obs) == 2
    @test kp.result === :conc
    @test [nm for (nm, _) in kp.assignments] == [:read_locs, :conc, :lam]
    bound, built = _pl_bind(two, _PL_PK_DATA, cols)
    # Each observation alone, in the legacy form, at the same draws: the
    # joint likelihood is their sum (shared chain, two response axes).
    gauss, gb = _pl_bind(_pl_pk_legacy(:(dv .~ Normal.(mu, sigma))),
        _PL_PK_DATA, cols; dims = Dict{Symbol,Int}(:kernel_nsub_conc => 2))
    pois, pb = _pl_bind(_pl_pk_legacy(:(lam = exp.(mu)), :(cc .~ Poisson.(lam))),
        _PL_PK_DATA, cols; dims = Dict{Symbol,Int}(:kernel_nsub_conc => 2))
    names = coordinate_names(built.layout)
    @test names == coordinate_names(gb.layout)
    u = 0.03 .* (1:length(names)) .- 0.05
    pnames = coordinate_names(pb.layout)
    up = [u[findfirst(==(n), names)] for n in pnames]
    @test _pl_q(built, bound, :likelihood, u) ≈
        _pl_q(gb, gauss, :likelihood, u) + _pl_q(pb, pois, :likelihood, up) rtol = 1e-12
    # A chain value is a cell value, never a sampled name or a response.
    # refused: `conc` bound by `=` then sampled with `.~` (single assignment)
    @test_throws "schedule-chain value" lower_rkppl(_pl_pk_chain(
        :(dv .~ Normal.(conc, sigma)), :(conc .~ Normal.(0.0, 1.0))),
        _PL_PK_DATA; conditioned = _PL_PK_DATA)
    # capability: schedule chain that feeds no observation (unused deterministic value) (todo `1qlbn5b`)
    free_chain = lower_rkppl(_pl_pk_chain(), _PL_PK_DATA; conditioned = _PL_PK_DATA)
    @test isempty(only(free_chain.kernel_plates).obs)
    @test isempty(free_chain.responses)
end

# --- The 99_plate_* corpus re-spellings against the forms they replace ----

# Bind a corpus program (test_corpus.jl's loader) and build it.
function _pl_corpus(file, cols; dims = Dict{Symbol,Int}())
    ast, data = _load_corpus_case(joinpath(_CORPUS_DIR, file))
    return _pl_bind(ast, data, Dict(k => v for (k, v) in cols if k in data);
        dims)
end

# Same model, same draws: the coordinates line up position by position
# (only the names the lowering recovers differ), so every preset agrees
# at one point.
function _pl_same_density(legacy, new, u)
    (lb, lbuilt), (nb, nbuilt) = legacy, new
    @test length(coordinate_names(nbuilt.layout)) ==
        length(coordinate_names(lbuilt.layout)) == length(u)
    for preset in (:likelihood, :prior, :sampler)
        @test _pl_q(nbuilt, nb, preset, u) ≈ _pl_q(lbuilt, lb, preset, u) rtol = 1e-12
    end
end

@testset "panel re-spellings match the panel form" begin
    # Panel layout: 2 subjects x 3 timepoints, per-subject `dose`; the
    # per-index form reads the same values per row (`dose[i]`).
    t = [0.5, 1.0, 2.0, 0.5, 1.0, 2.0]
    dose = [1.0, 2.0]
    drow = repeat(dose; inner = 3)
    dims = Dict{Symbol,Int}(:kernel_nsub_pred => 2, :kernel_T_pred => 3)
    cases = (
        ("32_kernel_plate.jl", "99_plate_32_gaussian.jl",
            [0.5, 1.2, 2.1, 0.8, 1.5, 2.5]),
        ("69_kernel_plate_poisson.jl", "99_plate_69_poisson.jl",
            [1, 0, 2, 3, 1, 0]),
        ("70_kernel_plate_bernoulli.jl", "99_plate_70_bernoulli.jl",
            [1, 0, 1, 1, 0, 0]),
        ("71_kernel_plate_nb2.jl", "99_plate_71_nb2.jl", [1, 0, 2, 3, 1, 0]),
        ("72_kernel_plate_gamma.jl", "99_plate_72_gamma.jl",
            [0.5, 1.2, 2.1, 0.8, 1.5, 2.5]),
        ("73_kernel_plate_beta.jl", "99_plate_73_beta.jl",
            [0.2, 0.7, 0.5, 0.3, 0.8, 0.4]),
        ("74_kernel_plate_studentt.jl", "99_plate_74_studentt.jl",
            [0.5, 1.2, 2.1, 0.8, 1.5, 2.5]))
    for (lf, nf, obs) in cases
        @testset "$nf" begin
            legacy = _pl_corpus(lf, Dict(:t => t, :dose => dose, :obs => obs);
                dims)
            new = _pl_corpus(nf, Dict(:t => t, :dose => drow, :obs => obs))
            n = length(coordinate_names(new[2].layout))
            _pl_same_density(legacy, new, 0.1 .* (1:n))
        end
    end
    # Distributions.jl oracle for one of them (Poisson log mean b0*dose*t).
    pb, pbuilt = _pl_corpus("99_plate_69_poisson.jl",
        Dict(:t => t, :dose => drow, :obs => [1, 0, 2, 3, 1, 0]))
    @test _pl_q(pbuilt, pb, :likelihood, [0.25]) ≈
        sum(logpdf.(Poisson.(exp.(0.25 .* drow .* t)), [1, 0, 2, 3, 1, 0])) rtol = 1e-12
    # All-scalar panel (one row per subject).
    cols = Dict(:dose => [10.0, 20.0, 5.0, 8.0], :dv => [1.1, 2.2, 0.4, 0.9],
        :ls => [0.1, -0.2, 0.3, 0.0])
    legacy = _pl_corpus("33_kernel_scalar.jl", cols;
        dims = Dict{Symbol,Int}(:kernel_nsub_pred => 4))
    new = _pl_corpus("99_plate_33_offset.jl", cols)
    _pl_same_density(legacy, new, [0.3])
    sigma = exp(0.3)
    mu = (cols[:dose] ./ 10.0) .* exp.(cols[:ls])
    @test _pl_q(new[2], new[1], :likelihood, [0.3]) ≈
        sum(logpdf.(Normal.(mu, sigma), cols[:dv])) rtol = 1e-12
    # Two plates over different rows, one shared coefficient (corpus 76):
    # each observation statement reads its own rows (4 Gaussian, 3
    # Poisson); the panel form needs one dims key per plate.
    cols = Dict(:x1 => [0.5, 1.0, 1.5, 2.0], :y1 => [0.4, 1.1, 1.4, 2.2],
        :x2 => [0.2, 0.4, 0.6], :y2 => [1, 0, 2])
    legacy = _pl_corpus("76_kernel_plates_gaussian_poisson.jl", cols;
        dims = Dict{Symbol,Int}(:kernel_nsub_pred1 => 4,
            :kernel_nsub_pred2 => 3))
    new = _pl_corpus("99_plate_76_gaussian_poisson.jl", cols)
    @test coordinate_names(new[2].layout) ==
        coordinate_names(legacy[2].layout) == [:b0, :sigma]
    @test new[1].n_obs == legacy[1].n_obs == 7
    _pl_same_density(legacy, new, [0.25, 0.3])
    _pl_same_density(legacy, new, [-0.4, 0.1])
    b0, sigma = 0.25, exp(0.3)
    @test _pl_q(new[2], new[1], :likelihood, [0.25, 0.3]) ≈
        sum(logpdf.(Normal.(b0 .* cols[:x1], sigma), cols[:y1])) +
        sum(logpdf.(Poisson.(exp.(b0 .* cols[:x2])), cols[:y2])) rtol = 1e-12
end

@testset "responses observe their own rows" begin
    # A statement broadcasts over the columns it reads (standard Julia), so
    # two responses may observe different rows; n_obs is their total.
    data = (:k1, :n1, :k2, :n2)
    head = (:(theta1 ~ Beta(1.0, 1.0)), :(theta2 ~ Beta(1.0, 1.0)))
    plates = Expr(:block, head...,
        _pl_plate(:(eachindex(k1)), :(k1[i] ~ Binomial(n1[i], theta1))),
        _pl_plate(:(eachindex(k2)), :(k2[i] ~ Binomial(n2[i], theta2))))
    twin = Expr(:block, head..., :(k1 .~ Binomial.(n1, theta1)),
        :(k2 .~ Binomial.(n2, theta2)))
    @test _pl_canon(lower_rkppl(plates, data; conditioned = data)) ==
        _pl_canon(lower_rkppl(twin, data; conditioned = data))
    cols = Dict{Symbol,AbstractVector}(:k1 => [3, 1, 4], :n1 => [5, 5, 6],
        :k2 => [0, 2, 2, 1, 3], :n2 => [4, 4, 3, 2, 5])
    bound, built = _pl_bind(plates, data, cols)
    @test bound.n_obs == 8
    u = [0.3, -0.4]
    nt = constrain(built.layout, u)
    @test _pl_q(built, bound, :likelihood, u) ≈
        sum(logpdf.(Binomial.(cols[:n1], nt.theta1), cols[:k1])) +
        sum(logpdf.(Binomial.(cols[:n2], nt.theta2), cols[:k2])) rtol = 1e-12
    # refused: the columns one statement reads share its rows — Julia's
    # broadcast throws a DimensionMismatch there (standard-Julia semantics,
    # language principle 3).
    bad = merge(cols, Dict{Symbol,AbstractVector}(:n2 => [4, 4, 3]))
    @test_throws "column length 3 ≠ the 5 rows of k2" _pl_bind(plates, data,
        bad)
    # refused: responses reading a common column observe one axis, so a
    # shared covariate cannot serve rows of two lengths (principle 3).
    shared = Expr(:block, :(b ~ Normal(0, 1)), :(s ~ Exponential(1)),
        :(y1 .~ Normal.(b .* x, s)), :(y2 .~ Normal.(b .* x, s)))
    @test_throws "column length 4 ≠ the 3 rows of y2" bind_data(
        lower_rkppl(shared, (:y1, :y2, :x); conditioned = (:y1, :y2, :x)), Dict{Symbol,AbstractVector}(
            :y1 => [0.1, 0.2, 0.3, 0.4], :y2 => [0.5, 0.6, 0.7],
            :x => [1.0, 2.0, 3.0, 4.0]))
    # A latent plate owns its authored axis beside an unrelated response.
    lat = Expr(:block, :(tau ~ Exponential(1)), :(s ~ Exponential(1)),
        :(m ~ Normal(0, 1)),
        _pl_plate(:(eachindex(y1)), :(theta[i] ~ Normal(0, tau)),
            :(y1[i] ~ Normal(theta[i], s))),
        _pl_plate(:(eachindex(y2)), :(y2[i] ~ Normal(m * x2[i], s))))
    latplan = lower_rkppl(lat, (:y1, :y2, :x2); conditioned = (:y1, :y2, :x2))
    cols = Dict{Symbol,AbstractVector}(:y1 => [0.1, 0.2, 0.3],
        :y2 => [0.4, 0.5], :x2 => [1.0, 2.0])
    bound = bind_data(latplan, cols)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.4; length = built.layout.total))
    nt = constrain(built.layout, u)
    @test bound.n_obs == 5
    @test length(nt.theta) == 3
    @test _pl_q(built, bound, :likelihood, u) ≈
        sum(logpdf.(Normal.(nt.theta, nt.s), cols[:y1])) +
        sum(logpdf.(Normal.(nt.m .* cols[:x2], nt.s), cols[:y2]))
end

@testset "dose-free grouped re-spellings need no schedule" begin
    # Radon: 8 counties, 2 rows each. The grouped form binds empty dose
    # columns and a per-county id; the per-index form observes rows with a
    # per-row varying intercept over the county index.
    cidx = repeat(1:8; inner = 2)
    ff = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0,
        0.0, 1.0, 0.0]
    yy = [1.2, 0.8, 0.5, 1.6, 1.4, 1.1, 0.3, 0.9, 1.7, 0.6, 0.2, 0.7, 1.3,
        1.5, 0.4, 1.0]
    legacy = _pl_corpus("77_kernel_radon_dosefree_grouped.jl", Dict(
        :county_id => collect(1:8), :county_idx => cidx, :time => zeros(16),
        :dsubj => Int[], :dtime => Float64[], :damt => Float64[], :ff => ff,
        :yy => yy))
    new = _pl_corpus("99_plate_77_radon.jl",
        Dict(:county_idx => cidx, :ff => ff, :yy => yy))
    n = length(coordinate_names(new[2].layout))
    _pl_same_density(legacy, new, 0.05 .* (1:n) .- 0.3)
    # Eight schools.
    y8 = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
    se8 = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
    legacy = _pl_corpus("78_kernel_eight_schools_dosefree_grouped.jl", Dict(
        :school_id => collect(1:8), :school_idx => collect(1:8),
        :time => zeros(8), :dsubj => Int[], :dtime => Float64[],
        :damt => Float64[], :yy => y8, :se => se8))
    new = _pl_corpus("99_plate_78_eight_schools.jl",
        Dict(:school_idx => collect(1:8), :yy => y8, :se => se8))
    n = length(coordinate_names(new[2].layout))
    _pl_same_density(legacy, new, 0.05 .* (1:n) .- 0.2)
end

@testset "schedule-chain re-spellings match the grouped form" begin
    cols = _pl_pk_cols()
    for (lf, nf, dims) in (
            ("49_kernel_grouped_pk.jl", "99_plate_49_grouped_pk.jl",
                Dict{Symbol,Int}(:kernel_nsub_conc => 2)),
            ("50_kernel_grouped_pk_logf.jl", "99_plate_50_grouped_pk_logf.jl",
                Dict{Symbol,Int}()),
            ("75_kernel_grouped_poisson.jl", "99_plate_75_grouped_poisson.jl",
                Dict{Symbol,Int}(:kernel_nsub_conc => 2)))
        @testset "$nf" begin
            legacy = _pl_corpus(lf, cols; dims)
            new = _pl_corpus(nf, cols)
            @test coordinate_names(new[2].layout) ==
                coordinate_names(legacy[2].layout)
            n = length(coordinate_names(new[2].layout))
            _pl_same_density(legacy, new, 0.02 .* (1:n) .- 0.05)
        end
    end
end

@testset "extracted schedule cells retain their predictor reads" begin
    small = _pl_pk_cols()
    # More subjects and observations, repeated/ragged dose schedules, and
    # one subject with no doses. The subject covariate keeps its own axis.
    large = Dict{Symbol,AbstractVector}(k => vcat(v, v) for (k, v) in small)
    large[:subj] = vcat(small[:subj], small[:subj] .+ 2, [5, 5, 5])
    large[:dsubj] = vcat(small[:dsubj], small[:dsubj] .+ 2)
    large[:time] = vcat(large[:time], [0.0, 1.0, 3.0])
    large[:dv] = vcat(large[:dv], [0.1, 0.2, 0.3])
    large[:cc] = vcat(large[:cc], [0, 1, 0])
    large[:age_s] = vcat(large[:age_s], 43.0)
    # Count the executable expression's nodes, including nested bodies;
    # growth of a data axis must not replicate the generated program.
    function inventory(ex, counts = Dict{Symbol,Int}())
        ex isa Expr || return counts
        counts[ex.head] = get(counts, ex.head, 0) + 1
        foreach(a -> inventory(a, counts), ex.args)
        return counts
    end
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for file in ("99_plate_49_grouped_pk.jl",
            "99_plate_50_grouped_pk_logf.jl", "99_plate_75_grouped_poisson.jl",
            "99_plate_grouped_two_obs.jl")
        ast, names = _load_corpus_case(joinpath(_CORPUS_DIR, file))
        plan = lower_rkppl(ast, names; conditioned = names)
        shapes = Dict{Symbol,Int}[]
        for (cols, subjects) in ((small, 2), (large, 5))
            data = Dict(k => v for (k, v) in cols if k in names)
            @test _pl_canon(lower_rkppl(ast, data; conditioned = data)) == _pl_canon(plan)
            bound = bind_data(plan, data)
            @test only(bound.kernel_plates).subjects == subjects
            @test length(bound.columns[:age_s]) == subjects
            built = build_kernel(bound)
            u = collect(0.002 .* (1:built.layout.total) .- 0.01)
            query = prepare_query(built, bound, :sampler)
            push!(shapes, inventory(code_expr(query)))
            sampler = prepare_sampler(built, bound, u; backend)
            value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
            @test isfinite(value) && all(isfinite, grad)
            @test value ≈ Base.invokelatest(query, u)
            @test grad ≈ _kernel_findiff(x -> Base.invokelatest(query, x), u) rtol = 1e-5 atol = 1e-5
        end
        @test shapes[1] == shapes[2]
    end
    # The read remains visible through another deterministic definition.
    ast = _pl_pk_chain(:(dv .~ Normal.(conc, sigma)))
    insert!(ast.args, 1, :(age_centered = age_s .- 40.0))
    i = findfirst(st -> st isa Expr && st.head === :(=) &&
        st.args[1] === :log_Vc, ast.args)
    ast.args[i] = :(log_Vc = b0_vc .+ b1_vc .* age_centered)
    bound, built = _pl_bind(ast, _PL_PK_DATA, small)
    @test only(bound.kernel_plates).subjects == 2
    @test isfinite(_pl_q(built, bound, :sampler, zeros(built.layout.total)))
end

@testset "a latent plate's size is its range" begin
    # `eachindex(v)` iterates `v`: the latent has one cell per entry of `v`.
    # A subject-level range has two entries beside four observations.
    # Binding must retain the authored latent axis.
    cols = _pl_pk_cols()
    ast = _pl_pk_chain(:(dv .~ Normal.(conc, sigma)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, :(s = eachindex(age_s)), Expr(:block,
                :(eta[s] ~ Normal(0, 1))))))
    plan = lower_rkppl(ast, _PL_PK_DATA; conditioned = _PL_PK_DATA)
    @test only(plan.plate_parameters).range === :age_s
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = [0.1cos(i) for i in 1:built.layout.total]
    q = constrain(built.layout, u)
    @test length(q.eta) == length(cols[:age_s]) == 2
    @test bound.n_obs == length(cols[:dv]) == 4
    baseline = _pl_pk_chain(:(dv .~ Normal.(conc, sigma)))
    basebound, basebuilt = _pl_bind(baseline, _PL_PK_DATA, cols)
    @test built.layout.total == basebuilt.layout.total + 2
    names = filter(!=(:eta), propertynames(q))
    baseq = NamedTuple{Tuple(names)}(Tuple(getproperty(q, name) for name in names))
    baseu = unconstrain(basebuilt.layout, baseq)
    @test _pl_q(built, bound, :likelihood, u) ≈ _pl_q(basebuilt, basebound, :likelihood, baseu)
    @test _pl_q(built, bound, :prior, u) ≈ _pl_q(basebuilt, basebound, :prior, baseu) +
        sum(logpdf.(Normal(), q.eta))
    _check_gradient(built.spec, bound, u)
    # Over the observation axis it binds with one cell per row.
    ast = Expr(:block, :(tau ~ Exponential(1)), :(s ~ Exponential(1)),
        _pl_plate(:(eachindex(y)), :(theta[i] ~ Normal(0, tau)),
            :(y[i] ~ Normal(theta[i], s))))
    plan = lower_rkppl(ast, (:y,); conditioned = (:y,))
    @test only(plan.plate_parameters).range === :y
    ok = bind_data(plan, Dict{Symbol,AbstractVector}(:y => [0.1, 0.4, -0.2]))
    @test assign_layout(ok).total == 2 + 3
end

@testset "plans without a kernel refuse dims keys" begin
    plan = lower_rkppl(Expr(:block, :(a ~ Normal(0, 1)),
        :(b ~ Normal(0, 1)), :(s ~ Exponential(1)),
        _pl_plate(:(eachindex(y)), :(y[i] ~ Normal(a + b * x[i], s)))),
        (:y, :x); conditioned = (:y, :x))
    # refused: dims keys on a plan with no kernel (wrong data)
    @test_throws "not consumed" bind_data(plan,
        Dict{Symbol,AbstractVector}(:y => [0.1, 0.2], :x => [1.0, 2.0]);
        dims = Dict{Symbol,Int}(:nsub => 2, :whatever_typo => 3))
end
