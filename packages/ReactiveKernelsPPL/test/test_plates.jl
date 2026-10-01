using Distributions
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
    plan = lower_rkppl(ast, data)
    bound = bind_data(plan, Dict{Symbol,AbstractVector}(cols); dims)
    return bound, build_kernel(bound)
end

_pl_plate(R, cells...) = Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
    Expr(:for, Expr(:(=), :i, R), Expr(:block, cells...)))

@testset "plate cells mean one loop iteration" begin
    data = (:y, :x)
    head = (:(a ~ Normal(0, 1)), :(b ~ Normal(0, 2)), :(s ~ Exponential(1)))
    top = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
        :(y .~ Normal.(mu, s))), data)
    # A scalar observation object and its broadcast spelling are the same
    # statement in a loop body (broadcasting over scalars returns the
    # scalar); both are the vectorized observation.
    for obj in (:(Normal(mu[i], s)), :(Normal.(mu[i], s)))
        got = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
            _pl_plate(:(eachindex(y)), :(y[i] ~ $obj))), data)
        @test _pl_canon(got) == _pl_canon(top)
    end
    # Scalar arithmetic in a cell local: `a + b * x[i]` is the cell value;
    # the dotted spelling of the same line lowers identically.
    for rhs in (:(a + b * x[i]), :(a .+ b .* x[i]))
        got = lower_rkppl(Expr(:block, head...,
            _pl_plate(:(eachindex(y)), :(mu = $rhs), :(y[i] ~ Normal(mu, s)))),
            data)
        @test _pl_canon(got) == _pl_canon(top)
    end
    # Response wrappers draw once per index too.
    wraptop = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
        :(y .~ truncated.(Normal.(mu, s), 0, 10))), data)
    wrap = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
        _pl_plate(:(eachindex(y)), :(y[i] ~ truncated(Normal(mu[i], s), 0, 10)))),
        data)
    @test _pl_canon(wrap) == _pl_canon(wraptop)
    # Operators over scalar-only operands stay scalar (`2 * s` is the same
    # value in every iteration).
    sc = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
        _pl_plate(:(eachindex(y)), :(y[i] ~ Normal(mu[i], 2 * s)))), data)
    sctop = lower_rkppl(Expr(:block, head..., :(mu = a .+ b .* x),
        :(y .~ Normal.(mu, 2 * s))), data)
    @test _pl_canon(sc) == _pl_canon(sctop)
end

@testset "plate gathers through a data index column" begin
    data = (:y, :g)
    head = (:(c[levels(g)] .~ Normal.(0, 2)), :(s ~ Exponential(1)))
    top = lower_rkppl(Expr(:block, head..., :(y .~ Normal.(c[g], s))), data)
    got = lower_rkppl(Expr(:block, head...,
        _pl_plate(:(eachindex(y)), :(y[i] ~ Normal(c[g[i]], s)))), data)
    @test _pl_canon(got) == _pl_canon(top)
    # The index column must be data.
    err = try
        lower_rkppl(Expr(:block, head..., :(h = g),
            _pl_plate(:(eachindex(y)), :(y[i] ~ Normal(c[h[i]], s)))), data)
        nothing
    catch e
        e
    end
    @test err isa SurfaceLoweringError && occursin("not bound data", err.message)
    # Other cross-index reads stay refused.
    @test_throws "cross-index reads" lower_rkppl(Expr(:block, head...,
        _pl_plate(:(eachindex(y)), :(y[i] ~ Normal(c[g[i - 1]], s)))), data)
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
        plan = lower_rkppl(_pl_pk_chain(obs), _PL_PK_DATA)
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
        for dims in (Dict(:kernel_nsub_conc => 2), Dict(:nsub => 2))
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
    plan = lower_rkppl(two, _PL_PK_DATA)
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
    @test_throws "schedule-chain value" lower_rkppl(_pl_pk_chain(
        :(dv .~ Normal.(conc, sigma)), :(conc .~ Normal.(0.0, 1.0))),
        _PL_PK_DATA)
    @test_throws "feeds no observation" lower_rkppl(_pl_pk_chain(
        :(dv .~ Normal.(log_Vc, sigma))), _PL_PK_DATA)
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
    _pl_same_density(legacy, new, 0.05 .* (1:12) .- 0.3)
    # Eight schools.
    y8 = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
    se8 = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
    legacy = _pl_corpus("78_kernel_eight_schools_dosefree_grouped.jl", Dict(
        :school_id => collect(1:8), :school_idx => collect(1:8),
        :time => zeros(8), :dsubj => Int[], :dtime => Float64[],
        :damt => Float64[], :yy => y8, :se => se8))
    new = _pl_corpus("99_plate_78_eight_schools.jl",
        Dict(:school_idx => collect(1:8), :yy => y8, :se => se8))
    _pl_same_density(legacy, new, 0.05 .* (1:10) .- 0.2)
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

@testset "a latent plate's size is its range" begin
    # `eachindex(w)` iterates `w`: the latent has one cell per entry of `w`.
    # Latents over a non-observation axis do not lower yet, so a plate whose
    # column is not the observation axis is refused instead of being sized
    # by the response rows.
    ast = Expr(:block, :(tau ~ Exponential(1)), :(s ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
            Expr(:for, :(j = eachindex(w)), Expr(:block,
                :(eta[j] ~ Normal(0, tau))))),
        _pl_plate(:(eachindex(y)), :(y[i] ~ Normal(eta[i], s))))
    plan = lower_rkppl(ast, (:y, :w))
    @test only(plan.plate_parameters).range === :w
    @test_throws "has 3 cells but n_obs is 4" bind_data(plan,
        Dict{Symbol,AbstractVector}(:y => [0.1, 0.4, -0.2, 0.3],
            :w => [1.0, 2.0, 3.0]))
    ok = bind_data(plan, Dict{Symbol,AbstractVector}(
        :y => [0.1, 0.4, -0.2, 0.3], :w => [1.0, 2.0, 3.0, 4.0]))
    @test assign_layout(ok).total == 2 + 4
end

@testset "plans without a kernel refuse dims keys" begin
    plan = lower_rkppl(Expr(:block, :(a ~ Normal(0, 1)),
        :(s ~ Exponential(1)),
        _pl_plate(:(eachindex(y)), :(y[i] ~ Normal(a, s)))), (:y,))
    @test_throws "not consumed" bind_data(plan,
        Dict{Symbol,AbstractVector}(:y => [0.1, 0.2]);
        dims = Dict{Symbol,Int}(:nsub => 2, :whatever_typo => 3))
end
