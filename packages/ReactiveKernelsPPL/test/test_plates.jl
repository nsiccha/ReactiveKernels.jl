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
