# A summed observation plate whose bound rows repeat runs once per distinct
# row, weighted by its row count (RK `src/duplicate_cells.jl`, snag
# `summed-plates-ov-67f319ae`): the PosteriorDB shapes it was reported on, as
# `@rkppl` writes them, against the same samplers with the plate's own loop.
using DifferentiationInterface: AutoEnzyme
using Enzyme: Enzyme
using ReactiveKernels: ReactiveKernels
using ReactiveKernelsPPL
using Test

# Dogs: three coefficients over two repeated covariates.
const _DUP_DOGS = @rkppl begin
    beta[1:3] .~ Normal.(0, 100)
    eta = beta[1] .+ beta[2] .* avoids .+ beta[3] .* shocks
    y .~ BernoulliLogit.(eta)
end
# NESLogit: flat coefficients over one discrete covariate.
const _DUP_NES = @rkppl begin
    alpha ~ Flat()
    beta[1:1] .~ Flat.()
    eta = alpha .+ beta[1] .* income
    vote .~ BernoulliLogit.(eta)
end
# Election88-like: intercepts gathered through a bound group index.
const _DUP_GROUPS = @rkppl begin
    b ~ Normal(0, 1)
    a[1:ngroup] .~ Normal.(0, 1)
    eta = a[group] .+ b .* x
    y .~ BernoulliLogit.(eta)
end

_dup_grouped(q) = occursin("_multiplicity", string(ReactiveKernels.code_expr(q.kernel)))

function _dup_samplers(plan, u)
    built = build_kernel(plan)
    sampler() = prepare_sampler(built, plan, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse), native_forward_batch = false)
    sampler(), ReactiveKernels._with_simplify(sampler, false)
end

function _dup_agree(grouped, flat, points)
    for u in points
        @test grouped(u) ≈ flat(u) rtol = 1e-12
        gg = similar(u); gf = similar(u)
        vg = sampler_value_and_gradient!(grouped, gg, u)
        vf = sampler_value_and_gradient!(flat, gf, u)
        @test first(vg) ≈ first(vf) rtol = 1e-12
        @test gg ≈ gf rtol = 1e-10 atol = 1e-10
    end
end

@testset "duplicate cells: summed Bernoulli-logit plates over repeated rows" begin
    # 300 rows, 30 distinct (avoids, shocks, y).
    avoids = repeat([0, 1, 2, 3, 4], 60); shocks = repeat([0, 1], 150)
    y = repeat([0, 1, 1], 100)
    dogs = _DUP_DOGS(; avoids, shocks) | (; y)
    grouped, flat = _dup_samplers(dogs, [0.1, -0.2, 0.05])
    @test _dup_grouped(grouped)
    @test !_dup_grouped(flat)
    _dup_agree(grouped, flat, ([0.1, -0.2, 0.05], [-1.0, 0.3, -0.4]))

    income = Float64.(repeat(1:5, 40)); vote = repeat([0, 1, 1, 0], 50)
    nes = _DUP_NES(; income) | (; vote)
    grouped, flat = _dup_samplers(nes, [0.2, -0.1])
    @test _dup_grouped(grouped)
    _dup_agree(grouped, flat, ([0.2, -0.1], [-2.0, 0.6]))

    group = repeat([1, 2, 3, 4], 60); x = repeat([-1.0, 0.5, 2.0], 80)
    y = repeat([1, 0, 0, 1, 0], 48)
    groups = _DUP_GROUPS(; ngroup = 4, group, x) | (; y)
    u = [0.3, -0.2, 0.1, 0.4, -0.5]
    grouped, flat = _dup_samplers(groups, u)
    @test _dup_grouped(grouped)
    _dup_agree(grouped, flat, (u, -u))

    # Rows that barely repeat keep the plate's own loop.
    nes = _DUP_NES(; income = collect(range(-1, 1; length = 40))) |
          (; vote = repeat([0, 1], 20))
    grouped, _ = _dup_samplers(nes, [0.2, -0.1])
    @test !_dup_grouped(grouped)
end
