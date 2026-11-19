using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test
import LogDensityProblems

# A sampler prepared on another binding of one build takes everything but
# that binding's data from the build's earlier bindings: RK's bound template
# (the plan, its data-only prefix and compiled residual), the non-allocating
# program of that residual, and the differentiated program of its gradient.
# Each binding still evaluates its own data prefix, seeds its own buffers and
# owns them. Before, every binding prepared all three afresh: about 0.2 s of
# each batch of RKPPLBench's batched SMC (`update_smc` on a target prepared
# from the cumulative rows), against 5 ms now (snag rkppl-smc-with-d-80655b87).

const _RP_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
const _RP_MODEL = @rkppl begin
    beta[1:P] .~ Normal.(0, s0)
    y .~ Normal.(X * beta, 1.0)
end

function _rp_data(n)
    X = hcat(ones(n), sin.(1:n), cos.(0.5 .* (1:n)))
    (; X, y = X * [0.5, -1.0, 2.0] .+ 0.1 .* sin.(3 .* (1:n)))
end
_rp_plan(n) = (d = _rp_data(n); _RP_MODEL(; P = 3, s0 = 2.0, X = d.X) | (; y = d.y))

_rp_nonallocating(built, plan, u) = Base.invokelatest(ReactiveKernels.prepare_nonallocating,
    ReactiveKernels._kernel_latest_prepared(
        prepare_query(built, plan, :sampler; exemplars = (u,))), u)

@testset "a sampler on another binding of one build reuses its preparation" begin
    u = zeros(3)
    sizes = (40, 100, 7, 250)
    built = build_kernel(_rp_plan(first(sizes)))
    points = ([0.3, -0.2, 1.1], [-1.0, 0.4, 0.0], [2.0, 2.0, -3.0])
    for n in sizes
        plan = _rp_plan(n)
        rebound = prepare_sampler(built, plan, u; backend = _RP_BACKEND)
        own = prepare_sampler(build_kernel(plan), plan, u; backend = _RP_BACKEND)
        for x in points
            v1, g1 = LogDensityProblems.logdensity_and_gradient(rebound, x)
            v2, g2 = LogDensityProblems.logdensity_and_gradient(own, x)
            @test v1 === v2
            @test g1 == g2
        end
    end

    # The bindings share one compiled residual, one non-allocating program and
    # one differentiated program; each owns its step buffers.
    kernels = [ReactiveKernels._kernel_latest_prepared(
        prepare_query(built, _rp_plan(n), :sampler; exemplars = (u,))) for n in sizes]
    @test all(k -> k.f === first(kernels).f, kernels)
    nonallocating = [_rp_nonallocating(built, _rp_plan(n), u) for n in sizes]
    first_na = first(nonallocating)
    for k in nonallocating[2:end]
        @test getfield(k, :f) === getfield(first_na, :f)
        @test getfield(k, :ast) === getfield(first_na, :ast)
        @test all(i -> getfield(k, :caches)[i] === nothing ||
                       getfield(k, :caches)[i] !== getfield(first_na, :caches)[i],
                  eachindex(getfield(k, :caches)))
    end
    # Interleaved calls keep each binding's values: the buffers are not shared.
    for (k, n) in zip(nonallocating, sizes), x in points
        plan = _rp_plan(n)
        @test Base.invokelatest(k, x) ===
              Base.invokelatest(_rp_nonallocating(build_kernel(plan), plan, u), x)
    end
    operators = [Base.invokelatest(prepare_ad, k, _RP_BACKEND, u; active = :unconstrained)
                 for k in nonallocating]
    @test all(op -> typeof(op.call) === typeof(first(operators).call), operators)
end
