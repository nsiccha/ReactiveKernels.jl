module ADAtomicShadowTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# A prepared operator serves one caller at a time, so the shadows its reverse
# mode accumulates into are private to the running call (snag
# prepared-rk-grad-788e46db). Where the loaded Enzyme has `set_atomic_add`, a
# reverse mode that leaves atomic updates to Enzyme is prepared with plain
# updates, and a mode whose caller chose either way is kept. Elsewhere every
# backend is stored as given.

const SWITCH = isdefined(Enzyme, :set_atomic_add)

@kernel shadow_objective(q::Vector{Float64}, scale::Float64) = begin
    total = scale * sum(abs2, q) / 2
end

stored(prepared) = prepared.backend
atomic_setting(backend::AutoEnzyme) = Enzyme.atomic_add(backend.mode)

@testset "prepared reverse modes and atomic shadow updates" begin
    q, scale = [0.3, -1.2, 2.0], 1.7
    gradient_of(backend) =
        prepare_ad(shadow_objective, backend, q, scale; want = :total, active = :q)
    automatic = AutoEnzyme()
    reverse = AutoEnzyme(; mode = Enzyme.Reverse)
    annotated = AutoEnzyme(; mode = Enzyme.Reverse, function_annotation = Enzyme.Const)
    backends = SWITCH ? (automatic, reverse, annotated,
        AutoEnzyme(; mode = Enzyme.set_atomic_add(Enzyme.Reverse, true)),
        AutoEnzyme(; mode = Enzyme.set_atomic_add(Enzyme.Reverse, false))) :
        (automatic, reverse, annotated)
    for backend in backends
        prepared = gradient_of(backend)
        @test ad_value_and_gradient!(prepared, zeros(3), q, scale)[2] ≈ scale .* q
        @test ad_gradient(prepared, [1.0, 0.5, -0.25], scale) ≈
            scale .* [1.0, 0.5, -0.25]
    end

    if SWITCH
        # Left to Enzyme: plain updates, every other mode setting unchanged.
        for backend in (automatic, reverse)
            mode = stored(gradient_of(backend)).mode
            @test Enzyme.atomic_add(mode) === false
            @test Enzyme.set_atomic_add(mode, nothing) === Enzyme.Reverse
        end
        @test stored(gradient_of(annotated)) === AutoEnzyme(;
            mode = Enzyme.set_atomic_add(Enzyme.Reverse, false),
            function_annotation = Enzyme.Const)
        # The caller's choice is kept, atomic updates included.
        for setting in (true, false)
            mode = Enzyme.set_atomic_add(Enzyme.Reverse, setting)
            @test stored(gradient_of(AutoEnzyme(; mode))) === AutoEnzyme(; mode)
        end
        pullback = prepare_ad_pullback(shadow_objective, automatic, 1.0, q, scale;
            want = :total, active = :q)
        @test atomic_setting(stored(pullback)) === false
        # Without a mode, a pushforward stays in forward mode.
        pushforward = prepare_ad_pushforward(shadow_objective, automatic,
            (ones(3),), q, scale; want = :total, active = :q)
        @test stored(pushforward) === automatic
        # A second-order backend: its reverse part, not its forward part.
        second_order = SecondOrder(reverse, AutoEnzyme(; mode = Enzyme.Forward))
        hvp = prepare_ad_hvp(shadow_objective, second_order, (ones(3),), q, scale;
            want = :total, active = :q)
        @test atomic_setting(DifferentiationInterface.outer(stored(hvp))) === false
        @test DifferentiationInterface.inner(stored(hvp)) ===
            DifferentiationInterface.inner(second_order)
        @test only(ad_hvp(hvp, (ones(3),), q, scale)) ≈ scale .* ones(3)
    else
        for backend in backends
            @test stored(gradient_of(backend)) === backend
        end
    end
end

end
