module ScanEndpointScopeTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test
include(joinpath(@__DIR__, "fixtures", "scan_endpoint_scope.jl"))
const F = ScanEndpointScopeFixtures

function reference(xs, step)
    previous = 0.0
    result = Float64[]
    for x in xs
        previous = step(previous, x)
        push!(result, previous)
    end
    result
end

@testset "object endpoints bind scan-local ports" begin
    object, control = prepare(F.object_scan), prepare(F.function_scan)
    for xs in (Float64[], [0.3, 0.9, 1.5], sin.(1:17))
        saved = copy(xs)
        @test object(xs) == control(xs)
        @test object(xs) isa Vector{Float64}
        @test prepare(F.object_scan; bound=(; times=xs))() == control(xs)
        @test prepare(F.object_scan; want=:total)(xs) == sum(control(xs))
        @test xs == saved
    end
    step = scan_body(only(r for r in object.plan.recipes
                         if r.op isa ReactiveKernels._AuthoredScanOp))
    @test any(v.name === :dt for r in step.recipes for v in r.outputs)
    @test all(!(r.op isa Union{KernelSpec,KernelObjectSpec,PreparedKernel})
              for r in step.recipes)

    for xs in (Float64[], [0.25, -0.5, 1.5], sin.(1:19))
        @test prepare(F.shifted_scan)(xs) == reference(xs,
            (previous, x) -> (previous + 2x) + x / 2)
        @test prepare(F.computed_scan)(xs) == reference(xs,
            (previous, x) -> 2 * ((previous + (2x + 1)) + x / 2))
    end
    for xs in (Float64[], [-1.0, 2.0, 0.0, 4.0, -2.0])
        @test prepare(F.lazy_scan)(xs) == reference(xs,
            (previous, x) -> x > 0 ? previous + log(x) : previous)
    end
end

@testset "scan endpoints preserve ordinary native reverse and ownership" begin
    q = [0.7]
    for n in (3, 17), bound in (false, true)
        xs = sin.(1:n)
        saved_q, saved_xs = copy(q), copy(xs)
        weight = sum(((n - i + 1) * xs[i] for i in eachindex(xs)); init=0.0)
        k = bound ? prepare(F.scaled_scan; bound=(; xs)) : prepare(F.scaled_scan)
        args = bound ? (q,) : (q, xs)
        ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), args...; active=:q)
        @test k(args...) ≈ q[1] * weight
        @test ad_gradient(ad, args...) ≈ [weight]
        @test q == saved_q && xs == saved_xs
    end
    # Empty materialized child scans have a separate native static-activity
    # capability gap (ReactiveKernels todo 1acmh42), also in the function control.
    for spec in (F.scaled_scan, F.scaled_function_scan), bound in (false, true)
        xs = Float64[]
        k = bound ? prepare(spec; bound=(; xs)) : prepare(spec)
        args = bound ? (q,) : (q, xs)
        ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), args...; active=:q)
        @test k(args...) === 0.0
        @test_broken ad_gradient(ad, args...) ≈ [0.0]
        @test q == [0.7] && isempty(xs)
    end
end
end
