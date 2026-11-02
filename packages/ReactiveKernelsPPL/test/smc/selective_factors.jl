isdefined(@__MODULE__, :SMCSelectiveFixture) || include("selective_fixture.jl")
module SMCSelectiveFactorTests
using ..SMCSelectiveFixture
using ReactiveKernels, ReactiveKernelsPPL, Random, LinearAlgebra, Test
const F = SMCSelectiveFixture

struct ReadPoint <: AbstractVector{Float64}
    values::Vector{Float64}
    reads::Vector{Int}
end
Base.size(point::ReadPoint) = size(point.values)
Base.IndexStyle(::Type{ReadPoint}) = IndexLinear()
Base.getindex(point::ReadPoint, i::Int) = (push!(point.reads, i); point.values[i])

@testset "An old factor reading a fresh parent remains replaced by the carry" begin
    source = quote
        a[1:nb] .~ Normal.(0, 1)
        z ~ Normal(sum(a), 2)
        y .~ Normal.(a[block], 1)
    end
    oldbuild = F.build(1, Int[], Float64[]; source)
    old = F.calibration(oldbuild, SMCReference([0.3, -0.7], [0.4, 0.6]))
    next = F.build(2, [2], [1.4]; source)
    target = F.window(next, old)
    names = coordinate_names(next.built.layout)
    descriptor = smc_window_coordinates(target)
    @test isempty(descriptor.active_old)
    @test length(descriptor.fresh_slots) == 1
    @test names[only(descriptor.current_indices)] != :z
    for x in (-0.4, 0.0, 1.2)
        gradient = zeros(1)
        expected = F.normal(x, 0, 1) + F.normal(1.4, x, 1)
        @test F.EXT._density(target, [x]) ≈ expected atol=1e-12
        @test F.EXT._gradient(target, gradient, [x]) ≈ expected atol=1e-12
        @test gradient ≈ [1.4-2x] atol=1e-10
    end
end

@testset "Fresh coupled priors agree with full-model values and native Reverse" begin
    for orientation in (:rows, :cols)
        lhs = orientation === :rows ? :(eachrow(z[1:nb, 1:3])) :
            :(eachcol(z[1:3, 1:nb]))
        observed = orientation === :rows ? :(z[block, 1]) : :(z[1, block])
        source = quote
            $lhs .~ Dirichlet([1.5, 2.0, 3.0])
            y .~ Normal.($observed, 1)
        end
        oldbuild = F.build(2, Int[], Float64[]; source)
        oldpoint = [0.1, -0.2, 0.3, -0.4]
        old = F.calibration(oldbuild, SMCReference(oldpoint, ones(4)))
        next = F.build(3, Int[], Float64[]; source)
        target = F.window(next, old)
        descriptor = smc_window_coordinates(target)
        @test length(descriptor.current_indices) == 2
        @test isempty(descriptor.active_old)
        oldq = prepare_sampler(oldbuild.built, oldbuild.plan, oldpoint; backend=F.BACKEND)
        full = prepare_sampler(next.built, next.plan, zeros(6); backend=F.BACKEND)
        current = ReactiveKernelsPPL._smc_coordinate_records(next.built.layout)
        matching = ReactiveKernelsPPL._smc_match_coordinates(old.frame.coordinates, current)
        for u in ([0.2, -0.1], [-0.7, 0.4])
            point = zeros(6)
            for i in eachindex(point)
                iszero(matching.old[i]) || (point[i] = oldpoint[matching.old[i]])
            end
            point[descriptor.current_indices] = u
            fullgradient, gradient = zeros(6), zeros(2)
            fullvalue = first(sampler_value_and_gradient!(full, fullgradient, point))
            expected = fullvalue - oldq(oldpoint)
            @test F.EXT._density(target, u) ≈ expected atol=1e-11
            @test F.EXT._gradient(target, gradient, u) ≈ expected atol=1e-11
            @test gradient ≈ fullgradient[descriptor.current_indices] atol=1e-10
        end
        # One coupled simplex event cannot be split into old and fresh factors.
        @test_throws ArgumentError ReactiveKernelsPPL._smc_fresh_window_terms(
            next.built, next.plan, descriptor.current_indices[1:1], zeros(6))
        # The current whole-array slice transform has no observation row
        # lowering. Refuse this shape rather than materializing old coordinates.
        @test_throws ReactiveKernels.PartitionLocalityError F.window(
            F.build(3, [3], [0.4]; source), old)
    end
end

@testset "Fresh flat coordinates enter even without prior read support" begin
    source = quote
        z[1:nb] .~ Flat()
        y .~ Normal.(z[block], 1)
    end
    old = F.calibration(F.build(1, Int[], Float64[]; source),
        SMCReference([0.3], [0.4]))
    target = F.window(F.build(2, [2], [1.4]; source), old)
    descriptor = smc_window_coordinates(target)
    @test descriptor.current_indices == [2]
    @test descriptor.fresh_slots == [1]
    gradient = zeros(1)
    @test F.EXT._gradient(target, gradient, [0.2]) ≈ F.normal(1.4, 0.2, 1)
    @test gradient ≈ [1.2]
end

@testset "Derived densities cannot silently disappear at a fresh boundary" begin
    source = quote
        z[1:nb] .~ Normal.(0, 1)
        squared = sum(z)^2
        squared ~ Normal(0.3, 1.7)
        y .~ Normal.(z[block], 1)
    end
    old = F.calibration(F.build(1, Int[], Float64[]; source),
        SMCReference([0.3], [0.4]))
    # The extra density owns no coordinate. There is no automatic assignment
    # of it to q_old versus p(fresh | active), so fail instead of omitting it.
    @test_throws ArgumentError F.window(F.build(2, [2], [1.4]; source), old)
    # Retained-only densities are already represented by the carried summary.
    old_only = F.window(F.build(1, [1], [1.4]; source), old)
    @test F.EXT._density(old_only, [0.2]) ≈
        F.normal(0.2, 0.3, 0.4) + F.normal(1.4, 0.2, 1)
    retained_source = quote
        z[1:nb] .~ Normal.(0, 1)
        squared = z[1]^2
        squared ~ Normal(0.3, 1.7)
        y .~ Normal.(z[block], 1)
    end
    retained_old = F.calibration(F.build(1, Int[], Float64[]; source=retained_source),
        SMCReference([0.3], [0.4]))
    fresh_only = F.window(F.build(2, [2], [1.4]; source=retained_source), retained_old)
    @test smc_window_coordinates(fresh_only).current_indices == [2]
    @test F.EXT._density(fresh_only, [0.2]) ≈
        F.normal(0.2, 0, 1) + F.normal(1.4, 0.2, 1)
end

@testset "Fresh native work stays local as inactive history grows" begin
    source = quote
        m ~ Normal(0, 4)
        z[1:nb] .~ Normal.(m, 2)
        y .~ Normal.(z[block], 1)
    end
    nodes(x) = x isa Expr ? 1 + sum(nodes, x.args; init=0) : 1
    emitted, reads, allocated = Int[], Int[], Int[]
    for n in (101, 10_001)
        old = F.calibration(F.build(n, Int[], Float64[]; source),
            SMCReference(zeros(n+1), ones(n+1)))
        target = F.window(F.build(n+1, [n+1], [0.5]; source), old)
        @test smc_window_coordinates(target).current_indices == [1, n+2]
        point, gradient = [0.2, -0.1], zeros(2)
        F.EXT._gradient(target, gradient, point)
        push!(allocated, @allocated F.EXT._gradient(target, gradient, point))
        push!(emitted, nodes(code_expr(target.query.kernel)))
        recording = ReadPoint(point, Int[])
        Base.invokelatest(target.query.kernel, recording)
        @test sort(unique(recording.reads)) == [1, 2]
        push!(reads, length(recording.reads))
    end
    @test emitted[1] == emitted[2]
    @test reads[1] == reads[2]
    @test allocated[2] - allocated[1] < 16_384
    @info "Fresh selective work" emitted reads allocated
end

@testset "Repeated selective boundaries retain a flat bounded Gaussian" begin
    scale = (; diagonal=[1.1, 0.7, 0.9], indices=[[1, 2]],
        reflections=[[cos(pi/8), sin(pi/8)]])
    old = F.calibration(F.build(3, Int[], Float64[]), SMCReference(zeros(3), scale))
    earlybytes = 0
    for window in 1:12
        target = F.window(F.build(3, [mod1(window, 3)], [0.2]), old;
            binding_id="boundary-$window")
        old = carry_smc(target, SMCReference([0.1window], [0.5]);
            calibration_id="flat", maxrank=3)
        @test length(old.reference.location) == 3
        @test length(old.reference.scale.values) <= 3
        @test propertynames(old.frame) == (:model_id, :layout, :coordinates)
        window == 3 && (earlybytes = Base.summarysize(old))
    end
    @test Base.summarysize(old) <= earlybytes + 256
end
end
