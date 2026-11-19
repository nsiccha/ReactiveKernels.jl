module PlateReverseNestedTests
using ReactiveKernels, DifferentiationInterface, Enzyme, MutatingFunctions, Test
const RK = ReactiveKernels

@kernel grouped_loss(groups::Vector{Vector{Int}}, z::Vector{Float64},
                     offsets::Vector{Float64}, scale::Float64, weight::Float64) = begin
    group_totals = plate(groups, offsets) do indices, offset
        xs = z[indices]
        terms = plate(xs) do x
            u = scale * x + offset
            u > 0 ? log1p(u) : sin(u)
        end
        sum(terms)
    end
    result = weight * sum(group_totals)
    return result
end

# Scalar indexing reads the stored coordinate, but this type defines a
# different vector-index operation. Its pullback multiplies by two.
struct ScaledGather <: AbstractVector{Float64}
    data::Vector{Float64}
end
Base.size(x::ScaledGather) = size(x.data)
Base.getindex(x::ScaledGather, i::Int) = x.data[i]
Base.getindex(x::ScaledGather, i::Vector{Int}) = 2 .* x.data[i]

@testset "nested gather dispatch respects the indexed operation" begin
    indices, x = [[2, 2, 1]], [0.1, 0.7]
    projected = ScaledGather(x)
    @test sum(projected[only(indices)]) == 2sum(x[only(indices)])
    # The same compiled caller can receive either array. Only the plain
    # projection can add the unscaled cell partials directly to its parent.
    primal = (source, groups) -> sum(source[only(groups)])
    rule = RK._PlateSumRule{(),(1,),typeof(primal),typeof(identity),false,((1,2,true),)}(primal, identity)
    @test RK._pr_rule_target(rule, (x, indices)) isa RK._PlateSumRuleBody
    @test RK._pr_rule_target(rule, (projected, indices)) === primal
    @test rule(projected, indices) == primal(projected, indices)
end

@kernel weighted_groups(groups::Vector{Vector{Float64}}, weights::Vector{Float64},
                        scale::Float64) = begin
    terms = plate(groups, weights) do xs, w
        ys = plate(xs) do x
            sin(scale * x)
        end
        w * sum(ys)
    end
    result = sum(terms)
    return result
end

function reference(groups, z, offsets, scale, weight)
    total = ds = 0.0
    dz, doff = zero(z), zero(offsets)
    for (g, indices) in pairs(groups), j in indices
        u = scale * z[j] + offsets[g]
        total += u > 0 ? log1p(u) : sin(u)
        du = weight * (u > 0 ? inv(1 + u) : cos(u))
        dz[j] += scale * du
        doff[g] += du
        ds += z[j] * du
    end
    weight * total, (dz, doff, ds)
end

function structure(ex)
    ex isa RK._PlateSumRule && return structure(RK._program_expression(ex.reverse.program))
    ex isa Expr || return (0, 0, 1)
    n, depth, size = 0, 0, 1
    for arg in ex.args
        k, d, s = structure(arg)
        n += k
        depth = max(depth, d)
        size += s
    end
    loop = ex.head in (:for, :while)
    (n + loop, depth + loop, size)
end

@testset "ordinary reverse through a sum of ragged group totals" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    active = (:z, :offsets, :scale)
    groups = [[1, 2, 1], Int[], [3], [1, 3, 1, 2]]
    z, offsets = [0.5, -2.1, 0.1, 0.9], [0.1, 0.6, -1.2, -0.2]
    args = (groups, z, offsets, 0.7, -2.4)
    gradients = map((false, true)) do on
        with_plate_reverse(on) do
            k = prepare(grouped_loss)
            if on
                @test last(only(plate_reverse_report(k, (2, 3, 4)))) == "rule"
                count, depth, _ = structure(RK._ad_inline_plate_programs(k, (2, 3, 4)).ast)
                @test count >= depth >= 2
            end
            prepare_ad(k, backend, args...; active)
        end
    end
    for current in (args, ([[2], [3, 3, 1], Int[]], -z, offsets[1:3], -0.9, 1.3),
                    (Vector{Int}[], z, Float64[], 1.7, -0.3),
                    ([Int[], Int[]], z, offsets[1:2], -0.5, 3.2))
        saved = deepcopy(current)
        value, expected = reference(current...)
        for grad in gradients
            v, actual = ad_value_and_gradient(grad, current...)
            @test v ≈ value
            @test all(isapprox(a, b; atol = 1e-12) for (a, b) in zip(actual, expected))
        end
        @test current == saved
    end
    with_plate_reverse(true) do
        large = ([repeat([1, 2, 1], g) for g in 1:18], z, sin.(1:18), -1.7, 0.4)
        buffered = prepare_nonallocating(grouped_loss, large...)
        @test last(only(plate_reverse_report(buffered, (2, 3, 4)))) == "rule"
        ad = prepare_ad(buffered, backend, large...; active)
        value, expected = reference(large...)
        for _ in 1:2
            v, actual = ad_value_and_gradient(ad, large...)
            @test v ≈ value
            @test all(isapprox(a, b; atol = 1e-12) for (a, b) in zip(actual, expected))
        end
        shapes = map((args, large)) do inputs
            k = prepare(grouped_loss; exemplars = inputs)
            @test last(only(plate_reverse_report(k, (2, 3, 4)))) == "rule"
            structure(RK._ad_inline_plate_programs(k, (2, 3, 4)).ast)
        end
        @test shapes[1] == shapes[2]
        for indices in (groups, [Int[], [2, 1], [3, 3], [2]])
            k = prepare(grouped_loss; bound = (; groups = indices))
            inputs = (z, offsets, 0.7, -2.4)
            ad = prepare_ad(k, backend, inputs...; active)
            v, actual = ad_value_and_gradient(ad, inputs...)
            value, expected = reference(indices, inputs...)
            @test v ≈ value
            @test all(isapprox(a, b; atol = 1e-12) for (a, b) in zip(actual, expected))
        end
    end
end

@testset "a transformed group total receives its own cotangent" begin
    groups, weights, s = [[0.2, -1.1], Float64[], [1.8]], [-1.5, 0.7, 2.1], 0.4
    sums = [sum(x -> sin(s * x), xs; init = 0.0) for xs in groups]
    ds = sum(w * sum(x -> x * cos(s * x), xs; init = 0.0) for (w, xs) in zip(weights, groups))
    with_plate_reverse(true) do
        k = prepare(weighted_groups)
        @test any(r -> occursin("intermediate plate total", last(r)), plate_reverse_report(k, (2, 3)))
        ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), groups, weights, s;
                        active = (:weights, :scale))
        v, (dw, actual_ds) = ad_value_and_gradient(ad, groups, weights, s)
        @test v ≈ sum(weights .* sums)
        @test dw ≈ sums
        @test actual_ds ≈ ds
    end
end
end
