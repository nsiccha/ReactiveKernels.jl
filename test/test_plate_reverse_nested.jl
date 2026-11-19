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
    projection = (RK._PRGroupProjection{Any}(1, 2, true, true),)
    rule = RK._PlateSumRule{(),(1,),typeof(primal),typeof(identity),false,projection}(primal, identity)
    @test RK._pr_rule_target(rule, (x, indices)) isa RK._PlateSumRuleBody
    @test RK._pr_rule_target(rule, (projected, indices)) === primal
    @test rule(projected, indices) == primal(projected, indices)
    types = (Vector{Float64}, Vector{Int})
    @test RK._pr_group_projection_covers(RK._PRGroupProjection{Float64}(1, 2, true, false), types)
    @test !RK._pr_group_projection_covers(RK._PRGroupProjection{Int}(1, 2, true, false), types)
    # A slot and its indices can both be refilled between the two passes.
    ext = Base.get_extension(RK, :ReactiveKernelsEnzymeExt)
    shared = [1, 2]
    values = ([[0.4]], [shared, shared])
    tape = ext._pr_tape(Val((false, true, true)), values)
    values[1][1][1] = 1.7
    shared[1] = 2
    @test tape == ([[0.4]], [[1, 2], [1, 2]])
    @test tape[2][1] === tape[2][2]
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

@kernel integer_totals(groups::Vector{Vector{Float64}}, scale::Float64) = begin
    totals = plate(groups) do xs
        cells = plate(xs) do x
            scale * x
        end
        count::Int = sum(cells)
        count
    end
    result = 1.0 * sum(totals)
    return result
end

@kernel float_totals(groups::Vector{Vector{Float64}}, scale::Float64) = begin
    totals = plate(groups) do xs
        cells = plate(xs) do x
            scale * x
        end
        subtotal::Float64 = sum(cells)
        subtotal
    end
    result = sum(totals)
    return result
end

@kernel indexed_group_totals(rows::Vector{Vector{Float64}}, ids::Vector{Int},
                             offsets::Vector{Float64}, scale::Float64) = begin
    scaled = plate(rows) do xs
        xs .* scale
    end
    totals = plate(ids) do i
        xs = scaled[i]
        offset = offsets[i]
        terms = plate(xs) do x
            sin(x + offset)
        end
        subtotal::Float64 = sum(terms)
        subtotal
    end
    result = sum(totals)
    return result
end

@testset "repeated scalar and ragged row lookups in a cached nested plate" begin
    rows = [[0.4, -1.2], Float64[], [0.8]]
    offsets = [0.12, -0.31, 0.7]
    scale = -0.8
    ids = [3, 1, 2, 1, 3]
    for enabled in (false, true)
        grad = with_plate_reverse(enabled) do
            k = prepare_nonallocating(indexed_group_totals, rows, ids, offsets, scale)
            if enabled
                @test last(last(plate_reverse_report(k, (3, 4)))) == "rule"
            end
            prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), rows, ids, offsets, scale;
                       active = (:offsets, :scale))
        end
        for order in (ids, [2, 2, 3], Int[])
            want = ds = 0.0
            db = zero(offsets)
            for i in order, x in rows[i]
                z = x * scale + offsets[i]
                want += sin(z)
                db[i] += cos(z)
                ds += x * cos(z)
            end
            val, (doffsets, dscale) = ad_value_and_gradient(grad, rows, order, offsets, scale)
            @test val ≈ want
            @test doffsets ≈ db
            @test dscale ≈ ds
        end
    end
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

@testset "typed group totals preserve the assignment's conversion" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    groups, scale = [[-1.0, 0.5], [2.0], Float64[]], 2.0
    for (model, expected) in ((integer_totals, 0.0), (float_totals, 1.5)), flag in (false, true)
        with_plate_reverse(flag) do
            k = prepare(model)
            if flag
                reason = last(only(plate_reverse_report(k, 2)))
                @test model === float_totals ? reason == "rule" : occursin("intermediate", reason)
            end
            ad = prepare_ad(k, backend, groups, scale; active = :scale)
            v, ds = ad_value_and_gradient(ad, groups, scale)
            @test v == 3.0
            @test only(ds) == expected
        end
    end
end
end
