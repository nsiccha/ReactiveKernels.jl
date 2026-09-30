using ReactiveKernels, Test

module SharedPreparationFixture
using ReactiveKernels
const calls = zeros(Int, 3)
basis(data) = (calls[1] += 1; data .^ 2)
weights(q, scale) = (calls[2] += 1; q .* scale)
normalizer(w, b) = (calls[3] += 1; sum(w) + sum(b))

@kernel explicit(samples, data, q, scale) = begin
    b = basis(data)
    w = weights(q, scale)
    offset = normalizer(w, b)
    values = plate(samples, Ref(w), Ref(offset)) do x, shared_w, shared_offset
        x * sum(shared_w) - shared_offset
    end
    return values
end

function opaque(samples, data, q, scale)
    out = similar(samples)
    for i in eachindex(samples)
        b = basis(data)
        w = weights(q, scale)
        out[i] = samples[i] * sum(w) - normalizer(w, b)
    end
    return out
end
end

@testset "explicit shared dependencies and bound preparation" begin
    C = SharedPreparationFixture
    data, q, scale = [2., 3.], [.5, -.2], 1.7
    original_data, original_q = copy(data), copy(q)
    for n in (3, 30)
        samples = collect(1.:n)
        fill!(C.calls, 0)
        reference = C.opaque(samples, data, q, scale)
        @test C.calls == [n, n, n]
        fill!(C.calls, 0)
        live = prepare(C.explicit; bound=(; data))
        @test C.calls == [1, 0, 0]
        @test live(samples, q, scale) == reference
        @test C.calls == [1, 1, 1]
        @test live(samples, 2q, scale) == C.opaque(samples, data, 2q, scale)
        fill!(C.calls, 0)
        @test live(samples, q, 2scale) == C.opaque(samples, data, q, 2scale)
        @test C.calls == [n, n+1, n+1]

        fill!(C.calls, 0)
        fixed = prepare(C.explicit; bound=(; data, q, scale))
        @test C.calls == [1, 1, 1]
        @test fixed(samples) == reference
        @test fixed(samples .+ 1) == (samples .+ 1) .* sum(q .* scale) .-
            (sum(q .* scale) + sum(data .^ 2))
        @test C.calls == [1, 1, 1]
        fill!(C.calls, 0)
        rebound = prepare(C.explicit; bound=(; data=2data, q, scale))
        @test C.calls == [1, 1, 1]
        @test rebound(samples) == samples .* sum(q .* scale) .-
            (sum(q .* scale) + sum((2data) .^ 2))

        fill!(C.calls, 0)
        weights_only = prepare(C.explicit; want=:w, bound=(; data))
        @test C.calls == [0, 0, 0]
        @test weights_only(samples, q, scale) == q .* scale
        @test C.calls == [0, 1, 0]
    end
    @test data == original_data && q == original_q
end

# A prepared child spliced into the outer graph, the shape of the ShinyRK
# simulation graph (three prepared children behind a bound schedule).
module CachedBoundFixture
using ReactiveKernels
const calls = Ref(0)
basis(data) = (calls[] += 1; data .^ 2)
const child = prepare(@kernel scaled_child(w, q) = begin
    r = w .* q
    return r
end)
@kernel nested(data, q) = begin
    b = basis(data)
    r = child(b, q)
    total = sum(r)
    return total
end
# A plate cell whose lazy branch reads a bound plate argument: the partial
# evaluator splits the plate per binding, so its residual is value-dependent.
@kernel masked(x, mask) = begin
    values = plate(x, mask) do xi, m
        m > 0 ? 2xi : -xi
    end
    return values
end
end

@testset "cached bound preparation reuses the plan and compiled residual" begin
    C = SharedPreparationFixture
    data, q, scale = [2., 3.], [.5, -.2], 1.7
    samples = collect(1.:5)
    cache = PreparationCache()

    # The residual (weights, normalizer, the plate) is value-independent here:
    # a second binding reruns only the prefix and shares the compiled callable.
    fill!(C.calls, 0)
    k1 = prepare!(cache, C.explicit; bound = (; data))
    @test C.calls == [1, 0, 0]
    @test k1(samples, q, scale) == C.opaque(samples, data, q, scale)
    fill!(C.calls, 0)
    k2 = prepare!(cache, C.explicit; bound = (; data = 2data))
    @test C.calls == [1, 0, 0]
    @test k2.f === k1.f
    @test typeof(k2) === typeof(k1)
    @test k2(samples, q, scale) == C.opaque(samples, 2data, q, scale)
    @test k1(samples, q, scale) == C.opaque(samples, data, q, scale)
    @test k2(samples, q, scale) == prepare(C.explicit; bound = (; data = 2data))(samples, q, scale)
    @test length(cache) == 1

    # Another bound port set is its own entry, again shared across bindings.
    fill!(C.calls, 0)
    k3 = prepare!(cache, C.explicit; bound = (; data, q, scale))
    @test C.calls == [1, 1, 1]
    k4 = prepare!(cache, C.explicit; bound = (; data = 2data, q = 3q, scale))
    @test C.calls == [2, 2, 2]
    @test k4.f === k3.f
    @test k3(samples) == C.opaque(samples, data, q, scale)
    @test k4(samples) == C.opaque(samples, 2data, 3q, scale)
    @test length(cache) == 2

    # A binding that hoists nothing the residual reads returns one kernel.
    w1 = prepare!(cache, C.explicit; want = :w, bound = (; data))
    w2 = prepare!(cache, C.explicit; want = :w, bound = (; data = 2data))
    @test w1 === w2
    @test w1(samples, q, scale) == q .* scale

    # The graph-level form.
    g = Graph()
    x = value!(g, :x, Float64)
    d = value!(g, :d, Vector{Float64})
    s = value!(g, :s, Float64)
    r = value!(g, :r, Float64)
    add!(g; inputs = (d,), outputs = (s,), op = sum)
    add!(g; inputs = (x, s), outputs = (r,), op = *)
    gk1 = prepare!(cache, g; have = (x, d), want = (r,), bound = (d => [1.0, 2.0],))
    gk2 = prepare!(cache, g; have = (x, d), want = (r,), bound = (d => [5.0, 5.0],))
    @test gk1(2.0) == 6.0 && gk2(2.0) == 20.0
    @test gk2.f === gk1.f
    # Bound-port validation is the same as `prepare`'s and caches nothing.
    @test_throws ArgumentError prepare!(cache, g; have = (x, d), want = (r,), bound = (s => 1.0,))
    @test_throws ArgumentError prepare!(cache, g; have = (x, d), want = (r,),
                                        bound = (d => [1.0], d => [2.0]))
end

@testset "cached bound preparation with a spliced prepared child" begin
    F = CachedBoundFixture
    cache = PreparationCache()
    data, q = [1.0, 2.0, 3.0], [2.0, 0.5, 1.0]
    F.calls[] = 0
    k1 = prepare!(cache, F.nested; bound = (; data))
    @test F.calls[] == 1
    @test k1(q) == sum((data .^ 2) .* q)
    k2 = prepare!(cache, F.nested; bound = (; data = 2data))
    @test F.calls[] == 2
    @test k2.f === k1.f
    @test k2(q) == sum(((2data) .^ 2) .* q)
    @test k1(q) == sum((data .^ 2) .* q)
    # The readable residual reports the new binding's constant.
    @test any(recipe -> recipe.op isa ReactiveKernels._BoundConstant &&
                        recipe.op.value == (2data) .^ 2, k2.plan.recipes)
end

@testset "cached bound preparation specializes a value-dependent residual per binding" begin
    F = CachedBoundFixture
    cache = PreparationCache()
    x = [1.0, 2.0, 3.0, 4.0]
    for mask in ([1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0])
        cached = prepare!(cache, F.masked; bound = (; mask))
        expected = [m > 0 ? 2xi : -xi for (xi, m) in zip(x, mask)]
        @test cached(x) == expected
        @test cached(x) == prepare(F.masked; bound = (; mask))(x)
    end
    @test length(cache) == 1
end
