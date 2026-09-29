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
