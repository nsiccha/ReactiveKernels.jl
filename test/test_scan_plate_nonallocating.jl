module ScanPlateNonallocatingTests
using ReactiveKernels, MutatingFunctions, Test

@kernel collected_paths(q::Vector{Float64}, X::Matrix{Float64}) = begin
    paths = plate(eachcol(X), Ref(q)) do xs, p
        seed = (value=p[1],)
        history = scan(xs, Ref(p); init=seed) do carry, x, params
            next = carry.value + params[2]*x
            ((value=next,), next)
        end
        history
    end
    return paths
end

bytes(k::K, q) where K = @allocated k(q)
@testset "nested scan caches own their per-cell buffers" begin
    q = [0.3, 0.7]
    for n in (0, 1, 17)
        X = reshape(sin.(1:3n), n, 3)
        saved = copy(X)
        k = prepare(collected_paths; bound=(; X))
        na = prepare_nonallocating(k)
        expected = [q[1] .+ q[2].*cumsum(x) for x in eachcol(X)]
        value = na(q)
        @test all(isapprox.(value, expected))
        @test value[1] !== value[2]
        changed = [0.1, 0.4]
        got = na(changed)
        @test all(isapprox.(got, [changed[1] .+ changed[2].*cumsum(x) for x in eachcol(X)]))
        @test X == saved
        @test q == [0.3, 0.7]
        na(q)
        bytes(na, q)
        @test bytes(na, q) == 0
    end
end
end
