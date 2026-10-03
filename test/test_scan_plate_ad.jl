module ScanPlateADTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

@kernel subject_scans(q::Vector{Float64}, X::Matrix{Float64}) = begin
    cells = plate(eachcol(X), Ref(q)) do xs, p
        seed = (value=p[1],)
        history = scan(xs, Ref(p); init=seed) do carry, x, params
            next = carry.value + params[2]*x
            ((value=next,), next)
        end
        sum(history)
    end
    total = sum(cells)
    return total
end

@testset "nested plate scans preserve ordinary reverse AD" begin
    q = [0.3, 0.7]
    for (n, G) in ((1, 2), (17, 7)), bound in (false, true)
        X = reshape(sin.(1:n*G), n, G)
        weight = sum((n-i+1)*X[i,s] for s in 1:G for i in 1:n)
        expected = n*G*q[1] + weight*q[2]
        k = bound ? prepare(subject_scans; bound=(; X)) : prepare(subject_scans)
        args = bound ? (q,) : (q, X)
        ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), args...; active=:q)
        @test k(args...) ≈ expected
        @test ad_gradient(ad, args...) ≈ [n*G, weight]
    end
end
end
