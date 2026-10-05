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
    for (n, G) in ((0, 2), (1, 2), (17, 7)), bound in (false, true)
        X = reshape(sin.(1:n*G), n, G)
        weight = sum((n-i+1)*X[i,s] for s in 1:G for i in 1:n; init=0.0)
        expected = n*G*q[1] + weight*q[2]
        k = bound ? prepare(subject_scans; bound=(; X)) : prepare(subject_scans)
        args = bound ? (q,) : (q, X)
        ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), args...; active=:q)
        @test k(args...) ≈ expected
        @test ad_gradient(ad, args...) ≈ [n*G, weight]
    end
end

# Each scan result below is summed by Base `sum`; an empty sequence must
# differentiate under ordinary Reverse like a nonempty one (see
# `benchmark/repro_enzyme_branch_allocation_phi.jl`).
@kernel summed(xs, gain) = begin
    history = scan(xs, Ref(gain); init=0.0) do carry, x, g
        next = carry + x*g
        (next, next)
    end
    total = sum(history)
    return total
end
@kernel trajectory(xs, gain) = begin
    path = scan(xs, Ref(gain); init=gain, include_init=true) do carry, x, g
        next = carry + x*g
        (next, next)
    end
    total = sum(path)
    return total
end
# The plate fuses into the scan loop; the second reader keeps its pointwise
# vector materialized.
@kernel fused(xs, gain) = begin
    history = scan(xs, Ref(gain); init=0.0) do carry, x, g
        next = carry + x*g
        (next, next)
    end
    pointwise = plate(history, Ref(gain)) do h, s
        h*s
    end
    total = sum(pointwise)
    squares = sum(abs2, pointwise)
    objective = total + squares
    return objective
end

@testset "empty scans keep ordinary reverse AD" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    gain = 0.7
    for n in (0, 1, 5)
        xs = sin.(1:n)
        original = copy(xs)
        prefix = cumsum(xs)
        cases = (
            (summed, :total, gain*sum(prefix), sum(prefix)),
            (trajectory, :total, gain*(1 + sum(prefix .+ 1)), 1 + sum(prefix .+ 1)),
            (fused, :objective, gain^2*sum(prefix) + gain^4*sum(abs2, prefix),
             2gain*sum(prefix) + 4gain^3*sum(abs2, prefix)))
        for (spec, want, value, derivative) in cases
            k = prepare(spec; want)
            ad = prepare_ad(k, backend, xs, gain; active=:gain)
            got, gradient = ad_value_and_gradient(ad, xs, gain)
            @test k(xs, gain) ≈ value atol=1e-12
            @test got ≈ value atol=1e-12
            @test gradient ≈ derivative atol=1e-12
        end
        # The runtime scan op that tensorized bodies call.
        op = only(r.op for r in summed.graph.recipes
                  if r.op isa ReactiveKernels._AuthoredScanOp)
        @test op(0.0, xs, gain) ≈ gain .* prefix
        runtime = Enzyme.autodiff(Enzyme.Reverse, g -> sum(op(0.0, xs, g)),
                                  Enzyme.Active, Enzyme.Active(gain))
        @test only(only(runtime)) ≈ sum(prefix) atol=1e-12
        @test xs == original
    end
end

# A declared scan output converts every assignment to its type, so the
# lowering must never bind it to a placeholder when the step's inferred output
# type is not concrete (here `Union{Float64, Int}`).
@kernel declared_unstable(xs, gain) = begin
    history::Vector{Float64} = scan(xs, Ref(gain); init=0.0) do carry, x, g
        next = x > 0 ? carry + x*g : 0
        (next, next)
    end
    total = sum(history)
    return total
end

@testset "declared scan outputs accept a non-concrete step type" begin
    k = prepare(declared_unstable; want=:total)
    for (xs, expected) in ((Float64[], 0.0), ([0.5, -1.0, 2.0], 0.35 + 0.0 + 1.4))
        original = copy(xs)
        @test k(xs, 0.7) ≈ expected
        @test xs == original
    end
end
end
