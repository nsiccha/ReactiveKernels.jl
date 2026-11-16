# Consumer acceptance for snag emit-brm-s-type-3e8e1a1b.
# Run with the src directory archived from BayesianRegressionModels commit
# 5b68a6b5 as ARGS[1], in an environment with RK and SpecialFunctions.
# No downstream source is rewritten or copied into RK's implementation.
module JSBRMHSGP
using ReactiveKernels, Test, SHA
const consumer = joinpath(only(ARGS), "statistical_preparation.jl")
@assert bytes2hex(sha256(read(consumer))) ==
    "612b64eeb004032e0d399755333cc07b172d34b3ecf518c499b6fdb38749f0d4"
include(consumer)
const Prep = StatisticalPreparation

basis2(x, y, k, lo, hi) = Prep.hsgp_basis(x, y; k, domain = ((lo, hi), (lo, hi)))
basis1(x, k, lo, hi) = Prep.hsgp_basis(x; k, domain = ((lo, hi),))
basis_tuple(x, y, k1, k2, lo, hi) = basis2(x, y, (k1, k2), lo, hi)
basis_bad_count(x, y, k) = Prep.hsgp_basis(x, y; k, domain = ((-2.0, 2.0),))

@kernel two_axes(x, y, k, lo, hi) = begin
    result = basis2(x, y, k, lo, hi)
    phi = result[1]
    lambda = result[2]
    return phi
end
@kernel one_axis(x, k, lo, hi) = begin
    result = basis1(x, k, lo, hi)
    phi = result[1]
    lambda = result[2]
    return phi
end
@kernel tuple_modes(x, y, k1, k2, lo, hi) = begin
    result = basis_tuple(x, y, k1, k2, lo, hi)
    phi = result[1]
    lambda = result[2]
    return phi
end
@kernel bad_count(x, y, k) = begin
    result = basis_bad_count(x, y, k)
    phi = result[1]
    lambda = result[2]
    return phi
end

const V = Vector{Float64}
function check(spec, name, types, cases)
    have, want = Tuple(keys(types)), (:phi, :lambda)
    source = ReactiveKernels.js_source(spec; have, want, types, name)
    native = prepare(spec; have, want)
    println(name, ": ", sizeof(source), " bytes; ", count("while (", source),
            " while loops; ", length(cases), " cases per vector representation")
    # Every length, mode count and domain below runs through this one module.
    @test occursin("while (", source)
    for vectors in (:array, :float64array)
        outcomes = ReactiveKernels.js_run(source, name, cases; vectors)
        for (c, r) in zip(cases, outcomes)
            expected = try native(c...) catch e; e end
            @test !r.inputs_mutated
            if expected isa Exception
                @test r.julia_error && r.error == string(nameof(typeof(expected)))
            else
                @test r.error === nothing
                r.error === nothing || continue
                @test all(size(a) == size(b) && isapprox(a, b; rtol = 1e-12, atol = 1e-14)
                          for (a, b) in zip(expected, r.value))
            end
        end
    end
end

@testset "exact BRM fixed-domain HSGP consumer" begin
    x, y = [-1.0, 0.0, 1.0], [0.5, 0.0, -0.5]
    long = collect(range(-1.0, 1.0; length = 37))
    common = [(x, y, -2.0, 2.0), (long, reverse(long), -3.0, 4.0),
              (Float64[], Float64[], -2.0, 2.0), (x, [0.0], -2.0, 2.0),
              (x, y, 0.0, 2.0), (x, y, 2.0, -2.0),
              (x, y, NaN, 2.0), (x, y, -2.0, Inf)]
    for (label, KT, ks) in [(:scalar_basis, Int, [1, 4]),
                            (:vector_basis, Vector{Int}, [[2, 3], [5, 4]])]
        cases = [(a, b, k, lo, hi) for (a, b, lo, hi) in common for k in ks]
        append!(cases, [(x, y, KT === Int ? 0 : [2, 0], -2.0, 2.0),
                        (x, y, KT === Int ? -1 : [2, -1], -2.0, 2.0)])
        KT === Int || push!(cases, (x, y, [2], -2.0, 2.0))
        check(two_axes, label, (; x=V, y=V, k=KT, lo=Float64, hi=Float64), cases)
    end
    check(one_axis, :single_basis, (; x=V, k=Int, lo=Float64, hi=Float64),
          [(x, 3, -2.0, 2.0), (long, 17, -3.0, 4.0), (Float64[], 2, -2.0, 2.0)])
    check(tuple_modes, :tuple_basis, (; x=V, y=V, k1=Int, k2=Int, lo=Float64, hi=Float64),
          [(x, y, 2, 3, -2.0, 2.0), (long, long, 5, 4, -3.0, 4.0),
           (x, y, 2, 0, -2.0, 2.0)])
    check(bad_count, :bad_domain_count, (; x=V, y=V, k=Int), [(x, y, 3)])
end
end
