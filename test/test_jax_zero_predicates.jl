module JAXZeroPredicateTests
using ReactiveKernels, Test
include("jax_python.jl")

@kernel zero_predicates(x) = begin
    all_zero = all(iszero, x)
    no_zero = all(!iszero, x)
    some_zero = any(iszero, x)
    some_nonzero = any(!iszero, x)
    return all_zero, no_zero, some_zero, some_nonzero
end
@kernel guarded(a::Float64, x::Vector{Float64}) = begin
    total = all(!iszero, x) ? a * sum(x) : a * a
    return total
end
@kernel guarded_rows(a::Float64, rows::Vector{Vector{Float64}}) = begin
    values = plate(rows) do row
        all(!iszero, row) ? a * sum(row) : a * a
    end
    total = sum(values)
    return total
end
throwing_predicate(x) = x == 0 ? false : error("predicate visited")
@kernel unsupported(x::Vector{Float64}) = begin
    value = all(throwing_predicate, x)
    return value
end

@testset "JAX zero-predicate reductions" begin
    wants = (:all_zero, :no_zero, :some_zero, :some_nonzero)
    inputs = (0.0, -0.0, 0.7, Inf, NaN, Float64[], [0.0, -0.0],
              [0.7, Inf, NaN], [0.7, -0.0], zeros(0, 2),
              [0.0 0.0; -0.0 0.0], [0.7 1.3; 2.1 4.0],
              Int[], [0, 2], Bool[], [false, true])
    sources = Dict(T => jax_source(zero_predicates; have=(:x,), want=wants,
                                   types=(x=T,)) for T in unique(typeof.(inputs)))
    # CONTRIBUTING.md's lazy-branch contract requires preserving exceptions:
    # a general eager predicate reduction would visit the throwing second item.
    @test_throws JAXEmissionError jax_source(unsupported; want=:value)
    @test !unsupported([0.0, 1.0])
    dynamic = jax_source(guarded; have=(:a, :x), want=:total)
    datasets = (Float64[], [0.7], [0.7, 1.3], [0.0, 1.3],
                [0.7 + i / 100 for i in 1:257])
    prepared = [jax_source(guarded; want=:total, data=(; x)) for x in datasets]
    @test all(s -> s.text == first(prepared).text, prepared)
    ragged = jax_source(guarded_rows; have=(:a, :rows), want=:total)
    if rk_jax_available()
        for x in inputs
            got = rk_jax_evaluate(sources[typeof(x)], :kernel, (x,))
            expected = (all(iszero, x), all(!iszero, x), any(iszero, x), any(!iszero, x))
            @test Tuple(got.value) == expected
        end
        for x in datasets, a in (-0.7, 0.2)
            expected = all(!iszero, x) ? a * sum(x) : a * a
            derivative = all(!iszero, x) ? sum(x) : 2a
            for (src, mode) in ((dynamic, false), (first(prepared), :arguments))
                got = rk_jax_evaluate(src, :kernel, (a, x); grad=true, prepare=mode)
                @test got.value ≈ expected
                @test got.grad ≈ derivative
            end
        end
        for rows in (Vector{Float64}[], [Float64[], [0.7, 1.3], [0.0]],
                     [Float64[], Float64[]], [[0.7], [0.7, 1.3, 2.1]])
            got = rk_jax_evaluate(ragged, :kernel, (0.2, rows); grad=true)
            @test got.value ≈ sum((all(!iszero, row) ? 0.2sum(row) : 0.2^2
                                  for row in rows); init=0.0)
            @test got.grad ≈ sum((all(!iszero, row) ? sum(row) : 0.4
                                 for row in rows); init=0.0)
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end
end
