module JAXSourceControlTests
using ReactiveKernels, Test
include("jax_python.jl")

guarded_pair(p) = (length(p) == 2 || throw(DimensionMismatch("two entries")); p[1] * p[2])
piecewise(x) = (x <= 0 && return exp(x); log1p(x))
joined(x) = (y = x > 0 ? x * x : -x; y + 1)
flagged(x, ::Val{flag}) where {flag} = flag ? x * x : 2x
flagged_outer(x) = flagged(x, Val(false)) + flagged(x, Val(true))
@traceable function simd_sum(x)
    total = 0.0
    @inbounds @simd for k in firstindex(x):lastindex(x)
        total += x[k]^2
    end
    total
end
@traceable parity(x, k) = (isodd(k) ? x * x : -x) + (iseven(k) ? x : 0.0)
checked_log(x) = x > 0 ? log(x) : throw(DomainError(x, "log visited"))
@traceable between(x) = 0.0 < x < checked_log(x) + 2.0 ? x * x : 0.0
@traceable chain_middle(x) = -1.0 < checked_log(x) < 1.0

@kernel guarded(p::Vector{Float64}) = begin y = guarded_pair(p); return y end
@kernel pieces(x::Float64) = begin y = piecewise(x); return y end
@kernel joins(x::Float64) = begin y = joined(x); return y end
@kernel flags(x::Float64) = begin y = flagged_outer(x); return y end
@kernel sums(x::Vector{Float64}) = begin y = simd_sum(x); return y end
@kernel odds(x::Float64, k::Int) = begin y = parity(x, k); return y end
@kernel comparisons(x::Float64) = begin y = between(x); return y end
@kernel middle(x::Float64) = begin y = chain_middle(x); return y end

@testset "JAX helper control flow and static tags" begin
    sources = Dict(name => jax_source(spec; have, want=:y) for (name, spec, have) in
        ((:guarded, guarded, (:p,)), (:pieces, pieces, (:x,)), (:joins, joins, (:x,)),
         (:flags, flags, (:x,)), (:sums, sums, (:x,)), (:odds, odds, (:x, :k)),
         (:comparisons, comparisons, (:x,)), (:middle, middle, (:x,))))
    @test count("jax.lax.fori_loop(", string(sources[:sums])) == 1
    # A definition and one call: the middle operand must not be evaluated twice.
    @test count("checked_log(", string(sources[:middle])) == 2
    datasets = (Float64[], [0.7], [i / 17 for i in 1:129])
    bound = [jax_source(sums; have=(:x,), want=:y, data=(; x)) for x in datasets]
    @test all(s -> s.text == first(bound).text, bound)
    if rk_jax_available()
        for x in (-0.7, 0.3, 2.4)
            for (name, f, derivative) in
                ((:pieces, piecewise, x <= 0 ? exp(x) : inv(1 + x)),
                 (:joins, joined, x > 0 ? 2x : -1.0),
                 (:flags, flagged_outer, 2 + 2x),
                 (:comparisons, between, between(x) == 0 ? 0.0 : 2x))
                got = rk_jax_evaluate(sources[name], :kernel, (x,); grad=true)
                @test got.value ≈ f(x)
                @test got.grad ≈ derivative
            end
            for k in (-3, -2, 0, 1, 2)
                got = rk_jax_evaluate(sources[:odds], :kernel, (x, k); grad=true)
                @test got.value ≈ parity(x, k)
                @test got.grad ≈ (isodd(k) ? 2x : 0.0)
            end
        end
        for x in datasets
            got = rk_jax_evaluate(sources[:sums], :kernel, (x,); grad=true)
            @test got.value ≈ simd_sum(x)
            @test got.grad ≈ 2 .* x
        end
        got = rk_jax_evaluate(sources[:guarded], :kernel, ([0.7, 1.3],); grad=true)
        @test got.value ≈ guarded_pair([0.7, 1.3])
        @test got.grad ≈ [1.3, 0.7]
        @test_throws "two entries" rk_jax_evaluate(sources[:guarded], :kernel, ([0.7],))
        for x in (0.2, 0.7, 4.0)
            @test rk_jax_evaluate(sources[:middle], :kernel, (x,)).value == chain_middle(x)
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end
end
