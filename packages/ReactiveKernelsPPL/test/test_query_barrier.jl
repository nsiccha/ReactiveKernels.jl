using DifferentiationInterface
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# The query call paths carry a `Base.invokelatest` world-age barrier. It passes
# the query itself (a mutable object), so the prepared kernel or AD operator is
# not copied. A gradient call returns its value through the query's field and
# allocates exactly what the prepared gradient allocates when called directly;
# a value call returns its value through the barrier, which boxes one
# `Float64` (16 bytes) and keeps the call differentiable. Before, the inline
# kernel or operator was copied to the heap on every call (1.1-2.5 KB per
# sampler gradient of these models) and the `(value, gradient)` result boxed.

const _BARRIER_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# A scalar binomial rate: its gradient allocates nothing.
_barrier_rate = @rkppl begin
    theta ~ Uniform(0, 1)
    k ~ Binomial(n, theta)
end

# A hierarchical normal with per-group latents: its kernel allocates its
# pointwise buffer.
_barrier_groups = @rkppl begin
    mu ~ Normal(0, 5)
    tau ~ HalfCauchy(5)
    @plate for j in eachindex(y)
        theta[j] ~ Normal(mu, tau)
        y[j] ~ Normal(theta[j], sigma[j])
    end
end

_barrier_cases() = (
    (_barrier_rate(; n = 12) | (; k = 5), nothing),
    (_barrier_groups(; sigma = [1.5, 0.8, 2.0, 1.1, 0.9]) |
        (; y = [0.4, -1.2, 2.3, 0.1, 0.7]), :y),
)

# Each helper calls twice, then measures a third call. The direct helpers run
# through `invokelatest` themselves, so their calls inside are static and the
# barrier is outside the measurement. The vararg helpers specialize on their
# argument count, so the measurement holds no dynamic dispatch of their own.
_measure(f::F) where {F} = (f(); f(); @allocated f())
_direct_density(kernel, args::Vararg{Any,N}) where {N} = _measure(() -> kernel(args...))
_direct_gradient(ad, g, args::Vararg{Any,N}) where {N} =
    _measure(() -> ad_value_and_gradient!(ad, g, args...))
_query_density(q, args::Vararg{Any,N}) where {N} = _measure(() -> q(args...))
_public_sampler_density(q, u) = q(u)
_public_cell_density(q, u, cell) = q(u, cell)
_query_gradient(q, g, u) = _measure(() -> sampler_value_and_gradient!(q, g, u))
_cell_gradient(s, g, u, cell) = _measure(() -> cell_value_and_gradient!(s, g, u, cell))
_query_ad_gradient(q, g, u) = _measure(() -> ad_value_and_gradient!(q, g, u))

@testset "query barrier allocates nothing of its own" begin
    for (plan, observation) in _barrier_cases()
        built = build_kernel(plan)
        n = built.layout.total
        u = [0.1 * sin(k) for k in 1:n]
        q = prepare_sampler(built, plan, u; backend = _BARRIER_BACKEND, warm = true)
        g = zeros(n)

        value, returned = sampler_value_and_gradient!(q, g, u)
        @test returned === g
        expected = Base.invokelatest(ad_value_and_gradient!, q.ad, zeros(n), u)
        @test (value, g) == expected
        @test q(u) == value
        @test (@inferred _public_sampler_density(q, u)) == value
        @test q(Float32.(u)) == q(Vector{Float64}(Float32.(u)))

        @test _query_density(q, u) <= Base.invokelatest(_direct_density, q.kernel, u) + 16
        @test _query_gradient(q, g, u) == Base.invokelatest(_direct_gradient, q.ad, g, u)

        # `QueryAD` verbs return the operator's own result through the barrier,
        # which boxes it once: one `(value, gradient)` tuple of 32 bytes. The
        # operator itself is not copied.
        qa = prepare_query_ad(prepare_ad, built, plan, :sampler, _BARRIER_BACKEND, u)
        @test ad_value_and_gradient!(qa, zeros(n), u) == expected
        @test _query_ad_gradient(qa, g, u) <=
              Base.invokelatest(_direct_gradient, qa.prepared, g, u) + 32

        observation === nothing && continue
        cq = prepare_cell_query(built, plan, observation)
        cs = prepare_cell_sampler(built, plan, observation, u; backend = _BARRIER_BACKEND)
        for cell in (1, 3)
            cell_value, cell_g = cell_value_and_gradient!(cs, zeros(n), u, cell)
            @test cq(u, cell) == cell_value
            @test (@inferred _public_cell_density(cq, u, cell)) == cell_value
            @test (cell_value, cell_g) ==
                  Base.invokelatest(ad_value_and_gradient!, cs.ad, zeros(n), u, cell)
        end
        @test _query_density(cq, u, 2) <=
              Base.invokelatest(_direct_density, cq.kernel, u, 2) + 16
        @test _cell_gradient(cs, g, u, 2) == Base.invokelatest(_direct_gradient, cs.ad, g, u, 2)
    end
end
