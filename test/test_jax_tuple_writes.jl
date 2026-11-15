using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# A tuple assignment that writes arrays by index (`a[i], b[i] = u, v`) inside
# a loop or a branch rebinds those arrays, so the loop or branch carries them
# as it carries a single `a[i] = u` write. Before, only the tuple's bare names
# were carried and the writes were silently dropped (found while handling
# snag emit-an-rk-kerne-6bbd4365: a helper's `loc[p], scale[p] = l, s` left
# both arrays zero in JAX).

@traceable function jax_tuple_writes(x)
    n = length(x)
    a = zeros(eltype(x), n)
    b = zeros(eltype(x), n)
    s = zero(eltype(x))
    for i in 1:n
        a[i], b[i] = 2 * x[i], x[i]^2
        if x[i] > 0
            s, b[i] = s + x[i], b[i] + 1
        end
    end
    sum(a) + 3 * sum(b) + s
end

@kernel jax_tuple_writes_kernel(x::Vector{Float64}) = begin
    total = jax_tuple_writes(x)
    return total
end

@testset "jax_source: tuple writes to indexed arrays are carried" begin
    src = jax_source(jax_tuple_writes_kernel; have = (:x,), want = :total)
    # Both written arrays and the scalar are the loop's carry.
    @test occursin(r"\(a\w*, b\w*, s\w*\) = carry", string(src))
    if !rk_jax_available()
        @test_skip false
    else
        native = prepare(jax_tuple_writes_kernel; have = (:x,), want = :total)
        for x in ([0.5, -1.25, 2.0], [-0.3, -0.7, 0.4, 1.1])
            r = rk_jax_evaluate(src, "kernel", (x,); grad = true)
            @test r.value ≈ native(x) rtol = 1e-12
            @test r.grad ≈ rk_fd_gradient(native, x) rtol = 1e-6
        end
    end
end
