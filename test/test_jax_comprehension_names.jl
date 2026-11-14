module JAXComprehensionNameTests
# A comprehension's Python names come from the source (snag
# jax-source-on-rk-d32d60bd): `c = [f(k) for k in 1:K]` binds the `jax.vmap`
# straight to `c`, over `jnp.arange` itself, with a `lambda` element when the
# element is one expression and a `def c_element` when it has statements. No
# `element`/`indices`/`values` scaffolding and no `c = values` alias. Values
# and gradients are the native kernel's, also for no elements.
using ReactiveKernels, Test
include("jax_python.jl")

@kernel jax_comprehension_mix(w::Vector{Float64}, trans::Matrix{Float64},
                              b::Vector{Float64}) = begin
    K = length(b)
    c = [sum(w[i] * trans[i, k] for i in 1:K) for k in 1:K]
    m = [c[k] + b[k] for k in 1:K]
    total = sum(m)
    return total
end
# An element with a statement is a nested function named after the target.
@kernel jax_comprehension_block(w::Vector{Float64}, b::Vector{Float64}) = begin
    d = [begin
             t = w[k]^2
             t + b[k]
         end for k in eachindex(b)]
    total = sum(d)
    return total
end

const MIX = (have = (:w, :trans, :b), want = :total,
             types = (; w = Vector{Float64}, trans = Matrix{Float64}, b = Vector{Float64}))

@testset "jax_source: a comprehension binds its target" begin
    mix = string(jax_source(jax_comprehension_mix; MIX...))
    @test occursin("    c = jax.vmap(lambda k: jnp.sum(jax.vmap(lambda i: w[i - 1] * trans[i - 1, k - 1])(jnp.arange(1, K + 1))", mix)
    @test occursin("    m = jax.vmap(lambda k", mix)
    generated = r"(?m)^\s*(?:def element(_\d+)?\(|(element|indices|values)(_\d+)? = )"
    @test !occursin(generated, mix)
    @test !occursin(r"(?m)^\s*c = values", mix)
    block = string(jax_source(jax_comprehension_block; have = (:w, :b), want = :total))
    @test occursin(r"(?m)^    def d_element\(k\):\n        t = w\[k - 1\] \*\* 2\n", block)
    @test occursin("    d = jax.vmap(d_element)(jnp.arange(1, ", block)
    @test !occursin(generated, block)
    if !rk_jax_available()
        @test_skip false
    else
        native = prepare(jax_comprehension_mix; have = MIX.have, want = MIX.want)
        src = jax_source(jax_comprehension_mix; MIX...)
        for K in (0, 3, 19)
            w = collect(range(-0.4, 0.9; length = K))
            trans = [sin(i + 2k) for i in 1:K, k in 1:K]
            b = collect(range(0.1, -0.3; length = K))
            r = rk_jax_evaluate(src, "kernel", (w, trans, b); grad = true)
            @test r.value ≈ native(w, trans, b) rtol = 1e-14 atol = 1e-15
            @test r.value ≈ sum(trans' * w) + sum(b) rtol = 1e-14 atol = 1e-15
            # d total / d w = row sums of trans.
            @test r.grad ≈ vec(sum(trans; dims = 2)) rtol = 1e-14 atol = 1e-15
        end
        native = prepare(jax_comprehension_block; have = (:w, :b), want = :total)
        src = jax_source(jax_comprehension_block; have = (:w, :b), want = :total)
        for K in (0, 3, 19)
            w = collect(range(-0.4, 0.9; length = K))
            b = collect(range(0.1, -0.3; length = K))
            r = rk_jax_evaluate(src, "kernel", (w, b); grad = true)
            @test r.value ≈ native(w, b) rtol = 1e-14 atol = 1e-15
            @test r.grad ≈ 2 .* w rtol = 1e-14 atol = 1e-15
        end
    end
end

end
