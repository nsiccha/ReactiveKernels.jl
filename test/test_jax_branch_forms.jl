module JAXBranchFormTests
# The Python a branch emits names nothing it need not (snag
# rkppl-jax-source-681f7aff): a traced selection between values already bound
# is `jnp.where`, an arm without statements is a `lambda`, a branch decided
# while tracing is Python's own `if`, and its guard `n == k || throw(...)` is
# `if not (...): raise`. Values and gradients are the native kernel's,
# including at a degenerate scale, where the untaken arm is non-finite.
using ReactiveKernels, Test
include("jax_python.jl")

# A Gaussian log density whose scale guard selects between bound values.
@kernel jax_form_guarded(x::Float64, m::Float64, s::Float64, ls::Float64) = begin
    z = (x - m) / (s == 0 && x == m ? 1.0 : s)
    lp = -0.5 * z^2 - (s == 0 && x != m ? 0.0 : ls)
    return lp
end
# An arm that computes stays a lazy `jax.lax.cond`: it is not evaluated where
# the other arm is taken (`x / s` at `s == 0`).
@kernel jax_form_lazy(x::Float64, s::Float64, ls::Float64) = begin
    lp = x < 0 ? -Inf : -ls - x / s
    return lp
end
# An `if` whose arm assigns an outer local keeps a lazy branch.
@kernel jax_form_carried(x::Float64, s::Float64) = begin
    lp = begin
        k = 1.0
        if x > s
            k = x - s
        end
        k * x
    end
    return lp
end
# A guard decided while tracing.
@kernel jax_form_static(x::Vector{Float64}) = begin
    total = begin
        length(x) == 3 || throw(DimensionMismatch("x must hold three values"))
        sum(x)
    end
    return total
end

closures(text) = count(r"(?m)^\s*def (then|else)_arm", text)
fd(f, x; h = 1e-6) = (f(x + h) - f(x - h)) / 2h

@testset "jax_source: branch forms name nothing they need not" begin
    guarded = string(jax_source(jax_form_guarded; have = (:x, :m, :s, :ls), want = :lp))
    @test closures(guarded) == 0
    @test !occursin("jax.lax.cond(", guarded)
    @test occursin("jnp.where(jnp.logical_and(s == 0, x == m), 1.0, s)", guarded)
    @test occursin("jnp.where(jnp.logical_and(s == 0, x != m), 0.0, ls)", guarded)
    lazy = string(jax_source(jax_form_lazy; have = (:x, :s, :ls), want = :lp))
    @test closures(lazy) == 0
    @test occursin("jax.lax.cond(x < 0, lambda: -jnp.inf, lambda: ", lazy)
    # The arm that assigns the local has a statement: a nested function, in
    # a lazy `jax.lax.cond`; the other arm is an expression.
    carried = string(jax_source(jax_form_carried; have = (:x, :s), want = :lp))
    @test occursin(r"jax\.lax\.cond\(x > s, then_arm, lambda: ", carried)
    @test !occursin("def else_arm", carried)
    static = string(jax_source(jax_form_static; have = (:x,), want = :total))
    @test closures(static) == 0
    @test !occursin(r"\bvalue\b", static)
    @test occursin("    if not (x.size == 3):\n        raise ValueError(\"x must hold three values\")", static)
    if !rk_jax_available()
        @test_skip false
    else
        native = prepare(jax_form_guarded; have = (:x, :m, :s, :ls), want = :lp)
        src = jax_source(jax_form_guarded; have = (:x, :m, :s, :ls), want = :lp)
        # Ordinary, and degenerate: `s == 0` with the log scale `-Inf`, at and
        # away from the location.
        for args in ((0.4, -0.3, 1.7, log(1.7)), (0.4, 0.4, 0.0, -Inf), (0.4, -0.3, 0.0, -Inf))
            r = rk_jax_evaluate(src, "kernel", args; grad = true)
            @test isequal(r.value, native(args...))
            # The derivative in `x`; at `s == 0` away from the location the
            # native density is the constant `-0.5 * Inf`.
            slope = isfinite(native(args...)) ?
                fd(u -> native(u, args[2:end]...), args[1]) : NaN
            isfinite(slope) && @test r.grad ≈ slope rtol = 1e-6
        end
        @test only(rk_jax_evaluate(src, "kernel", (0.4, 0.4, 0.0, -Inf); grad = true).grad) == 0
        native = prepare(jax_form_lazy; have = (:x, :s, :ls), want = :lp)
        src = jax_source(jax_form_lazy; have = (:x, :s, :ls), want = :lp)
        for args in ((0.4, 1.3, log(1.3)), (-0.4, 1.3, log(1.3)), (-0.4, 0.0, -Inf))
            r = rk_jax_evaluate(src, "kernel", args; grad = true)
            @test isequal(r.value, native(args...))
            # The untaken arm's `x / 0` reaches neither the value nor the gradient.
            args[1] < 0 && @test only(r.grad) == 0
        end
        native = prepare(jax_form_carried; have = (:x, :s), want = :lp)
        src = jax_source(jax_form_carried; have = (:x, :s), want = :lp)
        for args in ((0.4, 1.3), (2.4, 1.3))
            r = rk_jax_evaluate(src, "kernel", args; grad = true)
            @test r.value ≈ native(args...) rtol = 1e-14
            @test r.grad ≈ fd(u -> native(u, args[2]), args[1]) rtol = 1e-6
        end
        native = prepare(jax_form_static; have = (:x,), want = :total)
        src = jax_source(jax_form_static; have = (:x,), want = :total)
        @test rk_jax_evaluate(src, "kernel", ([0.1, 0.2, 0.4],)).value ≈ native([0.1, 0.2, 0.4])
        @test_throws DimensionMismatch native([0.1, 0.2])
        @test_throws "x must hold three values" rk_jax_evaluate(src, "kernel", ([0.1, 0.2],))
    end
end

end
