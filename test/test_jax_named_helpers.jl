module JAXNamedHelpersTests
using ReactiveKernels, Test
include(joinpath(@__DIR__, "jax_python.jl"))

# Plain authored helpers, recorded or not, are named, documented Python
# functions with their method's argument names (snag
# jax-source-drops-5f297211): a plain helper's straight-line lowered code is
# emitted once as `def scaled_square(x, s)`, a broadcast of one vectorizes
# that function itself, and an emitter-internal helper stays private.

"""
    scaled_square(x, s)

Square of `x` times `s`.
"""
named_scaled_square(x, s) = s * x^2

"""
    named_clipped(x, lo)

`x`, but at least `lo`.
"""
@traceable function named_clipped(x, lo)
    y = x
    if x < lo
        y = lo
    end
    return y
end

"""
    named_mean(level, trend, t)

The mean at step `t`.
"""
named_mean(level, trend, t) = level + trend * t

named_undocumented(a, b) = a * b - b
_named_private(x) = 2 * x + 1

@kernel named_toy(x, s, lo) = begin
    v = named_clipped(named_scaled_square(x, s), lo)
    return v
end

@kernel named_broadcast(level::Float64, trend::Float64, t::Vector{Float64}, lo::Float64) = begin
    mu = named_mean.(level, trend, t)
    floor = named_clipped.(mu, lo)
    direct = named_mean(level, trend, 2.0)
    other = named_undocumented(level, trend) + _named_private(trend)
    total = sum(floor) + direct + other
    return total
end

# `sigma = exp(u)` read by a division and a log: the stable rewrites read `u`
# through the recorded `_rk_div_by_of` and `_rk_log_of`, emitter-internal
# helpers.
@kernel named_rewrite(u::Float64, y::Float64) = begin
    sigma = exp(u)
    z = y / sigma
    total = z * z + log(sigma)
    return total
end

# A named helper returning a ragged field of its tuple argument unchanged:
# the caller keeps that field's own lengths, whose prepared flat positions a
# later `reduce(vcat, ...)` reads (the same for a plain or recorded helper).
@traceable named_second_field(pair) = getfield(pair, 2)
@kernel named_retained(a::Float64, x::Vector{Float64}, S::Vector{Vector{Float64}}) = begin
    paired = plate(eachindex(x), Ref(x), Ref(S)) do i, xs, ss
        (a * xs[i], (a + xs[i]) .* ss[i])
    end
    seconds = plate(paired) do pair
        named_second_field(pair)
    end
    flat = isempty(seconds) ? Float64[] : reduce(vcat, seconds)
    total = sum(abs2, flat)
    return total
end

pydefs(text) = [m[1] for m in eachmatch(r"(?m)^def (\w+)\(", text)]
docstring(text, name) = split(split(text, "def " * name * "(")[2], "\"\"\"")[2]

@testset "a plain helper is a named, documented function" begin
    types = (; x = Float64, s = Float64, lo = Float64)
    src = jax_source(named_toy; have = (:x, :s, :lo), want = :v, types)
    text = src.text
    @test occursin("def named_scaled_square(x: ArrayLike, s: ArrayLike) -> jax.Array:", text)
    @test occursin("Square of `x` times `s`.", docstring(text, "named_scaled_square"))
    @test occursin("def named_clipped(x: ArrayLike, lo: ArrayLike)", text)
    @test occursin("named_scaled_square(x, s)", split(text, "def kernel(")[2])
    @test count(==("named_scaled_square"), pydefs(text)) == 1
    if rk_jax_available()
        native = prepare(named_toy; have = (:x, :s, :lo), want = :v)
        for (x, s, lo) in ((1.5, 2.0, 0.5), (0.1, 3.0, 1.0), (-2.0, 0.5, 0.0))
            r = rk_jax_evaluate(src, :kernel, (x, s, lo); grad = true)
            @test r.value ≈ native(x, s, lo)
            @test r.grad ≈ (s * x^2 < lo ? 0.0 : 2 * s * x)
        end
    end
end

@testset "a broadcast vectorizes the named function itself" begin
    have = (:level, :trend, :t, :lo)
    src = jax_source(named_broadcast; have, want = :total)
    text = src.text
    defs = pydefs(text)
    @test "named_mean" in defs && "named_clipped" in defs
    # No private wrapper around a call of the named function.
    @test !any(d -> occursin("named_mean", d) && startswith(d, "_"), defs)
    @test !any(d -> occursin("named_clipped", d) && startswith(d, "_"), defs)
    @test occursin("jnp.vectorize(named_mean)(", text)
    @test occursin("jnp.vectorize(named_clipped)(", text)
    # The direct scalar call shares the broadcast's definition.
    @test count(==("named_mean"), defs) == 1
    @test occursin("def named_mean(level: ArrayLike, trend: ArrayLike, t: ArrayLike)", text)
    @test occursin("The mean at step `t`.", docstring(text, "named_mean"))
    @test occursin("Compute `named_undocumented` from its Julia definition.",
                   docstring(text, "named_undocumented"))
    # An authored name that marks a helper private keeps it private.
    @test "_named_private" in defs
    @test !any(d -> startswith(d, "rk_") || d == "named_private", defs)
    # One emitted module serves every length of `t` below (0, 3 and 19).
    if rk_jax_available()
        native = prepare(named_broadcast; have, want = :total)
        expected(level, trend, t, lo) =
            sum(max.(level .+ trend .* t, lo)) + level + 2trend + (level * trend - trend) +
            (2trend + 1)
        for t in (Float64[], [1.0, 2.0, 3.0], collect(range(-1.0, 2.0; length = 19)))
            r = rk_jax_evaluate(src, :kernel, (0.3, 0.7, t, 0.5); grad = true)
            @test r.value ≈ native(0.3, 0.7, t, 0.5)
            @test r.value ≈ expected(0.3, 0.7, t, 0.5)
            # d/dlevel: one per unclipped entry, plus the direct call and `other`.
            @test r.grad ≈ count(>=(0.5), 0.3 .+ 0.7 .* t) + 1 + 0.7
        end
    end
end

@testset "a named helper's returned field keeps its prepared lengths" begin
    cases = ((; x = [0.2, -0.5, 0.9], S = [[0.5], [1.5, 2.5], Float64[]]),
             (; x = Float64[], S = Vector{Float64}[]),
             (; x = collect(range(-0.4, 0.7; length = 19)),
              S = [collect(0.5:(i % 5)) for i in 1:19]))
    have = (:a, :x, :S)
    sources = [jax_source(named_retained; have, want = :total, data) for data in cases]
    @test all(src -> src.text == first(sources).text, sources)
    @test count(==("named_second_field"), pydefs(first(sources).text)) == 1
    if rk_jax_available()
        for data in cases
            r = rk_jax_evaluate(first(sources), :kernel, (0.4, data.x, data.S);
                                grad = true, prepare = :arguments)
            @test r.value ≈ sum(((0.4 + x) * s)^2 for (x, row) in zip(data.x, data.S)
                                for s in row; init = 0.0)
            @test r.grad ≈ 2 * sum((0.4 + x) * s^2 for (x, row) in zip(data.x, data.S)
                                   for s in row; init = 0.0)
        end
    end
end

@testset "emitter-internal recorded helpers stay private" begin
    src = jax_source(named_rewrite; have = (:u, :y), want = :total)
    text = src.text
    defs = pydefs(text)
    @test !any(d -> startswith(d, "rk_"), defs)
    @test any(d -> startswith(d, "_") && occursin("div_by", d), defs)
    @test any(d -> startswith(d, "_") && occursin("log_of", d), defs)
    if rk_jax_available()
        for y in (0.5, -1.0, 2.0)
            r = rk_jax_evaluate(src, :kernel, (0.2, y); grad = true)
            @test r.value ≈ (y / exp(0.2))^2 + 0.2
            @test r.grad ≈ -2 * (y / exp(0.2))^2 + 1
        end
    end
end

end
