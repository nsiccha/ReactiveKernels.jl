module JAXCollectionLoopTests
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

# A `for` over a vector's elements, or over `enumerate` of one, is the loop
# over the vector's positions: one `jax.lax.fori_loop` (or lazy steps to a
# padded row's length) whose body binds the targets at each position, so the
# emitted source does not grow with the data and `jax.grad` differentiates
# it (snag jax-enumerate-nu-546b65cb).

# The reported case: one-based positions and the elements of a computed vector.
@traceable function numbered_sum(values)
    total = 0.0
    for (i, value) in enumerate(values)
        total += i * value
    end
    return total
end
@kernel numbered(x::Float64, y::Vector{Float64}) = begin
    values = x .* y
    value = numbered_sum(values)
    return value
end

# The elements alone.
@traceable function element_squares(a, values)
    total = 0.0
    for v in values
        total += (a * v)^2
    end
    total
end
# `enumerate`'s pair bound to one name.
@traceable function whole_pairs(a, values)
    total = 0.0
    for p in enumerate(values)
        total += p[1] * a * p[2]
    end
    total
end
# Nested `enumerate`, destructured as Julia destructures it.
@traceable function nested_pairs(a, values)
    total = 0.0
    for (i, (j, v)) in enumerate(enumerate(values))
        total += (i + j) * sin(a * v)
    end
    total
end
# The loop's targets are its own: an outer local of the same name keeps its
# value after the loop, as Julia's `for` binds its targets afresh.
@traceable function shadowed_target(a, values)
    value = a
    total = 0.0
    for (i, value) in enumerate(values)
        total += i * value * a
    end
    total + value
end
# A body that writes the vector it iterates reads its writes at later
# positions, as Julia's array iteration does.
@traceable function running_writes(a, values)
    xs = a .* values
    total = 0.0
    for (i, x) in enumerate(xs)
        if i < length(xs)
            xs[i + 1] = xs[i + 1] + x
        end
        total += x
    end
    total
end
# Exits from an enumerated loop: `continue` skips a step, `break` the rest.
@traceable function enumerated_exits(a, values)
    total = 0.0
    for (i, x) in enumerate(values)
        x < 0 && break
        x == 0 && continue
        total += i * log(a * x)
    end
    total
end
# `zip` runs to its shortest vector.
@traceable function zipped_steps(a, values)
    total = 0.0
    for (i, (x, y)) in enumerate(zip(values, values[2:end]))
        total += i * a * (y - x)^2
    end
    total
end
# Generators and comprehensions take the same iterators.
@traceable numbered_generator(a, values) = sum(i * a * v for (i, v) in enumerate(values); init = 0.0)
@traceable zipped_comprehension(a, values) = sum([x * sin(a * y) for (x, y) in zip(values, values[2:end])])
@kernel collection_loops(a::Float64, values::Vector{Float64}) = begin
    squares = element_squares(a, values)
    pairs = whole_pairs(a, values)
    nested = nested_pairs(a, values)
    shadowed = shadowed_target(a, values)
    running = running_writes(a, values)
    exits = enumerated_exits(a, values)
    zipped = zipped_steps(a, values)
    generator = numbered_generator(a, values)
    comprehension = zipped_comprehension(a, values)
    return squares, pairs, nested, shadowed, running, exits, zipped, generator, comprehension
end
const COLLECTION_HELPERS = (squares = element_squares, pairs = whole_pairs,
                            nested = nested_pairs, shadowed = shadowed_target,
                            running = running_writes, exits = enumerated_exits,
                            zipped = zipped_steps, generator = numbered_generator,
                            comprehension = zipped_comprehension)

# A padded row's elements: lazy steps to the padded width, so a term that is
# not finite past the row's end (`log(0)` and below) reaches neither value
# nor gradient (CONTRIBUTING.md, "Lazy branches").
@traceable function enumerated_row_terms(r, a)
    total = zero(a)
    for (t, v) in enumerate(r)
        total += a * log(length(r) - t + 1) + v
    end
    return total
end
@traceable enumerated_row_sum(r, a) = sum(t * a * v for (t, v) in enumerate(r); init = zero(a))
@kernel enumerated_rows(x::Vector{Float64}, R::Vector{Vector{Float64}}) = begin
    cells = plate(R) do r
        enumerated_row_terms(r, x[1]) + enumerated_row_sum(r, x[1])
    end
    total = sum(cells)
    return total
end

reverse_derivative(f, a, values) =
    Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active, Enzyme.Active(a),
                    Enzyme.Const(values))[1][1]

@testset "jax_source: loops over a vector's elements and enumerate" begin
    datasets = (Float64[], [0.2, 0.7], [0.0], [-1.0, NaN], [1.0, 0.0, 2.0, -1.0, NaN],
                [i / 17 for i in 1:129])

    # The reported call: one loop, and a source that does not grow with `y`.
    types = (; x = Float64, y = Vector{Float64})
    reported = [jax_source(numbered; types, data = (; y), functions = (; value = :value))
                for y in datasets]
    @test all(s -> s.text == first(reported).text, reported)
    @test count("jax.lax.fori_loop(", first(reported).text) == 1
    src = jax_source(numbered; have = (:x, :y), want = :value)
    @test count("jax.lax.fori_loop(", string(src)) == 1

    functions = NamedTuple{keys(COLLECTION_HELPERS)}(keys(COLLECTION_HELPERS))
    loops = jax_source(collection_loops; have = (:a, :values), functions)
    bound = [string(jax_source(collection_loops; have = (:a, :values), functions,
                               data = (; values))) for values in datasets]
    @test all(==(first(bound)), bound)

    # The native results the JAX ones must equal.
    native = prepare(numbered; have = (:x, :y), want = :value)
    @test native(0.4, [0.2, 0.7]) ≈ 0.64
    @test reverse_derivative((x, y) -> numbered_sum(x .* y), 0.4, [0.2, 0.7]) ≈ 1.6
    @test shadowed_target(0.5, [1.0, 2.0]) == 0.5 * (1 + 4) + 0.5

    if rk_jax_available()
        for y in datasets
            value = numbered_sum(0.4 .* y)
            derivative = reverse_derivative((x, y) -> numbered_sum(x .* y), 0.4, y)
            for (source, prepare) in ((src, false), (reported[1], true))
                got = rk_jax_evaluate(source, prepare ? :value : :kernel, (0.4, y);
                                      grad = true, prepare)
                @test isequal(got.value, value) || got.value ≈ value
                @test isequal(got.grad, derivative) || got.grad ≈ derivative
            end
        end
        for values in datasets, (name, f) in pairs(COLLECTION_HELPERS)
            value = f(0.7, values)
            derivative = reverse_derivative(f, 0.7, values)
            got = rk_jax_evaluate(loops, name, (0.7, values); grad = true)
            @test isequal(got.value, value) || got.value ≈ value
            @test isequal(got.grad, derivative) || got.grad ≈ derivative
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end

@testset "jax_source: enumerate over a padded row runs no body past its end" begin
    src = jax_source(enumerated_rows; have = (:x, :R), want = :total)
    @test occursin("jax.lax.cond(kept, step_body, lambda c: c, carry)", string(src))
    if rk_jax_available()
        f = prepare(enumerated_rows; have = (:x, :R), want = :total)
        for R in ([[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]], [[0.5, 1.5], Float64[], [2.5]])
            x = [0.7]
            got = rk_jax_evaluate(src, "kernel", (x, R); grad = true)
            @test got.value ≈ f(x, R) rtol = 1e-12
            @test all(isfinite, got.grad)
            @test got.grad ≈ rk_fd_gradient(u -> f(u, R), x) rtol = 1e-6 atol = 1e-6
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end
end
