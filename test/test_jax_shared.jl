using ReactiveKernels, Test
include(joinpath(@__DIR__, "jax_python.jl"))

@kernel jax_shared_outputs(x::Vector{Float64}, y::Vector{Float64}) = begin
    squared = sum(abs2, x)
    # This access must never execute for the squared-only entry.
    first_y = y[1]
    shifted = squared + first_y
    return shifted
end

@traceable jax_bound_identity(n) = n
@traceable jax_history_value(history, t) = history[t]
@traceable jax_ragged_identity(flag, rows) = rows

@kernel jax_returned_ragged(flag::Float64, rows::Vector{Vector{Float64}}) = begin
    values = jax_ragged_identity(flag, rows)
    total = sum(reduce(vcat, values))
    return total
end

@testset "named helpers retain returned ragged positions" begin
    region = ReactiveKernels._JaxSourceRegion(:ragged_group, [:flag, :rows], [:values],
        Dict(:values => :value), "A ragged value returned by an authored region.")
    for regions in (ReactiveKernels._JaxSourceRegion[], [region])
        src = jax_source(jax_returned_ragged; have = (:flag, :rows), want = :total,
            data = (; rows = [[1.0, 2.0], Float64[], [3.0]]), source_regions = regions)
        @test count(r"(?m)^def jax_ragged_identity\w*\(", src.text) == 1
        if rk_jax_available()
            for rows in ([[1.0, 2.0], Float64[], [3.0]], [Float64[], Float64[]],
                         [[-1.0], [2.0, 3.0, 4.0], Float64[], [5.0]])
                result = rk_jax_evaluate(src, :kernel, (1.0, rows); grad = true, prepare = :arguments)
                @test result.value ≈ sum(sum, rows)
                @test result.grad ≈ 0.0
            end
        end
    end
end

@kernel jax_returned_bound(x::Vector{Float64}, stops::Vector{Int}) = begin
    total = begin
        acc = 0.0
        for r in eachindex(stops)
            n = jax_bound_identity(stops[r])
            for t in 1:n
                acc += x[t]
            end
        end
        acc
    end
    return total
end

@kernel jax_helper_history(a::Float64, x::Vector{Float64}) = begin
    ys = scan(eachindex(x); init = 0.0, history = 2.0) do carry, t, history
        previous = t > 1 ? jax_history_value(history, t - 1) : 0.0
        (carry, a * previous + x[t])
    end
    total = sum(ys)
    return total
end

@testset "named helpers retain returned bounds and scan history proofs" begin
    src = jax_source(jax_returned_bound; have = (:x, :stops), want = :total,
        data = (; stops = [2, 4]))
    @test count(r"(?m)^\s*def jax_bound_identity\w*\(", src.text) == 1
    if rk_jax_available()
        result = rk_jax_evaluate(src, :kernel, (collect(1.0:6.0), [2, 4]);
            grad = true, prepare = :arguments)
        @test result.value ≈ 13.0
        @test result.grad ≈ [2.0, 2.0, 1.0, 1.0, 0.0, 0.0]
    end
    src = jax_source(jax_helper_history; have = (:a, :x), want = :total,
        data = (; x = ones(3)))
    @test count(r"(?m)^\s*def jax_history_value\w*\(", src.text) == 1
    @test !occursin("compact_history = False", src.text)
    native = prepare(jax_helper_history; have = (:a, :x), want = :total)
    if rk_jax_available()
        for n in (0, 3, 19)
            x = collect(range(0.1, 0.9; length = n))
            result = rk_jax_evaluate(src, :kernel, (0.4, x); grad = true, prepare = :arguments)
            @test result.value ≈ native(0.4, x)
            # Differentiate the same recurrence, independently of the emitter.
            value = derivative = gradient = 0.0
            for element in x
                derivative = value + 0.4 * derivative
                value = 0.4 * value + element
                gradient += derivative
            end
            @test result.grad ≈ gradient
        end
    end
end

@testset "interleaved regions keep named executable portions" begin
    g = Graph()
    x, a, b, c = (value!(g, name, Float64) for name in (:x, :a, :b, :c))
    add!(g, x => a, abs2)
    add!(g, a => b, sin)
    add!(g, b => c, cos)
    p = plan(g; have = x, want = (a, c))
    regions = [
        ReactiveKernels._JaxSourceRegion(:outer, [:x, :b], [:a, :c],
            Dict(:a => :early, :c => :value), "Interleaved outer computation."),
        ReactiveKernels._JaxSourceRegion(:middle, [:a], [:b],
            Dict(:b => :value), "Middle computation.")]
    schedule = ReactiveKernels._jax_region_schedule(p, regions)
    @test length(schedule) == 3
    @test all(unit -> unit.region !== nothing, schedule)
    src = ReactiveKernels._jax_module_source([:kernel => (p, (:a, :c))],
        (Float64,), NamedTuple(), nothing; source_regions = regions)
    @test count(r"(?m)^def outer\w*\(", src.text) == 2
    @test count(r"(?m)^def middle\w*\(", src.text) == 1
    if rk_jax_available()
        result = rk_jax_evaluate(src, :kernel, (0.4,))
        @test result.value ≈ [0.16, cos(sin(0.16))]
    end
end

"Sum the squared entries with an authored loop."
@traceable function jax_authored_energy(x)
    total = 0.0
    for i in eachindex(x)
        total += abs2(x[i])
    end
    total
end

@kernel jax_repeated_helper(x::Vector{Float64}, y::Vector{Float64}) = begin
    left = jax_authored_energy(x)
    right = jax_authored_energy(y)
    total = left + right
    return total
end

@testset "recorded helpers retain named reusable functions" begin
    src = jax_source(jax_repeated_helper; want = :total, name = :total)
    @test count(r"(?m)^def jax_authored_energy\w*\(", src.text) == 1
    @test occursin("Sum the squared entries with an authored loop.", src.text)
    if rk_jax_available()
        for n in (0, 3, 13)
            x, y = fill(0.4, n), fill(-0.7, n + 1)
            got = rk_jax_evaluate(src, :total, (x, y); grad = true)
            @test got.value ≈ jax_authored_energy(x) + jax_authored_energy(y)
            @test got.grad ≈ 2x
        end
    else
        @test_skip false
    end
end

@traceable function jax_assert_positive(x)
    x > 0 || throw(ArgumentError("expected a positive helper input"))
    nothing
end

@kernel jax_cached_checks(x::Float64, y::Float64) = begin
    a = if x > 0
        jax_assert_positive(x)
        x
    else
        x
    end
    b = if y > 0
        jax_assert_positive(-y)
        y
    else
        y
    end
    total = a + b
    return total
end

@testset "reused helper checks retain their calling conditions" begin
    src = jax_source(jax_cached_checks; want = :total)
    @test count(r"(?m)^def jax_assert_positive\w*\(", src.text) == 1
    if rk_jax_available()
        got = rk_jax_evaluate(src, :kernel, (1.0, -2.0); grad = true)
        @test got.value ≈ -1.0
        @test got.grad ≈ 1.0
        @test_throws "expected a positive helper input" rk_jax_evaluate(src, :kernel, (1.0, 2.0))
    else
        @test_skip false
    end
end

@kernel jax_selector_collision(_entry::Float64, shift::Float64) = begin
    squared = _entry^2
    shifted = squared + shift
    return shifted
end

@testset "shared selector preserves authored input names" begin
    src = jax_source(jax_selector_collision; data = (; shift = 2.0),
        functions = (squared = :squared, shifted = :shifted))
    @test first(only(f for f in src.functions if f.name == "squared").inputs).first == "_entry"
    if rk_jax_available()
        got = rk_jax_evaluate(src, :shifted, (0.3, 2.0); grad = true, prepare = :arguments)
        @test got.value ≈ 2.09
        @test got.grad ≈ 0.6
    else
        @test_skip false
    end
end

@testset "shared JAX entry computations" begin
    src = jax_source(jax_shared_outputs;
        functions = (squared = :squared, shifted = :shifted,
                     values = (squared = :squared, shifted = :shifted)))
    text = src.text
    @test count("jnp.sum", text) == 1
    @test occursin("def model(", text)
    @test [f.name for f in src.functions] == ["squared", "shifted", "values"]
    x = [0.3, -0.7, 1.2]
    if rk_jax_available()
        for y in (Float64[], [2.5])
            out = rk_jax_evaluate(src, :squared, (x, y); grad = true)
            @test out.value ≈ sum(abs2, x)
            @test out.grad ≈ 2x
        end
        out = rk_jax_evaluate(src, :shifted, (x, [2.5]); grad = true)
        @test out.value ≈ sum(abs2, x) + 2.5
        @test out.grad ≈ 2x
        vals = rk_jax_evaluate(src, :values, (x, [2.5])).value
        @test vals["squared"] ≈ sum(abs2, x)
        @test vals["shifted"] ≈ sum(abs2, x) + 2.5
    else
        @test_skip false
    end
end

jax_joint_alternative(x) = (sin(x), cos(x))

@testset "entry sharing preserves alternative producers" begin
    g = Graph()
    x, a, b = (value!(g, n, Float64) for n in (:x, :a, :b))
    add!(g, x => a, abs2; cost = 1)
    # Joint and independent queries choose different producers for a.
    add!(g; inputs = x, outputs = (a, b), op = jax_joint_alternative, cost = 1.5)
    independent = plan(g; have = x, want = a)
    joint = plan(g; have = x, want = (a, b))
    @test !ReactiveKernels._jax_compatible_plans(independent, joint)
    mod = ReactiveKernels._JaxModule()
    bodies, functions = String[], ReactiveKernels.JAXFunction[]
    ReactiveKernels._jax_functions!(mod, bodies, functions,
        [:independent => (independent, :a), :joint => (joint, (:a, :b))], (Float64,))
    @test length(bodies) == 2
    @test [f.name for f in functions] == ["independent", "joint"]
end

@traceable function jax_prepared_check(x)
    x > 0 || throw(ArgumentError("prepared helper requires positive x"))
    x^2
end
@kernel jax_checked_endpoint(x::Float64) = begin
    value = jax_prepared_check(x)
    return value
end
const jax_checked_prepared = prepare(jax_checked_endpoint)
@kernel jax_conditional_prepared(x::Float64, enabled::Float64) = begin
    value = enabled > 0 ? jax_checked_prepared(x) : x
    return value
end

@testset "prepared callees transport the active check guard" begin
    src = jax_source(jax_conditional_prepared; have = (:x, :enabled), want = :value)
    native = prepare(jax_conditional_prepared)
    if rk_jax_available()
        for (x, enabled, gradient) in ((-2.0, -1.0, 1.0), (2.0, 1.0, 4.0))
            result = rk_jax_evaluate(src, :kernel, (x, enabled); grad = true)
            @test result.value ≈ native(x, enabled)
            @test result.grad ≈ gradient
        end
        @test_throws r"prepared helper requires positive x" rk_jax_evaluate(
            src, :kernel, (-2.0, 1.0); grad = true)
    else
        @test_skip false
    end
end

@traceable jax_tuple_identity(rows) = rows

@kernel jax_returned_tuple(a::Float64, x::Vector{Float64},
                           R::Vector{Vector{Float64}}, S::Vector{Vector{Float64}}) = begin
    paired = plate(eachindex(x), Ref(x), Ref(R), Ref(S)) do i, xs, rs, ss
        (a .* rs[i] .+ xs[i], (a + xs[i]) .* ss[i], a * xs[i])
    end
    retained = jax_tuple_identity(paired)
    firsts = plate(retained) do pair
        pair[1]
    end
    seconds = plate(retained) do pair
        pair[2]
    end
    first_flat = isempty(firsts) ? Float64[] : reduce(vcat, firsts)
    second_flat = isempty(seconds) ? Float64[] : reduce(vcat, seconds)
    total = sum(first_flat) + sum(second_flat)
end

@testset "named boundaries retain each ragged tuple field" begin
    cases = (
        (; x = [0.2, -0.5, 0.9], R = [[1., 2.], Float64[], [3.]],
           S = [[0.5], [1.5, 2.5], Float64[]]),
        (; x = Float64[], R = Vector{Float64}[], S = Vector{Float64}[]),
        (; x = collect(range(-0.4, 0.7; length = 23)),
           R = [collect(1.0:(i % 7)) for i in 1:23],
           S = [collect(0.5:(i % 5)) for i in 1:23]),
    )
    have = (:a, :x, :R, :S)
    native = prepare(jax_returned_tuple; have, want = :total)
    region = ReactiveKernels._JaxSourceRegion(:tuple_group, [:paired], [:retained],
        Dict(:retained => :value), "Retain independently sized tuple fields.")
    for regions in (ReactiveKernels._JaxSourceRegion[], [region])
        sources = [jax_source(jax_returned_tuple; have, want = :total, data,
                              source_regions = regions) for data in cases]
        structure(text) = [count(pattern, text) for pattern in
            (r"(?m)^\s*def ", "jax.vmap(", "jax.lax.cond(", "jax.lax.fori_loop(")]
        @test all(src -> structure(src.text) == structure(first(sources).text), sources)
        @test count(r"(?m)^def jax_tuple_identity\w*\(", first(sources).text) == 1
        if rk_jax_available()
            for data in cases
                args = (0.4, data.x, data.R, data.S)
                result = rk_jax_evaluate(first(sources), :kernel, args;
                                        grad = true, prepare = :arguments)
                @test result.value ≈ native(args...)
                @test result.grad ≈ sum(sum, data.R; init = 0.0) + sum(sum, data.S; init = 0.0)
            end
        else
            @test_skip false
        end
    end
end
