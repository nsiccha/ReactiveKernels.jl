module JAXArgumentWriteTests
# A device helper that writes its array argument in place writes its caller's
# array, as Julia passes arrays by reference: the helper's function returns
# the written array after its value and the caller rebinds the local it
# passed, so a later read (another helper's, or the caller's own) sees the
# write (snag jax-tuple-splat-05c8497a). Each value and ordinary reverse
# gradient is compared with the native kernel's.
using ReactiveKernels, Test, Enzyme
using LogExpFunctions: logsumexp
include("jax_python.jl")

# A helper that writes its argument by index and returns another value.
@traceable function write_bump!(w, x)
    for s in eachindex(w)
        w[s] += x * s
    end
    return x
end
@traceable function write_reader(x, w0)
    w = copy(w0)
    t = write_bump!(w, x)
    return t + sum(abs2, w)
end
# Through a helper that passes the array on, twice: each call sees the last.
@traceable write_twice!(w, x) = write_bump!(w, x) + write_bump!(w, 2x)
@traceable function write_nested(x, w0)
    w = copy(w0)
    t = write_twice!(w, x)
    return t * sum(w) + w[end]
end
# A new array passed directly: the write reaches nothing else.
@traceable write_fresh(x, w0) = write_bump!(copy(w0), x) + sum(w0)
# A helper that rebinds its argument to a copy first writes only the copy;
# one that writes, then rebinds, wrote the caller's array until the rebinding.
@traceable function write_own_copy!(w, x)
    w = copy(w)
    w[1] += x
    return sum(w)
end
@traceable function write_then_copy!(w, x)
    w[1] += x
    w = copy(w)
    w[end] += 3x
    return sum(w)
end
@traceable function write_rebinding(x, w0)
    w = copy(w0)
    a = write_own_copy!(w, x)
    b = write_then_copy!(w, x)
    return a * b + sum(abs2, w)
end

# The reported sequential factorization, written as two calls: field `b`'s
# step starts from the weights field `a`'s step left behind.
@traceable function write_advance!(column, weights)
    total = 0.0
    for i in eachindex(first(column))
        before = logsumexp(weights)
        for (s, value) in enumerate(column)
            weights[s] += value[i]
        end
        total += logsumexp(weights) - before
    end
    return total
end
@traceable function write_two_steps(records, w0)
    values = vec(records)
    weights = copy(w0)
    a = map(values) do value
        value.a
    end
    sa = write_advance!(a, weights)
    b = map(values) do value
        value.b
    end
    sb = write_advance!(b, weights)
    return sa + 2 * sb + sum(weights)
end

for (name, helper) in ((:jax_write_reader, :write_reader), (:jax_write_nested, :write_nested),
                       (:jax_write_fresh, :write_fresh), (:jax_write_rebinding, :write_rebinding))
    @eval @kernel $name(x::Float64, w0::Vector{Float64}) = begin
        value = $helper(x, w0)
        return value
    end
end
@kernel jax_write_two_steps(x::Float64, indices::Vector{Int}, y::Vector{Float64},
                            w0::Vector{Float64}) = begin
    records = plate(indices, x, Ref(y)) do i, s, ys
        (; a = ys .* (s + i), b = ys .* (s - i))
    end
    value = write_two_steps(records, w0)
    return value
end

# A loop or branch that calls a writing helper without assigning the array
# itself carries the array (`_jax_carried_names!`), so the write reaches the
# reads after it, as Julia's does.
@traceable function write_in_loop(x, w0)
    w = copy(w0)
    total = 0.0
    for k in 1:2
        total += write_bump!(w, x)
    end
    return total + sum(w)
end
@traceable function write_in_nested_loops(x, w0)
    w = copy(w0)
    for a in 1:2
        for b in 1:a
            write_bump!(w, x * b)
        end
    end
    return sum(abs2, w)
end
# One arm writes, the other leaves the array as it entered.
@traceable function write_in_branch(x, w0)
    w = copy(w0)
    if x > 0
        write_bump!(w, x)
    end
    return sum(abs2, w)
end
# A counted `while` (a `for` that `jax.grad` reverses).
@traceable function write_in_while(x, w0)
    w = copy(w0)
    k = 0
    while k < 3
        write_bump!(w, x)
        k = k + 1
    end
    return sum(abs2, w)
end
# A `&&` or `||` whose right side writes runs it only where Julia's does.
@traceable function write_short_circuit(x, w0)
    w = copy(w0)
    t = x > 0 && write_bump!(w, x) > 0
    u = x > 1 || write_bump!(w, 2x) < 0
    return sum(abs2, w) + t + 2u
end
# ReactiveKernelsPPL's nested pointwise marginalization: the method a nesting
# selects loops over positions and calls the writing method at each, a loop
# that never assigns `weights`.
@traceable function write_pointwise(cells::AbstractArray{<:Number}, weights)
    before = logsumexp(weights)
    for (s, c) in enumerate(cells)
        weights[s] += c
    end
    return logsumexp(weights) - before
end
@traceable function write_pointwise(values::AbstractArray{<:AbstractArray}, weights)
    template = first(values)
    result = similar(template, Float64)
    for i in eachindex(template)
        cells = map(values) do value
            value[i]
        end
        result[i] = write_pointwise(cells, weights)
    end
    return result
end
@traceable function write_nested_pointwise(records, w0)
    weights = copy(w0)
    a = map(vec(records)) do value
        value.a
    end
    pointwise = write_pointwise(a, weights)
    return sum(pointwise) + 2 * sum(weights)
end
for (name, helper) in ((:jax_write_in_loop, :write_in_loop),
                       (:jax_write_in_nested_loops, :write_in_nested_loops),
                       (:jax_write_in_branch, :write_in_branch),
                       (:jax_write_in_while, :write_in_while),
                       (:jax_write_short_circuit, :write_short_circuit))
    @eval @kernel $name(x::Float64, w0::Vector{Float64}) = begin
        value = $helper(x, w0)
        return value
    end
end
@kernel jax_write_nested_pointwise(x::Float64, indices::Vector{Int}, y::Vector{Float64},
                                   w0::Vector{Float64}) = begin
    records = plate(indices, x, Ref(y)) do i, s, ys
        (; a = ys .* (s + i), b = ys .* (s - i))
    end
    value = write_nested_pointwise(records, w0)
    return value
end
# A `while` whose condition reads the array its body's helper writes:
# `jax.lax.while_loop`, which JAX differentiates forward only.
@traceable function write_until(x, w0)
    w = copy(w0)
    while w[1] < 1.0
        write_bump!(w, abs(x) + 0.25)
    end
    return sum(w)
end
@kernel jax_write_until(x::Float64, w0::Vector{Float64}) = begin
    value = write_until(x, w0)
    return value
end

# Where the caller's rebinding cannot reach every later read, emission raises
# rather than drop the write, which would change the value Julia computes
# (CONTRIBUTING.md, "Acceptance evidence": effects are semantic correctness):
# a loop calling a writing helper that no name outside the loop body names
# before translation, and a kernel value that later recipes read.
@traceable function write_through_local(x, w0)
    w = copy(w0)
    total = 0.0
    for k in 1:2
        g = write_bump!
        total += g(w, x)
    end
    return total + sum(w)
end
@kernel jax_write_through_local(x::Float64, w0::Vector{Float64}) = begin
    value = write_through_local(x, w0)
    return value
end
# Rebound on one path only: whether the write reaches the caller is data.
@traceable function write_maybe_copy!(w, x)
    if x > 0
        w = copy(w)
    end
    w[1] += x
    return x
end
@traceable function write_branch_rebinding(x, w0)
    w = copy(w0)
    t = write_maybe_copy!(w, x)
    return t + sum(w)
end
@kernel jax_write_branch_rebinding(x::Float64, w0::Vector{Float64}) = begin
    value = write_branch_rebinding(x, w0)
    return value
end
@kernel jax_write_recipe_value(x::Float64, w0::Vector{Float64}) = begin
    weights = copy(w0)
    step = write_bump!(weights, x)
    value = step + sum(weights)
    return value
end

function jax_write_parity(spec, have, datasets)
    data(d) = NamedTuple{Base.tail(have)}(d)
    native = prepare(spec; have, want = :value)
    src = jax_source(spec; have, data = data(first(datasets)), functions = (value = :value,))
    @test all(string(jax_source(spec; have, data = data(d), functions = (value = :value,))) ==
              string(src) for d in datasets)
    if rk_jax_available()
        for d in datasets, mode in (true, :arguments), x in (0.4, -1.3)
            got = rk_jax_evaluate(src, :value, (x, d...); grad = :joint, prepare = mode)
            @test got.value ≈ native(x, d...) atol = 1e-10
            @test got.joint_value ≈ got.value atol = 1e-10
            derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
                Enzyme.Active(x), map(Enzyme.Const, d)...)[1][1]
            @test got.grad ≈ derivative atol = 1e-10
        end
    else
        @test_skip false
    end
    src
end

# A value without `jax.grad` (a `jax.lax.while_loop`), against the native kernel's.
function jax_write_values(spec, have, datasets)
    data(d) = NamedTuple{Base.tail(have)}(d)
    native = prepare(spec; have, want = :value)
    src = jax_source(spec; have, data = data(first(datasets)), functions = (value = :value,))
    @test all(string(jax_source(spec; have, data = data(d), functions = (value = :value,))) ==
              string(src) for d in datasets)
    if rk_jax_available()
        for d in datasets, x in (0.4, -1.3)
            @test rk_jax_evaluate(src, :value, (x, d...); prepare = true).value ≈
                  native(x, d...) atol = 1e-10
        end
    else
        @test_skip false
    end
    src
end

emission_error(spec, have) = try
    jax_source(spec; have, want = :value)
    nothing
catch e
    e isa JAXEmissionError || rethrow()
    sprint(showerror, e)
end

@testset "jax_source: a helper writing its array argument writes the caller's" begin
    weights = [([0.0, -0.5, 0.3],), ([1.0],), ([0.2, 0.1, -0.4, 0.9, 1.5],)]
    native = prepare(jax_write_reader; have = (:x, :w0), want = :value)
    @test native(0.4, [0.0, -0.5, 0.3]) ≈ 0.4 + 0.4^2 + 0.3^2 + 1.5^2
    native = prepare(jax_write_rebinding; have = (:x, :w0), want = :value)
    @test native(0.4, [1.0, 2.0]) ≈ (3.4 * 4.6) + (1.4^2 + 2.0^2)
    for spec in (jax_write_reader, jax_write_nested, jax_write_fresh, jax_write_rebinding)
        jax_write_parity(spec, (:x, :w0), weights)
    end
    src = jax_write_parity(jax_write_two_steps, (:x, :indices, :y, :w0),
        [([1, 2, 3], [0.2, 0.7], [0.0, -0.5, 0.3]),
         ([4, 1], [0.3, -0.1, 1.5, 2.0], [0.1, 0.2]),
         ([2, 5, 1], Float64[], [0.0, 1.0, -1.0])])
    # The authored loops stay runtime loops, the written weights their carry.
    @test occursin("jax.lax.fori_loop(", string(src))

    message = emission_error(jax_write_through_local, (:x, :w0))
    @test message !== nothing && occursin("inside a loop or branch that does not carry `w`", message)
    message = emission_error(jax_write_branch_rebinding, (:x, :w0))
    @test message !== nothing && occursin("rebinds it inside a loop, branch or `let`", message)
    message = emission_error(jax_write_recipe_value, (:x, :w0))
    @test message !== nothing &&
          occursin("this call (a kernel statement, or lowered code) has no local", message)
end

@testset "jax_source: a helper's write inside a loop or branch leaves it through the carry" begin
    weights = [([0.0, -0.5, 0.3],), ([1.0],), ([0.2, 0.1, -0.4, 0.9, 1.5],)]
    native = prepare(jax_write_in_loop; have = (:x, :w0), want = :value)
    @test native(0.4, [1.0, 2.0]) ≈ 0.8 + (1.8 + 3.6)
    native = prepare(jax_write_short_circuit; have = (:x, :w0), want = :value)
    @test native(-1.3, [1.0]) ≈ (1.0 - 2.6)^2 + 2
    for spec in (jax_write_in_loop, jax_write_in_nested_loops, jax_write_in_branch,
                 jax_write_in_while, jax_write_short_circuit)
        src = jax_write_parity(spec, (:x, :w0), weights)
        # The authored loop stays one runtime loop, beside the helper's own.
        spec in (jax_write_in_loop, jax_write_in_while) &&
            @test count("jax.lax.fori_loop(", string(src)) == 2
    end
    src = jax_write_parity(jax_write_nested_pointwise, (:x, :indices, :y, :w0),
        [([1, 2, 3], [0.2, 0.7], [0.0, -0.5, 0.3]),
         ([4, 1], [0.3, -0.1, 1.5, 2.0], [0.1, 0.2]),
         ([2, 5, 1], Float64[], [0.0, 1.0, -1.0])])
    @test occursin("jax.lax.fori_loop(", string(src))
    src = jax_write_values(jax_write_until, (:x, :w0),
        [([0.0, -0.5, 0.3],), ([1.0],), ([-2.0, 0.1, -0.4, 0.9, 1.5],)])
    @test occursin("jax.lax.while_loop(", string(src))
end

end
