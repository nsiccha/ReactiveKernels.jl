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

# Where the caller's rebinding cannot reach every later read, emission raises
# rather than drop the write, which would change the value Julia computes
# (CONTRIBUTING.md, "Acceptance evidence": effects are semantic correctness):
# a loop that does not itself assign the array, and a kernel value that later
# recipes read.
@traceable function write_in_loop(x, w0)
    w = copy(w0)
    total = 0.0
    for k in 1:2
        total += write_bump!(w, x)
    end
    return total + sum(w)
end
@kernel jax_write_in_loop(x::Float64, w0::Vector{Float64}) = begin
    value = write_in_loop(x, w0)
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

    message = emission_error(jax_write_in_loop, (:x, :w0))
    @test message !== nothing && occursin("inside a loop or branch that does not assign `w`", message)
    message = emission_error(jax_write_branch_rebinding, (:x, :w0))
    @test message !== nothing && occursin("rebinds it inside a loop, branch or `let`", message)
    message = emission_error(jax_write_recipe_value, (:x, :w0))
    @test message !== nothing &&
          occursin("this call (a kernel statement, or lowered code) has no local", message)
end

end
