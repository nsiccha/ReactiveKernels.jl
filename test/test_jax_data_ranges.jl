using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# Ranges whose ends are entries of the prepared data (snag
# jax-data-indexed-76ff8d4a): each record names an inclusive interval
# `starts[r]:stops[r]` of a series, possibly empty. JAX needs a static number
# of steps for `jax.grad`; `prepare_data` computes the longest interval's.

@kernel jax_ranges_loop(x::Vector{Float64}, starts::Vector{Int}, stops::Vector{Int}) = begin
    result = begin
        out = zeros(length(starts))
        for r in 1:length(starts)
            acc = 0.0
            for t in starts[r]:stops[r]
                acc = acc + x[t]
            end
            out[r] = acc
        end
        sum(abs2, out)
    end
    return result
end
# A generator reduction, a comprehension and a slice over the same ranges.
@kernel jax_ranges_forms(x::Vector{Float64}, starts::Vector{Int}, stops::Vector{Int}) = begin
    result = begin
        acc = 0.0
        for r in 1:length(starts)
            v = [x[t] * t for t in starts[r]:stops[r]]
            acc = acc + r * sum(x[t]^2 for t in starts[r]:stops[r]; init = 0.0) +
                sum(v) + length(v) + sum(abs2, x[starts[r]:stops[r]])
        end
        acc
    end
    return result
end
# A plate over the records: its cells' entries of `starts` and `stops` are
# one record's.
@kernel jax_ranges_plate(x::Vector{Float64}, starts::Vector{Int}, stops::Vector{Int}) = begin
    cells = plate(starts, stops) do a, b
        sum(x[t] for t in a:b; init = 0.0)
    end
    total = sum(abs2, cells)
    return total
end
# One data end: bounded by the least start or the largest stop. Shifted ends
# and ends bound to names first keep their bound.
@kernel jax_ranges_open(x::Vector{Float64}, starts::Vector{Int}, stops::Vector{Int}) = begin
    result = begin
        acc = 0.0
        for r in 1:length(starts)
            for t in starts[r]:length(x)
                acc = acc + r * x[t]
            end
            for t in 1:stops[r]
                acc = acc + x[t]^2
            end
            a = starts[r]
            b = stops[r]
            for t in (a + 1):(b + 1)
                acc = acc + r * x[t - 1]^3
            end
        end
        acc
    end
    return result
end
# Indexed writes: a step past the stop writes nothing.
@kernel jax_ranges_writes(x::Vector{Float64}, starts::Vector{Int}, stops::Vector{Int}) = begin
    result = begin
        y = zeros(length(x))
        for r in 1:length(starts)
            for t in starts[r]:stops[r]
                y[t] = y[t] + x[t] * r
            end
        end
        sum(abs2, y)
    end
    return result
end
# Nested ranges from a record's start to an element of its range, and a
# triangular loop from another loop's index.
@kernel jax_ranges_nested(x::Vector{Float64}, starts::Vector{Int}, stops::Vector{Int}) = begin
    result = begin
        acc = 0.0
        for r in 1:length(starts)
            for t in starts[r]:stops[r]
                for s in starts[r]:t
                    acc = acc + x[s] * x[t]
                end
            end
        end
        for i in 1:length(x)
            for j in i:length(x)
                acc = acc + x[i] * x[j] * j
            end
        end
        acc
    end
    return result
end
# A start that is a free integer input has no bound.
@kernel jax_ranges_free(x::Vector{Float64}, first::Int) = begin
    result = begin
        acc = 0.0
        for t in first:length(x)
            acc = acc + x[t]
        end
        acc
    end
    return result
end
# A comprehension from a static start past 1 to a padded row's length has
# as many entries as its range (it used to count from 1), and a slice of a
# padded row up to its length is a padded gather (it was refused).
@kernel jax_ranges_padded_tail(R::Vector{Vector{Float64}}) = begin
    cells = plate(R) do r
        s = begin
            v = [r[t] for t in 2:length(r)]
            sum(v) + 10.0 * length(v) + sum(abs2, r[1:length(r)]) + length(r[2:length(r)])
        end
        s
    end
    total = sum(cells)
    return total
end

const JAX_RANGES_KERNELS = (jax_ranges_loop, jax_ranges_forms, jax_ranges_plate, jax_ranges_open,
                            jax_ranges_writes, jax_ranges_nested)
_jax_ranges_want(k) = k === jax_ranges_plate ? :total : :result

@testset "jax_source: ranges whose ends are prepared data" begin
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    starts, stops = [2, 4, 4, 1, 5], [3, 4, 3, 5, 4]
    have = (:x, :starts, :stops)
    for k in JAX_RANGES_KERNELS
        src = jax_source(k; have, want = _jax_ranges_want(k), data = (; starts, stops))
        text = string(src)
        # A static number of steps, so the loop is a scan `jax.grad` reverses.
        @test occursin("_rk_prepared[\"_rk_", text)
        @test !occursin("while_loop", text)
    end
    loop = jax_source(jax_ranges_loop; have, want = :result, data = (; starts, stops))
    @test occursin("jax.lax.fori_loop(0, max(_rk_span_starts_stops + 1, 0), ", string(loop))
    prepare_data = first(loop.functions)
    @test prepare_data.name == "prepare_data"
    @test first.(prepare_data.outputs) == ["starts", "stops", "_rk_span_starts_stops"]
    @test "_rk_span_starts_stops" in first.(last(loop.functions).inputs)
    open_ends = string(jax_source(jax_ranges_open; have, want = :result, data = (; starts, stops)))
    @test occursin("_rk_bound_min(_rk_prepared[\"starts\"])", open_ends)
    @test occursin("_rk_bound_max(_rk_prepared[\"stops\"])", open_ends)

    # Without the data the ends are traced inputs, with no static bound.
    @test_throws "Ends that are entries of integer arrays of the prepared data" jax_source(
        jax_ranges_loop; have, want = :result)
    @test_throws "a loop whose first index depends on data" jax_source(jax_ranges_free;
        have = (:x, :first), want = :result)
    # A prepared start is static.
    @test occursin("jax.lax.fori_loop(first, ", string(jax_source(jax_ranges_free;
        have = (:x, :first), want = :result, data = (; first = 2))))

    if !rk_jax_available()
        @test_skip false
    else
        datasets = [(starts, stops), ([3, 5, 2], [2, 4, 1]), ([1, 5, 3], [1, 5, 3]),
                    (Int[], Int[])]
        for k in JAX_RANGES_KERNELS, (s, e) in datasets
            # A loop over no records traces its body's gathers from empty
            # arrays, which JAX refuses (as before this change).
            isempty(s) && k !== jax_ranges_plate && continue
            want = _jax_ranges_want(k)
            f = prepare(k; have, want)
            src = jax_source(k; have, want, data = (; starts = s, stops = e))
            for prepare_mode in (true, :arguments)
                r = rk_jax_evaluate(src, "kernel", (x, s, e); prepare = prepare_mode, grad = true)
                @test r.value ≈ f(x, s, e) rtol = 1e-12
                @test r.grad ≈ rk_fd_gradient(u -> f(u, s, e), x) rtol = 1e-6 atol = 1e-6
            end
        end
        R = [[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]]
        tail = jax_source(jax_ranges_padded_tail; have = (:R,), want = :total)
        @test rk_jax_evaluate(tail, "kernel", (R,)).value ≈
            prepare(jax_ranges_padded_tail; have = (:R,), want = :total)(R)
        # Eager calls, and loops that are scans (not while loops) under
        # `jax.grad`.
        mktempdir() do dir
            write(joinpath(dir, "loop.py"), jax_source(jax_ranges_loop; have, want = :result,
                                                       data = (; starts, stops)))
            write(joinpath(dir, "check.py"), raw"""
import jax
import numpy as np
import loop
x = np.arange(1.0, 6.0)
prepared = loop.prepare_data(starts=np.array([2, 4, 4, 1, 5]), stops=np.array([3, 4, 3, 5, 4]))
assert int(prepared["_rk_span_starts_stops"]) == 4
assert float(loop.kernel(x, **prepared)) == 25.0 + 16.0 + 0.0 + 225.0 + 0.0
def primitives(jaxpr):
    jaxpr = getattr(jaxpr, 'jaxpr', jaxpr)
    names = []
    for eqn in jaxpr.eqns:
        names.append(eqn.primitive.name)
        for child in eqn.params.values():
            for part in child if isinstance(child, (tuple, list)) else (child,):
                if hasattr(part, 'jaxpr') or hasattr(part, 'eqns'):
                    names += primitives(part)
    return names
names = primitives(jax.make_jaxpr(jax.grad(lambda u: loop.kernel(u, **prepared)))(x))
assert 'while' not in names and names.count('scan') >= 2, names
print("EAGER_SCAN_OK")
""")
            output = read(Cmd(`$RK_JAX_PYTHON $(joinpath(dir, "check.py"))`; dir), String)
            @test occursin("EAGER_SCAN_OK", output)
        end
    end
end
