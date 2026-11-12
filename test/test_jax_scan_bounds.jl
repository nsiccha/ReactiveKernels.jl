using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

@traceable function jax_scan_bound_repeat(a, n)
    state = (a, a / 3)
    for i in 2:n
        state = (0.5 * state[1] + a, state[2] + a / 7)
    end
    state[1] + state[2]
end
@traceable function jax_scan_bound_indexed(a, counts, j)
    state = (a, a / 3)
    for i in 2:counts[j]
        state = (0.5 * state[1] + a, state[2] + a / 7)
    end
    state[1] + state[2]
end
@traceable function jax_scan_bound_peak(counts)
    peak = 1
    for j in eachindex(counts)
        peak = max(peak, counts[j])
    end
    peak
end

@kernel jax_scan_bound_flat(a::Float64, counts::Vector{Int}) = begin
    path = scan(counts; init = 0.0) do acc, n
        next = acc + jax_scan_bound_repeat(a, n)
        (next, next)
    end
    total = sum(path)
    total
end
@kernel jax_scan_bound_rows(a::Float64, counts::Vector{Vector{Int}}) = begin
    cells = plate(counts) do ns
        path = scan(ns; init = 0.0) do acc, n
            next = acc + jax_scan_bound_repeat(a, n)
            (next, next)
        end
        sum(path)
    end
    total = sum(cells)
    total
end
@kernel jax_scan_bound_history(a::Float64, counts::Vector{Int}) = begin
    path = scan(counts; init = 0.0, history = 0.0) do acc, n, earlier
        next = acc + jax_scan_bound_repeat(a, n)
        (next, next)
    end
    total = sum(path)
    total
end
@kernel jax_scan_bound_indices(a::Float64, counts::Vector{Vector{Int}}) = begin
    cells = plate(counts) do ns
        path = scan(eachindex(ns), Ref(ns); init = 0.0) do acc, j, ns
            next = if ns[j] > 1
                acc + jax_scan_bound_indexed(a, ns, j)
            else
                acc
            end
            (next, next)
        end
        sum(path)
    end
    total = sum(cells)
    total
end
@kernel jax_scan_bound_maximum(a::Float64, counts::Vector{Vector{Int}}) = begin
    cells = plate(counts) do ns
        peak = jax_scan_bound_peak(ns)
        path = scan(ns, Ref(peak); init = 0.0) do acc, n, peak
            next = acc + jax_scan_bound_repeat(a, peak)
            (next, next)
        end
        sum(path)
    end
    total = sum(cells)
    total
end
@kernel jax_scan_bound_pairs(a::Float64, starts::Vector{Vector{Int}},
                            stops::Vector{Vector{Int}}) = begin
    cells = plate(starts, stops) do firsts, lasts
        path = scan(firsts, lasts; init = 0.0) do acc, lo, hi
            next = begin
                out = acc
                for j in lo:hi
                    out = out + a * j
                end
                out
            end
            (next, next)
        end
        sum(path)
    end
    total = sum(cells)
    total
end
@kernel jax_scan_bound_crossrows(a::Float64, starts::Vector{Vector{Int}},
                                stops::Vector{Vector{Int}}) = begin
    path = scan(eachindex(starts[1]), Ref(starts), Ref(stops); init = 0.0) do acc, j, starts, stops
        lo = starts[1][j]
        hi = stops[2][j]
        next = begin
            out = acc
            for j in lo:hi
                out = out + a * j
            end
            out
        end
        (next, next)
    end
    total = sum(path)
    total
end

@testset "jax_source: prepared bounds through scans and ragged cells" begin
    have = (:a, :counts)
    rows = [[2, 4], [1], Int[]]
    kernels = (jax_scan_bound_rows, jax_scan_bound_indices, jax_scan_bound_maximum)
    for k in (jax_scan_bound_flat, kernels...)
        data = k === jax_scan_bound_flat ? (; counts = rows[1]) : (; counts = rows)
        src = jax_source(k; have, want = :total, data)
        @test occursin("jax.lax.scan", string(src))
        @test occursin("jax.lax.fori_loop", string(src))
        @test occursin("_rk_bound_max", string(src))
        @test !occursin("jax.lax.while_loop", string(src))
    end
    pairs = jax_source(jax_scan_bound_pairs; have = (:a, :starts, :stops), want = :total,
                       data = (; starts = [[2, 8], [4]], stops = [[3, 7], [6]]))
    @test occursin("_rk_bound_span", string(pairs))

    if !rk_jax_available()
        @test_skip false
    else
        @testset "ragged kernel $kind" for (kind, k) in enumerate(kernels)
            f = prepare(k; have, want = :total)
            src = jax_source(k; have, want = :total, data = (; counts = rows))
            for counts in (rows, [[1, 0], Int[]], [Int[], Int[]], Vector{Int}[],
                           [fill(13, 17), [3, 5]])
                # Neither data values nor the number of rows/steps replicates source.
                @test string(jax_source(k; have, want = :total, data = (; counts))) == string(src)
                for mode in (true, :arguments)
                    r = rk_jax_evaluate(src, "kernel", (0.7, counts); prepare = mode, grad = true)
                    @test r.value ≈ f(0.7, counts) atol = 1e-12
                    # This recurrence is linear in a, so f(1) is its exact derivative.
                    @test r.grad ≈ f(1.0, counts) atol = 1e-12
                end
            end
        end
        for k in (jax_scan_bound_flat, jax_scan_bound_history)
            f = prepare(k; have, want = :total)
            src = jax_source(k; have, want = :total, data = (; counts = rows[1]))
            for counts in ([2, 4, 1], Int[], [1], fill(19, 23))
                r = rk_jax_evaluate(src, "kernel", (0.7, counts); prepare = :arguments, grad = true)
                @test r.value ≈ f(0.7, counts) atol = 1e-12
                @test r.grad ≈ f(1.0, counts) atol = 1e-12
            end
        end
        f = prepare(jax_scan_bound_pairs; have = (:a, :starts, :stops), want = :total)
        starts, stops = [[2, 8], [4], Int[]], [[3, 7], [6], Int[]]
        r = rk_jax_evaluate(pairs, "kernel", (0.7, starts, stops); prepare = :arguments, grad = true)
        @test r.value ≈ f(0.7, starts, stops)
        @test r.grad ≈ f(1.0, starts, stops)
        # Same column, different rows: paired spans would bound 1:102 by
        # just three steps. Preserve both row selections in the provenance.
        starts, stops = [[1], [100]], [[2], [102]]
        crossed = jax_source(jax_scan_bound_crossrows;
            have = (:a, :starts, :stops), want = :total, data = (; starts, stops))
        @test !occursin("_rk_bound_span", string(crossed))
        r = rk_jax_evaluate(crossed, "kernel", (0.7, starts, stops);
                            prepare = :arguments, grad = true)
        @test r.value ≈ 0.7 * sum(1:102)
        @test r.grad ≈ sum(1:102)
        mktempdir() do dir
            write(joinpath(dir, "model.py"), jax_source(jax_scan_bound_rows;
                have, want = :total, data = (; counts = rows)))
            write(joinpath(dir, "check.py"), raw"""
import jax
import numpy as np
import model
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
inventories = []
for n in (2, 17, 53):
    data = model.prepare_data(counts=[np.full(n, n, dtype=np.int64), np.array([3])])
    f = lambda a: model.kernel(a, **data)
    names = primitives(jax.make_jaxpr(jax.grad(f))(0.7))
    assert 'while' not in names and names.count('scan') >= 4, names
    inventories.append((len(names), names.count('scan')))
assert len({scans for _, scans in inventories}) == 1, inventories
assert max(size for size, _ in inventories) <= 2 * min(size for size, _ in inventories), inventories
print('RETAINED_SCAN_BODIES', inventories)
""")
            output = read(Cmd(`$RK_JAX_PYTHON $(joinpath(dir, "check.py"))`; dir), String)
            @test occursin("RETAINED_SCAN_BODIES", output)
            println(output)
        end
    end
end
