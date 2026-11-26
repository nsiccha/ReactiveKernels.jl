using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

@kernel function jax_row_trajectories(a::Float64, labels::Vector{Vector{Int}}, visits::Vector{Vector{Int}})
    totals = plate(labels, visits, Ref(a)) do keys, counts, scale
        history = scan(keys, counts, Ref(scale); init = 0.0) do state, k, c, x
            next = k > 0 ? state + x * c : state - x * c
            (next, next)
        end
        sum(history)
    end
    total = sum(totals)
    return total
end

@kernel function jax_terminal_rows(a::Float64, rows::Vector{Vector{Float64}})
    cells = plate(rows, Ref(a)) do row, x
        history = scan(row, Ref(x); init = x, include_init = true) do state, y, scale
            next = state * scale + y
            (next, next)
        end
        last(history)
    end
    total = sum(cells)
    return total
end

@testset "A terminal scan retains the initial state of empty rows" begin
    have = (:a, :rows)
    native = prepare(jax_terminal_rows; have, want = :total)
    src = jax_source(jax_terminal_rows; have, want = :total, data = (:rows,), layout = :chains)
    @test occursin("= _rk_scan_rows(", string(src))
    if rk_jax_available()
        for rows in ([Float64[], [1.5, 4.0], [-2.0]], [Float64[], Float64[]],
                     Vector{Float64}[], [[0.4, 1.0], [-0.2, 0.7]])
            a = 0.45
            derivative = sum(rows; init = 0.0) do row
                value, gradient = a, 1.0
                for y in row
                    value, gradient = value * a + y, gradient * a + value
                end
                gradient
            end
            r = rk_jax_evaluate(src, "kernel", (a, rows); prepare = :arguments, grad = true)
            @test r.value ≈ native(a, rows)
            @test r.grad ≈ derivative
        end
    end
end

@traceable function jax_row_prefix(codes)
    indices = zeros(Int, length(codes))
    index = 0
    for i in eachindex(codes)
        index += codes[i] > 0
        indices[i] = index
    end
    indices
end

@kernel function jax_row_index_sum(a::Float64, codes::Vector{Vector{Int}})
    terms = plate(codes) do row
        indices = jax_row_prefix(row)
        a * sum(indices)
    end
    total = sum(terms)
    return total
end

@kernel function jax_row_read_history(a::Float64, events::Vector{Vector{Int}},
                                      visits::Vector{Vector{Int}})
    cells = plate(events, visits, Ref(a)) do kinds, reads, x
        history = scan(kinds, Ref(x); init = 0.0) do state, k, scale
            next = k > 0 ? state + scale * k : state - scale
            (next, (next, -2next))
        end
        values = plate(reads, Ref(history)) do index, path
            path[index][1] - path[index][2]
        end
        sum(values)
    end
    total = sum(cells)
    return total
end

@kernel function jax_scaled_rows(a::Float64, rows::Vector{Vector{Float64}})
    scaled = plate(rows, Ref(a)) do row, x
        x .* row
    end
    joined = reduce(vcat, scaled)
    total = sum(joined)
    return total
end

@kernel function jax_subject_reads(a::Float64, events::Vector{Vector{Int}},
                                   visits::Vector{Vector{Int}})
    subjects = plate(events, visits, Ref(a)) do row, reads, scale
        history = scan(row, Ref(scale); init = 0.0) do s, k, x
            n = s + k*x
            (n, n)
        end
        values = plate(reads, Ref(history)) do j, h
            h[j]
        end
        (values, scale .* row)
    end
    observed = plate(subjects) do subject
        subject[1]
    end
    joined = reduce(vcat, observed)
    total = sum(joined)
    return total
end

@testset "A subject plate publishes record fields on separate event and observation schedules" begin
    have = (:a, :events, :visits)
    src = jax_source(jax_subject_reads; have, want = :total, data = (:events, :visits), layout = :chains)
    native = prepare(jax_subject_reads; have, want = :total)
    if rk_jax_available()
        for (events, visits) in (([[2,-1,3], Int[], [7]], [[3,1,3,2], Int[], [1]]),
                                 ([[9], [8,3]], [Int[], [2,1]]), ([[2,3], [7,9]], [[2], [1]]))
            r = rk_jax_evaluate(src, "kernel", (0.4, events, visits); prepare = :arguments, grad = true)
            @test r.value ≈ native(0.4, events, visits)
            @test r.grad ≈ native(1.0, events, visits)
        end
    end
end

@testset "Variable-length plate outputs retain their packed order" begin
    have = (:a, :rows)
    src = jax_source(jax_scaled_rows; have, want = :total, data = (:rows,), layout = :chains)
    native = prepare(jax_scaled_rows; have, want = :total)
    @test occursin("else _rk_collect_rows(", string(src))
    if rk_jax_available()
        for rows in ([Float64[], [1.5, -2, 4], [3.0]], [[1.0], [2.0]], [Float64[], Float64[]])
            r = rk_jax_evaluate(src, "kernel", (2.0, rows); prepare = :arguments, grad = true)
            @test r.value == native(2.0, rows)
            @test r.grad == native(1.0, rows)
        end
        exposed = jax_source(jax_scaled_rows; have, want = :joined, data = (:rows,), layout = :chains)
        rows = [[4.0], Float64[], [2.0, 7.0]]
        r = rk_jax_evaluate(exposed, "kernel", (3.0, rows); prepare = :arguments)
        @test r.value == [12.0, 6.0, 21.0]
    end
end

@testset "A second row table reads selected steps of the first row's history" begin
    have = (:a, :events, :visits)
    src = jax_source(jax_row_read_history; have, want = :total, data = (:events, :visits), layout = :chains)
    native = prepare(jax_row_read_history; have, want = :total)
    if rk_jax_available()
        for (events, visits) in (([[2, -1, 3], Int[], [7]], [[3, 1, 3, 2], Int[], [1]]),
                                 ([Int[], Int[]], [Int[], Int[]]), ([[1,3], [2,-1]], [[2], [1]]))
            r = rk_jax_evaluate(src, "kernel", (0.3, events, visits); prepare = :arguments, grad = true)
            @test r.value ≈ native(0.3, events, visits)
            @test r.grad ≈ native(1.0, events, visits)
        end
    end
end

@testset "A chain scans only real events and starts each row with its own carry" begin
    have = (:a, :labels, :visits)
    labels = [Int[], [1, -1, 1], Int[], [-1], Int[]]
    visits = [Int[], [2, 3, 5], Int[], [7], Int[]]
    emit(rows, counts) = jax_source(jax_row_trajectories; have, want = :total, layout = :chains,
                                    data = (; labels = rows, visits = counts))
    source = emit(labels, visits)
    native = prepare(jax_row_trajectories; have, want = :total)
    @test occursin("= _rk_scan_rows(", string(source))
    @test string(emit([repeat([1,-1], 35)], [ones(Int, 70)])) == string(source)
    @test_throws ArgumentError jax_source(jax_row_trajectories; have, want = :total, layout = :unknown)
    if rk_jax_available()
        for (ks, cs) in ((labels, visits), (reverse(labels), reverse(visits)),
                         ([[1, -1], [-1, 1]], [[3, 4], [5, 6]]),
                         ([Int[], Int[]], [Int[], Int[]]), (Vector{Int}[], Vector{Int}[]))
            r = rk_jax_evaluate(source, "kernel", (0.7, ks, cs); prepare = :arguments, grad = true)
            @test r.value ≈ native(0.7, ks, cs)
            @test r.grad ≈ native(1.0, ks, cs) # The authored recurrence is linear in scale.
        end
        @test_throws "DimensionMismatch" rk_jax_evaluate(source, "kernel", (0.7, labels, reverse(visits));
                                                         prepare = :arguments)
        mktempdir() do dir
            write(joinpath(dir, "model.py"), source)
            write(joinpath(dir, "verify.py"), raw"""
import jax
import numpy as np
import model

def scan_lengths(jaxpr):
    for eq in getattr(jaxpr, "jaxpr", jaxpr).eqns:
        if eq.primitive.name == "scan":
            yield eq.params["length"]
        for v in eq.params.values():
            for child in v if isinstance(v, (tuple, list)) else (v,):
                if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
                    yield from scan_lengths(child)

for n in (5, 23, 97):
    ks = [[], [1, -1], [1] * n, []]
    data = model.prepare_data(labels=ks, visits=ks)
    f = jax.jit(jax.vmap(jax.value_and_grad(lambda x: model.kernel(x, **data))))
    traces = list(scan_lengths(jax.make_jaxpr(f)(np.ones(7))))
    assert traces == [n + 2, n + 2], traces
""")
            @test success(Cmd(`$RK_JAX_PYTHON $(joinpath(dir, "verify.py"))`; dir))
        end
    else
        @test_skip false
    end
end

@testset "Fixed cell indices are prepared once in compact storage" begin
    rows = [[1, -1, 3], Int[], [4]]
    have = (:a, :codes)
    src = jax_source(jax_row_index_sum; have, want = :total, data = (; codes = rows), layout = :chains)
    hot = split(string(src), "def kernel(")[end]
    @test !occursin("jax_row_prefix", hot)
    @test !occursin("prepared[\"codes\"]", hot)
    if rk_jax_available()
        native = prepare(jax_row_index_sum; have, want = :total)
        for codes in (rows, [[-3, 7], [13, -2]], [Int[], Int[]])
            r = rk_jax_evaluate(src, "kernel", (0.5, codes); prepare = :arguments, grad = true)
            @test r.value == native(0.5, codes)
            @test r.grad == native(1.0, codes)
        end
    end
end
