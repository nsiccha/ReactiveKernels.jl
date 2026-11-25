using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

@kernel jax_scan_plate_reads(a::Float64, labels::Vector{Vector{Int}},
                             visits::Vector{Vector{Int}}) = begin
    totals = plate(labels, visits) do keys, order
        weights = [a, -2a]
        entries = plate(keys, Ref(weights)) do key, w
            (w[key] + a, w[key] - a)
        end
        path = scan(keys, order, Ref(entries); init = 0.0) do state, key, index, table
            next = if key == 1
                state + table[index][1]
            else
                state - table[index][2]
            end
            (next, next)
        end
        sum(path)
    end
    total = sum(totals)
    total
end

@traceable function jax_checked_entry(a, x)
    x > 0 || throw(ArgumentError("table entry must be positive"))
    a * x
end

@kernel jax_scan_checked_table(a::Float64, xs::Vector{Float64}, visits::Vector{Int}) = begin
    entries = plate(xs) do x
        jax_checked_entry(a, x)
    end
    path = scan(xs, visits, Ref(entries); init = 0.0) do acc, x, j, table
        next = acc + table[j]
        (next, next)
    end
    total = sum(path)
    total
end

@kernel jax_scan_partitioned_table(a::Float64, keys::Vector{Int}, visits::Vector{Int}) = begin
    entries = plate(keys) do k
        k == 1 ? a : -2a
    end
    path = scan(keys, visits, Ref(entries); init = 0.0) do acc, k, j, table
        next = acc + table[j]
        (next, next)
    end
    total = sum(path)
    total
end

@testset "JAX scan reads one numeric plate cell at the requested index" begin
    have = (:a, :labels, :visits)
    native = prepare(jax_scan_plate_reads; have, want = :total)
    emit(labels, visits) = jax_source(jax_scan_plate_reads; have, want = :total, data = (; labels, visits))
    labels = [[1, 2, 1, 2], [2], Int[]]
    visits = [[2, 4, 2, 1], [1], Int[]]
    src = emit(labels, visits)
    @test occursin(r"plate_entry\w* = .*entries_cell\(", string(src))
    if rk_jax_available()
        for (ks, js) in ((labels, visits), (reverse(labels), reverse(visits)),
                         ([fill(2, 17)], [collect(17:-1:1)]),
                         ([Int[], Int[]], [Int[], Int[]]), (Vector{Int}[], Vector{Int}[]))
            @test string(emit(ks, js)) == string(src)
            r = rk_jax_evaluate(src, "kernel", (0.7, ks, js); prepare = :arguments, grad = true)
            @test r.value ≈ native(0.7, ks, js)
            # Each recurrence is linear in a, so the native unit response is its derivative.
            @test r.grad ≈ native(1.0, ks, js)
        end
        mktempdir() do dir
            write(joinpath(dir, "model.py"), src)
            write(joinpath(dir, "trace.py"), raw"""
import jax
import numpy as np
import model

def equations(jaxpr):
    for eq in getattr(jaxpr, "jaxpr", jaxpr).eqns:
        yield eq
        for param in eq.params.values():
            for child in param if isinstance(param, (list, tuple)) else (param,):
                if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
                    yield from equations(child)

sizes = []
for n in (7, 31, 127):
    labels = [np.resize([1, 2], n - i) for i in range(3)]
    visits = [np.arange(len(row), 0, -1) for row in labels]
    data = model.prepare_data(labels=labels, visits=visits)
    read = jax.vmap(lambda a: model.kernel(a, **data))
    eqns = list(equations(jax.make_jaxpr(read)(np.ones(5))))
    scans = [e for e in eqns if e.primitive.name == "scan"]
    assert len(scans) == 1
    # A table column carried into this scan would span operations, draws and rows.
    for v in scans[0].invars:
        a = v.aval
        assert not (np.issubdtype(a.dtype, np.floating) and set((3, 5, n)) <= set(a.shape)), a
    sizes.append(len(eqns))
assert max(sizes) <= 2 * min(sizes), sizes
""")
            @test success(Cmd(`$RK_JAX_PYTHON $(joinpath(dir, "trace.py"))`; dir))
        end
    else
        @test_skip false
    end
end

@testset "A scan leaves checks on unvisited plate cells in force" begin
    have = (:a, :xs, :visits)
    xs, visits = [1.0, -2.0], [1, 1]
    src = jax_source(jax_scan_checked_table; have, want = :total)
    @test !occursin("plate_entry =", string(src))
    native = prepare(jax_scan_checked_table; have, want = :total)
    @test_throws ArgumentError native(0.7, xs, visits)
    if rk_jax_available()
        @test_throws "table entry must be positive" rk_jax_evaluate(src, "kernel", (0.7, xs, visits))
        r = rk_jax_evaluate(src, "kernel", (0.7, abs.(xs), visits); grad = true)
        @test r.value ≈ native(0.7, abs.(xs), visits)
        @test r.grad ≈ native(1.0, abs.(xs), visits)
    end
end

@testset "A scan indexes partitioned plate results in their original order" begin
    have = (:a, :keys, :visits)
    src = jax_source(jax_scan_partitioned_table; have, want = :total, data = (:keys, :visits))
    @test occursin("entries_lane_order", string(src))
    if rk_jax_available()
        keys, visits = [2, 1, 2, 1], [4, 2, 4, 1]
        native = prepare(jax_scan_partitioned_table; have, want = :total)
        r = rk_jax_evaluate(src, "kernel", (0.7, keys, visits); prepare = :arguments, grad = true)
        @test r.value ≈ native(0.7, keys, visits)
        @test r.grad ≈ native(1.0, keys, visits)
    end
end
