using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

include("jax_guarded_scan_fixture.jl")

@traceable jax_apply_scan_rows(p, state, offset) = p[1] * state + offset[2]

@kernel function jax_requested_scan_producer(a::Float64, kinds::Vector{Vector{Int}},
                                              visits::Vector{Vector{Int}})
    totals = plate(kinds, visits) do ks, js
        offsets = plate(ks) do k
            (a * k, -2a * k)
        end
        path = scan(ks, js, Ref(offsets); init = 0.0) do carry, kind, j, table
            next = kind == 3 ? carry + table[j][2] : carry
            (next, next)
        end
        sum(path)
    end
    total = sum(totals)
    return total
end

@testset "A prepared scan guard batches the requested producer rows" begin
    have = (:a, :kinds, :visits)
    src = jax_source(jax_requested_scan_producer; have, want=:total, data=(:kinds, :visits))
    @test occursin(r" = _rk_(requested|scan)_cell_rows\(", string(src))
    @test occursin(" = _rk_match_count(", string(src))
    native = prepare(jax_requested_scan_producer; have, want=:total)
    if rk_jax_available()
        # A repeated visit to a different row, inactive invalid indices, an
        # empty subject and a short subject whose final active tag is padded.
        tiles = [[fill(3, n); fill(2, 25 - n)] for n in (0, 7, 8, 9, 16, 17)]
        indices = [reverse(collect(eachindex(row))) for row in tiles]
        for (ks, js) in (([[2, 3, 2, 3], [3], Int[]], [[99, 1, -1, 1], [1], Int[]]),
                         ([[1, 2], Int[]], [[-1, 99], Int[]]),
                         ([[3, 3]], [[2, 1]]), ([Int[]], [Int[]]), (tiles, indices))
            r = rk_jax_evaluate(src, "kernel", (0.75, ks, js); prepare=:arguments, grad=true)
            @test r.value ≈ native(0.75, ks, js)
            @test r.grad ≈ native(1.0, ks, js)
        end
        @test_throws "BoundsError" rk_jax_evaluate(src, "kernel", (0.75, [[2, 3]], [[0, 3]]);
                                                   prepare=:arguments)
        mktempdir() do dir
            for layout in (:groups, :padded)
                model = jax_source(jax_requested_scan_producer; have, want=:total,
                                   data=(:kinds, :visits), plate_layout=layout)
                if layout === :padded
                    @test occursin(" = _rk_requested_cell_rows(", model.text)
                end
                write(joinpath(dir, string(layout) * ".py"), model)
            end
            # This fixture is linear in a. Supply the native slope for each
            # schedule, including a repeated read, inactive bad indices and an empty row.
            open(joinpath(dir, "cases.py"), "w") do io
                println(io, "cases = [")
                for (ks, js) in (([[2, 3, 2, 3], [3], Int[]], [[99, 1, -1, 1], [1], Int[]]),
                                 ([[3, 2, 3, 2], [3], Int[]], [[2, 99, 2, -1], [1], Int[]]),
                                 ([Int[]], [Int[]]), (tiles, indices))
                    println(io, "    (", _rk_py(ks), ", ", _rk_py(js), ", ", native(1.0, ks, js), "),")
                end
                println(io, "]")
            end
            driver = joinpath(@__DIR__, "jax_guarded_reads.py")
            @test success(`$RK_JAX_PYTHON $driver $dir`)
        end
    end
end

@kernel function jax_guarded_scan_scale(scale::Float64, bases::Vector{Vector{Float64}},
                                        kinds::Vector{Vector{Int}}, visits::Vector{Vector{Int}},
                                        weights::Vector{Float64})
    totals = plate(bases, kinds, visits, Ref(weights)) do base, ks, js, ws
        path = scan(ks, js, Ref(base), Ref(ws); init = 0.0) do state, k, j, table, w
            numerator = scale * w[k]
            variance = k == 3 ? table[j] : 1.0
            next = state + numerator / sqrt(variance)
            (next, next)
        end
        sum(path)
    end
    total = sum(totals)
    return total
end

@kernel function jax_scan_two_tables(a::Float64, xs::Vector{Float64}, kinds::Vector{Int},
                                     steps::Vector{Int}, spans::Vector{Int}, visits::Vector{Int})
    propagators = plate(xs) do x
        (a + x, a - x)
    end
    offsets = plate(kinds) do k
        (a * k, 2a * k)
    end
    path = scan(kinds, steps, spans, visits; init = 0.0) do state, kind, step, span, j
        moved = propagators[step][1] + state
        next = kind == 3 ? jax_apply_scan_rows(propagators[span], moved, offsets[j]) : moved
        (next, next)
    end
    total = sum(path)
    return total
end

@traceable function jax_scan_positive_order(x)
    x >= 0 || throw(ArgumentError("checked scan operation"))
    x
end

@traceable function jax_checked_record(x)
    x > 0 || throw(ArgumentError("nonpositive record input"))
    (x, 2x)
end

@kernel function jax_checked_record_scan(a::Float64, bases::Vector{Vector{Float64}},
                                         visits::Vector{Vector{Int}})
    totals = plate(bases, visits) do base, js
        records = plate(base) do x
            jax_checked_record(a * x)
        end
        path = scan(js, Ref(records); init = 0.0) do carry, j, table
            next = carry + table[j][1]
            (next, next)
        end
        sum(path)
    end
    total = sum(totals)
    return total
end

@testset "Scheduled record checks visit real producer entries before the scan" begin
    have = (:a, :bases, :visits)
    src = jax_source(jax_checked_record_scan; have, want=:total, data=(:bases, :visits), plate_layout=:padded)
    @test occursin("schedule=_rk_record_rows_", src.text)
    native = prepare(jax_checked_record_scan; have, want=:total)
    if rk_jax_available()
        bases, visits = [[1., 2.], Float64[], [3.]], [[2, 1], Int[], [1]]
        result = rk_jax_evaluate(src, "kernel", (0.75, bases, visits); prepare=:arguments, grad=true)
        @test result.value ≈ native(0.75, bases, visits)
        @test result.grad ≈ native(1.0, bases, visits)
        # The second producer entry is invalid even if the scan never reads it.
        bad, later = [[1., 0.], Float64[], [3.]], [[99, 1], Int[], [1]]
        @test_throws "nonpositive record input" native(0.75, bad, later)
        @test_throws "nonpositive record input" rk_jax_evaluate(src, "kernel", (0.75, bad, later); prepare=:arguments)
    end
end

@kernel function jax_scan_row_order(a::Float64, kinds::Vector{Int}, visits::Vector{Int},
                                    table::Vector{Float64})
    path = scan(kinds, visits, Ref(table); init = a) do state, kind, j, rows
        checked = jax_scan_positive_order(kind)
        moved = state + checked
        next = kind == 3 ? moved + rows[j] : moved
        (next, next)
    end
    total = sum(path)
    return total
end

@testset "A one-arm scan read takes the requested rows as step inputs" begin
    have = (:a, :bases, :types, :orders)
    src = jax_source(jax_guarded_scan_rows; have, want = :total, data = (:bases, :types, :orders))
    padded = jax_source(jax_guarded_scan_rows; have, want = :total, data = (:bases, :types, :orders),
                        plate_layout = :padded)
    packed = jax_source(jax_guarded_scan_rows; have, want = :total, data = (:bases, :types, :orders),
                        plate_layout = :chains)
    native = prepare(jax_guarded_scan_rows; have, want = :total)
    grouped = jax_source(jax_guarded_scan_rows; have, want = :total, data = (:bases, :types, :orders),
                         plate_layout = :groups)
    @test occursin(" = _rk_scan_record_rows(", string(grouped))
    named = jax_source(jax_guarded_scan_rows; have, data = (:bases, :types, :orders),
        functions = (prepare_plate_layout = :total, select_plate_layout = :total, bind_gradient = :total),
        constants = (SUBJECT_SCAN_SCHEDULES = 17,), plate_layout = :padded)
    @test named.functions[2].inputs == padded.functions[2].inputs
    @test [f.name for f in named.functions] == ["prepare_data", "prepare_plate_layout", "select_plate_layout",
                                                "bind_gradient", "prepare_plate_layout_2", "select_plate_layout_2", "bind_gradient_2"]
    @test occursin("SUBJECT_SCAN_SCHEDULES = 17", named.text)
    @test occursin("for name, inputs in SUBJECT_SCAN_SCHEDULES_2.items()", named.text)
    @test occursin(" = _rk_subject_scan(", string(padded))
    @test occursin("_rk_record_plate(offsets_cell,", string(padded))
    @test occursin(" = _rk_event_table_rows(_rk_scan_record_rows, ", string(packed))
    @test occursin(" = _rk_scan_events(", string(packed))
    checked_have = (:a, :kinds, :visits, :table)
    checked = jax_source(jax_scan_row_order; have = checked_have, want = :total)
    @test !occursin("_rk_scan_rows(", string(checked))
    captured_have = (:a, :xs, :kinds, :steps, :spans, :visits)
    captured = jax_source(jax_scan_two_tables; have = captured_have, want = :total)
    @test count(" = _rk_scan_record_rows(", string(captured)) == 2
    ref = prepare(jax_scan_two_tables; have = captured_have, want = :total)
    if rk_jax_available()
        scale_have = (:scale, :bases, :kinds, :visits, :weights)
        scaled = jax_source(jax_guarded_scan_scale; have = scale_have, want = :total,
                            data = (:bases, :kinds, :visits, :weights))
        scale_native = prepare(jax_guarded_scan_scale; have = scale_have, want = :total)
        @test count(" = _rk_scan_rows(", string(scaled)) == 2
        # Both the taken and untaken last event get repeated as physical
        # padding. Only real events contribute even when a missing row
        # would make arithmetic downstream of the conditional singular.
        for args in (([[4.0], [9.0], Float64[]], [[3], [2, 3], Int[]], [[1], [999, 1], Int[]]),
                     ([[4.0], [9.0]], [[2], [2, 3]], [[999], [-1, 1]]),
                     ([Float64[], Float64[]], [[2], [2, 2]], [[-1], [0, 9]]))
            weights = [0.25, 0.5, 0.75]
            result = rk_jax_evaluate(scaled, "kernel", (2.0, args..., weights); prepare = :arguments, grad = :joint)
            @test result.joint_value ≈ scale_native(2.0, args..., weights)
            @test result.grad ≈ scale_native(1.0, args..., weights)
        end
        for data in (([[0.2, -1.0, 3.1], [4.0], Float64[]],
                      [[3, 2, 3, 3], [2, 3], [2]], [[3, -9, 1, 3], [999, 1], [0]]),
                     ([Float64[]], [[2, 2]], [[0, 7]]),
                     ([Float64[]], [Int[]], [Int[]]),
                     (Vector{Float64}[], Vector{Int}[], Vector{Int}[]))
            for a in (0.65, -0.7), layout in (src, packed)
                r = rk_jax_evaluate(layout, "kernel", (a, data...); prepare = :arguments, grad = true)
                @test r.value ≈ native(a, data...)
                # The recurrence and readout are affine in a.
                @test r.grad ≈ native(1.0, data...) - native(0.0, data...)
            end
        end
        for data in (([Float64[]], [[3]], [[1]]), ([[1.0], [1.0, 2.0]], [[3], [2]], [[2], [1]]))
            @test_throws BoundsError native(0.5, data...)
            @test_throws "BoundsError" rk_jax_evaluate(src, "kernel", (0.5, data...); prepare = :arguments)
            @test_throws "BoundsError" rk_jax_evaluate(packed, "kernel", (0.5, data...); prepare = :arguments)
        end
        @test_throws "checked scan operation" rk_jax_evaluate(checked, "kernel", (0.5, [-1, 3], [1, 2], [1.0]))
        args = ([0.5, -0.3], [2, 3, 2], [1, 2, 1], [99, 1, 0], [99, 2, -1])
        for a in (-0.3, 0.7)
            result = rk_jax_evaluate(captured, "kernel", (a, args...); grad = true)
            @test result.value ≈ ref(a, args...)
            # This schedule has one multiplicative update, so the exact
            # derivative of its quadratic is its centered unit difference.
            @test result.grad ≈ (ref(a + 1, args...) - ref(a - 1, args...)) / 2
        end
        mktempdir() do dir
            write(joinpath(dir, "model.py"), padded)
            script = joinpath(@__DIR__, "jax_scan_row_inputs.py")
            @test success(`$RK_JAX_PYTHON $script $dir`)
        end
    end
end

@kernel function jax_subject_two_tables(a::Float64, xs::Vector{Vector{Float64}}, kinds::Vector{Vector{Int}},
                                        steps::Vector{Vector{Int}}, spans::Vector{Vector{Int}},
                                        visits::Vector{Vector{Int}})
    totals = plate(xs, kinds, steps, spans, visits) do x, ks, st, sp, js
        propagators = plate(x) do v
            (a + v, a - v)
        end
        offsets = plate(ks) do k
            (a * k, 2a * k)
        end
        path = scan(ks, st, sp, js, Ref(propagators), Ref(offsets); init = 0.0) do state, kind, step, span, j, P, O
            moved = P[step][1] + state
            next = kind == 3 ? P[span][2] * moved + O[j][2] : moved
            (next, next)
        end
        sum(path)
    end
    total = sum(totals)
    return total
end

@testset "prepare_data proves a subject scan's table reads" begin
    have = (:a, :xs, :kinds, :steps, :spans, :visits)
    data = (:xs, :kinds, :steps, :spans, :visits)
    padded = jax_source(jax_subject_two_tables; have, want = :total, data, plate_layout = :padded)
    # One verdict per gathered read: every step's propagator, a run's span and its offset.
    @test count("= _rk_scan_index_bound(", padded.text) == 3
    @test occursin(r"if not \(\(_rk_scan_index_bound_\w+ and x\.shape\[0\] <= jax\.tree\.leaves\(propagators\)", padded.text)
    @test !occursin("_rk_scan_index_bound", jax_source(jax_subject_two_tables; have, want = :total, data,
                                                        plate_layout = :chains).text)
    native = prepare(jax_subject_two_tables; have, want = :total)
    if rk_jax_available()
        # Invalid spans and offsets only at steps that are not runs, and an empty subject.
        good = ([[1., 2.], Float64[], [3.]], [[3, 2, 3], Int[], [2]], [[1, 2, 2], Int[], [1]],
                [[2, 99, 1], Int[], [0]], [[1, 99, 3], Int[], [-5]])
        mktempdir() do dir
            write(joinpath(dir, "model.py"), padded)
            write(joinpath(dir, "check.py"), """
import sys
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, sys.argv[1])
import model

def inventory(jaxpr, found=None):
    found = {} if found is None else found
    for eq in getattr(jaxpr, "jaxpr", jaxpr).eqns:
        found[eq.primitive.name] = found.get(eq.primitive.name, 0) + 1
        for p in eq.params.values():
            for c in p if isinstance(p, (tuple, list)) else (p,):
                if hasattr(c, "jaxpr") or hasattr(c, "eqns"):
                    inventory(c, found)
    return found.get("debug_callback", 0)

good = $(_rk_py(collect(good)))
expected = $(_rk_py([native(a, good...) for a in (0.4, -1.3)]))
proofs = sorted(k for k in model.PREPARED if k.startswith("_rk_scan_index_bound"))
assert len(proofs) == 3, list(model.PREPARED)
vg = jax.value_and_grad(lambda a, d: model.kernel(a, **d))
data = model.prepare_data(*good)
assert [data[p] for p in proofs] == [1, 1, 1], [data[p] for p in proofs]
# On a GPU the check would copy its predicate to the host mid-program.
assert inventory(jax.make_jaxpr(vg)(0.4, data)) == 0
for a, value in zip((0.4, -1.3), expected):
    np.testing.assert_allclose(jax.jit(vg)(a, data)[0], value)
# A verdict of 0 restores the check and changes no value or gradient.
off = {**data, **{p: model.StaticInt(0) for p in proofs}}
assert inventory(jax.make_jaxpr(vg)(0.4, off)) > 0
np.testing.assert_allclose(jax.jit(vg)(0.4, off), jax.jit(vg)(0.4, data))
# A run's span past its subject's rows, and a zero step read at every live
# step: prepare_data does not raise, the check stays and the function does.
for which, row in ((3, [2, 99, 3]), (2, [1, 0, 2])):
    bad = list(good)
    bad[which] = [row] + good[which][1:]
    data = model.prepare_data(*bad)
    assert [data[p] for p in proofs].count(0) == 1, [data[p] for p in proofs]
    try:
        jax.block_until_ready(jax.jit(vg)(0.4, data))
        jax.effects_barrier()
    except Exception as err:
        assert "BoundsError" in str(err), str(err)
    else:
        raise AssertionError("accepted an out-of-range table index")

# The verdict: requested steps only, live lengths, the table's own rows.
b = model._rk_scan_index_bound
idx, live = np.array([[1, 9, 2], [3, 0, 0]]), np.array([3, 1])
tab, rows = np.zeros((2, 3)), np.array([2, 3])
tags = np.array([[3, 2, 3], [3, 1, 1]])
assert b(idx, live, live, tab, rows, tags, live, 3) == 1
assert b(idx, live, live, tab, rows, None, None, None) == 0  # 9 is live without the guard
assert b(idx, live, live, tab, np.array([1, 3]), tags, live, 3) == 0  # 2 past a row of 1
assert b(idx, np.array([3, 2]), live, tab, rows, tags, live, 3) == 0  # index row longer than the scan
assert b(idx, live, live, tab, rows, tags, np.array([3, 2]), 3) == 0  # tag row longer than the scan
assert b(idx, live, live, tab, np.array([2, 4]), tags, live, 3) == 0  # table lengths exceed storage
assert b(np.zeros((2, 0), int), np.array([0, 0]), np.array([0, 0]), np.zeros((2, 0)),
         np.array([0, 0]), None, None, None) == 1
""")
            @test success(pipeline(`$RK_JAX_PYTHON $(joinpath(dir, "check.py")) $dir`, stderr=stderr))
        end
    end
end
