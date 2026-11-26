using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

@kernel function jax_guarded_scan_rows(a::Float64, bases::Vector{Vector{Float64}},
                                      types::Vector{Vector{Int}}, orders::Vector{Vector{Int}})
    subjects = plate(bases, types, orders) do base, kinds, visits
        offsets = plate(base) do x
            (a * x, 2a * x)
        end
        path = scan(kinds, visits, Ref(offsets); init = 0.5) do carry, kind, j, table
            moved = 0.75 * carry + a
            next = kind == 3 ? moved + table[j][1] : moved
            (next, next)
        end
        sum(path)
    end
    total = sum(subjects)
    return total
end

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
    @test occursin(" = _rk_scan_cell_rows(", string(src))
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
    @test occursin(" = _rk_scan_record_rows(", string(src))
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
