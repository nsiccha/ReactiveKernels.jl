using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

@kernel function jax_field_readout(a::Float64, steps::Vector{Vector{Float64}},
                                   slots::Vector{Vector{Int}}, tags::Vector{Vector{Int}})
    totals = plate(steps, slots, tags) do xs, js, qs
        walk = scan(xs; init = 0.0) do state, x
            next = state + a * x
            (next, (next, 2.0 * next + x))
        end
        reads = plate(js, qs, Ref(walk)) do j, q, w
            q == 1 ? w[j][1] : w[j][2]
        end
        sum(reads)
    end
    total = sum(totals)
    return total
end

@kernel function jax_guarded_field(a::Float64, xs::Vector{Float64}, enabled::Bool,
                                   i::Int, j::Int, tag::Int)
    rows = plate(xs) do x
        (a * x, a + x)
    end
    total = if enabled
        tag == 1 ? rows[i][1] : rows[j][2]
    else
        -a
    end
    return total
end

@kernel function jax_short_circuit_fields(rows::Vector{Tuple{Float64,Float64}}, i::Int,
                                          tag::Int, enabled::Bool)
    selected = if tag == 1
        enabled && rows[i][1] > 0
    else
        enabled && rows[i][2] > 0
    end
    return selected
end

@testset "A batched field choice gathers the common record before selecting" begin
    have = (:a, :steps, :slots, :tags)
    native = prepare(jax_field_readout; have, want = :total)
    src = jax_source(jax_field_readout; have, want = :total, data = (:steps, :slots, :tags))
    ghave = (:a, :xs, :enabled, :i, :j, :tag)
    guarded = jax_source(jax_guarded_field; have = ghave, want = :total)
    reference = prepare(jax_guarded_field; have = ghave, want = :total)
    short = jax_source(jax_short_circuit_fields; have = (:rows, :i, :tag, :enabled), want = :selected)
    # Distinct indices and short-circuit reads stay under their guards.
    for source in (guarded, short)
        @test occursin(r"record_entry\w* = ", string(source))
        @test !occursin(r"(?m)^    record_entry\w* = ", string(source))
    end
    if rk_jax_available()
        for (xs, js, qs) in (([[1.0, 0.5, -0.2], [-1.5], Float64[]],
                              [[3, 1, 3, 2], [1, 1], Int[]], [[2, 1, 2, 1], [1, 2], Int[]]),
                             ([Float64[]], [Int[]], [Int[]]),
                             (Vector{Float64}[], Vector{Int}[], Vector{Int}[]))
            for a in (-0.6, 0.8)
                r = rk_jax_evaluate(src, "kernel", (a, xs, js, qs); prepare = :arguments, grad = true)
                @test r.value ≈ native(a, xs, js, qs)
                # The two fields and their recurrence are affine in a.
                @test r.grad ≈ native(1.0, xs, js, qs) - native(0.0, xs, js, qs)
            end
        end
        for spec in ((false, 0, 0, 1), (true, 1, 0, 1), (true, 0, 1, 2))
            args = ([1.5], spec...)
            r = rk_jax_evaluate(guarded, "kernel", (0.8, args...); grad = true)
            @test r.value ≈ reference(0.8, args...)
            @test r.grad ≈ reference(1.0, args...) - reference(0.0, args...)
        end
        mktempdir() do dir
            write(joinpath(dir, "model.py"), src)
            script = joinpath(@__DIR__, "jax_shared_record_reads.py")
            @test success(`$RK_JAX_PYTHON $script $dir`)
        end
    else
        @test_skip false
    end
end
