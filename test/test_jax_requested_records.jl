using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

@kernel function jax_requested_records(a::Float64, bases::Vector{Vector{Float64}},
                                      kinds::Vector{Vector{Int}}, firsts::Vector{Vector{Int}},
                                      seconds::Vector{Vector{Int}}, visits::Vector{Vector{Int}})
    totals = plate(bases, kinds, firsts, seconds, visits) do base, ks, is, js, vs
        records = plate(base) do x
            (a * x, exp(-a * x))
        end
        offsets = plate(ks, is, js) do kind, i, j
            r = records[i]
            s = records[j]
            (r[1] * s[2], r[2] + s[1])
        end
        path = scan(ks, vs, Ref(offsets); init=0.0) do carry, kind, v, table
            next = kind == 3 ? 0.75carry + table[v][1] + table[v][2] : carry
            (next, next)
        end
        sum(path)
    end
    total = sum(totals)
    return total
end

@kernel function jax_guarded_requested_records(a::Float64, base::Vector{Float64},
                                              kinds::Vector{Int}, indices::Vector{Int})
    records = plate(base) do x
        (a * x, exp(-a * x))
    end
    offsets = plate(kinds, indices) do kind, i
        kind == 3 ? records[i] : (0.0, 0.0)
    end
    path = scan(kinds, indices, Ref(offsets); init=0.0) do carry, kind, i, table
        next = kind == 3 ? carry + table[i][1] : carry
        (next, next)
    end
    total = sum(path)
    return total
end

@testset "A nested guard retains its record read" begin
    have = (:a, :base, :kinds, :indices)
    src = jax_source(jax_guarded_requested_records; have, want=:total,
                     data=have[2:end], plate_layout=:padded)
    @test !occursin("requested_record=None", src.text)
    if rk_jax_available()
        native = prepare(jax_guarded_requested_records; have, want=:total)
        args = (0.7, [0.2, 0.4], [3, 2], [1, 99])
        result = rk_jax_evaluate(src, "kernel", args; prepare=:arguments, grad=true)
        @test result.value ≈ native(args...)
        @test result.grad ≈ native(1.0, args[2:end]...)
    end
end

@testset "Requested cells compose subject and record indices" begin
    have = (:a, :bases, :kinds, :firsts, :seconds, :visits)
    data = have[2:end]
    src = jax_source(jax_requested_records; have, want=:total, data, plate_layout=:padded)
    @test occursin("requested_record=None", src.text)
    @test occursin("requested_record_2=None", src.text)
    @test occursin(" = _rk_requested_cell_rows(", src.text)
    native = prepare(jax_requested_records; have, want=:total)
    if rk_jax_available()
        mktempdir() do dir
            write(joinpath(dir, "model.py"), src)
            open(joinpath(dir, "cases.py"), "w") do io
                println(io, "cases = [")
                for n in (9, 17)
                    bases = [collect(range(0.2, 0.8; length=n-2)), [0.3, 0.9], Float64[]]
                    kinds = [[i % 3 == 0 ? 3 : 2 for i in 1:n], [3, 2, 3], Int[]]
                    firsts = [[mod1(i, n-2) for i in 1:n], [1, 2, 1], Int[]]
                    seconds = [[mod1(n-i, n-2) for i in 1:n], [2, 1, 2], Int[]]
                    # Duplicate requested producer rows and inactive bad indices.
                    visits = [[i % 3 == 0 ? 2 : -99 for i in 1:n], [1, 99, 1], Int[]]
                    for shift in (0, 1)
                        firsts[1] = circshift(firsts[1], shift)
                        args = (bases, kinds, firsts, seconds, visits)
                        a, h = 0.7, 1e-5
                        value = native(a, args...)
                        gradient = (native(a+h, args...) - native(a-h, args...)) / (2h)
                        println(io, "(", join(_rk_py.(args), ", "), ", ", value, ", ", gradient, "),")
                    end
                end
                # Empty subjects, including no requested cells anywhere.
                for args in (([[0.2], Float64[]], [[2], Int[]], [[1], Int[]], [[1], Int[]], [[99], Int[]]),
                             ([Float64[]], [Int[]], [Int[]], [Int[]], [Int[]]))
                    println(io, "(", join(_rk_py.(args), ", "), ", ", native(0.7, args...), ", 0.0),")
                end
                println(io, "]")
            end
            @test success(`$RK_JAX_PYTHON $(joinpath(@__DIR__, "jax_requested_records.py")) $dir`)
        end
    else
        @test_skip "RK_JAX_PYTHON is not available"
    end
end
