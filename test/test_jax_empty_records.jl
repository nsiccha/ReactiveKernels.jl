module JAXEmptyRecordTests
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

@kernel constructor_record(x::Float64) = begin
    empty_record = NamedTuple()
    value = (; square = x * x, empty_record)
    square = value.square
    return value, square, empty_record
end
@kernel literal_record(x::Float64) = begin
    empty_record = (;)
    value = (; square = x * x, empty_record)
    square = value.square
    return value, square, empty_record
end
@kernel empty_record() = begin
    value = NamedTuple()
    return value
end
@kernel typed_empty_record() = begin
    value = NamedTuple{()}(())
    return value
end
const EmptyRecord = typeof(NamedTuple())
@kernel concrete_empty_record() = begin
    value = EmptyRecord(())
    return value
end
@traceable function helper_record(x)
    (; square = x * x, empty_record = NamedTuple())
end
@kernel helper_kernel(x::Float64) = begin
    value = helper_record(x)
    square = value.square
    return value, square
end
@kernel typed_record(x::Float64) = begin
    value = NamedTuple{(:square, :empty_record)}((x * x, NamedTuple()))
    square = value.square
    return value, square
end
@kernel converted_scalar(x::Int) = begin
    value = Float64(x)
    return value
end

@kernel record_rows(x::Float64, rows::Vector{Vector{Float64}}) = begin
    records = plate(rows, x) do row, s
        terms = plate(row) do y
            (; empty_record = NamedTuple(), term = y > 0 ? log(s * y) : (s * y)^2)
        end
        scores = plate(terms) do r
            r.term
        end
        (; empty_record = (; inner = NamedTuple()), total = sum(scores))
    end
    totals = plate(records) do r
        r.total
    end
    value = sum(totals)
    return value
end

@testset "jax_source: empty named records" begin
    sources = [jax_source(spec; functions = (value = :value, square = :square))
               for spec in (constructor_record, literal_record, helper_kernel, typed_record)]
    # Constructor and literal have exactly the same emitted representation.
    @test string(sources[1]) == string(sources[2])
    for (spec, src) in zip((constructor_record, literal_record, helper_kernel, typed_record), sources)
        native = prepare(spec; want = :value)
        scalar = prepare(spec; want = :square)
        for x in (-0.7, 0.0, 0.4)
            @test native(x) == (; square = x * x, empty_record = NamedTuple())
            derivative = Enzyme.autodiff(Enzyme.Reverse, scalar, Enzyme.Active,
                                         Enzyme.Active(x))[1][1]
            @test derivative ≈ 2x
            if rk_jax_available()
                got = rk_jax_evaluate(src, :value, (x,))
                @test got.value["square"] ≈ native(x).square
                @test got.value["empty_record"] == Dict()
                got_scalar = rk_jax_evaluate(src, :square, (x,); grad = :joint)
                @test got_scalar.value ≈ scalar(x)
                @test got_scalar.grad ≈ derivative
            end
        end
    end
    for spec in (empty_record, typed_empty_record, concrete_empty_record)
        @test prepare(spec; have = (), want = :value)() === NamedTuple()
        src = jax_source(spec; have = (), functions = (value = :value,))
        if rk_jax_available()
            mktempdir() do dir
                path = joinpath(dir, "empty.py")
                write(path, src)
                script = "import runpy, jax; m=runpy.run_path(" * repr(path) *
                         "); assert m['value']() == {}; assert jax.jit(m['value'])() == {}"
                @test success(pipeline(`$RK_JAX_PYTHON -c $script`; stdout, stderr))
            end
        end
    end
    converted = jax_source(converted_scalar; want = :value)
    if rk_jax_available()
        for x in (-3, 0, 5)
            @test rk_jax_evaluate(converted, :kernel, (x,)).value === Float64(x)
        end
    end
    datasets = [Vector{Float64}[], [Float64[]], [[0.2, -0.4], Float64[], [0.6]],
                [collect(range(-0.8, 0.9; length = 12)), [0.5]]]
    srcs = [jax_source(record_rows; want = :value, data = (; rows)) for rows in datasets]
    @test all(src -> string(src) == string(first(srcs)), srcs)
    native = prepare(record_rows; want = :value)
    for rows in datasets
        x = 0.4
        derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
                                     Enzyme.Active(x), Enzyme.Const(rows))[1][1]
        if rk_jax_available()
            for mode in (true, :arguments)
                got = rk_jax_evaluate(first(srcs), :kernel, (x, rows); grad = :joint, prepare = mode)
                @test got.value ≈ native(x, rows) atol = 1e-10
                @test got.joint_value ≈ native(x, rows) atol = 1e-10
                @test got.grad ≈ derivative atol = 1e-10
            end
        end
    end
    if rk_jax_available()
        mktempdir() do dir
            path = joinpath(dir, "rows.py")
            write(path, first(srcs))
            driver = joinpath(@__DIR__, "jax_empty_records.py")
            @test success(pipeline(`$RK_JAX_PYTHON $driver $path`; stdout, stderr))
        end
    end
    rk_jax_available() || @test_skip "RK_JAX_PYTHON not set"
end
end
