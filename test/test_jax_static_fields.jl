using ReactiveKernels, Test, Enzyme
using DifferentiationInterface: AutoEnzyme

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

@traceable function jax_typed_field(record::NamedTuple{names}) where {names}
    name = first(names)
    return getproperty(record, name)
end
@traceable function jax_keyed_field(record::NamedTuple)
    names = keys(record)
    return getproperty(record, first(names))
end
@traceable jax_direct_field(record) = record.a
@traceable jax_field_name(record) = first(keys(record))
@traceable jax_read_named_field(record, name) = getproperty(record, name)
@traceable function jax_reordered_field(record)
    names = keys(record)
    selected = (last(names), first(names))
    return getproperty(record, selected[1])
end
function jax_plain_keyed_field(record::NamedTuple)
    names = keys(record)
    name = last(names)
    getproperty(record, name)
end

@kernel jax_typed_field_kernel(x::Float64) = begin
    record = (; a = x * x, b = x + 1)
    value = jax_typed_field(record)
    return value
end
@kernel jax_keyed_field_kernel(x::Float64) = begin
    record = (; a = x * x, b = x + 1)
    value = jax_keyed_field(record)
    return value
end
@kernel jax_direct_field_kernel(x::Float64) = begin
    record = (; a = x * x, b = x + 1)
    value = jax_direct_field(record)
    return value
end
@kernel jax_composed_field_kernel(x::Float64) = begin
    record = (; a = x * x, b = x + 1)
    reversed = (; b = x + 1, a = x * x)
    value = jax_read_named_field(record, jax_field_name(record)) +
        jax_read_named_field(record, last(keys(record))) +
        jax_typed_field(reversed) + jax_plain_keyed_field(record) +
        jax_reordered_field(record)
    return value
end
@kernel jax_host_field_kernel(x::Float64, y::Vector{Float64}) = begin
    record = (; a = sum(y), b = sum(abs2, y))
    coefficient = jax_keyed_field(record) +
        jax_read_named_field(record, jax_field_name(record)) +
        jax_read_named_field(record, last(keys(record)))
    value = x * coefficient
    return value
end
@kernel jax_ragged_field_kernel(x::Float64, rows::Vector{Vector{Float64}}) = begin
    terms = plate(rows, x) do row, s
        if isempty(row)
            0.0
        else
            record = (; a = row .* s, b = row .* (s * s))
            sum(jax_typed_field(record)) + sum(jax_keyed_field(record)) +
                sum(jax_read_named_field(record, last(keys(record))))
        end
    end
    value = sum(terms)
    return value
end

@testset "jax_source: structural field names through locals and helpers" begin
    specs = (jax_direct_field_kernel, jax_typed_field_kernel, jax_keyed_field_kernel,
             jax_composed_field_kernel)
    for spec in specs
        source = jax_source(spec; types = (x = Float64,), functions = (value = :value,))
        native = prepare(spec; have = (:x,), want = :value)
        ad = prepare_ad(native, AutoEnzyme(; mode = Enzyme.Reverse), 0.4; active = :x)
        for x in (0.4, -0.7)
            expected = spec === jax_composed_field_kernel ? x * x + 4 * (x + 1) : x * x
            derivative = spec === jax_composed_field_kernel ? 2 * x + 4 : 2 * x
            @test native(x) ≈ expected
            @test only(ad_gradient(ad, x)) ≈ derivative
            if rk_jax_available()
                result = rk_jax_evaluate(source, :value, (x,); grad = :joint)
                @test result.value ≈ native(x)
                @test result.joint_value ≈ result.value
                @test result.grad ≈ only(ad_gradient(ad, x))
            else
                @test_skip false
            end
        end
    end
end

@testset "jax_source: structural names keep host values and sizes runtime" begin
    datasets = (Float64[], [0.3], [0.2, -0.4, 1.2], collect(0.1:0.1:2.9))
    sources = [jax_source(jax_host_field_kernel; data = (; y), functions = (value = :value,))
               for y in datasets]
    @test all(s -> string(s) == string(first(sources)), sources)
    @test haskey(first(sources).prepared, :coefficient)
    native = prepare(jax_host_field_kernel; have = (:x, :y), want = :value)
    for y in datasets
        original = copy(y)
        derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
            Enzyme.Active(0.4), Enzyme.Const(y))[1][1]
        @test derivative ≈ 2sum(y) + sum(abs2, y)
        if rk_jax_available()
            result = rk_jax_evaluate(first(sources), :value, (0.4, y);
                grad = :joint, prepare = :arguments)
            @test result.value ≈ native(0.4, y)
            @test result.grad ≈ derivative
        else
            @test_skip false
        end
        @test y == original
    end
end

@testset "jax_source: structural names in empty and ragged plates" begin
    datasets = (Vector{Float64}[], [Float64[]], [Float64[], [0.2, 0.7], [-0.3]],
                [collect(range(0.1; step = 0.02, length = k)) for k in 1:19])
    sources = [jax_source(jax_ragged_field_kernel; data = (; rows),
                         functions = (value = :value,)) for rows in datasets]
    @test all(s -> string(s) == string(first(sources)), sources)
    @test occursin("jax.vmap(", string(first(sources)))
    native = prepare(jax_ragged_field_kernel; have = (:x, :rows), want = :value)
    for rows in datasets
        original = deepcopy(rows)
        derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
            Enzyme.Active(0.4), Enzyme.Const(rows))[1][1]
        if rk_jax_available()
            result = rk_jax_evaluate(first(sources), :value, (0.4, rows);
                grad = :joint, prepare = :arguments)
            @test result.value ≈ native(0.4, rows)
            @test result.grad ≈ derivative
        else
            @test_skip false
        end
        @test rows == original
    end
    if rk_jax_available()
        mktempdir() do dir
            write(joinpath(dir, "model.py"), first(sources))
            cp(joinpath(@__DIR__, "jax_static_fields.py"), joinpath(dir, "check.py"))
            @test success(`$RK_JAX_PYTHON $(joinpath(dir, "check.py"))`)
        end
    end
end
