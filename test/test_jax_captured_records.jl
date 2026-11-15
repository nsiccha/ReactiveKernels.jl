using ReactiveKernels, Test

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# A kernel that captures a named tuple of numeric arrays and passes it whole to
# an authored helper (snag emit-an-rk-kerne-6bbd4365): the record is the
# Python tuple of its fields, each array field a typed module constant with
# Julia's shape, and the helper reads its fields as it reads a named-tuple
# HAVE's.

@traceable function jax_record_weighted(x, metadata)
    metadata.weights[1] * x[metadata.indices[1]]
end

function jax_record_weighted_spec()
    metadata = (; weights = [2.0], indices = [1])
    @kernel jax_record_weighted_kernel(x) = begin
        result = jax_record_weighted(x, metadata)
    end
end

# Matrices and a rank-3 integer array, read at loop indices and at indices
# read from other fields; a zero index marks an unused slot, as a padded
# table of a structure's members does.
@traceable function jax_record_table(x, m)
    length(x) >= length(m.scale) ||
        throw(DimensionMismatch("one position per table row is required"))
    y = copy(x)
    rows = zeros(eltype(x), length(m.scale))
    total = zero(eltype(x))
    for p in eachindex(m.scale)
        value = m.scale[p] * m.offset
        for j in axes(m.members, 2)
            q = m.members[p, j]
            if q > 0
                value += m.weights[p, j] * x[q]
            end
        end
        for j in axes(m.pairs, 2)
            a = m.pairs[1, j, p]
            b = m.pairs[2, j, p]
            if a > 0
                value += x[a] * x[b]
            end
        end
        rows[p] = value
        y[m.target[p]] = y[m.target[p]] + value
        total += m.inner.factor * value^2
    end
    total + sum(rows) + sum(abs2, y)
end

function jax_record_table_spec()
    metadata = (; scale = [1.5, -0.5], offset = 0.25,
                members = [1 3 0; 2 0 0], weights = [0.5 -1.0 9.0; 2.0 9.0 9.0],
                pairs = cat([1 2; 3 3], [2 0; 1 0]; dims = 3), target = [3, 1],
                inner = (; factor = 0.5))
    @kernel jax_record_table_kernel(x) = begin
        total = jax_record_table(x, metadata)
        # A field read in the kernel itself names the same constant.
        shifted = total + metadata.scale[1]
        return shifted
    end
end

# A record of literals stays static: it sizes an allocation.
@traceable jax_record_sized(x, dims) = sum(zeros(eltype(x), dims.n) .+ x[1]) * dims.s

function jax_record_static_spec()
    dims = (; n = 3, s = 2.0)
    @kernel jax_record_static_kernel(x) = begin
        result = jax_record_sized(x, dims)
    end
end

@traceable jax_record_apply(x, ops) = ops.f(x[1])

function jax_record_function_spec()
    ops = (; f = exp, w = [1.0])
    @kernel jax_record_function_kernel(x) = begin
        result = jax_record_apply(x, ops)
    end
end

@testset "jax_source: a captured named tuple of arrays passed to a helper" begin
    spec = jax_record_weighted_spec()
    native = prepare(spec; have = (:x,), want = :result)
    @test native([3.0]) == 6.0
    src = jax_source(spec; have = (:x,), want = :result, types = (; x = Vector{Float64}))
    text = string(src)
    @test occursin("METADATA_WEIGHTS = jnp.asarray([2.0], dtype=jnp.float64)", text)
    @test occursin("METADATA_INDICES = jnp.asarray([1], dtype=jnp.int64)", text)
    @test occursin("METADATA = (METADATA_WEIGHTS, METADATA_INDICES)", text)
    @test occursin(r"jax_record_weighted\w*\(x\w*, METADATA\)", text)

    table = jax_record_table_spec()
    native = prepare(table; have = (:x,), want = :shifted)
    table_src = jax_source(table; have = (:x,), want = :shifted,
                           types = (; x = Vector{Float64}))
    table_text = string(table_src)
    @test occursin("METADATA_MEMBERS = jnp.asarray([[1, 3, 0], [2, 0, 0]], dtype=jnp.int64)",
                   table_text)
    @test occursin("METADATA_PAIRS = jnp.asarray(", table_text)
    # The nested record of a literal stays a literal inside the record.
    @test occursin("(0.5,)", table_text)
    @test count("METADATA_SCALE = ", table_text) == 1

    static_src = jax_source(jax_record_static_spec(); have = (:x,), want = :result,
                            types = (; x = Vector{Float64}))
    @test !occursin("DIMS = ", string(static_src))

    err = try
        jax_source(jax_record_function_spec(); have = (:x,), want = :result,
                   types = (; x = Vector{Float64}))
        nothing
    catch e
        e
    end
    @test err isa JAXEmissionError
    @test occursin("the field `f` of the captured value `ops` is a function", sprint(showerror, err))

    if !rk_jax_available()
        @test_skip false
    else
        r = rk_jax_evaluate(src, "kernel", ([3.0],); grad = true)
        @test r.value ≈ 6.0
        @test r.grad ≈ [2.0]
        for x in ([0.3, -1.2, 0.8], [1.1, 0.4, -0.6])
            r = rk_jax_evaluate(table_src, "kernel", (x,); grad = true)
            @test r.value ≈ native(x) rtol = 1e-12
            @test r.grad ≈ rk_fd_gradient(native, x) rtol = 1e-6
        end
        static_native = prepare(jax_record_static_spec(); have = (:x,), want = :result)
        r = rk_jax_evaluate(static_src, "kernel", ([0.7],))
        @test r.value ≈ static_native([0.7]) rtol = 1e-12
    end
end
