using ReactiveKernels, Test, Enzyme

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

@kernel jax_record_flat(x::Float64, indices::Vector{Int}) = begin
    records = plate(indices, x) do i, s
        (; prior = (s + i)^2, likelihood = s * i)
    end
    terms = plate(records) do r
        r.prior + r.likelihood
    end
    value = sum(terms)
    return value
end

@kernel jax_record_nested(x::Float64, indices::Vector{Int}, y::Vector{Float64}) = begin
    records = plate(indices, x, Ref(y)) do i, s, ys
        (; prior = (s + i)^2, likelihood = s * i, pointwise = (; y = ys .* s))
    end
    terms = plate(records) do r
        r.prior + r.likelihood + sum(r.pointwise.y)
    end
    value = sum(terms)
    return value
end

@kernel jax_record_product(x::Float64, a::Vector{Int}, b::Vector{Int}, y::Vector{Float64}) = begin
    states = collect(Iterators.product(a, b))
    records = plate(states, x, Ref(y)) do ij, s, ys
        (; prior = (s + ij[1])^2, likelihood = s * ij[2], pointwise = (; y = ys .* s))
    end
    terms = plate(records) do r
        r.prior + r.likelihood + sum(r.pointwise.y)
    end
    ordered = vec(terms)
    value = sum(terms)
    return value, ordered
end

@kernel jax_record_ragged(x::Float64, rows::Vector{Vector{Float64}}) = begin
    records = plate(rows, x) do row, s
        (; prior = s^2, pointwise = (; y = row .* s, z = row .+ s))
    end
    terms = plate(records) do r
        r.prior + sum(r.pointwise.y) + sum(r.pointwise.z)
    end
    value = sum(terms)
    return value
end

@kernel jax_record_padded(x::Float64, rows::Vector{Vector{Float64}}) = begin
    totals = plate(rows, x) do row, s
        records = plate(row) do y
            (; prior = s * y, pointwise = (; likelihood = y + 1))
        end
        terms = plate(records) do r
            r.prior + r.pointwise.likelihood
        end
        sum(terms)
    end
    value = sum(totals)
    return value
end

@kernel jax_record_rank3(x::Float64, indices::Array{Int,3}) = begin
    records = plate(indices, x) do i, s
        (; pointwise = (; y = (s + i, s - i)), prior = s * i)
    end
    terms = plate(records) do r
        r.pointwise.y[1]^2 + r.pointwise.y[2] + r.prior
    end
    ordered = vec(terms)
    value = sum(terms)
    return value, ordered
end

@kernel jax_record_padded_tuple(x::Float64, rows::Vector{Vector{Float64}}) = begin
    totals = plate(rows, x) do row, s
        records = plate(row) do y
            (s * y, (; likelihood = y + 1))
        end
        terms = plate(records) do r
            r[1] + r[2].likelihood
        end
        sum(terms)
    end
    value = sum(totals)
    return value
end

@kernel jax_record_guarded(x::Float64, indices::Vector{Int}, y::Vector{Float64}) = begin
    records = plate(indices, x, Ref(y)) do i, s, ys
        prior = i > 0 ? log(s + i) : s^2
        (; prior, pointwise = (; y = ys .* s), shared = s * ys[1])
    end
    terms = plate(records) do r
        r.prior + sum(r.pointwise.y) + r.shared
    end
    value = sum(terms)
    return value
end

function jax_record_parity(spec, have, datasets; ordered = false, reverse_reference = nothing)
    data(d) = NamedTuple{Base.tail(have)}(d)
    native = prepare(spec; have, want = :value)
    functions = ordered ? (value = :value, ordered = :ordered) : (value = :value,)
    src = jax_source(spec; have, data = data(first(datasets)), functions)
    @test all(string(jax_source(spec; have, data = data(d), functions)) == string(src)
              for d in datasets)
    if rk_jax_available()
        for d in datasets, mode in (true, :arguments)
            x = 0.4
            got = rk_jax_evaluate(src, :value, (x, d...); grad = :joint, prepare = mode)
            @test got.value ≈ native(x, d...) atol = 1e-8
            @test got.joint_value ≈ got.value atol = 1e-8
            reference = reverse_reference === nothing ? native : reverse_reference
            reverse_reference === nothing || @test reference(x, d...) ≈ native(x, d...)
            derivative = Enzyme.autodiff(Enzyme.Reverse, reference, Enzyme.Active,
                Enzyme.Active(x), map(Enzyme.Const, d)...)[1][1]
            @test got.grad ≈ derivative atol = 1e-8
            if ordered
                ordered_native = prepare(spec; have, want = :ordered)
                got_ordered = rk_jax_evaluate(src, :ordered, (x, d...); prepare = mode)
                @test got_ordered.value ≈ ordered_native(x, d...)
            end
        end
    else
        @test_skip false
    end
    src
end

@testset "jax_source: numeric record plates" begin
    jax_record_parity(jax_record_flat, (:x, :indices), [([1, 2, 3],), (collect(1:30),), (Int[],)])
    jax_record_parity(jax_record_nested, (:x, :indices, :y),
        [([1, 2, 3], [0.2, 0.7]), (collect(1:30), [0.1, 0.5, 0.9]),
         (Int[], Float64[]), (Int[], [0.2]), ([1, 2], Float64[])])
    product = jax_record_parity(jax_record_product, (:x, :a, :b, :y),
        [([2, 0], [4, 1], [0.2, 0.7]), (collect(1:8), collect(1:7), [0.3]),
         (Int[], [2, 3], Float64[]), ([1, 2], Int[], [0.1]), (Int[], Int[], Float64[])];
        ordered = true)
    ragged = [([[0.2, 0.7], Float64[], [0.3]],), ([[0.1], [0.2, 0.5, 0.9]],),
              ([Float64[], Float64[]],), (Vector{Float64}[],)]
    ragged_source = jax_record_parity(jax_record_ragged, (:x, :rows), ragged)
    if rk_jax_available()
        mktempdir() do dir
            product_path = joinpath(dir, "product.py")
            ragged_path = joinpath(dir, "ragged.py")
            write(product_path, product)
            write(ragged_path, ragged_source)
            driver = joinpath(@__DIR__, "jax_record_plates.py")
            @test success(pipeline(`$RK_JAX_PYTHON $driver $product_path $ragged_path`; stdout, stderr))
        end
    end
    jax_record_parity(jax_record_padded, (:x, :rows), ragged)
    jax_record_parity(jax_record_padded_tuple, (:x, :rows), ragged)
    jax_record_parity(jax_record_rank3, (:x, :indices),
        [(reshape(collect(1:24), 2, 3, 4),), (reshape(collect(1:120), 4, 5, 6),),
         (Array{Int}(undef, 0, 2, 3),), (Array{Int}(undef, 2, 0, 3),),
         (Array{Int}(undef, 2, 3, 0),)]; ordered = true)
    jax_record_parity(jax_record_guarded, (:x, :indices, :y),
        [([-1, 0, 1, 2], [0.2, 0.7]), (Int[], Float64[]), ([-2, 3], [0.5])])
end

isdefined(@__MODULE__, :MatrixRecordPlateFixture) || include("fixtures/matrix_record_plate.jl")

jax_record_right_field(record) = record.pointwise.b

@kernel jax_record_independent(x::Float64, left::Vector{Vector{Float64}}, right::Vector{Vector{Float64}}) = begin
    records = plate(left, right, x) do a, b, s
        (; pointwise = (; a = a .* s, b = b .+ s), prior = s^2)
    end
    terms = plate(records) do r
        sum(r.pointwise.a) + sum(jax_record_right_field(r)) + r.prior
    end
    value = sum(terms)
    return value
end

@kernel jax_record_padded_matrix(x::Float64, rows::Vector{Vector{Float64}}, weights::Vector{Float64}) = begin
    totals = plate(rows, x, Ref(weights)) do row, s, w
        matrix = row .* transpose(w)
        records = plate(matrix) do y
            (; prior = s * y, pointwise = (; likelihood = y + 1))
        end
        terms = plate(records) do r
            r.prior + r.pointwise.likelihood
        end
        sum(terms) + length(records) + size(records, 1)
    end
    value = sum(totals)
    return value
end

@testset "jax_source: record shape and padding metadata" begin
    jax_record_parity(MatrixRecordPlateFixture.matrix_records, (:x, :a, :b, :y),
        [([1, 2], [3, 4, 5], [0.2 0.3; 0.5 0.7]),
         ([2], [3], zeros(0, 3)), (Int[], [1, 2], zeros(2, 0)),
         ([1, 2], Int[], zeros(0, 0))]; reverse_reference = MatrixRecordPlateFixture.reference)
    jax_record_parity(jax_record_independent, (:x, :left, :right),
        [([[0.2, 0.7], Float64[], [0.3]], [[0.9], [0.4, 0.2], Float64[]]),
         ([Float64[]], [[0.2, 0.3]]), (Vector{Float64}[], Vector{Float64}[])])
    jax_record_parity(jax_record_padded_matrix, (:x, :rows, :weights),
        [([[0.2, 0.7], Float64[], [0.3]], [0.5, 0.8, 1.2]),
         ([Float64[], Float64[]], [0.2, 0.3]), (Vector{Float64}[], [0.3]),
         ([[0.2], Float64[]], Float64[])])
end
