using ReactiveKernels, Test, Enzyme

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

@kernel jax_product_single(x::Float64, n::Int) = begin
    states = collect(Iterators.product(0:n))
    terms = plate(states) do state
        (x + state[1])^2
    end
    value = sum(terms)
    return value
end

@kernel jax_product_two(x::Float64, n::Int, m::Int) = begin
    states = collect(Iterators.product(0:n, m:-2:1))
    terms = plate(states) do state
        (x + state[1] + 10 * state[2])^2
    end
    ordered = vec(terms)
    value = sum(terms)
    return value, ordered
end

@kernel jax_product_three(x::Float64, a::Vector{Int}, b::Vector{Int}, c::Vector{Int}) = begin
    states = collect(Iterators.product(a, b, c))
    terms = plate(states) do state
        (x + state[1] + 10 * state[2] + 100 * state[3])^2
    end
    ordered = vec(terms)
    value = sum(terms)
    return value, ordered
end

@kernel jax_product_matrix(x::Float64, a::Matrix{Int}, b::Vector{Int}) = begin
    states = collect(Iterators.product(a, b))
    terms = plate(states) do state
        (x + state[1] + 10 * state[2])^2
    end
    ordered = vec(terms)
    value = sum(terms)
    return value, ordered
end

@traceable function jax_product_read(x, states)
    x * (length(states) + size(states, 1) + 10 * size(states, 2)) +
        states[2, 1][1] + states[1, 2][2] + vec(states)[2][1]
end
@kernel jax_product_shape(x::Float64, n::Int, m::Int) = begin
    states = collect(Iterators.product(0:n, 1:m))
    value = jax_product_read(x, states)
    return value
end

@traceable function jax_product_rows(rows)
    total = 0
    for row in rows
        for state in Iterators.product(row, 1:2)
            total += state[1] * state[2]
        end
    end
    total
end
@kernel jax_product_ragged(x::Float64, rows::Vector{Vector{Int}}) = begin
    total = jax_product_rows(rows)
    value = x^2 * total
    return value
end

@traceable function jax_product_host_read(n, m)
    iter = Iterators.product(0:n, 1:m)
    states = collect(iter)
    if length(iter) == 0
        0
    else
        dims = size(iter)
        length(iter) + dims[1] + 10 * dims[2] +
            states[1, 1][1] + states[end][2] + vec(states)[1][2]
    end
end
@kernel jax_product_host(x::Float64, n::Int, m::Int) = begin
    total = jax_product_host_read(n, m)
    value = x^2 * total
    return value
end

function jax_product_parity(spec, have, datasets; ordered = false)
    native = prepare(spec; have, want = :value)
    data(d) = NamedTuple{Base.tail(have)}(d)
    functions = ordered ? (value = :value, ordered = :ordered) : (value = :value,)
    src = jax_source(spec; have, data = data(first(datasets)), functions)
    # Data controls allocation and iteration at prepare_data, never the
    # number of emitted bodies. Compare complete source, not just its size.
    @test all(string(jax_source(spec; have, data = data(d), functions)) == string(src)
              for d in datasets)
    @test occursin("def prepare_data", string(src))
    @test occursin("_rk_product_rows", string(src))
    if rk_jax_available()
        for d in datasets, mode in (true, :arguments)
            x = 0.4
            got = rk_jax_evaluate(src, :value, (x, d...); grad = :joint, prepare = mode)
            @test got.value ≈ native(x, d...)
            @test got.joint_value ≈ got.value
            # Differentiate the same primal through each backend's ordinary
            # reverse path, including empty supports. No custom rule is used.
            reference = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
                                       Enzyme.Active(x), map(Enzyme.Const, d)...)[1][1]
            @test got.grad ≈ reference atol = 1e-7 rtol = 1e-7
            if ordered
                f = prepare(spec; have, want = :ordered)
                actual = rk_jax_evaluate(src, :ordered, (x, d...); prepare = mode)
                @test actual.value ≈ f(x, d...)
            end
        end
    else
        @test_skip false
    end
    src
end

@testset "jax_source: data-sized Cartesian products" begin
    @test prepare(jax_product_single; want = :value)(0.4, 5) ≈ 67.96
    jax_product_parity(jax_product_single, (:x, :n), [(5,), (20,), (0,), (-1,)])
    src = jax_product_parity(jax_product_two, (:x, :n, :m),
                             [(2, 3), (7, 9), (-1, 3), (2, 0), (-1, 0)]; ordered = true)
    if rk_jax_available()
        mktempdir() do dir
            path = joinpath(dir, "product_module.py")
            write(path, src)
            driver = joinpath(@__DIR__, "jax_products.py")
            @test success(pipeline(`$RK_JAX_PYTHON $driver $path`; stdout, stderr))
        end
    end
    jax_product_parity(jax_product_three, (:x, :a, :b, :c),
                       [([2, 0], [4, 1], [3, 5]), (collect(1:5), [1, 2, 3], [7]),
                        (Int[], [2], [3]), ([2], Int[], [3]), ([2], [3], Int[])]; ordered = true)
    jax_product_parity(jax_product_shape, (:x, :n, :m), [(2, 3), (8, 7)])
    jax_product_parity(jax_product_matrix, (:x, :a, :b),
                       [(reshape([3, 1, 4, 2], 2, 2), [2, 1]),
                        (reshape(collect(1:6), 2, 3), [1, 2, 3]),
                        (Matrix{Int}(undef, 2, 0), [1, 2])]; ordered = true)
    jax_product_parity(jax_product_ragged, (:x, :rows),
                       [([[1, 2], Int[], [3]],), ([[4], [5, 6, 7]],), (Vector{Int}[],)])
    jax_product_parity(jax_product_host, (:x, :n, :m), [(2, 3), (8, 7), (-1, 3), (2, 0)])
end
