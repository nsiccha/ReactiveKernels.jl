using ReactiveKernels, Test, Enzyme

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

@kernel jax_constant_product(x::Float64) = begin
    states = collect(Iterators.product(0:1, 0:1))
    terms = plate(states, x) do state, scale
        scale * (state[1] + 2 * state[2])
    end
    value = sum(terms)
    ordered = vec(terms)
    return value, ordered
end

@kernel jax_constant_product_chain(x::Float64) = begin
    axis = unique([3, 1, 3])
    states = collect(Iterators.product(axis, 5:-2:1, 2:3))
    terms = plate(states, x) do state, scale
        (scale + state[1] + 10 * state[2] + 100 * state[3])^2
    end
    value = sum(terms)
    ordered = vec(terms)
    return value, ordered
end

@kernel jax_constant_product_empty(x::Float64) = begin
    states = collect(Iterators.product(0:1, 1:0))
    terms = plate(states, x) do state, scale
        scale * (state[1] + 2 * state[2])
    end
    value = sum(terms)
    ordered = vec(terms)
    return value, ordered
end

@kernel jax_constant_findall(x::Float64) = begin
    selected = findall(y -> y < 4, [1, 4, 3])
    value = x^2 * sum(selected)
    return value
end

@kernel jax_constant_scalar(x::Float64) = begin
    scale = 3.0
    value = scale * x^2
    return value
end

@traceable function jax_constant_guarded_rows()
    rows = Int[]
    for state in Iterators.product(1:0, 0:1)
        push!(rows, [1][state[2] + 2])
    end
    isempty(rows) ? 7 : rows[1]
end
@kernel jax_constant_guard(x::Float64) = begin
    coefficient = jax_constant_guarded_rows()
    value = x^2 * coefficient
    return value
end

@testset "jax_source: host preparation without data inputs" begin
    @test prepare(jax_constant_product; want = :value)(0.4) ≈ 2.4
    @test prepare(jax_constant_product; want = :ordered)(0.4) ≈ [0.0, 0.4, 0.8, 1.2]
    for spec in (jax_constant_product, jax_constant_product_chain, jax_constant_product_empty)
        src = jax_source(spec; data = NamedTuple(), functions = (value = :value, ordered = :ordered))
        @test string(src) == string(jax_source(spec; data = (),
                                               functions = (value = :value, ordered = :ordered)))
        @test isempty(only(f for f in src.functions if f.name == "prepare_data").inputs)
        @test haskey(src.prepared, :states)
        @test !haskey(src.prepared, :x)
        if rk_jax_available()
            native = prepare(spec; want = :value)
            ordered = prepare(spec; want = :ordered)
            for x in (-0.7, 0.4), mode in (true, :arguments)
                got = rk_jax_evaluate(src, :value, (x,); grad = :joint, prepare = mode)
                @test got.value ≈ native(x)
                @test got.joint_value ≈ got.value
                reference = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active, Enzyme.Active(x))[1][1]
                @test got.grad ≈ reference atol = 1e-9 rtol = 1e-9
                got_ordered = rk_jax_evaluate(src, :ordered, (x,); prepare = mode)
                @test got_ordered.value ≈ ordered(x)
            end
        else
            @test_skip false
        end
    end
    # Host fallback belongs to the dependency split, not to product syntax.
    src = jax_source(jax_constant_findall; data = (), functions = (value = :value,))
    @test haskey(src.prepared, :selected)
    if rk_jax_available()
        for x in (-0.7, 0.4), mode in (true, :arguments)
            got = rk_jax_evaluate(src, :value, (x,); grad = :joint, prepare = mode)
            native = prepare(jax_constant_findall; want = :value)
            @test got.value ≈ native(x)
            @test got.grad ≈ Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active, Enzyme.Active(x))[1][1]
        end
    end
    # Device-supported constants still fold in the ordinary function.
    src = jax_source(jax_constant_scalar; data = (), functions = (value = :value,))
    @test isempty(src.prepared)

    # An empty host loop and the unchosen branch never read invalid entries.
    src = jax_source(jax_constant_guard; data = (), functions = (value = :value,))
    @test haskey(src.prepared, :coefficient)
    if rk_jax_available()
        native = prepare(jax_constant_guard; want = :value)
        for x in (-0.7, 0.4), mode in (true, :arguments)
            got = rk_jax_evaluate(src, :value, (x,); grad = :joint, prepare = mode)
            @test got.value ≈ native(x)
            @test got.grad ≈ Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active, Enzyme.Active(x))[1][1]
        end
    end
end
