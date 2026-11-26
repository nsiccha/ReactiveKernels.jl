module JAXIndependentMaps
using ReactiveKernels, Test
include("jax_python.jl")

@traceable function signed_map(table, a)
    y = fill(a, length(table))
    for i in 1:length(table)
        x = table[i]
        if x > 0
            y[i] = a / sqrt(x)
        elseif x < 0
            y[i] = -a / sqrt(-x)
        end
    end
    y
end

@traceable function signed_recurrence(table, a)
    y = fill(a, length(table))
    for i in 1:length(table)
        x = table[i]
        if x > 0
            y[i] = y[max(i-1, 1)] + a * x
        elseif x < 0
            y[i] = a * x
        end
    end
    y
end

@kernel function map_model(a::Float64, table::Vector{Float64})
    out = signed_map(table, a)
    total = sum(out)
    return total
end
@kernel function recurrence_model(a::Float64, table::Vector{Float64})
    out = signed_recurrence(table, a)
    total = sum(out)
    return total
end

@kernel function two_tables(a::Float64, first::Vector{Float64}, second::Vector{Float64})
    left = signed_map(first, a)
    right = signed_map(second, a)
    total = sum(left) - sum(right)
    return total
end

@testset "Independent stores partition prepared rows and retain their original order" begin
    have = (:a, :table)
    compile(k; want=:total, data=(:table,)) = jax_source(k; have, want, data)
    src = compile(map_model)
    @test occursin("_rk_sign_groups", string(src))
    @test !occursin("_rk_sign_groups", string(compile(recurrence_model)))
    @test !occursin("_rk_sign_groups", string(jax_source(map_model; have, want=:total)))
    @test string(compile(map_model; data=(table=[-2.,0.,1.],))) ==
          string(compile(map_model; data=(table=ones(47),)))
    if rk_jax_available()
        pair = jax_source(two_tables; have=(:a,:first,:second), want=:total, data=(:first,:second))
        p = rk_jax_evaluate(pair, :kernel, (2.0, [-4.,1.], [0.,4.,9.]); prepare=:arguments, grad=true)
        @test p.value ≈ 2 * (0.5 - (1 + 0.5 + 1/3))
        @test p.grad ≈ 0.5 - (1 + 0.5 + 1/3)
        result = compile(map_model; want=:out)
        f = prepare(map_model; have, want=:out)
        for table in ([-4.,0.,16.,-1.,4.], [9.,4.], [-4.,-1.], [0.,-0.,NaN], Float64[])
            value = rk_jax_evaluate(result, :kernel, (0.7,table); prepare=:arguments)
            @test value.value ≈ f(0.7,table)
            derivative = rk_jax_evaluate(src, :kernel, (0.7,table); prepare=:arguments, grad=true)
            @test derivative.grad ≈ sum(f(1.,table))
        end
    end
end
end
