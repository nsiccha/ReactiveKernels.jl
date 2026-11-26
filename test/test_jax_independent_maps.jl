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
@traceable function copy_corner(x, a)
    out = fill(a, size(x, 1), size(x, 2))
    for i in 1:size(x, 1)
        for j in 1:2
            out[i, j] = x[i, j]
        end
    end
    for i in 1:min(size(x, 1), size(x, 2))
        out[i, i] = 2.0
    end
    out
end

@kernel function corner_model(a::Float64, x::Matrix{Float64})
    out = copy_corner(x .* a, a)
    total = sum(out)
    return total
end

@kernel function guarded_corner(a::Float64, x::Matrix{Float64})
    total = a > 0 ? sum(copy_corner(x .* a, a)) : a
    return total
end

@testset "Aligned matrix stores preserve order, bounds, and gradients" begin
    have = (:a, :x)
    src = jax_source(corner_model; have, want = :total)
    @test occursin("BoundsError: block exceeds array dimensions", string(src))
    @test occursin(".at[:", string(src))
    x = [1.0 4.0 7.0; 0.5 2.0 -0.5]
    native = prepare(corner_model; have, want = :total)
    @test_throws BoundsError native(0.7, zeros(2, 1))
    if rk_jax_available()
        probe = rk_jax_evaluate(src, :kernel, (0.7, x); grad = true)
        @test probe.value ≈ native(0.7, x)
        @test probe.grad ≈ native(1.0, x) - native(0.0, x)
        @test_throws "BoundsError: block exceeds array dimensions" rk_jax_evaluate(
            src, :kernel, (0.7, zeros(2, 1)))
        skipped = jax_source(guarded_corner; have, want = :total)
        inactive = rk_jax_evaluate(skipped, :kernel, (-0.7, zeros(2, 1)); grad = true)
        @test inactive.value == -0.7
        @test inactive.grad == 1.0
    end
    @test native(0.7, zeros(0, 1)) == 0.0
end

# The store to (1,2) changes the value read through the reshape at (2,2).
@traceable function overlapping_copy(x)
    out = fill(x, 3, 2)
    out[3, 1] = 2x
    alias = reshape(out, 2, 3)
    for i in 1:2
        for j in 1:2
            out[i, j] = alias[i, j]
        end
    end
    out
end

@kernel function alias_sum(x::Float64)
    out = overlapping_copy(x)
    total = sum(out)
    return total
end

@traceable function overlapping_row(x)
    out = fill(x, 2, 3)
    out[1, 1] = 2x
    strip = vec(out)
    for j in 1:3
        out[2, j] = strip[j]
    end
    out
end

@kernel function alias_row_sum(x::Float64)
    out = overlapping_row(x)
    total = sum(out)
    return total
end

@testset "Overlapping reshape reads observe ordered writes" begin
    for (model, native) in ((alias_sum, overlapping_copy), (alias_row_sum, overlapping_row))
        src = jax_source(model; have=(:x,), want=:total)
        @test !occursin(".at[:", string(src))
        @test !occursin("jax.lax.dynamic_update_slice(", string(src))
        @test occursin("fori_loop(", string(src))
        if rk_jax_available()
            overlap = rk_jax_evaluate(src, :kernel, (1.0,); grad=true)
            @test overlap.value == sum(native(1.0)) == 9
            @test overlap.grad == 9
        end
    end
end
end
