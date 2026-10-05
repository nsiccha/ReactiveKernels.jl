using ReactiveKernels, Reactant, Test
using DifferentiationInterface: AutoEnzyme
import Enzyme

@kernel bound_context_array(x::Vector{Float64}, shift::Float64,
        scale::Float64, data::Vector{Float64}) = begin
    centered = x .- shift
    density::Float64 = sum(data .* centered) - scale * sum(abs2, centered)
end

@kernel bound_context_scalar(x::Float64, scale::Float64, offset::Float64) = begin
    density::Float64 = scale * x^2 + offset
end

@kernel bound_context_tuple(alpha::Vector{Float64}, beta::Vector{Float64},
        shift::Float64, scale::Float64, data::Vector{Float64}) = begin
    density::Float64 = sum(data .* (alpha .- shift)) + scale * sum(abs2, beta)
end

@testset "staged AD preserves bound numeric and array contexts" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for n in (3, 9, 17), scale in (0.7, -0.4)
        x = collect(range(-0.3, 0.5; length = n))
        data = collect(range(0.4, 1.1; length = n))
        shift = 0.35
        saved_data = copy(data)
        prepared = prepare_ad(bound_context_array, backend, x;
            active = :x, want = :density, bound = (; shift, scale, data))
        staged = let prepared = prepared
            v -> ad_value_and_gradient(prepared, v)
        end
        traced = Reactant.to_rarray(x)
        compiled = Reactant.compile(staged, (traced,))
        for v in (x, x .+ 0.2)
            saved_v = copy(v)
            rv = Reactant.to_rarray(v)
            saved_rv = Array(rv)
            value, gradient = compiled(rv)
            expected = sum(data .* (v .- shift)) - scale * sum(abs2, v .- shift)
            derivative = data .- 2scale .* (v .- shift)
            native_value, native_gradient = ad_value_and_gradient(prepared, v)
            @test Float64(value) ≈ expected
            @test Array(gradient) ≈ derivative
            @test native_value ≈ expected
            @test native_gradient ≈ derivative
            @test v == saved_v
            @test Array(rv) == saved_rv
            @test data == saved_data
        end
    end

    for scale in (0.7, -0.4)
        offset = 1.3
        prepared = prepare_ad(bound_context_scalar, backend, 0.2;
            active = :x, want = :density, bound = (; scale, offset))
        staged = let prepared = prepared
            v -> ad_value_and_gradient(prepared, v)
        end
        traced = Reactant.to_rarray(0.2; track_numbers = true)
        compiled = Reactant.compile(staged, (traced,))
        for v in (0.2, -0.6)
            rv = Reactant.to_rarray(v; track_numbers = true)
            value, gradient = compiled(rv)
            @test Float64(value) ≈ scale * v^2 + offset
            @test Float64(gradient) ≈ 2scale * v
            @test Float64(rv) == v
        end
    end

    for n in (3, 9, 17)
        alpha = collect(range(-0.3, 0.5; length = n))
        beta = collect(range(0.1, 0.4; length = n))
        data = collect(range(0.4, 1.1; length = n))
        shift, scale = 0.35, -0.4
        saved_data = copy(data)
        prepared = prepare_ad(bound_context_tuple, backend, alpha, beta;
            active = (:beta, :alpha), want = :density, bound = (; shift, scale, data))
        staged = let prepared = prepared
            (a, b) -> ad_value_and_gradient(prepared, a, b)
        end
        traced = map(Reactant.to_rarray, (alpha, beta))
        compiled = Reactant.compile(staged, traced)
        for (a, b) in ((alpha, beta), (alpha .+ 0.2, beta .- 0.3))
            saved_a, saved_b = copy(a), copy(b)
            ra, rb = map(Reactant.to_rarray, (a, b))
            value, gradient = compiled(ra, rb)
            @test Float64(value) ≈ sum(data .* (a .- shift)) + scale * sum(abs2, b)
            @test Array(gradient[1]) ≈ 2scale .* b
            @test Array(gradient[2]) ≈ data
            @test a == saved_a && b == saved_b
            @test Array(ra) == a && Array(rb) == b
            @test data == saved_data
        end
    end
end
