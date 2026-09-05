module FusedADReactantTests
using ReactiveKernels, Reactant, Test
import Enzyme
using DifferentiationInterface: AutoEnzyme

@kernel objective(q::Vector{Float64}, data::Vector{Float64}) = begin
    density = sum(q .* data) - sum(abs2, q) / 2
end

@kernel gradient_step(prepared, q::Vector{Float64}, data::Vector{Float64}) = begin
    (value, gradient) = ad_value_and_gradient(prepared, q, data)
    next_position = q .+ 0.1 .* gradient
    return (value, next_position)
end

@testset "prepared AD fuses into an enclosing kernel" begin
    q, data = [0.3, -0.4], [2.0, -1.0]
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    prepared = prepare_ad(prepare(objective), backend, q, data; active=:q)
    step = prepare(gradient_step)
    rq, rd = Reactant.to_rarray(q), Reactant.to_rarray(data)
    compiled = Reactant.compile(step, (prepared, rq, rd); sync=true)
    # Independent analytic derivatives, including a changed inactive input.
    # No numerical agreement between backend trajectories is required.
    for values in (data, [4.0, 3.0])
        value, next = compiled(prepared, rq, Reactant.to_rarray(values))
        @test Float64(value) ≈ sum(q .* values) - sum(abs2, q) / 2
        @test Array(next) ≈ q .+ 0.1 .* (values .- q)
    end
    native_value, native_gradient = ad_value_and_gradient(prepared, q, data)
    @test native_value ≈ sum(q .* data) - sum(abs2, q) / 2
    @test native_gradient ≈ data .- q
    @test Array(rq) == q

    bound = prepare_ad(objective, backend, q;
        active=:q, want=:density, bound=(; data))
    bound_operation = let bound=bound
        point -> begin
            value, gradient = ad_value_and_gradient(bound, point)
            (value, point .+ 0.1 .* gradient)
        end
    end
    bound_compiled = Reactant.compile(bound_operation, (rq,); sync=true)
    value, next = bound_compiled(rq)
    @test Float64(value) ≈ sum(q .* data) - sum(abs2, q) / 2
    @test Array(next) ≈ q .+ 0.1 .* (data .- q)
end
end
