using ReactiveKernels
using Reactant
using Test
import Reactant: @compile

include(joinpath(@__DIR__, "..", "examples", "nutpie_diagonal_adaptation.jl"))
using .NutpieDiagonalAdaptationExample:
    ORACLE_INPUTS, initialize_kernel, adaptation_kernel, initial_state, advance

@testset "nutpie diagonal mathematical kernels compile without a domain adapter" begin
    init_position = Reactant.to_rarray(ORACLE_INPUTS.init_position)
    init_gradient = Reactant.to_rarray(ORACLE_INPUTS.init_gradient)
    compiled_initialize = @compile initialize_kernel(init_position, init_gradient)
    initialized = compiled_initialize(init_position, init_gradient)
    native = initial_state(ORACLE_INPUTS.init_position, ORACLE_INPUTS.init_gradient)
    @test Array(initialized[1]) ≈ native.stds
    @test Array(initialized[2]) ≈ native.inv_stds
    @test Array(initialized[3]) ≈ native.transformation_mean
    @test initialized[4] ≈ native.logdet

    # Scalar state must remain traced across repeated calls: the first output is
    # a ConcretePJRTNumber, so compiling those inputs as static host Float64s
    # would produce a one-shot thunk with the wrong second-call signature.
    device = (
        Reactant.to_rarray(native.draw_mean),
        Reactant.to_rarray(native.draw_variance),
        Reactant.to_rarray(native.grad_mean),
        Reactant.to_rarray(native.grad_variance),
        Reactant.to_rarray(native.count; track_numbers = true),
        Reactant.to_rarray(native.background_draw_mean),
        Reactant.to_rarray(native.background_draw_variance),
        Reactant.to_rarray(native.background_grad_mean),
        Reactant.to_rarray(native.background_grad_variance),
        Reactant.to_rarray(native.background_count; track_numbers = true),
        Reactant.to_rarray(native.stds),
        Reactant.to_rarray(native.inv_stds),
        Reactant.to_rarray(native.transformation_mean),
        Reactant.to_rarray(native.logdet; track_numbers = true),
        Reactant.to_rarray(native.transformation_id; track_numbers = true),
    )
    first_step = first(ORACLE_INPUTS.steps)
    first_position = Reactant.to_rarray(first_step.position)
    first_gradient = Reactant.to_rarray(first_step.gradient)
    first_good = Reactant.to_rarray(first_step.is_good; track_numbers = true)
    first_switch = Reactant.to_rarray(first_step.switch_now; track_numbers = true)
    first_adapt = Reactant.to_rarray(first_step.adapt_now; track_numbers = true)
    compiled_adaptation = @compile adaptation_kernel(
        device..., first_position, first_gradient,
        first_good, first_switch, first_adapt)

    for step in ORACLE_INPUTS.steps
        position = Reactant.to_rarray(step.position)
        gradient = Reactant.to_rarray(step.gradient)
        is_good = Reactant.to_rarray(step.is_good; track_numbers = true)
        switch_now = Reactant.to_rarray(step.switch_now; track_numbers = true)
        adapt_now = Reactant.to_rarray(step.adapt_now; track_numbers = true)
        device = compiled_adaptation(
            device..., position, gradient, is_good, switch_now, adapt_now)
        native = advance(native, step.position, step.gradient;
            is_good = step.is_good,
            switch_now = step.switch_now,
            adapt_now = step.adapt_now)
    end

    array_fields = (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13)
    scalar_fields = (5, 10, 14, 15)
    native_values = values(native)
    for index in array_fields
        @test Array(device[index]) ≈ native_values[index] rtol = 2e-15
    end
    for index in scalar_fields
        @test device[index] ≈ native_values[index] rtol = 2e-15
    end
end
