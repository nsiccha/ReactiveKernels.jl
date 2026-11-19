module ReactiveKernelsPPLEnzymePreparationExt

using ReactiveKernelsPPL
import ReactiveKernels
using DifferentiationInterface: AutoEnzyme
using Enzyme: Reverse
using PrecompileTools: @compile_workload

# Include the triggers of DI's Enzyme extension and RK's Enzyme and
# MutatingFunctions extensions so they are available during image generation.
function _prepare_workload()
    y = [0.2, -0.1, 0.4]
    backend = AutoEnzyme(; mode = Reverse)
    mixture = @rkppl begin
        m ~ Normal(0, 3)
        z[axes(y, 1)] .~ Bernoulli(0.3)
        y .~ Normal.(m .+ 2.5 .* z, 1)
    end
    plan = mixture() | (; y)
    prepare_gibbs(build_kernel(plan), plan, [0.0]; discrete = (; z = zeros(Int, length(y))), backend)

    # SMC and ordinary samplers share this preparation.
    regression = @rkppl begin
        a ~ Normal(0, 2)
        b ~ Normal(0, 1)
        y .~ Normal.(a .+ b .* x, 0.5)
    end
    plan = regression(; x = [-0.1, 0.2, 0.7]) | (; y)
    prepare_sampler(build_kernel(plan), plan, zeros(2); backend)
    # Retain Julia's compiled preparation methods. Advertising the example
    # programs to RK's runtime lookup could substitute one of their generated
    # functions for a downstream image's identical body with a different type.
    owner = @__MODULE__
    if ReactiveKernels._image_target_module() === owner &&
            Base.invokelatest(isdefined, owner, ReactiveKernels._IMAGE_PROGRAMS)
        table = Base.invokelatest(getglobal, owner, ReactiveKernels._IMAGE_PROGRAMS)
        empty!(table.functions)
    end
    return nothing
end

@compile_workload _prepare_workload()

end
