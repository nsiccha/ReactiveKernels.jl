module ReactiveKernelsPPLEnzymePreparationExt

using ReactiveKernelsPPL
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
    return nothing
end

@compile_workload _prepare_workload()

end
