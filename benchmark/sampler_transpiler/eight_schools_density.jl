module EightSchoolsDensity
using ReactiveKernels, ReactiveKernelsPPLExamples, Enzyme, DifferentiationInterface
const RK = ReactiveKernels
const E = ReactiveKernelsPPLExamples.EightSchoolsExample

include("density_callbacks.jl")
using .DensityCallbacks: Potential, Gradient, CallbackHandle

function build_density()
    model = E.build_eight_schools_graph()
    q = [0.0, log(5.0), zeros(8)...]
    density = prepare(model;
        have=(:unconstrained, :observations, :observation_scales), want=:posterior,
        bound=(observations=E.EIGHT_SCHOOLS_Y,
               observation_scales=E.EIGHT_SCHOOLS_SIGMA))
    ad_backend = AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const)
    ad = prepare_ad(density, ad_backend, q; active=:unconstrained)
    (; density, ad, q)
end

end
