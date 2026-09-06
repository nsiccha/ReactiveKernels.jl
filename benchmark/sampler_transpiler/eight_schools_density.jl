module EightSchoolsDensity
using ReactiveKernels, ReactiveKernelsPPLExamples, Enzyme, DifferentiationInterface
const RK = ReactiveKernels
const E = ReactiveKernelsPPLExamples.EightSchoolsExample

struct Potential{K} <: Function
    kernel::K
end
(f::Potential)(q) = -f.kernel(q)
struct Gradient{A} <: Function
    ad::A
end
function (f::Gradient)(q)
    value, gradient = RK.ad_value_and_gradient(f.ad, q)
    -value, -gradient
end
function (f::Gradient)(destination, q)
    value, _ = RK.ad_value_and_gradient!(f.ad, destination, q)
    destination .*= -1
    -value
end

# Prototype of compiler-owned callback storage: numerical state carries a
# typed handle instead of copying a prepared program's metadata by value.
# The captured callback is initialized once and never replaced.
struct CallbackHandle{F} <: Function
    storage::Base.RefValue{F}
end
CallbackHandle(f::F) where {F} = CallbackHandle{F}(Ref(f))
(f::CallbackHandle)(args...) = getfield(f, :storage)[](args...)

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
