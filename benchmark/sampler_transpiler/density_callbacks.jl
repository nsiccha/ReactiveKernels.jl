module DensityCallbacks
using ReactiveKernels

# Shared sampler adapters. Context is fixed for a prepared sampler, while q is
# active. The density itself can retain runtime inputs (e.g. centeredness).
struct Potential{K,C} <: Function
    kernel::K
    context::C
end
Potential(kernel) = Potential(kernel, ())
(f::Potential)(q) = -f.kernel(q, f.context...)

struct Gradient{A,C} <: Function
    ad::A
    context::C
end
Gradient(ad) = Gradient(ad, ())
function (f::Gradient)(q)
    value, gradient = ad_value_and_gradient(f.ad, q, f.context...)
    -value, -gradient
end
function (f::Gradient)(destination, q)
    value, _ = ad_value_and_gradient!(f.ad, destination, q, f.context...)
    destination .*= -1
    -value
end

# Numerical state retains a typed handle rather than copying prepared-program
# metadata. The captured callback is initialized once and never replaced.
struct CallbackHandle{F} <: Function
    storage::Base.RefValue{F}
end
CallbackHandle(f::F) where {F} = CallbackHandle{F}(Ref(f))
(f::CallbackHandle)(args...) = getfield(f, :storage)[](args...)
end
