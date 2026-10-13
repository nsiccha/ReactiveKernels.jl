# Standalone Enzyme reproducer: the intermediate reverse multiplication by
# interval width overflows although the final derivative fits in Float64.
# Run with an environment containing Enzyme; no ReactiveKernels code is used.
using Enzyme

function bounded_poisson_log(u)
    beta = -10.0 + 20.0 / (1.0 + exp(-u))
    eta = -1000.0 * beta
    return eta - exp(eta)
end

u = log(0.465 / 0.535)
value = bounded_poisson_log(u)
gradient = only(only(Enzyme.autodiff(Enzyme.Reverse, bounded_poisson_log,
    Enzyme.Active, Enzyme.Active(u))))
rate = exp(700.0)
expected = (1.0 - rate) * (-1000.0) * (20.0 * 0.465 * 0.535)
println((; value, gradient, expected))
