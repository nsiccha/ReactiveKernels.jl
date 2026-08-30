# Source-faithful ReactiveKernels translation of ReactiveHMC.jl
# src/energies.jl:2-10 at ca9ea4ca41924bb0e1fadc01c717e1333916aba6.
#
# There are exactly two authoring-boundary changes:
#   1. ReactiveHMC object authoring becomes method-bearing `@kernel`.
#   2. `lambertw` is an ordinary captured callable port rather than a package
#      global, keeping LambertW out of ReactiveKernels core and exposing the
#      generic callable-port compiler seam.
#
# The mathematical assignments and nested method bodies otherwise retain the
# upstream spelling and order.  This file is an external compiler-acceptance
# fixture, not package API.
module ReactiveHMCRKEFixture

using ReactiveKernels

@kernel rke(lambertw; m=1.0, c=1.0) = begin
    c1 = m*c^2
    c2 = (m*c)^2
    e_sq(x_sq) = c1*sqrt(x_sq/c2+1)
    p_sq(x_sq) = exp(-e_sq(__self__, x_sq))
    P_sq(x_sq) = -(2 * c2 * exp(-c1 *sqrt((c2 + x_sq)/c2)) * (1 + c1 * sqrt((c2 + x_sq)/c2)))/c1^2
    P0_sq = -(2 * c2 * exp(-c1) * (1 + c1))/c1^2
    cdf_sq(x_sq) = (P0_sq - P_sq(__self__, x_sq)) / P0_sq
    quantile_sq(q) = (c2 - c1^2 *c2 + 2* c2* lambertw(-(c1^2* P0_sq* (-1 + q))/(2* c2* exp(1)), -1) + c2* lambertw(-(c1^2 *P0_sq* (-1 + q))/(2 *c2 *exp(1)), -1)^2)/c1^2
end

end # module ReactiveHMCRKEFixture
