# Source-faithful ReactiveKernels capture of ReactiveHMC.jl src/integrators.jl
# at ca9ea4ca41924bb0e1fadc01c717e1333916aba6.
#
# The two nonseparable integrators are ordinary mutating mathematical kernels;
# their bodies below differ only by `@kernel` capture. `multistep` remains the
# original ordinary higher-order Julia wrapper because its variadic positional
# and keyword forwarding are not reactive state and must stay generic.
module ReactiveHMCIntegratorFixture

using ReactiveKernels

@kernel generalized_leapfrog!(phasepoint; stepsize, n_fi_steps) = begin
    pos0, mom0 = map(copy, (phasepoint.pos, phasepoint.mom))
    for _ in 1:n_fi_steps
        @. phasepoint.mom = mom0 - 0.5 * stepsize * phasepoint.dham_dpos
    end
    dham_dmom0 = copy(phasepoint.dham_dmom)
    for _ in 1:n_fi_steps
        @. phasepoint.pos = pos0 + 0.5 * stepsize *
            (dham_dmom0 + phasepoint.dham_dmom)
    end
    @. phasepoint.mom -= 0.5 * stepsize * phasepoint.dham_dpos
end

@kernel implicit_midpoint!(phasepoint; stepsize, n_fi_steps) = begin
    pos0, mom0 = map(copy, (phasepoint.pos, phasepoint.mom))
    for _ in 1:n_fi_steps
        (; dham_dmom, dham_dpos) = phasepoint
        @. phasepoint.pos = pos0 + 0.5 * stepsize * dham_dmom
        @. phasepoint.mom = mom0 - 0.5 * stepsize * dham_dpos
    end
    @. phasepoint.pos = 2 * phasepoint.pos - pos0
    @. phasepoint.mom = 2 * phasepoint.mom - mom0
end

multistep(f, args...; n_steps, stepsize, kwargs...) = for _ in 1:n_steps
    f(args...; stepsize=stepsize / n_steps, kwargs...)
end
multistep(f; n_steps) =
    (args...; kwargs...) -> multistep(f, args...; n_steps, kwargs...)

end # module ReactiveHMCIntegratorFixture
