# Shared Reactant test helpers: imports, extension handles, traceable
# RHS forms and reference configs. Included once by test_reactant.jl.

using Reactant

const RExt = Base.get_extension(ReactiveKernelsReactantODESolvers,
    :ReactiveKernelsReactantODESolversReactantExt)
const TR = Reactant.to_rarray

# Traceable RHS: vectorized whole-array operations only (no scalar indexing
# into traced arrays, no mutation, no branches on traced values).
decay_traceable(u, p, t) = -p .* u

function lotka_traceable(u, p, t)
    # NOTE: `reverse(u)` is avoided deliberately — Reactant 0.2.285 lowers it
    # through an in-place method that reverses the input buffer as well
    # (`TracedRArray.jl`: `reverse(v, start, stop)` mutates `v`). The slice
    # below is pure and verified input-preserving under tracing.
    r = u[end:-1:1]
    [1.5, -3.0] .* u .+ [-1.0, 1.0] .* u .* r
end

const DECAY_R_U0 = [1.0, 2.0]
const DECAY_R_P = [0.5, 1.5]
const DECAY_R_TSPAN = (0.0, 3.0)
const DECAY_R_SAVEAT = [1.0, 2.0]
const DECAY_R_CFG = ReactantTsit5Config(DECAY_R_TSPAN; abstol=1e-10,
    reltol=1e-8, dt=0.05, maxiters=1000, saveat=DECAY_R_SAVEAT)
