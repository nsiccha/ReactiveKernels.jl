# Shared Reactant test helpers: imports, extension handles, traceable
# RHS forms, reference configs, and the compiled-gradient composition.
# Included once by test_reactant.jl; test_fixedn_traced.jl includes it
# guarded so the probe file also runs standalone.

using Reactant
import Enzyme

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

function compile_reactant_gradient(closure, u0, p, which)
    # Select the output with a concrete branch at construction time so the
    # traced loss indexes the result tuple with a constant.
    pick = which == 1 ? (ys -> ys[1]) : (ys -> ys[2])
    outer = (u0i, pi, du0i, dpi) -> begin
        Enzyme.autodiff(Enzyme.Reverse,
            (a, b) -> sum(pick(closure(a, b))), Enzyme.Active,
            Enzyme.Duplicated(u0i, du0i), Enzyme.Duplicated(pi, dpi))
        (dpi, du0i)
    end
    Reactant.compile(outer,
        (TR(u0), TR(p), TR(zero.(u0)), TR(zero.(p))))
end

function compile_reactant_gradient(closure, u0, ::Nothing, which)
    pick = which == 1 ? (ys -> ys[1]) : (ys -> ys[2])
    outer = (u0i, du0i) -> begin
        Enzyme.autodiff(Enzyme.Reverse, a -> sum(pick(closure(a))),
            Enzyme.Active, Enzyme.Duplicated(u0i, du0i))
        du0i
    end
    Reactant.compile(outer, (TR(u0), TR(zero.(u0))))
end
