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

# Right-hand sides for the backsolve adjoint: one graph each authoring `du`
# and the cotangents of `λᵀ du` (the rule constraint on
# docs/src/constraints.md); `t` is a port the mathematics does not use.
using ReactiveKernels: @kernel, derivative_rule

@kernel decay_graph(u::Vector{Float64}, p::Vector{Float64}, t::Float64,
        du_bar::Vector{Float64}) = begin
    du::Vector{Float64} = -p .* u
    u_bar::Vector{Float64} = -p .* du_bar
    p_bar::Vector{Float64} = -u .* du_bar
    t_bar::Float64 = 0.0
    return du, u_bar, p_bar, t_bar
end
const decay_rule = derivative_rule(decay_graph; primal = :du,
    covector = :du_bar, cotangents = (u = :u_bar, p = :p_bar, t = :t_bar),
    name = :decay)

@kernel lotka_graph(u::Vector{Float64}, t::Float64,
        du_bar::Vector{Float64}) = begin
    r::Vector{Float64} = u[end:-1:1]
    du::Vector{Float64} = [1.5, -3.0] .* u .+ [-1.0, 1.0] .* u .* r
    # d(λᵀdu)/du: the diagonal terms plus the swapped cross terms.
    cross_bar::Vector{Float64} = [-1.0, 1.0] .* u .* du_bar
    u_bar::Vector{Float64} = [1.5, -3.0] .* du_bar .+ [-1.0, 1.0] .* r .* du_bar .+
        cross_bar[end:-1:1]
    t_bar::Float64 = 0.0
    return du, u_bar, t_bar
end
const lotka_rule = derivative_rule(lotka_graph; primal = :du,
    covector = :du_bar, cotangents = (u = :u_bar, t = :t_bar), name = :lotka)

const DECAY_R_U0 = [1.0, 2.0]
const DECAY_R_P = [0.5, 1.5]
const DECAY_R_TSPAN = (0.0, 3.0)
const DECAY_R_SAVEAT = [1.0, 2.0]
const DECAY_R_CFG = ReactantTsit5Config(DECAY_R_TSPAN; abstol=1e-10,
    reltol=1e-8, dt=0.05, maxiters=1000, saveat=DECAY_R_SAVEAT)
