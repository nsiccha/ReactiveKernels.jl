# Host-side API for the Reactant-traced driver (implemented by the
# `ReactiveKernelsReactantODESolversReactantExt` extension).
#
# The traced solve compiles a fixed-shape program: the state dimension, the
# saveat count, the iteration bound, and every hyperparameter are compile-time
# constants. Only the initial state `u0` and (optionally) the parameter vector
# `p` are traced. Adaptivity itself — accept/reject, step-size updates, dense
# saveat emission — runs inside the traced region on traced values.

"""
    ReactantTsit5Config{T<:AbstractFloat}

Concrete hyperparameter bundle for [`compile_ode_solve`](@ref). All fields
are compile-time constants of the generated program:

- `t0`, `t1`, `tdir`: span endpoints and direction (`±1`).
- `abstol`, `reltol`: error tolerances (same scaling as [`solve_ode`](@ref)).
- `dt_init`: positive initial step magnitude (no auto-init inside the traced
  region; compute [`initial_dt`](@ref) natively on concrete inputs and pass
  it here).
- `dtmax`: maximum step; default `abs(t1 - t0)`.
- `maxiters`: static iteration bound (accepted plus rejected attempts).
- `saveat`: interior saveat times, direction-sorted (at least one; the
  endpoint state is always returned separately).
- `tab`, `dense`: the Tsit5 tableau and dense-output coefficients.

Construct with keywords:

```julia
ReactantTsit5Config(tspan; abstol=1e-6, reltol=1e-3, dt, dtmax=nothing,
                    maxiters=1_000_000, saveat)
```
"""
struct ReactantTsit5Config{T<:AbstractFloat}
    t0::T
    t1::T
    tdir::T
    abstol::T
    reltol::T
    dt_init::T
    dtmax::T
    maxiters::Int
    saveat::Vector{T}
    tab::Tsit5Tableau{T}
    dense::Tsit5DenseCoefficients{T}

    function ReactantTsit5Config{T}(t0::T, t1::T, tdir::T, abstol::T,
            reltol::T, dt_init::T, dtmax::T, maxiters::Integer,
            saveat::Vector{T}) where {T<:AbstractFloat}
        new{T}(t0, t1, tdir, abstol, reltol, dt_init, dtmax, Int(maxiters),
            saveat, Tsit5Tableau{T}(), Tsit5DenseCoefficients{T}())
    end
end

function ReactantTsit5Config(tspan; abstol::Real=1e-6, reltol::Real=1e-3,
        dt::Real, dtmax::Union{Real,Nothing}=nothing,
        maxiters::Integer=1_000_000, saveat)
    length(tspan) == 2 ||
        throw(ArgumentError("tspan must hold exactly two times"))
    t0_in, t1_in = tspan[1], tspan[2]
    (t0_in isa Real && t1_in isa Real && isfinite(t0_in) && isfinite(t1_in)) ||
        throw(ArgumentError("tspan times must be finite and real"))
    t0_in != t1_in || throw(ArgumentError("tspan endpoints must differ"))
    T = promote_type(typeof(float(t0_in)), typeof(float(t1_in)))
    T <: AbstractFloat ||
        throw(ArgumentError("tspan must resolve to a floating-point type"))
    t0, t1 = T(t0_in), T(t1_in)
    atol = T(abstol)
    rtol = T(reltol)
    (isfinite(atol) && atol >= zero(T) && isfinite(rtol) && rtol >= zero(T)) ||
        throw(ArgumentError("abstol and reltol must be finite and non-negative"))
    (dt isa Real && isfinite(dt) && dt > zero(dt)) ||
        throw(ArgumentError("dt must be finite and positive"))
    span = abs(t1 - t0)
    dtmax_T = dtmax === nothing ? span : T(dtmax)
    (isfinite(dtmax_T) && dtmax_T > zero(T)) ||
        throw(ArgumentError("dtmax must be finite and positive"))
    maxiters >= 0 || throw(ArgumentError("maxiters must be non-negative"))
    tdir = t1 > t0 ? one(T) : -one(T)
    interior = _prepare_saveat(saveat, t0, t1, tdir, T)
    isempty(interior) && throw(ArgumentError(
        "the compiled solve requires at least one interior saveat point"))
    ReactantTsit5Config{T}(t0, t1, tdir, atol, rtol, T(dt), dtmax_T, maxiters,
        interior)
end

"""
    compile_ode_solve(f, u0_example, p_example, ::Tsit5, config::ReactantTsit5Config)

Compile a fixed-shape adaptive Tsit5 solve with Reactant (requires
Reactant.jl; implemented by the package extension). `u0_example` fixes the
state dimension and element type; `p_example` is either `nothing` (the RHS
closes over concrete parameters) or an example parameter vector, which is
then traced alongside `u0`.

Returns a callable: `solved(u0)` (or `solved(u0, p)`) returns
`(endpoint, saveat_matrix, status)` with plain Julia values, where
`saveat_matrix` has one column per configured saveat point and `status` is
`0` (reached `t1`), `1` (iteration bound exhausted), or `2` (NaN error
estimate observed).

The RHS `f(u, p, t)` must be Reactant-traceable: vectorized whole-array
operations only — no scalar indexing into traced arrays, no mutation, no
branches on traced values. Hyperparameters (`dt_init`, tolerances,
`maxiters`, saveat times) are compile-time constants; the adaptive
step-size sequence itself is computed inside the program.

Gradient path: [`compile_backsolve_gradient`](@ref) is the supported
way to differentiate this solve. It re-solves the augmented adjoint ODE
backward in backward early-exit compiled programs; the only
`Enzyme.autodiff` differentiates the loop-free RHS once per stage
evaluation, so nothing differentiates through the adaptive while loop.

Direct differentiation through the solve with
`Enzyme.autodiff(::Reverse, ...)` is NOT supported: Reactant's reverse
mode cannot lower through a `while` loop whose exit is data dependent
(see [`traceable_ode_closure`](@ref) for the limitation and its
reproducer).
"""
function compile_ode_solve end

function compile_ode_solve(args...; kwargs...)
    throw(ArgumentError(
        "compile_ode_solve requires Reactant.jl to be loaded " *
        "(the Reactant extension is inactive)"))
end

"""
    compile_backsolve_gradient(f, u0_example, p_example, ::Tsit5,
                               config::ReactantTsit5Config; loss=:endpoint)

Compile a backsolve-style adjoint gradient for the fixed-shape solve
described by `config` (requires Reactant.jl; implemented by the package
extension). `u0_example` fixes the state dimension and element type;
`p_example` is either `nothing` or an example parameter vector, as in
[`compile_ode_solve`](@ref).

`loss` selects the scalar loss over the compiled forward solve:

- `:endpoint`: `sum(endpoint)`, one backward segment `t1 → t0`;
- `:saveat`: `sum(saveat_matrix)`, one backward segment per saveat interval
  with the adjoint jumping by `1` at each saveat point.

Returns a callable: `grad(u0)` (or `grad(u0, p)`) runs the forward compiled
solve, then re-solves the augmented `[u; λ; μ]` adjoint system backward in
compiled reverse-time programs, and returns `(grad_u0, grad_p)` with plain
Julia values (`grad_p === nothing` when `p_example === nothing`).

Both directions run the early-exit primal loop. The only
`Enzyme.autodiff` in this path differentiates the loop-free RHS `f` once
per stage evaluation (a vector-Jacobian product lowered to straight-line
code inside the step); nothing differentiates through the adaptive while
loop. A non-successful forward or backward solve throws an `ErrorException`
(there is no trajectory to adjoin).

That explicit backend call is an interim (decision
`2026-09-23T02-22-38-939-1gh4snu`): the right-hand-side VJP is to come from
the derivative-rule generator as graph mathematics once its slice covers
vector right-hand sides (todo `2026-09-23T03-04-45-362-1w4062g`).
"""
function compile_backsolve_gradient end

function compile_backsolve_gradient(args...; kwargs...)
    throw(ArgumentError(
        "compile_backsolve_gradient requires Reactant.jl to be loaded " *
        "(the Reactant extension is inactive)"))
end

"""
    traceable_ode_closure(f, config::ReactantTsit5Config, u0_example, p_example)

Build the raw traced closure compiled by [`compile_ode_solve`](@ref)
(requires Reactant.jl; implemented by the package extension). Takes traced
`(u0,)` or `(u0, p)` and returns traced `(endpoint, saveat_matrix, status)`.
Exposed for IR inspection (`Reactant.@code_hlo`) and custom compilation.

The program is one retained `@trace while` loop that exits lazily on
`(n < maxiters) & (t < t1)`; its body is emitted once, so the program size
is independent of `maxiters` and of the number of saveat points (dense
output accumulates into one `n × nsave` buffer through a vectorized window
mask).

Limitation: Enzyme reverse through this loop does not lower — Reactant's
reverse-mode `while` handling needs a statically known iteration count,
which a data-dependent exit cannot provide (Reactant/Enzyme-only
reproducer: `benchmark/repro_reactant_adaptive_while_reverse.jl` at the
repository root). Differentiate with
[`compile_backsolve_gradient`](@ref) instead.
"""
function traceable_ode_closure end

function traceable_ode_closure(args...; kwargs...)
    throw(ArgumentError(
        "traceable_ode_closure requires Reactant.jl to be loaded " *
        "(the Reactant extension is inactive)"))
end
