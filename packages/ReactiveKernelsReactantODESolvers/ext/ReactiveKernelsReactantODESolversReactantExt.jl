# Reactant-traced adaptive Tsit5 driver.
#
# The traced program is a single `@trace while` loop over fixed-shape buffers:
# the adaptive accept/reject logic blends via `ifelse` (no traced branches),
# all stage math is the shared vectorized core from the parent package, and
# saveat emission accumulates into a tuple of per-point columns threaded by
# structural recursion. Hyperparameters are concrete closure constants; only
# `u0` and (optionally) `p` are traced.
#
# Verified Reactant constraints (probed against Reactant 0.2.285, recorded
# here so the next reader does not re-derive them):
# - scalar getindex/setindex! on traced arrays errors (`assertscalar`);
# - plain `if`/`while` on traced values errors; `@trace while` lowers;
# - `@trace if` with array branches misbehaves — every selection here uses
#   `ifelse` (scalar and broadcast), which lowers to `select`;
# - loop-carried scalars must derive from traced inputs (concrete literals
#   do not carry), hence the `z = sum(u0 .* 0)` traced-zero origin;
# - integer locals referenced in a `@trace while` body promote to traced
#   integers, so loop bounds/counts must not appear there at all (not even
#   via globals) — the emission below threads tuples by structural
#   recursion instead of indexing a flat buffer;
# - parametric structs captured by the loop need a bound the traced
#   promotion satisfies (`Number`, not `AbstractFloat`);
# - `Enzyme.autodiff(::Reverse, ...)` inside traced code is intercepted by
#   Reactant's overlay and lowers the pullback into the program.
# - Enzyme reverse through a `@trace while` needs `checkpointing =
#   Binomial(b)` (revolve over a fixed budget), `track_numbers = false`,
#   a single-comparison condition over a pure `+1` counter, and a
#   constant bound: `Periodic(n)` fails even on a trivial loop, a
#   compound `(a < b) & (c < d)` cond fails analysis even with `Binomial`
#   ("no known iteration count"), and a select-on-IV saturating counter
#   segfaults the Binomial reverse transform. Early exit is therefore a
#   per-iteration freeze, not a second clause or a saturated counter.
# - never place complementary comparisons (`x <= c` and `x > c`) on one
#   traced value that can be NaN: the optimizer derives one from the
#   other, wrong for NaN. Test `isnan` explicitly instead.

module ReactiveKernelsReactantODESolversReactantExt

using ReactiveKernelsReactantODESolvers
import Reactant

const RKRO = ReactiveKernelsReactantODESolvers

function RKRO.traceable_ode_closure(f, cfg::RKRO.ReactantTsit5Config,
        u0_example::AbstractVector, p_example)
    T = typeof(cfg.t0)
    eltype(u0_example) == T ||
        throw(ArgumentError("u0 element type $(eltype(u0_example)) must match config type $T"))
    n = length(u0_example)
    n >= 1 || throw(ArgumentError("state dimension must be positive"))
    if p_example !== nothing
        p_example isa AbstractVector ||
            throw(ArgumentError("traced parameters must be a vector or nothing"))
        eltype(p_example) == T || throw(ArgumentError(
            "parameter element type $(eltype(p_example)) must match config type $T"))
    end
    nsave = length(cfg.saveat)
    nsave >= 1 || throw(ArgumentError(
        "the compiled solve requires at least one interior saveat point"))

    tab = cfg.tab
    dense = cfg.dense
    t0, t1, tdir = cfg.t0, cfg.t1, cfg.tdir
    atol, rtol = cfg.abstol, cfg.reltol
    dtmax = cfg.dtmax
    maxiters = T(cfg.maxiters)
    # Checkpointing for Enzyme reverse through the traced while: it must be
    # `Binomial` (revolve over a fixed checkpoint budget), not `Periodic` —
    # probed against Reactant 0.2.285, `Periodic(n)` still fails reverse
    # with "WhileOp does not have known iteration count for cache removal",
    # and upstream tests reverse-through-while only with `Binomial`.
    # Built here — concrete, once — so the `@trace` site below splices a
    # plain value, not an expression.
    ckpt = Reactant.Binomial(8)
    saveat_tup = Tuple(cfg.saveat)
    dt_init_signed = tdir * cfg.dt_init
    one_T, zero_T = one(T), zero(T)
    two_T = one_T + one_T
    half_T = one_T / two_T
    qoldinit_T = T(RKRO.TSIT5_QOLDINIT)

    function body(u0, p)
        # Traced-zero origin: every loop-carried scalar derives from it so
        # the values stay traced (concrete literals do not carry).
        z = sum(u0 .* zero_T)
        maxiters_tr = z + maxiters
        one_tr = z + one_T
        two_tr = z + two_T
        half_tr = z + half_T
        qoldinit_tr = z + qoldinit_T
        t = z + t0
        dt = z + dt_init_signed
        # Each carried scalar gets its own traced object (`+ zero_T`): the
        # `@trace` aliasing check rejects loop-carried variables that start
        # aliased and diverge.
        qold = qoldinit_tr + zero_T
        n_iter = z + zero_T
        bad = z + zero_T
        u = u0
        k1 = f(u0, p, t)
        out = ntuple(_ -> u0 .* zero_T, nsave)

        # Single-comparison condition over a pure `+1` counter and a
        # constant bound — the only `@trace while` shape whose
        # Binomial-checkpointed reverse Enzyme lowers: a compound
        # `(a < b) & (c < d)` cond fails analysis ("no known iteration
        # count"), and a select-on-IV saturating counter segfaults the
        # Binomial reverse transform (`reverseBinomial`/`popCache`; minimal
        # repro filed upstream). Early exit is therefore a per-iteration
        # `active` freeze: once `t` reaches `t1` every update below
        # becomes an exact no-op while the counter runs out the bound.
        # Costs up to `maxiters` no-op iterations per solve; values are
        # bitwise identical to early exit.
        Reactant.@trace checkpointing = ckpt track_numbers = false while n_iter < maxiters_tr
            active = tdir * (t1 - t) > zero_T
            dt_use = ifelse(tdir * (t + dt - t1) > zero_T, t1 - t, dt)
            # Finite dummy step when frozen: `dt_use` is 0 there, and the
            # dense fraction `(ts - t) / dt_use` would be Inf/NaN. The
            # primal discards it via the freeze selects, but reverse-mode
            # forms 0 * non-finite partials into live shadows (NaN
            # gradients). Every update from the dummy is discarded below.
            dt_step = ifelse(active, dt_use, one_tr)
            taken = RKRO.tsit5_step(f, u, k1, p, t, dt_step, tab, atol, rtol)
            EEst = taken.EEst
            accept = EEst <= one_tr
            q, q11 = RKRO.pi_factors(EEst, qold)
            dt_next = min(
                ifelse(accept, RKRO.pi_accept_dt(dt_use, q),
                    RKRO.pi_reject_dt(dt_use, q11)), dtmax)
            u_next = ifelse.(accept, taken.u, u)
            k1_next = ifelse.(accept, taken.k[7], k1)
            t_next = ifelse(accept, t + dt_use, t)
            qold_next = ifelse(accept, max(EEst, qoldinit_tr), qold)
            # A rejected step with finite EEst > 1 is ordinary; a NaN
            # estimate latches `bad` (every float is <= 1, > 1, or NaN).
            # Spelled `isnan`-first deliberately: placing both `EEst <= 1`
            # and `EEst > 1` lets the optimizer derive one comparison from
            # the other, which is wrong for NaN (probed: identical NaN
            # input reads `(<=, >)` as `(false, true)` in one program and
            # `(true, false)` in another).
            bad_next = ifelse(accept, bad, ifelse(isnan(EEst), one_tr, bad))
            t_new = t + dt_use
            out = RKRO._emit_saveat_cols(out, saveat_tup, u, taken.k, t,
                dt_step, min(t, t_new), max(t, t_new), accept & active, dense)
            u = ifelse.(active, u_next, u)
            k1 = ifelse.(active, k1_next, k1)
            t = ifelse(active, t_next, t)
            dt = ifelse(active, dt_next, dt)
            qold = ifelse(active, qold_next, qold)
            n_iter = n_iter + one_tr
            bad = ifelse(active, bad_next, bad)
        end
        reached = tdir * (t1 - t) <= zero_T
        status = ifelse(bad > half_tr, two_tr,
            ifelse(reached, z, one_tr))
        (u, out, status)
    end

    p_example === nothing ? (u0,) -> body(u0, nothing) : body
end

function RKRO.compile_ode_solve(f, u0_example::AbstractVector,
        p_example, ::RKRO.Tsit5, cfg::RKRO.ReactantTsit5Config)
    closure = RKRO.traceable_ode_closure(f, cfg, u0_example, p_example)
    example_args = p_example === nothing ? (Reactant.to_rarray(u0_example),) :
        (Reactant.to_rarray(u0_example), Reactant.to_rarray(p_example))
    thunk = Reactant.compile(closure, example_args)
    function solved(args...)
        traced = map(Reactant.to_rarray, args)
        u_end, cols, status = thunk(traced...)
        (Array(u_end), hcat(map(Array, cols)...), Int(Float64(status)))
    end
    p_example === nothing ? (u0,) -> solved(u0) : (u0, p) -> solved(u0, p)
end

end # module ReactiveKernelsReactantODESolversReactantExt
