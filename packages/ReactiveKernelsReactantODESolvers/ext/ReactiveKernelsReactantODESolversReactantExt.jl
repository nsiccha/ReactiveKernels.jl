# Reactant-traced adaptive Tsit5 driver.
#
# The traced program is a single `@trace while` loop over fixed-shape buffers:
# the adaptive accept/reject logic blends via `ifelse` (no traced branches),
# each step calls the standard prepared `tsit5_stage` kernel, and saveat
# emission accumulates into a tuple of per-point columns threaded by
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
#   a single-comparison condition over a pure `+1` counter, and a constant
#   bound. The three rejected shapes fail differently (each probed minimal
#   on a trivial loop): `Periodic(n)` dies at XLA lowering
#   (`stablehlo.dynamic_pad` untranslatable); a compound `(a < b) & (c < d)`
#   cond fails analysis under either scheme ("no known iteration count");
#   a select-on-IV saturating counter segfaults the Binomial reverse
#   transform (`reverseBinomial`/`popCache`). The reverse-compatible loop
#   shape is therefore a per-iteration freeze, not a second clause or a
#   saturated counter — but ONLY the reverse path needs it. Primal
#   `@trace while` lowers a compound `(n < max) & (t < t1)` cond and
#   exits early (probed minimal), so `early_exit=true` (the default) runs
#   no frozen iterations at all. Measured freeze cost (strato2,
#   2026-09-18, Reactant 0.2.285; 2-state decay converging in ~60 native
#   attempts): maxiters=80 executes in 0.10 ms, maxiters=1000 in 0.55 ms
#   — execution wall time of the freeze shape scales with the bound, not
#   the difficulty. Both shapes produce bitwise-identical values (locked
#   by the agreement test); status semantics are unchanged (exhaustion
#   still reports from the unreached `t1`).
# - never place complementary comparisons (`x <= c` and `x > c`) on one
#   traced value that can be NaN: the optimizer derives one from the
#   other, wrong for NaN. Test `isnan` explicitly instead.

module ReactiveKernelsReactantODESolversReactantExt

using ReactiveKernelsReactantODESolvers
import Reactant

const RKRO = ReactiveKernelsReactantODESolvers

function RKRO.traceable_ode_closure(f, cfg::RKRO.ReactantTsit5Config,
        u0_example::AbstractVector, p_example; early_exit::Bool=true)
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
    # The step runs through the standard prepared kernel (concrete, once):
    # prepared calls lower through Reactant via the core traced-slot
    # machinery, primal and reverse. `inv_n` likewise concrete.
    kstep = RKRO.prepare_tsit5_stage()
    inv_n = T(inv(n))

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

        # Loop shape by consumer; the two bodies below are INTENTIONALLY
        # identical (a shared step closure capturing traced values breaks
        # `@trace while` operand/block-arg matching — probed). The only
        # difference is the condition. The agreement test locks both shapes
        # to bitwise-identical values, so any future divergence fails loudly.
        #
        # The primal exits early on a compound `(n < max) & (t < t1)`
        # condition (probed: lowers and exits in primal Reactant).
        # Reverse-through-while cannot use that shape: the only `@trace
        # while` form whose Binomial-checkpointed reverse Enzyme lowers is
        # a single comparison over a pure `+1` counter and a constant
        # bound — a compound cond fails analysis ("no known iteration
        # count"), and a select-on-IV saturating counter segfaults the
        # Binomial reverse transform (`reverseBinomial`/`popCache`;
        # minimal repro filed upstream). The freeze variant therefore keeps
        # the single-comparison cond and turns arrival at `t1` into a
        # per-iteration no-op freeze, at up to `maxiters` wasted iterations
        # per solve. Status semantics are unchanged (exhaustion still
        # reports from the unreached `t1`).
        if early_exit
            Reactant.@trace checkpointing = ckpt track_numbers = false while (n_iter < maxiters_tr) & (tdir * (t1 - t) > zero_T)
            active = tdir * (t1 - t) > zero_T
            dt_use = ifelse(tdir * (t + dt - t1) > zero_T, t1 - t, dt)
            # Finite dummy step when frozen: `dt_use` is 0 there, and the
            # dense fraction `(ts - t) / dt_use` would be Inf/NaN. The
            # primal discards it via the freeze selects, but reverse-mode
            # forms 0 * non-finite partials into live shadows (NaN
            # gradients). Every update from the dummy is discarded below.
            dt_step = ifelse(active, dt_use, one_tr)
            taken_u, taken_k, taken_EEst = kstep(f, u, k1, p, t,
                dt_step, tab, atol, rtol, inv_n)
            taken = (u=taken_u, k=taken_k, EEst=taken_EEst)
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
        else
            Reactant.@trace checkpointing = ckpt track_numbers = false while n_iter < maxiters_tr
            active = tdir * (t1 - t) > zero_T
            dt_use = ifelse(tdir * (t + dt - t1) > zero_T, t1 - t, dt)
            # Finite dummy step when frozen: `dt_use` is 0 there, and the
            # dense fraction `(ts - t) / dt_use` would be Inf/NaN. The
            # primal discards it via the freeze selects, but reverse-mode
            # forms 0 * non-finite partials into live shadows (NaN
            # gradients). Every update from the dummy is discarded below.
            dt_step = ifelse(active, dt_use, one_tr)
            taken_u, taken_k, taken_EEst = kstep(f, u, k1, p, t,
                dt_step, tab, atol, rtol, inv_n)
            taken = (u=taken_u, k=taken_k, EEst=taken_EEst)
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

# Augmented backsolve RHS over `w = [u; λ; μ]`: the state re-solves `f`
# backward in time, the adjoint integrates `dλ/dt = -Jᵀλ`, and the parameter
# quadrature `dμ/dt = -(λᵀf_p)` so `μ(t0) = dL/dp`. The VJP pair comes from
# one `Enzyme.autodiff` over the loop-free RHS per stage evaluation (lowered
# to straight-line code inside the step by Reactant's Enzyme overlay — the
# same closure-in-autodiff pattern the through-while recipe uses, probed
# minimal inside a `@trace while` body before building on it). Nothing here
# differentiates through the adaptive loop. Reached through
# `Reactant.Enzyme` (identical to the `Enzyme` module object), so the
# extension needs no new dependency.
function _backsolve_rhs(f, n::Int, m::Int, ::Type{T}) where {T<:AbstractFloat}
    zT = zero(T)
    sdot = (uu, pp, ll, tt) -> sum(f(uu, pp, tt) .* ll)
    sdot_nop = (uu, ll, tt) -> sum(f(uu, nothing, tt) .* ll)
    r1 = 1:n
    r2 = (n + 1):(2n)
    EA = Reactant.Enzyme
    function aug(w, p, t)
        u = w[r1]
        lam = w[r2]
        fwd = f(u, p, t)
        jtu = u .* zT
        if m == 0
            # Concrete branch: `m` is a closure constant, never traced.
            EA.autodiff(EA.Reverse, EA.Const(sdot_nop), EA.Active,
                EA.Duplicated(u, jtu), EA.Const(lam), EA.Const(t))
            vcat(fwd, .-jtu)
        else
            jtp = p .* zT
            EA.autodiff(EA.Reverse, EA.Const(sdot), EA.Active,
                EA.Duplicated(u, jtu), EA.Duplicated(p, jtp),
                EA.Const(lam), EA.Const(t))
            vcat(fwd, .-jtu, .-jtp)
        end
    end
    aug
end

function RKRO.compile_backsolve_gradient(f, u0_example::AbstractVector,
        p_example, ::RKRO.Tsit5, cfg::RKRO.ReactantTsit5Config;
        loss::Symbol=:endpoint)
    loss === :endpoint || loss === :saveat ||
        throw(ArgumentError("loss must be :endpoint or :saveat, got $loss"))
    T = typeof(cfg.t0)
    n = length(u0_example)
    m = p_example === nothing ? 0 : length(p_example)
    forward = RKRO.compile_ode_solve(f, u0_example, p_example, RKRO.Tsit5(),
        cfg)
    aug = _backsolve_rhs(f, n, m, T)
    w_example = zeros(T, 2n + m)
    # Backward journey, latest first: `:endpoint` re-solves `t1 → t0` in one
    # segment with `λ(t1) = 1`; `:saveat` walks `t1 → … → t0` segment by
    # segment and jumps `λ` by `1` at each saveat point (sum loss). Every
    # segment is an early-exit primal; each compiles one interior saveat time
    # (a driver requirement), a midpoint column that is discarded.
    bounds = loss === :endpoint ? [cfg.t1, cfg.t0] :
        reverse!([cfg.t0; cfg.saveat; cfg.t1])
    nseg = length(bounds) - 1
    segs = map(1:nseg) do i
        a, b = bounds[i], bounds[i + 1]
        lo, hi = minmax(a, b)
        mid = (lo + hi) / 2
        lo < mid < hi || throw(ArgumentError(
            "backsolve segment ($a → $b) admits no interior saveat time"))
        bcfg = RKRO.ReactantTsit5Config((a, b); abstol=cfg.abstol,
            reltol=cfg.reltol, dt=cfg.dt_init, dtmax=cfg.dtmax,
            maxiters=cfg.maxiters, saveat=[mid])
        seg = RKRO.compile_ode_solve(aug, w_example, p_example, RKRO.Tsit5(),
            bcfg)
        (seg, loss === :saveat && i < nseg)
    end
    gr2 = (n + 1):(2n)
    gr3 = (2n + 1):(2n + m)
    function run(u0v, pv)
        ep, _, fst = pv === nothing ? forward(u0v) : forward(u0v, pv)
        fst == 0 || error("backsolve gradient needs a successful forward " *
                          "solve, got status $fst (t1 not reached)")
        w = vcat(ep, loss === :endpoint ? ones(T, n) : zeros(T, n),
            zeros(T, m))
        for (seg, jump) in segs
            wend, _, bst = pv === nothing ? seg(w) : seg(w, pv)
            bst == 0 || error("backsolve gradient: a backward segment " *
                              "failed with status $bst")
            w = wend
            jump && (w[gr2] .+= one(T))
        end
        (w[gr2], m == 0 ? nothing : w[gr3])
    end
    p_example === nothing ? (u0v,) -> run(u0v, nothing) :
        (u0v, pv) -> run(u0v, pv)
end

end # module ReactiveKernelsReactantODESolversReactantExt
