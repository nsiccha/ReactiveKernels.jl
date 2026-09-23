# Reactant-traced adaptive Tsit5 driver.
#
# The traced program is a single `@trace while` loop over fixed-shape buffers
# that exits as soon as the span is covered: the adaptive accept/reject
# logic blends via `ifelse` (a selection between two valid values), each
# step calls the standard prepared `tsit5_stage` kernel, and saveat emission
# updates one `n × nsave` dense buffer with a vectorized window mask.
# Hyperparameters are concrete closure constants; only `u0` and (optionally)
# `p` are traced. The program size is independent of `maxiters` and of the
# number of saveat points (locked by the test suite).
#
# Verified Reactant constraints (probed against Reactant 0.2.285, recorded
# here so the next reader does not re-derive them):
# - scalar getindex/setindex! on traced arrays errors (`assertscalar`);
# - plain `if`/`while` on traced values errors; `@trace while` lowers;
# - `@trace if` with array branches misbehaves — every selection here uses
#   `ifelse` (scalar and broadcast), which lowers to `select`;
# - loop-carried scalars must derive from traced inputs (concrete literals
#   do not carry), hence the `z = sum(u0 .* 0)` traced-zero origin;
# - a shared step closure capturing traced values breaks `@trace while`
#   operand/block-arg matching, so the step body is spelled inline;
# - parametric structs captured by the loop need a bound the traced
#   promotion satisfies (`Number`, not `AbstractFloat`);
# - a `DerivativeRule` reverse cut is plain graph mathematics, so it traces
#   inside the step like any other array expression;
# - never place complementary comparisons (`x <= c` and `x > c`) on one
#   traced value that can be NaN: the optimizer derives one from the
#   other, wrong for NaN. Test `isnan` explicitly instead.
#
# Known limitation (upstream): Enzyme reverse THROUGH the retained
# `@trace while` does not lower. The loop's exit condition is data
# dependent (`(n < maxiters) & (t < t1)`), and Reactant's reverse-mode
# while handling requires a statically known iteration count (or a
# single-comparison counter with `Binomial` checkpointing; the compound
# condition fails with "no known iteration count"). Reactant/Enzyme-only
# reproducer: `benchmark/repro_reactant_adaptive_while_reverse.jl` at the
# repository root. The supported gradient path is the backsolve adjoint
# (`compile_backsolve_gradient`), whose right-hand-side VJP is the authored
# reverse cut of a `DerivativeRule`, evaluated inside the step: nothing
# differentiates anything. The former workarounds — a fixed-N straight-line
# unroll and post-exit dummy-`dt` masked iterations that kept a
# single-comparison loop shape alive — were removed: both are the shapes
# `docs/src/constraints.md` forbids.

module ReactiveKernelsReactantODESolversReactantExt

using ReactiveKernelsReactantODESolvers
import Reactant
import ReactiveKernels

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
    saveat_vec = copy(cfg.saveat)
    dt_init_signed = tdir * cfg.dt_init
    one_T, zero_T = one(T), zero(T)
    two_T = one_T + one_T
    half_T = one_T / two_T
    qoldinit_T = T(RKRO.TSIT5_QOLDINIT)
    # The step runs through the standard prepared kernel (concrete, once):
    # prepared calls lower through Reactant via the core traced-slot
    # machinery. `inv_n` likewise concrete.
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
        # The saveat times as one traced vector (read-only inside the loop)
        # and the dense buffer as one `n × nsave` matrix carried through it.
        saveat_tr = z .+ saveat_vec
        out = u0 .* reshape(saveat_tr .* zero_T, 1, nsave)

        # One retained loop with lazy early exit: the body runs only while
        # the span is uncovered, so no iteration is ever a masked no-op.
        Reactant.@trace track_numbers = false while (n_iter < maxiters_tr) & (tdir * (t1 - t) > zero_T)
            dt_use = ifelse(tdir * (t + dt - t1) > zero_T, t1 - t, dt)
            taken_u, taken_k, taken_EEst = kstep(f, u, k1, p, t, dt_use, tab,
                atol, rtol, inv_n)
            EEst = taken_EEst
            accept = EEst <= one_tr
            q, q11 = RKRO.pi_factors(EEst, qold)
            dt_next = min(
                ifelse(accept, RKRO.pi_accept_dt(dt_use, q),
                    RKRO.pi_reject_dt(dt_use, q11)), dtmax)
            t_new = t + dt_use
            out = RKRO._emit_saveat(out, saveat_tr, u, taken_k, t, dt_use,
                min(t, t_new), max(t, t_new), accept, dense)
            u = ifelse.(accept, taken_u, u)
            k1 = ifelse.(accept, taken_k[7], k1)
            t = ifelse(accept, t_new, t)
            dt = dt_next
            qold = ifelse(accept, max(EEst, qoldinit_tr), qold)
            # A rejected step with finite EEst > 1 is ordinary; a NaN
            # estimate latches `bad` (every float is <= 1, > 1, or NaN).
            # Spelled `isnan`-first deliberately: placing both `EEst <= 1`
            # and `EEst > 1` lets the optimizer derive one comparison from
            # the other, which is wrong for NaN (probed: identical NaN
            # input reads `(<=, >)` as `(false, true)` in one program and
            # `(true, false)` in another).
            bad = ifelse(accept, bad, ifelse(isnan(EEst), one_tr, bad))
            n_iter = n_iter + one_tr
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
        u_end, smat, status = thunk(traced...)
        (Array(u_end), Array(smat), Int(Float64(status)))
    end
    p_example === nothing ? (u0,) -> solved(u0) : (u0, p) -> solved(u0, p)
end

# Augmented backsolve RHS over `w = [u; λ; μ]`: the state re-solves the
# right-hand side backward in time, the adjoint integrates `dλ/dt = -Jᵀλ`,
# and the parameter quadrature `dμ/dt = -(λᵀf_p)` so `μ(t0) = dL/dp`. The VJP
# pair `(Jᵀλ, f_pᵀλ)` is the right-hand side's authored reverse cut — the
# rule's graph owns that mathematics (ReactiveKernels `derivative_rule`) — so
# the augmented system is ordinary graph arithmetic inside the step and
# nothing differentiates anything, neither the adaptive loop nor the RHS.
# With no parameter block the rule takes `(u, t)`; with one it takes
# `(u, p, t)`; `t` is never active, so only the `u` (and `p`) cotangents are
# selected.
function _backsolve_rhs(rule::ReactiveKernels.DerivativeRule, n::Int, m::Int,
        ::Type{T}) where {T<:AbstractFloat}
    r1 = 1:n
    r2 = (n + 1):(2n)
    if m == 0
        function aug_nop(w, p, t)
            u = w[r1]
            lam = w[r2]
            fwd = rule(u, t)
            (u_bar,) = ReactiveKernels.reverse_cut(rule, Val(1), u, t, lam)
            vcat(fwd, .-u_bar)
        end
        return aug_nop
    end
    function aug(w, p, t)
        u = w[r1]
        lam = w[r2]
        fwd = rule(u, p, t)
        u_bar, p_bar = ReactiveKernels.reverse_cut(rule, Val(3), u, p, t, lam)
        vcat(fwd, .-u_bar, .-p_bar)
    end
    aug
end

# The right-hand side of a backsolve gradient must own its reverse
# mathematics: a plain function has nothing to hand the adjoint.
function _backsolve_rule(f, m::Int)
    f isa ReactiveKernels.DerivativeRule || throw(ArgumentError(
        "compile_backsolve_gradient needs a ReactiveKernels.DerivativeRule " *
        "right-hand side (one graph authoring du and the u/p cotangents of " *
        "λᵀ du; see `derivative_rule`), got $(typeof(f)); a plain function " *
        "cannot supply the adjoint's vector-Jacobian products"))
    ReactiveKernels.has_reverse_branch(f) || throw(ArgumentError(
        "compile_backsolve_gradient: $(f) has no reverse branch (covector + " *
        "cotangents) to supply the adjoint's vector-Jacobian products"))
    expected = m == 0 ? 2 : 3
    inputs = ReactiveKernels.rule_inputs(f)
    length(inputs) == expected || throw(ArgumentError(
        "compile_backsolve_gradient: the right-hand-side rule must take " *
        (m == 0 ? "(u, t)" : "(u, p, t)") * " but $(f) takes $(inputs)"))
    f
end

function RKRO.compile_backsolve_gradient(f, u0_example::AbstractVector,
        p_example, ::RKRO.Tsit5, cfg::RKRO.ReactantTsit5Config;
        loss::Symbol=:endpoint)
    loss === :endpoint || loss === :saveat ||
        throw(ArgumentError("loss must be :endpoint or :saveat, got $loss"))
    T = typeof(cfg.t0)
    n = length(u0_example)
    m = p_example === nothing ? 0 : length(p_example)
    rule = _backsolve_rule(f, m)
    # The forward solve takes the driver's `(u, p, t)` shape; a parameter-free
    # rule takes `(u, t)`.
    forward_rhs = m == 0 ? (u, p, t) -> rule(u, t) : rule
    forward = RKRO.compile_ode_solve(forward_rhs, u0_example, p_example,
        RKRO.Tsit5(), cfg)
    aug = _backsolve_rhs(rule, n, m, T)
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
