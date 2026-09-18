# Reactant-compiled adaptive solve: primal parity, IR shape, status flags,
# and ordinary reverse gradients through the compiled program (plain
# `Enzyme.autodiff(::Reverse, ...)` inside the traced region, lowered by
# Reactant's Enzyme overlay — no `runtime_activity`, no priming).
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

@testset "reactant extension loads" begin
    @test RExt !== nothing
end

@testset "traceable Lotka reformulation" begin
    # The vectorized form is exactly the native RHS (same operations, same
    # order, commuting exact negations), checked natively before compiling.
    for u in ([1.0, 1.0], [2.5, 0.25], [0.1, 4.0])
        @test lotka_traceable(u, nothing, 0.0) ==
              lotka_volterra(u, LOTKA_PARAMS, 0.7)
    end
end

@testset "traced config validation" begin
    good = ReactantTsit5Config((0.0, 3.0); dt=0.05, saveat=[1.0, 2.0])
    @test good.tdir == 1.0
    @test good.saveat == [1.0, 2.0]
    @test good.dtmax == 3.0
    back = ReactantTsit5Config((3.0, 0.0); dt=0.05, saveat=[2.0, 1.0])
    @test back.tdir == -1.0
    @test back.saveat == [2.0, 1.0]

    @test_throws ArgumentError ReactantTsit5Config((0.0, 3.0); dt=0.0,
        saveat=[1.0])
    @test_throws ArgumentError ReactantTsit5Config((0.0, 3.0); dt=0.05,
        saveat=[4.0])
    @test_throws ArgumentError ReactantTsit5Config((0.0, 3.0); dt=0.05,
        saveat=Float64[])
    @test_throws ArgumentError ReactantTsit5Config((0.0, 3.0); dt=0.05,
        saveat=[0.0, 3.0])
    @test_throws ArgumentError ReactantTsit5Config((0.0, 3.0); dt=0.05,
        dtmax=0.0, saveat=[1.0])
    @test_throws ArgumentError ReactantTsit5Config((0.0, 3.0); dt=0.05,
        maxiters=-1, saveat=[1.0])
end

const DECAY_R_U0 = [1.0, 2.0]
const DECAY_R_P = [0.5, 1.5]
const DECAY_R_TSPAN = (0.0, 3.0)
const DECAY_R_SAVEAT = [1.0, 2.0]
const DECAY_R_CFG = ReactantTsit5Config(DECAY_R_TSPAN; abstol=1e-10,
    reltol=1e-8, dt=0.05, maxiters=1000, saveat=DECAY_R_SAVEAT)

@testset "compiled decay primal" begin
    solved = compile_ode_solve(decay_traceable, DECAY_R_U0, DECAY_R_P, Tsit5(),
        DECAY_R_CFG)
    endpoint, smat, status = solved(DECAY_R_U0, DECAY_R_P)
    @test status == 0
    @test size(smat) == (2, 2)

    # Analytic reference (exact): decay has a closed form, so no ODE
    # saveat-semantics guessing is needed here.
    exact(t) = DECAY_R_U0 .* exp.(-DECAY_R_P .* t)
    bar = agreement_bar(1e-10, 1e-8, 2.0)
    @test max_abs_diff([endpoint], [exact(3.0)]) < bar
    @test max_abs_diff([smat[:, j] for j in 1:2],
        [exact(t) for t in DECAY_R_SAVEAT]) < bar

    # Same fixed initial step on both drivers: the formulations agree to
    # near roundoff (any divergence here is a real formulation mismatch).
    native = solve_ode(exponential_decay, DECAY_R_U0, DECAY_R_TSPAN, Tsit5();
        p=DECAY_R_P, abstol=1e-10, reltol=1e-8, dt=0.05, saveat=DECAY_R_SAVEAT)
    @test native.retcode == :Success
    @test endpoint ≈ native.u[end] rtol = 1e-9
    for (j, k) in enumerate(2:3)
        @test smat[:, j] ≈ native.u[k] rtol = 1e-9
    end
end

@testset "compiled Lotka primal" begin
    saveat = collect(0.5:0.5:9.5)
    cfg = ReactantTsit5Config(LOTKA_TSPAN; abstol=1e-10, reltol=1e-8, dt=0.01,
        maxiters=5000, saveat=saveat)
    solved = compile_ode_solve(lotka_traceable, LOTKA_U0, nothing, Tsit5(),
        cfg)
    endpoint, smat, status = solved(LOTKA_U0)
    @test status == 0
    @test size(smat) == (2, length(saveat))
    # The OrdinaryDiffEq reference saves exactly the requested points (no
    # automatic endpoints in this configuration); assert the times loudly
    # so a semantics change fails here instead of silently misaligning.
    ref_saveat = vcat(saveat, [10.0])
    ref = ode_reference(lotka_volterra, LOTKA_U0, LOTKA_TSPAN, LOTKA_PARAMS;
        saveat=ref_saveat)
    @test ref.t == ref_saveat
    bar = agreement_bar(1e-10, 1e-8, solution_scale(ref.u))
    @test max_abs_diff([endpoint], [ref.u[end]]) < bar
    cols = [smat[:, j] for j in 1:length(saveat)]
    @test max_abs_diff(cols, ref.u[1:end-1]) < bar
end

@testset "compiled program IR shape" begin
    closure = RKRO.traceable_ode_closure(decay_traceable, DECAY_R_CFG,
        DECAY_R_U0, DECAY_R_P)
    tm = Reactant.@code_hlo closure(TR(DECAY_R_U0), TR(DECAY_R_P))
    ir = String(tm)
    @test occursin("stablehlo", ir)
    # The adaptive loop lowers to a data-dependent while, not an unrolled
    # fixed-step chain.
    @test occursin("stablehlo.while", ir)
end

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

@testset "compiled endpoint gradient" begin
    # Reverse-through-while requires the freeze shape (single-comparison
    # cond); the primal early-exit cond fails Binomial analysis.
    closure = RKRO.traceable_ode_closure(decay_traceable, DECAY_R_CFG,
        DECAY_R_U0, DECAY_R_P; early_exit=false)
    grad_compiled = compile_reactant_gradient(closure, DECAY_R_U0, DECAY_R_P,
        1)
    dp_ad, du0_ad = grad_compiled(TR(DECAY_R_U0), TR(DECAY_R_P),
        TR(zero.(DECAY_R_U0)), TR(zero.(DECAY_R_P)))
    solved = compile_ode_solve(decay_traceable, DECAY_R_U0, DECAY_R_P, Tsit5(),
        DECAY_R_CFG)
    # Finite differences of the COMPILED solve: a pure pullback check that
    # shares the primal exactly (each point is a cheap execution).
    @test Array(dp_ad) ≈ central_gradient(
        p -> sum(first(solved(DECAY_R_U0, p))), DECAY_R_P) atol = 1e-6
    @test Array(du0_ad) ≈ central_gradient(
        x -> sum(first(solved(x, DECAY_R_P))), DECAY_R_U0) atol = 1e-6
end

@testset "compiled saveat gradient" begin
    # Reverse-through-while requires the freeze shape (single-comparison
    # cond); the primal early-exit cond fails Binomial analysis.
    closure = RKRO.traceable_ode_closure(decay_traceable, DECAY_R_CFG,
        DECAY_R_U0, DECAY_R_P; early_exit=false)
    grad_compiled = compile_reactant_gradient(closure, DECAY_R_U0, DECAY_R_P,
        2)
    dp_ad, du0_ad = grad_compiled(TR(DECAY_R_U0), TR(DECAY_R_P),
        TR(zero.(DECAY_R_U0)), TR(zero.(DECAY_R_P)))
    solved = compile_ode_solve(decay_traceable, DECAY_R_U0, DECAY_R_P, Tsit5(),
        DECAY_R_CFG)
    @test Array(dp_ad) ≈ central_gradient(p -> sum(solved(DECAY_R_U0, p)[2]),
        DECAY_R_P) atol = 1e-6
    @test Array(du0_ad) ≈ central_gradient(x -> sum(solved(x, DECAY_R_P)[2]),
        DECAY_R_U0) atol = 1e-6
end

@testset "backsolve endpoint gradient" begin
    # Forward compiled solve records the trajectory; the pullback re-solves
    # the augmented adjoint ODE backward in a second early-exit compiled
    # program — no differentiation through the adaptive while loop, no freeze
    # shape anywhere in this path.
    grad = compile_backsolve_gradient(decay_traceable, DECAY_R_U0, DECAY_R_P,
        Tsit5(), DECAY_R_CFG; loss=:endpoint)
    du0_ad, dp_ad = grad(DECAY_R_U0, DECAY_R_P)
    solved = compile_ode_solve(decay_traceable, DECAY_R_U0, DECAY_R_P, Tsit5(),
        DECAY_R_CFG)
    @test dp_ad ≈ central_gradient(
        p -> sum(first(solved(DECAY_R_U0, p))), DECAY_R_P) atol = 1e-6
    @test du0_ad ≈ central_gradient(
        x -> sum(first(solved(x, DECAY_R_P))), DECAY_R_U0) atol = 1e-6
    # Analytic cross-check (exact): L = Σ u0_i e^{-p_i t1}.
    t1 = DECAY_R_TSPAN[2]
    @test du0_ad ≈ exp.(-DECAY_R_P .* t1) atol = 1e-6
    @test dp_ad ≈ -t1 .* DECAY_R_U0 .* exp.(-DECAY_R_P .* t1) atol = 1e-6
end

@testset "backsolve saveat gradient" begin
    # Multi-segment backward journey with an adjoint jump at each saveat
    # point (sum loss).
    grad = compile_backsolve_gradient(decay_traceable, DECAY_R_U0, DECAY_R_P,
        Tsit5(), DECAY_R_CFG; loss=:saveat)
    du0_ad, dp_ad = grad(DECAY_R_U0, DECAY_R_P)
    solved = compile_ode_solve(decay_traceable, DECAY_R_U0, DECAY_R_P, Tsit5(),
        DECAY_R_CFG)
    @test dp_ad ≈ central_gradient(p -> sum(solved(DECAY_R_U0, p)[2]),
        DECAY_R_P) atol = 1e-6
    @test du0_ad ≈ central_gradient(x -> sum(solved(x, DECAY_R_P)[2]),
        DECAY_R_U0) atol = 1e-6
    # Analytic cross-check (exact): L = Σ_j Σ_i u0_i e^{-p_i s_j}.
    n = length(DECAY_R_U0)
    @test du0_ad ≈ [sum(exp(-DECAY_R_P[i] * s) for s in DECAY_R_SAVEAT)
        for i in 1:n] atol = 1e-6
    @test dp_ad ≈ [-DECAY_R_U0[i] *
        sum(s * exp(-DECAY_R_P[i] * s) for s in DECAY_R_SAVEAT) for i in 1:n] atol = 1e-6
end

@testset "backsolve endpoint gradient without parameters" begin
    # Nonlinear RHS, parameters closed over: no μ block, `grad_p === nothing`.
    cfg = ReactantTsit5Config(LOTKA_TSPAN; abstol=1e-10, reltol=1e-8, dt=0.01,
        maxiters=5000, saveat=[5.0])
    grad = compile_backsolve_gradient(lotka_traceable, LOTKA_U0, nothing,
        Tsit5(), cfg; loss=:endpoint)
    du0_ad, dp_ad = grad(LOTKA_U0)
    @test dp_ad === nothing
    solved = compile_ode_solve(lotka_traceable, LOTKA_U0, nothing, Tsit5(),
        cfg)
    @test du0_ad ≈ central_gradient(x -> sum(first(solved(x))), LOTKA_U0) atol = 1e-6
end

@testset "backsolve saveat gradient without parameters" begin
    # Multi-segment saveat journey on a nonlinear non-toy system, no μ block.
    # Reference: the freeze-shape through-while gradient on the same loss —
    # an independent AD path over the identical forward program. The bar is
    # the repo's tolerance-scaled convention: backsolve re-solves rather
    # than differentiating the trajectory, so optimise-vs-discretise error
    # at the ~1e-6 level is expected (measured 2.5e-6 here; central
    # differences agree with the freeze reference to 3e-8, confirming the
    # backsolve value rather than the test bar).
    cfg = ReactantTsit5Config(LOTKA_TSPAN; abstol=1e-10, reltol=1e-8, dt=0.01,
        maxiters=5000, saveat=[3.0, 6.0])
    grad = compile_backsolve_gradient(lotka_traceable, LOTKA_U0, nothing,
        Tsit5(), cfg; loss=:saveat)
    du0_ad, dp_ad = grad(LOTKA_U0)
    @test dp_ad === nothing
    closure = RKRO.traceable_ode_closure(lotka_traceable, cfg, LOTKA_U0,
        nothing; early_exit=false)
    grad_frozen = compile_reactant_gradient(closure, LOTKA_U0, nothing, 2)
    du0_fz = Array(grad_frozen(TR(LOTKA_U0), TR(zero.(LOTKA_U0))))
    bar = agreement_bar(1e-10, 1e-8, maximum(abs, du0_fz))
    @test max_abs_diff([du0_ad], [du0_fz]) < bar
end

@testset "backsolve loss validation" begin
    @test_throws ArgumentError compile_backsolve_gradient(decay_traceable,
        DECAY_R_U0, DECAY_R_P, Tsit5(), DECAY_R_CFG; loss=:bogus)
end

@testset "compiled status flags" begin
    # Exhausted iteration bound reports the last-good state, not success.
    short_cfg = ReactantTsit5Config(DECAY_R_TSPAN; abstol=1e-10, reltol=1e-8,
        dt=0.05, maxiters=2, saveat=DECAY_R_SAVEAT)
    short_solved = compile_ode_solve(decay_traceable, DECAY_R_U0, DECAY_R_P,
        Tsit5(), short_cfg)
    _, _, short_status = short_solved(DECAY_R_U0, DECAY_R_P)
    @test short_status == 1

    # NaN error estimates flag status 2 and freeze the state.
    nan_solved = compile_ode_solve((u, p, t) -> u .* NaN, DECAY_R_U0,
        DECAY_R_P, Tsit5(), DECAY_R_CFG)
    nan_end, _, nan_status = nan_solved(DECAY_R_U0, DECAY_R_P)
    @test nan_status == 2
    @test nan_end == DECAY_R_U0
end

@testset "compiled backward primal" begin
    forward = solve_ode(exponential_decay, DECAY_R_U0, DECAY_R_TSPAN, Tsit5();
        p=DECAY_R_P, abstol=1e-10, reltol=1e-8)
    u_at_3 = forward.u[end]
    cfg = ReactantTsit5Config((3.0, 0.0); abstol=1e-10, reltol=1e-8, dt=0.05,
        maxiters=1000, saveat=[2.0, 1.0])
    solved = compile_ode_solve(decay_traceable, u_at_3, DECAY_R_P, Tsit5(),
        cfg)
    endpoint, smat, status = solved(u_at_3, DECAY_R_P)
    @test status == 0
    @test max_abs_diff([endpoint], [DECAY_R_U0]) <
          agreement_bar(1e-10, 1e-8, 2.0)
    @test size(smat) == (2, 2)
end

@testset "early-exit and freeze shapes agree bit-for-bit" begin
    # The two loop shapes share step semantics by construction (identical
    # bodies); frozen iterations are exact no-ops, so skipping them must
    # not change a single bit. Any future divergence fails here loudly.
    early = Reactant.compile(
        RKRO.traceable_ode_closure(decay_traceable, DECAY_R_CFG, DECAY_R_U0,
            DECAY_R_P; early_exit=true),
        (TR(DECAY_R_U0), TR(DECAY_R_P)))
    frozen = Reactant.compile(
        RKRO.traceable_ode_closure(decay_traceable, DECAY_R_CFG, DECAY_R_U0,
            DECAY_R_P; early_exit=false),
        (TR(DECAY_R_U0), TR(DECAY_R_P)))
    for (u0, p) in ((DECAY_R_U0, DECAY_R_P), ([2.5, 0.25], [1.0, 2.0]))
        e_end, e_cols, e_stat = early(TR(u0), TR(p))
        f_end, f_cols, f_stat = frozen(TR(u0), TR(p))
        @test Array(e_end) == Array(f_end)
        @test Float64(e_stat) == Float64(f_stat)
        @test all(map((a, b) -> Array(a) == Array(b), e_cols, f_cols))
    end
end
