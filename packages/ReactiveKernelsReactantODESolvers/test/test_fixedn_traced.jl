# Traced fixed-N solve: unrolled-IR shape, primal parity with the native
# driver on guard-clean solves, and the through-reverse probe (plain
# Enzyme reverse over the unrolled program, checked against finite
# differences of the compiled solve).
#
# No emergency retries exist in the traced driver (a poisoned or
# guard-exceeding step latches failure and freezes the rest), so parity
# is asserted only where native takes no retries (locked per case).

# Guarded: the full suite defines these via test_reactant.jl already.
isdefined(@__MODULE__, :RExt) || include("test_reactant_helpers.jl")

const FIXEDN_R_CFG = ReactantTsit5Config(DECAY_R_TSPAN; abstol=1e-6,
    reltol=1e-3, dt=0.05, maxiters=25, saveat=DECAY_R_SAVEAT)

const LOTKA_R_TSPAN = (0.0, 10.0)
const LOTKA_R_SAVEAT = [5.0]
const LOTKA_R_DT = initial_dt(lotka_volterra, LOTKA_U0, LOTKA_R_TSPAN;
    p=LOTKA_PARAMS)
const LOTKA_R_CFG = ReactantTsit5Config(LOTKA_R_TSPAN; abstol=1e-6,
    reltol=1e-3, dt=LOTKA_R_DT, maxiters=40, saveat=LOTKA_R_SAVEAT)

@testset "fixed-N IR is unrolled" begin
    closure = RKRO.traceable_fixedn_closure(decay_traceable, FIXEDN_R_CFG,
        DECAY_R_U0, DECAY_R_P)
    tm = Reactant.@code_hlo closure(TR(DECAY_R_U0), TR(DECAY_R_P))
    ir = String(tm)
    @test occursin("stablehlo", ir)
    # Static bound: all N steps unroll, no adaptive while op survives.
    @test !occursin("stablehlo.while", ir)
end

@testset "traced fixed-N primal parity" begin
    native_d = solve_fixed_n(exponential_decay, DECAY_R_U0, DECAY_R_TSPAN,
        25, Tsit5(); p=DECAY_R_P, abstol=1e-6, reltol=1e-3, dt=0.05,
        saveat=DECAY_R_SAVEAT)
    @test native_d.retcode == :Success
    @test native_d.stats.nrejected == 0
    solved = compile_fixedn_solve(decay_traceable, DECAY_R_U0, DECAY_R_P,
        Tsit5(), FIXEDN_R_CFG)
    ep, smat, st = solved(DECAY_R_U0, DECAY_R_P)
    @test st == 0
    @test ep ≈ native_d.u[end] atol = 1e-10
    for (j, ts) in enumerate(DECAY_R_SAVEAT)
        @test smat[:, j] ≈ native_d.u[1 + j] atol = 1e-10
    end

    native_l = solve_fixed_n(lotka_volterra, LOTKA_U0, LOTKA_R_TSPAN, 40,
        Tsit5(); p=LOTKA_PARAMS, dt=LOTKA_R_DT, saveat=LOTKA_R_SAVEAT)
    @test native_l.retcode == :Success
    @test native_l.stats.nrejected == 0
    solved_l = compile_fixedn_solve(lotka_traceable, LOTKA_U0, nothing,
        Tsit5(), LOTKA_R_CFG)
    ep_l, smat_l, st_l = solved_l(LOTKA_U0)
    @test st_l == 0
    @test ep_l ≈ native_l.u[end] atol = 1e-8
    @test smat_l[:, 1] ≈ native_l.u[2] atol = 1e-8
end

@testset "fixed-N through-reverse probe (small)" begin
    closure = RKRO.traceable_fixedn_closure(decay_traceable, FIXEDN_R_CFG,
        DECAY_R_U0, DECAY_R_P)
    grad_compiled = compile_reactant_gradient(closure, DECAY_R_U0,
        DECAY_R_P, 1)
    dp_ad, du0_ad = grad_compiled(TR(DECAY_R_U0), TR(DECAY_R_P),
        TR(zero.(DECAY_R_U0)), TR(zero.(DECAY_R_P)))
    solved = compile_fixedn_solve(decay_traceable, DECAY_R_U0, DECAY_R_P,
        Tsit5(), FIXEDN_R_CFG)
    @test Array(dp_ad) ≈ central_gradient(
        p -> sum(first(solved(DECAY_R_U0, p))), DECAY_R_P) atol = 1e-6
    @test Array(du0_ad) ≈ central_gradient(
        x -> sum(first(solved(x, DECAY_R_P))), DECAY_R_U0) atol = 1e-6
end

@testset "fixed-N through-reverse probe (medium)" begin
    closure = RKRO.traceable_fixedn_closure(lotka_traceable, LOTKA_R_CFG,
        LOTKA_U0, nothing)
    grad_compiled = compile_reactant_gradient(closure, LOTKA_U0, nothing, 1)
    du0_ad = grad_compiled(TR(LOTKA_U0), TR(zero.(LOTKA_U0)))
    solved = compile_fixedn_solve(lotka_traceable, LOTKA_U0, nothing,
        Tsit5(), LOTKA_R_CFG)
    # h=1e-6, not the default 1e-8: each solve carries ~1e-12 of
    # h-independent rounding noise (40 unrolled steps on a
    # non-contractive system), so h=1e-8 resolves noise, not slope
    # (measured natively: FD(h=1e-8) errs 7e-5 vs Enzyme AD, FD(h=1e-6)
    # errs 3e-7). The compiled pullback matches native Enzyme AD to 2e-9.
    @test Array(du0_ad) ≈ central_gradient(
        x -> sum(first(solved(x))), LOTKA_U0; h=1e-6) atol = 1e-5
end

@testset "fixed-N traced validation" begin
    bad = ReactantTsit5Config(DECAY_R_TSPAN; dt=0.05, maxiters=0,
        saveat=DECAY_R_SAVEAT)
    @test_throws ArgumentError RKRO.traceable_fixedn_closure(decay_traceable,
        bad, DECAY_R_U0, DECAY_R_P)
end
