# Stub-counter routing fixture for the Fix C/D/E per-cell semantics (performance contract 2026-09-08).
# Exercises the ACTUAL production routing — All80Axes.native_single_eval! / hmc_rk_native! and the
# shared classify_boundary — with STUB inputs carrying CALL COUNTERS, so a BLOCKED RK cell is proven
# to leave its closure UNINVOKED (count 0), not merely to hold a string. Establishes directly the
# per-cell number-vs-diagnostic behavior + the primal→gradient PROPAGATION + the per-side boundary
# contract (with NaN / +Inf / both-fail controls).
# Run: julia --project=benchmark/all80-env benchmark/test_cell_routing_fixture.jl
include(joinpath(@__DIR__, "all80_parity.jl"))
include(joinpath(@__DIR__, "all80_axes.jl"))
using Test
using .All80Axes: native_single_eval!, hmc_rk_native!

isnum(x) = x isa Real && isfinite(x)
isdiag(x) = x isa AbstractString && !isempty(x)

mutable struct Calls; rk_primal::Int; rk_grad::Int; hmc::Int; end

# Stub `c` whose RK closures INCREMENT `calls` when actually invoked (so a blocked cell ⇒ count 0).
stubc(calls; rk_primal_diag = nothing, rk_grad_diag = nothing) = (;
    rk_primal_diag, rk_grad_diag,
    rk_primal = () -> (calls.rk_primal += 1; 1.0),
    tu_primal = () -> 1.0, stan_primal = () -> 1.0,
    rk_grad = () -> (calls.rk_grad += 1; (1.0, [1.0])),
    tu_grad = () -> (1.0, [1.0]), stan_grad = () -> (1.0, [1.0]),
    hmc_time_loop = (backend, T) -> (calls.hmc += 1; 5.0))

function route(c)
    r = Dict{String,Any}(); native_single_eval!(r, c)
    r["hmc_transitions"] = 4; hmc_rk_native!(r, c); r
end

@testset "Fix C/D per-cell routing + call counters (blocked closures NOT invoked)" begin
    # (iii) all verified ⇒ all RK cells NUMERIC; each RK closure called exactly once.
    calls = Calls(0, 0, 0); r = route(stubc(calls))
    @test isnum(r["primal_rk"]) && isnum(r["gradient_rk"]) && isnum(r["hmc_rk_native"])
    # CALLED ⇒ count > 0 (Chairmarks.@be invokes the timed closures many times; hmc once).
    @test calls.rk_primal > 0 && calls.rk_grad > 0 && calls.hmc > 0
    @test isnum(r["primal_turing"]) && isnum(r["primal_stan"]) &&
          isnum(r["gradient_turing"]) && isnum(r["gradient_stan"])

    # (ii) good primal + FAILED gradient ⇒ primal_rk NUMERIC (called once); gradient_rk + hmc DIAG (NOT called).
    calls = Calls(0, 0, 0); r = route(stubc(calls; rk_grad_diag = "gradient_rk: non-finite gradient"))
    @test isnum(r["primal_rk"]) && calls.rk_primal > 0             # primal CALLED
    @test isdiag(r["gradient_rk"]) && calls.rk_grad == 0           # gradient BLOCKED ⇒ NOT invoked
    @test isdiag(r["hmc_rk_native"]) && calls.hmc == 0
    @test isnum(r["primal_turing"]) && isnum(r["gradient_stan"])   # reference cells still numeric

    # (i) BAD primal ⇒ ALL RK cells DIAG; NO RK closure invoked (broken graph ⇒ nothing measured).
    calls = Calls(0, 0, 0); r = route(stubc(calls; rk_primal_diag = "primal_rk: not a constant offset"))
    @test isdiag(r["primal_rk"]) && isdiag(r["gradient_rk"]) && isdiag(r["hmc_rk_native"])
    @test (calls.rk_primal, calls.rk_grad, calls.hmc) == (0, 0, 0)
    @test occursin("primal", r["gradient_rk"]) && occursin("primal", r["hmc_rk_native"])
    @test isnum(r["primal_turing"]) && isnum(r["gradient_stan"])
end

@testset "Fix E per-side boundary classification (NaN / +Inf / both-fail controls)" begin
    @test classify_boundary(-Inf, -Inf, -Inf; discover = true) == (nothing, true, nothing)
    # Turing finite (GLMM) ⇒ turing invalid, RK clear; diag reports ACTUAL values (no false "finite").
    rkb, tok, tdg = classify_boundary(-Inf, -Inf, -4.7e26; discover = true)
    @test rkb === nothing && tok == false && isdiag(tdg) && occursin("-4.7", tdg)
    # +Inf and NaN Turing are ALSO non-rejections (predicate vb_t==-Inf).
    @test classify_boundary(-Inf, -Inf, Inf; discover = true)[2] == false
    @test classify_boundary(-Inf, -Inf, NaN; discover = true)[2] == false
    # RK finite (RK support defect) ⇒ rk diag, Turing clear.
    rkb, tok, tdg = classify_boundary(1.0, -Inf, -Inf; discover = true)
    @test isdiag(rkb) && tok == true && tdg === nothing
    # BOTH RK and Turing fail ⇒ BOTH diagnostics set (no false single-side assertion).
    rkb, tok, tdg = classify_boundary(1.0, -Inf, 2.0; discover = true)
    @test isdiag(rkb) && tok == false && isdiag(tdg)
    # Stan not -Inf (finite OR NaN) ⇒ ALWAYS throws (reference must reject).
    @test_throws ErrorException classify_boundary(-Inf, 1.0, -Inf; discover = true)
    @test_throws ErrorException classify_boundary(-Inf, NaN, -Inf; discover = true)
    # NON-DISCOVER fails closed on any RK/Turing boundary failure.
    @test_throws ErrorException classify_boundary(-Inf, -Inf, 1.0; discover = false)
    @test_throws ErrorException classify_boundary(1.0, -Inf, -Inf; discover = false)
end
println("CELL_ROUTING_FIXTURE_OK")
