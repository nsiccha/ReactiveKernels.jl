using DifferentiationInterface
using Enzyme
using ReactiveKernelsPPL
using Serialization
using Test

# Maintained coverage for report/transpile_report.jl: the machine-generated
# Layer 3 → Layer 4 evidence behind transpile-update briefs. The demo model
# is the peer brief's n=6 gaussian in NEW spelling; the posterior string is
# asserted bit-for-bit against the peer's old-spelling number
# (-15.886646898631646), which is the independent IR-unchanged oracle.

include(joinpath(@__DIR__, "..", "report", "transpile_report.jl"))

const _REPORT_SURFACE = """
@rkppl begin
    mu_b1 ~ Normal(0.0, 1.0)
    mu_b2 ~ Normal(0.0, 1.0)
    mu = mu_b1 .+ mu_b2 .* x
    sigma ~ Exponential(1.0)
    y .~ Normal.(mu, sigma)
end
"""

_report_demo_data() = Dict{Symbol,AbstractVector}(
    :y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
    :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0])

const _REPORT_FACTOR_SURFACE = """
@rkppl begin
    c[levels(g)] .~ Normal.(0.0, 2.0)
    sigma ~ Exponential(1.0)
    mu = c[g]
    y .~ Normal.(mu, sigma)
end
"""

_report_factor_data() = Dict{Symbol,AbstractVector}(
    :y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
    :g => [1, 2, 1, 3, 2, 3])

const _REPORT_U = [0.5, -0.25, 0.1]
# Peer's old-spelling value (brief 2026-09-18T13-50-37-077-124g6vh Layer 4).
const _REPORT_POSTERIOR = "posterior(u) = -15.886646898631646"

@testset "report surface mode demo gaussian" begin
    md = transpile_report(_REPORT_SURFACE, _report_demo_data();
        meta = (; model = "demo-gaussian"), u = _REPORT_U,
        backend = _GEN_BACKEND)
    @test occursin("## Layer 3 — `@rkppl` surface", md)
    @test occursin("mu = mu_b1 .+ mu_b2 .* x", md)
    @test occursin("y .~ Normal.(mu, sigma)", md)
    @test occursin("n_obs = 6", md)
    @test occursin("scale :sigma", md)
    @test occursin("Dict(:y => :response, :x => :predictor)", md)
    @test occursin("## Layer 4 — emitted RK kernel (verbatim)", md)
    @test occursin("ppl_model(unconstrained::Vector{Float64}", md)
    # Bit-for-bit vs the peer's old-spelling oracle: same IR, same kernel.
    @test occursin(_REPORT_POSTERIOR, md)
    @test occursin("gradient cross-check: AD vs central differences", md)
    @test occursin("PASS", md)
end

@testset "report levels demo" begin
    md = transpile_report(_REPORT_FACTOR_SURFACE, _report_factor_data();
        meta = (; model = "demo-factor"), backend = _GEN_BACKEND)
    @test occursin("c[levels(g)] .~ Normal.(0.0, 2.0)", md)
    @test occursin(
        "levelmaps   = [(mu, g, values [1, 2, 3], source levels, subset :)]",
        md)
    @test occursin("(finite)", md)
    @test occursin("PASS", md)
end

@testset "report AST mode fidelity + artifact round-trip" begin
    ast = _surface_block(_REPORT_SURFACE)
    @test ast isa Expr && ast.head === :block
    md = transpile_report(ast, _report_demo_data(); u = _REPORT_U)
    @test occursin(
        "rendered Layer 3 re-parses to the input AST (modulo line numbers)", md)
    @test occursin(_REPORT_POSTERIOR, md)
    @test occursin("gradient cross-check: not run (no AD backend", md)
    mktempdir() do dir
        art = joinpath(dir, "demo.jls")
        Serialization.serialize(art,
            (; ast, data = _report_demo_data(), meta = (; model = "demo")))
        loaded = load_artifact(art)
        @test loaded.ast == ast
        md2 = transpile_report(loaded.ast, loaded.data;
            meta = loaded.meta, u = _REPORT_U)
        @test occursin(_REPORT_POSTERIOR, md2)
        # Malformed artifacts fail closed.
        bad = joinpath(dir, "bad.jls")
        Serialization.serialize(bad, (; nope = 1))
        # refused: malformed report artifact (artifact schema contract)
        @test_throws ArgumentError load_artifact(bad)
    end
end

@testset "report bound-prepare split: behavior preserved" begin
    # The multi-probe internals reproduce the single-probe numbers exactly.
    cols = _norm_cols(_report_demo_data())
    ast = _surface_block(_REPORT_SURFACE)
    bound = bind_data(lower_rkppl(ast, keys(cols)), cols)
    prep = _prepare_bound_report(bound)
    @test prep.n == 3
    pr = _run_report_probe(prep, _REPORT_U; backend = _GEN_BACKEND)
    @test repr(pr.val) == "-15.886646898631646"
    @test pr.grad_ok === true
    @test pr.grad_maxdiff isa Float64 && isfinite(pr.grad_maxdiff)
    pr0 = _run_report_probe(prep, nothing)
    @test isfinite(pr0.val)
    @test pr0.grad_ok === nothing
    @test pr0.grad_maxdiff === nothing
    md = join([
        _layer3_block("X", "f")...,
        _boundary_block(bound)...,
        _layer4_block(kernel_expr(bound, prep.built.layout))...,
        _verification_multi_block([pr])...,
    ], "\n")
    @test occursin("n_obs = 6", md)
    @test occursin("ppl_model(unconstrained::Vector{Float64}", md)
    @test occursin("- probes: 1", md)
    @test occursin("- probe 1: `posterior(u) = -15.886646898631646`", md)
end

@testset "report v2 shape: load + fail-closed without BRM" begin
    mktempdir() do dir
        ast = _surface_block(_REPORT_SURFACE)
        good = (; case_id = "demo", ast, defs = Expr[],
            plan = nothing, meta = (; case_id = "demo", provenance = "test"))
        v2 = joinpath(dir, "v2.jls")
        Serialization.serialize(v2, good)
        loaded = load_artifact(v2)
        @test _is_v2_artifact(loaded)
        @test loaded.case_id == "demo"
        @test !_is_v2_artifact((; ast, data = _report_demo_data()))
        # This env has no BRM: the translate seam fails closed with guidance.
        err = try
            transpile_report_v2(loaded)
            nothing
        catch e
            e
        end
        # refused: the v2 translation seam requires the BRM package in this environment (dependency contract).
        @test err isa ArgumentError
        @test occursin("BayesianRegressionModels", sprint(showerror, err))
        # Malformed v2 fails closed at load.
        bad = joinpath(dir, "bad.jls")
        Serialization.serialize(bad,
            (; case_id = "x", ast, defs = Expr[], plan = nothing))
        # refused: v2 artifact missing meta (artifact schema contract)
        @test_throws ArgumentError load_artifact(bad)
        Serialization.serialize(bad, (; case_id = "x", ast = "not-expr",
            defs = Expr[], plan = nothing, meta = (;)))
        # refused: v2 artifact ast is not an Expr (artifact schema contract)
        @test_throws ArgumentError load_artifact(bad)
        # CLI --artifact dispatches v2 (and fails closed without BRM here).
        # refused: v2 translate seam needs BRM, absent in this env (fails closed with guidance)
        @test_throws ArgumentError main(["--artifact", v2])
        try
            main(["--artifact", v2])
            @test false
        catch e
            # refused: the v2 translation seam requires BRM (dependency contract).
            @test occursin("BayesianRegressionModels", sprint(showerror, e))
        end
    end
end

@testset "report CLI main + input errors" begin
    mktempdir() do dir
        model = joinpath(dir, "model.jl")
        data = joinpath(dir, "data.jl")
        out = joinpath(dir, "report.md")
        write(model, _REPORT_SURFACE)
        write(data, "DATA = Dict{Symbol,AbstractVector}(\n" *
                    "    :y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],\n" *
                    "    :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0])\n")
        @test main(["--surface", model, "--data", data, "--out", out,
            "--u", "0.5,-0.25,0.1", "--model", "demo"]) == 0
        md = read(out, String)
        @test occursin(_REPORT_POSTERIOR, md)
        @test occursin("model = \"demo\"", md)
        # Bare begin/end surface also accepted.
        write(model, "begin\n    a ~ Normal(0, 1)\n    s ~ Exponential(1.0)\n    eta = a\n    y .~ Normal.(eta, s)\nend\n")
        write(data, "DATA = (; y = [1.0, 2.0])\n")
        @test main(["--surface", model, "--data", data, "--out", out]) == 0
        @test occursin("n_obs = 2", read(out, String))
        # Errors fail closed, never half a report.
        # refused: mutually exclusive CLI flags --artifact and --surface
        @test_throws ArgumentError main(
            ["--artifact", "a", "--surface", model, "--data", data])
        # refused: missing required --data argument
        @test_throws ArgumentError main(["--surface", model])
        # refused: unknown CLI flag
        @test_throws ArgumentError main(["--bogus"])
        write(data, "NOTDATA = 1\n")
        # refused: data file defines no DATA
        @test_throws ArgumentError load_surface_files(model, data)
        # refused: u length != layout dimension
        @test_throws ArgumentError transpile_report(
            _REPORT_SURFACE, _report_demo_data(); u = [0.0])
    end
end
