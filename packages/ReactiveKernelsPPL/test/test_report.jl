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
        @test_throws ArgumentError load_artifact(bad)
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
        @test_throws ArgumentError main(
            ["--artifact", "a", "--surface", model, "--data", data])
        @test_throws ArgumentError main(["--surface", model])
        @test_throws ArgumentError main(["--bogus"])
        write(data, "NOTDATA = 1\n")
        @test_throws ArgumentError load_surface_files(model, data)
        @test_throws ArgumentError transpile_report(
            _REPORT_SURFACE, _report_demo_data(); u = [0.0])
    end
end
