using DifferentiationInterface
using Enzyme
using ReactiveKernelsPPL
using Serialization
using Test
using TOML

# Coverage for report/sweep_appends.jl: the fusion sweep driver over
# report/transpile_report.jl. Fixtures under test/sweep_appends/ are tiny
# (n ≤ 6) and self-contained; brm-probe cases run against stub_worker.jl
# (stdlib-only, no BRM, no Stan). No full-corpus data, no BridgeStan, no XLA.
# (`_GEN_BACKEND` comes from test_generator.jl, included first.)

include(joinpath(@__DIR__, "..", "report", "sweep_appends.jl"))

const _SWEEP_FIXTURES = joinpath(@__DIR__, "sweep_appends", "manifest.toml")
const _SWEEP_FIXDIR = joinpath(@__DIR__, "sweep_appends")
const _SWEEP_STUB = joinpath(_SWEEP_FIXDIR, "stub_worker.jl")
const _SWEEP_IDS =
    ["corpus/demo_gaussian", "corpus/demo_factor", "corpus/demo_poisson"]
# Peer's old-spelling value (brief 2026-09-18T13-50-37-077-124g6vh Layer 4),
# same independent IR-unchanged oracle as test_report.jl.
const _SWEEP_POSTERIOR = "posterior(u) = -15.886646898631646"

_toml_str(s) = "\"" * replace(s, "\\" => "\\\\", "\"" => "\\\"") * "\""

function _stub_worker_argv(extra::Vector{String} = String[])
    return ["\"$(replace(joinpath(Sys.BINDIR, "julia"), "\\" => "\\\\"))\"",
        "\"--startup-file=no\"", _toml_str(_SWEEP_STUB),
        [_toml_str(a) for a in extra]...]
end

function _write_brm_manifest(dir; id = "stubdemo",
        u_probes = "[[0.0]]", expected = "[-100.0]",
        extra_worker_args = String[], extra_case = "")
    mani = joinpath(dir, "brm.toml")
    worker = "[" * join(_stub_worker_argv(extra_worker_args), ", ") * "]"
    body = """
        [meta]
        pins = { rk = "fixture" }

        [[case]]
        id = $(_toml_str(id))
        kind = "brm-probe"
        brm_inputs = "stub:demo"
        worker = $worker
        """
    u_probes !== nothing && (body *= "u_probes = $u_probes\n")
    expected !== nothing && (body *= "expected = $expected\n")
    body *= extra_case
    write(mani, body)
    return mani
end

@testset "sweep happy path (fixtures, Enzyme backend)" begin
    mktempdir() do dir
        res = sweep_appends(_SWEEP_FIXTURES, dir; backend = _GEN_BACKEND)
        @test res.ok
        @test [r.id for r in res.results] == _SWEEP_IDS
        @test all(r -> r.verdict == "VERIFY_OK", res.results)
        @test all(r -> length(r.probes) == 1, res.results)
        @test all(r -> r.probes[1].grad == "PASS", res.results)
        @test res.results[1].probes[1].delta == 0.0
        @test isnan(res.results[2].probes[1].delta)
        # Slash ids mirror the hierarchy under the output dir.
        for id in _SWEEP_IDS
            @test isfile(joinpath(dir, "$id.md"))
            @test isfile(joinpath(dir, "$id.jls"))
        end
        md = read(joinpath(dir, "corpus/demo_gaussian.md"), String)
        @test occursin("# Closeout: corpus/demo_gaussian", md)
        @test occursin("kind = \"surface\"", md)
        @test occursin("generator_version = 1", md)
        @test occursin("rk = \"fixture\"", md)
        @test occursin("brm = \"fixture\"", md)
        @test occursin("## Layer 3 — `@rkppl` surface", md)
        @test occursin("## Boundary — `lower_rkppl` → `bind_data`", md)
        @test occursin("## Layer 4 — emitted RK kernel (verbatim)", md)
        @test occursin("## Verification", md)
        @test occursin("- probes: 1", md)
        @test occursin("## Re-sweep verdict", md)
        @test occursin(_SWEEP_POSTERIOR, md)
        @test occursin("Δ = 0.0 MATCH", md)
        @test occursin("case verdict: VERIFY_OK", md)
        # No supersedes line unless the manifest carries the ref.
        @test !occursin("Supersedes:", md)
        # The sweep artifact reloads and re-renders the same posterior.
        art = load_artifact(joinpath(dir, "corpus/demo_gaussian.jls"))
        md2 = transpile_report(art.ast, art.data; meta = art.meta,
            u = [0.5, -0.25, 0.1])
        @test occursin(_SWEEP_POSTERIOR, md2)
        # Machine-readable index.
        index = TOML.parsefile(joinpath(dir, "manifest.toml"))
        @test index["generator"]["name"] == "sweep_appends"
        @test index["generator"]["version"] == 1
        @test index["pins"] == Dict("rk" => "fixture", "brm" => "fixture")
        @test [c["id"] for c in index["case"]] == _SWEEP_IDS
        @test index["case"][1]["kind"] == "surface"
        @test index["case"][1]["file"] == "corpus/demo_gaussian.md"
        @test index["case"][1]["artifact"] == "corpus/demo_gaussian.jls"
        @test index["case"][1]["verdict"] == "VERIFY_OK"
        @test index["case"][1]["probe"][1]["posterior"] == -15.886646898631646
        @test index["case"][1]["probe"][1]["expected"] == -15.886646898631646
        @test index["case"][1]["probe"][1]["delta"] == 0.0
        @test index["case"][1]["probe"][1]["grad"] == "PASS"
        @test !haskey(index["case"][2]["probe"][1], "expected")
        # Verdict table: one row per probe.
        tab = read(joinpath(dir, "verdict-table.md"), String)
        @test occursin(
            "| corpus/demo_gaussian | 1 | surface | -15.886646898631646 " *
                "| -15.886646898631646 | 0.0 | PASS | — | VERIFY_OK |",
            tab)
        @test occursin("corpus/demo_factor", tab)
        @test occursin("corpus/demo_poisson", tab)
        @test occursin("cases = 3 (VERIFY_OK 3), probes = 3", tab)
    end
end

@testset "sweep multi-probe surface" begin
    mktempdir() do dir
        mani = joinpath(dir, "mani.toml")
        write(mani, """
            [[case]]
            id = "multi"
            kind = "surface"
            surface = $(_toml_str(joinpath(_SWEEP_FIXDIR, "demo_gaussian.jl")))
            data = $(_toml_str(joinpath(_SWEEP_FIXDIR, "demo_gaussian_data.jl")))
            u_probes = [[0.5, -0.25, 0.1], [0.0, 0.0, 0.0]]
            """)
        res = sweep_appends(mani, joinpath(dir, "out"); backend = _GEN_BACKEND)
        @test res.ok
        @test length(res.results[1].probes) == 2
        @test res.results[1].probes[1].posterior == -15.886646898631646
        @test isfinite(res.results[1].probes[2].posterior)
        md = read(joinpath(dir, "out", "multi.md"), String)
        @test occursin("- probes: 2", md)
        @test occursin("- probe 2: `posterior(u) =", md)
        tab = read(joinpath(dir, "out", "verdict-table.md"), String)
        @test occursin("| multi | 1 | surface |", tab)
        @test occursin("| multi | 2 | surface |", tab)
        # Surface expected-length is checked at manifest load (count known).
        write(mani, """
            [[case]]
            id = "multi"
            kind = "surface"
            surface = $(_toml_str(joinpath(_SWEEP_FIXDIR, "demo_gaussian.jl")))
            data = $(_toml_str(joinpath(_SWEEP_FIXDIR, "demo_gaussian_data.jl")))
            u_probes = [[0.5, -0.25, 0.1], [0.0, 0.0, 0.0]]
            expected = [-15.886646898631646]
            """)
        @test_throws ArgumentError load_manifest(mani)
    end
end

@testset "sweep without backend" begin
    mktempdir() do dir
        res = sweep_appends(_SWEEP_FIXTURES, dir)
        @test res.ok
        @test all(r -> r.probes[1].grad == "not run", res.results)
        md = read(joinpath(dir, "corpus/demo_gaussian.md"), String)
        @test occursin("gradient cross-check: not run", md)
        @test occursin(_SWEEP_POSTERIOR, md)
    end
end

@testset "sweep mismatch + supersedes + based_on" begin
    mktempdir() do dir
        mani = joinpath(dir, "mani.toml")
        write(mani, """
            [[case]]
            id = "demo-gaussian"
            kind = "surface"
            surface = $(_toml_str(joinpath(_SWEEP_FIXDIR, "demo_gaussian.jl")))
            data = $(_toml_str(joinpath(_SWEEP_FIXDIR, "demo_gaussian_data.jl")))
            u_probes = [[0.5, -0.25, 0.1]]
            expected = [0.0]
            supersedes_ref = "2026-09-18T20-40-56-130-61rfdz"
            based_on = "ReactiveKernels:brm:parity-fam-gaussian/briefs/2026-09-25T22-01-55-116-1q2zyjh"
            """)
        res = sweep_appends(mani, joinpath(dir, "out"); backend = _GEN_BACKEND)
        @test !res.ok
        @test res.results[1].verdict == "MISMATCH"
        @test res.results[1].probes[1].delta == -15.886646898631646
        md = read(joinpath(dir, "out", "demo-gaussian.md"), String)
        @test occursin("Δ = -15.886646898631646 MISMATCH", md)
        @test occursin("case verdict: MISMATCH", md)
        @test occursin("Supersedes: 2026-09-18T20-40-56-130-61rfdz", md)
        @test occursin("Based-on: ReactiveKernels:brm:parity-fam-gaussian/briefs/" *
            "2026-09-25T22-01-55-116-1q2zyjh", md)
        index = TOML.parsefile(joinpath(dir, "out", "manifest.toml"))
        @test index["case"][1]["supersedes"] ==
            "2026-09-18T20-40-56-130-61rfdz"
        @test index["case"][1]["based_on"] ==
            "ReactiveKernels:brm:parity-fam-gaussian/briefs/2026-09-25T22-01-55-116-1q2zyjh"
        tab = read(joinpath(dir, "out", "verdict-table.md"), String)
        @test occursin("MISMATCH", tab)
        @test sweep_main(["--manifest", mani,
            "--out", joinpath(dir, "out2")]) == 1
    end
end

@testset "sweep ERROR row" begin
    mktempdir() do dir
        mani = joinpath(dir, "mani.toml")
        write(mani, """
            [[case]]
            id = "demo-gaussian"
            kind = "surface"
            surface = $(_toml_str(joinpath(_SWEEP_FIXDIR, "demo_gaussian.jl")))
            data = $(_toml_str(joinpath(_SWEEP_FIXDIR, "demo_gaussian_data.jl")))
            u_probes = [[0.0]]
            """)
        res = sweep_appends(mani, joinpath(dir, "out"))
        @test !res.ok
        @test res.results[1].verdict == "ERROR"
        @test occursin("probe point", res.results[1].note)
        # ERROR cases still get a closeout file (loud failure, never silent).
        md = read(joinpath(dir, "out", "demo-gaussian.md"), String)
        @test occursin("## Sweep error", md)
        @test occursin("case verdict: ERROR", md)
        @test !isfile(joinpath(dir, "out", "demo-gaussian.jls"))
        index = TOML.parsefile(joinpath(dir, "out", "manifest.toml"))
        @test index["case"][1]["verdict"] == "ERROR"
        @test !haskey(index["case"][1], "posterior")
        @test !haskey(index["case"][1], "artifact")
        tab = read(joinpath(dir, "out", "verdict-table.md"), String)
        @test occursin("| demo-gaussian | — | surface | ERROR |", tab)
    end
end

@testset "sweep manifest errors fail fast" begin
    mktempdir() do dir
        surf = _toml_str(joinpath(_SWEEP_FIXDIR, "demo_gaussian.jl"))
        dat = _toml_str(joinpath(_SWEEP_FIXDIR, "demo_gaussian_data.jl"))
        good = """
            [[case]]
            id = "demo-gaussian"
            kind = "surface"
            surface = $surf
            data = $dat
            """
        write(joinpath(dir, "m.toml"), good)
        @test_nowarn load_manifest(joinpath(dir, "m.toml"))
        # Missing file / bad TOML / empty cases.
        @test_throws ArgumentError load_manifest(joinpath(dir, "nope.toml"))
        write(joinpath(dir, "bad.toml"), "[[case\n")
        @test_throws ArgumentError load_manifest(joinpath(dir, "bad.toml"))
        write(joinpath(dir, "empty.toml"), "[meta]\n")
        @test_throws ArgumentError load_manifest(joinpath(dir, "empty.toml"))
        # Duplicate ids — one namespace across kinds (cross-kind collides).
        write(joinpath(dir, "dup.toml"), good * good)
        @test_throws ArgumentError load_manifest(joinpath(dir, "dup.toml"))
        write(joinpath(dir, "dup2.toml"), good * """
            [[case]]
            id = "demo-gaussian"
            kind = "brm-probe"
            brm_inputs = "x"
            worker = ["true"]
            """)
        @test_throws ArgumentError load_manifest(joinpath(dir, "dup2.toml"))
        # Bad id charset (escapes fail closed); slash segments pass.
        for bad in ("../escape", "a//b", "/lead", "trail/", "has space")
            write(joinpath(dir, "id.toml"),
                replace(good, "demo-gaussian" => bad))
            @test_throws ArgumentError load_manifest(joinpath(dir, "id.toml"))
        end
        write(joinpath(dir, "slash.toml"),
            replace(good, "demo-gaussian" => "corpus/01_gaussian"))
        @test_nowarn load_manifest(joinpath(dir, "slash.toml"))
        # Kind rules.
        write(joinpath(dir, "nk.toml"), """
            [[case]]
            id = "x"
            surface = $surf
            data = $dat
            """)
        @test_throws ArgumentError load_manifest(joinpath(dir, "nk.toml"))
        write(joinpath(dir, "bk.toml"),
            replace(good, "\"surface\"" => "\"bogus\""))
        @test_throws ArgumentError load_manifest(joinpath(dir, "bk.toml"))
        write(joinpath(dir, "sw.toml"), good * "worker = [\"x\"]\n")
        @test_throws ArgumentError load_manifest(joinpath(dir, "sw.toml"))
        write(joinpath(dir, "bs.toml"), """
            [[case]]
            id = "x"
            kind = "brm-probe"
            brm_inputs = "x"
            worker = ["true"]
            surface = $surf
            """)
        @test_throws ArgumentError load_manifest(joinpath(dir, "bs.toml"))
        write(joinpath(dir, "bw.toml"), """
            [[case]]
            id = "x"
            kind = "brm-probe"
            brm_inputs = "x"
            """)
        @test_throws ArgumentError load_manifest(joinpath(dir, "bw.toml"))
        write(joinpath(dir, "bi.toml"), """
            [[case]]
            id = "x"
            kind = "brm-probe"
            brm_inputs = ""
            worker = ["true"]
            """)
        @test_throws ArgumentError load_manifest(joinpath(dir, "bi.toml"))
        write(joinpath(dir, "wn.toml"), """
            [[case]]
            id = "x"
            kind = "brm-probe"
            brm_inputs = "x"
            worker = ["true", 7]
            """)
        @test_throws ArgumentError load_manifest(joinpath(dir, "wn.toml"))
        write(joinpath(dir, "wf.toml"), """
            [[case]]
            id = "x"
            kind = "brm-probe"
            brm_inputs = "x"
            worker = ["/nonexistent/worker.jl"]
            """)
        @test_throws ArgumentError load_manifest(joinpath(dir, "wf.toml"))
        # Unknown keys (top level, meta, case).
        write(joinpath(dir, "k1.toml"), good * "[bogus]\n")
        @test_throws ArgumentError load_manifest(joinpath(dir, "k1.toml"))
        write(joinpath(dir, "k2.toml"), "[meta]\nnope = 1\n" * good)
        @test_throws ArgumentError load_manifest(joinpath(dir, "k2.toml"))
        write(joinpath(dir, "k3.toml"), good * "bogus = 1\n")
        @test_throws ArgumentError load_manifest(joinpath(dir, "k3.toml"))
        # Missing surface / data files; missing required key.
        write(joinpath(dir, "sf.toml"),
            replace(good, "demo_gaussian.jl" => "nope.jl"))
        @test_throws ArgumentError load_manifest(joinpath(dir, "sf.toml"))
        write(joinpath(dir, "rk.toml"), """
            [[case]]
            id = "x"
            kind = "surface"
            surface = $surf
            """)
        @test_throws ArgumentError load_manifest(joinpath(dir, "rk.toml"))
        # Non-finite / malformed u_probes / expected.
        write(joinpath(dir, "u1.toml"), good * "u_probes = []\n")
        @test_throws ArgumentError load_manifest(joinpath(dir, "u1.toml"))
        write(joinpath(dir, "u2.toml"), good * "u_probes = [[1.0, inf]]\n")
        @test_throws ArgumentError load_manifest(joinpath(dir, "u2.toml"))
        write(joinpath(dir, "u3.toml"), good * "u_probes = [\"x\"]\n")
        @test_throws ArgumentError load_manifest(joinpath(dir, "u3.toml"))
        write(joinpath(dir, "e1.toml"), good * "expected = [inf]\n")
        @test_throws ArgumentError load_manifest(joinpath(dir, "e1.toml"))
        write(joinpath(dir, "e2.toml"), good * "expected = 7.0\n")
        @test_throws ArgumentError load_manifest(joinpath(dir, "e2.toml"))
        # Malformed refs.
        write(joinpath(dir, "s.toml"), good * "supersedes_ref = \"has space\"\n")
        @test_throws ArgumentError load_manifest(joinpath(dir, "s.toml"))
        write(joinpath(dir, "b.toml"), good * "based_on = \"has space\"\n")
        @test_throws ArgumentError load_manifest(joinpath(dir, "b.toml"))
        # CLI flag errors.
        @test_throws ArgumentError sweep_main(["--manifest"])
        @test_throws ArgumentError sweep_main(["--bogus", "x"])
        @test_throws ArgumentError sweep_main(
            ["--manifest", joinpath(dir, "m.toml")])
        # Nothing is written when the manifest is bad.
        out = joinpath(dir, "out")
        @test_throws ArgumentError sweep_appends(
            joinpath(dir, "nope.toml"), out)
        @test !isdir(out)
    end
end

@testset "sweep CLI main exit codes" begin
    mktempdir() do dir
        @test sweep_main(["--manifest", _SWEEP_FIXTURES,
            "--out", joinpath(dir, "out")]) == 0
        @test isfile(joinpath(dir, "out", "verdict-table.md"))
    end
end

@testset "sweep determinism" begin
    mktempdir() do dir
        a, b = joinpath(dir, "a"), joinpath(dir, "b")
        ra = sweep_appends(_SWEEP_FIXTURES, a; backend = _GEN_BACKEND)
        rb = sweep_appends(_SWEEP_FIXTURES, b; backend = _GEN_BACKEND)
        @test ra.ok && rb.ok
        # Text outputs are byte-identical across runs; the .jls artifacts
        # are load-equal (serialization bytes are not a stability contract).
        for f in ["corpus/demo_gaussian.md", "corpus/demo_factor.md",
                "corpus/demo_poisson.md", "manifest.toml", "verdict-table.md"]
            @test read(joinpath(a, f), String) == read(joinpath(b, f), String)
        end
        for id in _SWEEP_IDS
            la = load_artifact(joinpath(a, "$id.jls"))
            lb = load_artifact(joinpath(b, "$id.jls"))
            @test la.ast == lb.ast
            ma = transpile_report(la.ast, la.data; meta = la.meta)
            mb = transpile_report(lb.ast, lb.data; meta = lb.meta)
            @test ma == mb
        end
    end
end

@testset "sweep brm-probe via stub worker" begin
    mktempdir() do dir
        mani = _write_brm_manifest(dir;
            extra_case = "supersedes_ref = \"2026-09-18T20-40-56-130-61rfdz\"\n")
        res = sweep_appends(mani, joinpath(dir, "out"))
        @test res.ok
        @test length(res.results) == 1
        r = res.results[1]
        @test r.kind === :brm
        @test r.verdict == "VERIFY_OK"
        @test length(r.probes) == 1
        pr = r.probes[1]
        @test pr.posterior == -100.0
        @test pr.delta == 0.0
        @test pr.grad == "PASS"
        @test pr.grad_maxdiff == 1.0e-9
        sb_want = -100.0 - (-100.0 + 1.0e-12)
        @test pr.sb_delta == sb_want
        @test abs(pr.sb_delta) < 1.0e-11
        # Closeout file: header + spliced worker sections + re-sweep.
        md = read(joinpath(dir, "out", "stubdemo.md"), String)
        @test occursin("# Closeout: stubdemo", md)
        @test occursin("kind = \"brm-probe\"", md)
        @test occursin("brm_inputs = \"stub:demo\"", md)
        @test occursin("worker = \"", md)
        @test occursin("Supersedes: 2026-09-18T20-40-56-130-61rfdz", md)
        @test occursin("## Layer SB (stub)", md)
        @test occursin("## Re-sweep verdict", md)
        @test occursin("Δ = 0.0 MATCH", md)
        @test occursin("case verdict: VERIFY_OK", md)
        # Worker artifact copied opaque; worker log captured; spec recorded.
        @test read(joinpath(dir, "out", "stubdemo.jls"), String) ==
            "STUB-V2-ARTIFACT:stubdemo"
        @test isfile(joinpath(dir, "out", "stubdemo.worker.log"))
        @test occursin("command:",
            read(joinpath(dir, "out", "stubdemo.worker.log"), String))
        spec = TOML.parsefile(joinpath(dir, "out", ".worker", "stubdemo.in.toml"))
        @test spec["id"] == "stubdemo"
        @test spec["brm_inputs"] == "stub:demo"
        @test length(spec["u_probes"]) == 1
        @test spec["u_probes"][1] == [0.0]
        # Index + table.
        index = TOML.parsefile(joinpath(dir, "out", "manifest.toml"))
        @test index["case"][1]["kind"] == "brm-probe"
        @test index["case"][1]["brm_inputs"] == "stub:demo"
        @test index["case"][1]["worker_pins"] == Dict("worker" => "stub")
        @test index["case"][1]["probe"][1]["posterior"] == -100.0
        @test index["case"][1]["probe"][1]["sb_delta"] == sb_want
        tab = read(joinpath(dir, "out", "verdict-table.md"), String)
        @test occursin("| stubdemo | 1 | brm-probe | -100.0 | -100.0 | 0.0 " *
            "| PASS | $(repr(sb_want)) | VERIFY_OK |", tab)
    end
end

@testset "sweep brm-probe multi-probe + null probes" begin
    mktempdir() do dir
        # Two manifest probes: the stub answers per probe.
        mani = _write_brm_manifest(dir;
            u_probes = "[[0.0], [1.0]]", expected = "[-100.0, -101.0]")
        res = sweep_appends(mani, joinpath(dir, "out"))
        @test res.ok
        @test [p.posterior for p in res.results[1].probes] ==
            [-100.0, -101.0]
        tab = read(joinpath(dir, "out", "verdict-table.md"), String)
        @test occursin("| stubdemo | 1 | brm-probe |", tab)
        @test occursin("| stubdemo | 2 | brm-probe |", tab)
        # Absent u_probes: the worker expands (stub: single canned probe).
        mani2 = _write_brm_manifest(dir; u_probes = nothing)
        res2 = sweep_appends(mani2, joinpath(dir, "out2"))
        @test res2.ok
        @test res2.results[1].probes[1].u_show ==
            "origin (worker-expanded)"
        spec = TOML.parsefile(joinpath(dir, "out2", ".worker", "stubdemo.in.toml"))
        @test !haskey(spec, "u_probes")
    end
end

@testset "sweep token wrap: argv shape (no token acquired)" begin
    worker = ["julia", "--startup-file=no", "/abs/worker.jl"]
    # Unwrapped default: exactly the pre-wrap shape (regression pin).
    @test _worker_cmd(worker, "s.toml", "o").exec ==
        ["julia", "--startup-file=no", "/abs/worker.jl",
        "--spec", "s.toml", "--out", "o"]
    # Wrapped: helper prefix coupled to --no-token (exactly one side gates).
    @test _worker_cmd(worker, "s.toml", "o"; wrap_token = true).exec ==
        ["kb-acquire-compute-token", "--",
        "julia", "--startup-file=no", "/abs/worker.jl",
        "--spec", "s.toml", "--out", "o", "--no-token"]
    # No execution happens above: safe in CI without the helper or a token.
end

@testset "sweep CLI --wrap-token parses (no execution)" begin
    args = _sweep_cli_args(["--manifest", "m.toml", "--out", "o"])
    @test args.wrap_token == false
    args = _sweep_cli_args(["--manifest", "m.toml", "--out", "o", "--wrap-token"])
    @test args.wrap_token == true
    @test args.manifest_path == "m.toml"
    @test args.out_dir == "o"
    @test_throws ArgumentError _sweep_cli_args(["--manifest", "m.toml"])
end

@testset "sweep stub worker accepts --no-token" begin
    mktempdir() do dir
        mani = _write_brm_manifest(dir; extra_worker_args = ["--no-token"])
        res = sweep_appends(mani, joinpath(dir, "out"))
        @test res.ok
        @test res.results[1].verdict == "VERIFY_OK"
    end
end

@testset "sweep brm-probe worker failures" begin
    mktempdir() do dir
        # Nonzero exit.
        mani = _write_brm_manifest(dir; extra_worker_args = ["--fail"])
        res = sweep_appends(mani, joinpath(dir, "out"))
        @test !res.ok
        @test res.results[1].verdict == "ERROR"
        @test occursin("nonzero", res.results[1].note)
        @test occursin("failing as requested",
            read(joinpath(dir, "out", "stubdemo.worker.log"), String))
        # Malformed numbers.
        for (kind, needle) in (("missing-key", "missing \"grad\""),
                ("bad-grad", "must be one of"), ("count", "returned 2 probe(s)"))
            m = _write_brm_manifest(dir;
                extra_worker_args = ["--bad-numbers", kind])
            out = joinpath(dir, "out-$kind")
            r = sweep_appends(m, out)
            @test !r.ok
            @test r.results[1].verdict == "ERROR"
            @test occursin(needle, r.results[1].note)
        end
        # Expected-length mismatch is checked at consume time.
        m = _write_brm_manifest(dir; expected = "[-100.0, -101.0]")
        r = sweep_appends(m, joinpath(dir, "out-exp"))
        @test !r.ok
        @test r.results[1].verdict == "ERROR"
        @test occursin("2 expected value(s)", r.results[1].note)
        tab = read(joinpath(dir, "out-exp", "verdict-table.md"), String)
        @test occursin("| stubdemo | — | brm-probe | ERROR |", tab)
    end
end
