# Stub BRM worker for sweep_appends tests (no BRM, no Stan — stdlib only).
#
# Speaks the worker contract (see report/sweep_appends.jl header):
#   julia --startup-file=no stub_worker.jl --spec <in.toml> --out <case-outdir>
# Reads the spec and writes canned-but-spec-derived sections.md +
# numbers.toml + artifact.jls. Failure-path fixtures: `--fail` exits 1;
# `--bad-numbers <kind>` writes a malformed numbers.toml with kind in
# `missing-key`, `bad-grad`, `count`.

using TOML

function _stub_args(argv)
    spec, out, fail, bad = nothing, nothing, false, nothing
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--spec"
            i += 1; i > length(argv) && error("stub worker: --spec needs a value")
            spec = argv[i]
        elseif a == "--out"
            i += 1; i > length(argv) && error("stub worker: --out needs a value")
            out = argv[i]
        elseif a == "--fail"
            fail = true
        elseif a == "--bad-numbers"
            i += 1; i > length(argv) && error("stub worker: --bad-numbers needs a kind")
            bad = argv[i]
        else
            error("stub worker: unknown arg $a")
        end
        i += 1
    end
    (spec === nothing || out === nothing) && error("stub worker: need --spec and --out")
    return spec, out, fail, bad
end

function _stub_main(argv)
    spec_path, out_dir, fail, bad = _stub_args(argv)
    if fail
        println(stderr, "stub worker: failing as requested (--fail)")
        return 1
    end
    spec = TOML.parsefile(spec_path)
    id = string(spec["id"])
    n = haskey(spec, "u_probes") ? length(spec["u_probes"]) : 1
    nprobes = bad == "count" ? n + 1 : n
    mkpath(out_dir)
    sections = join([
        "## Layer 1 — BRM source (stub)",
        "",
        "stub sections for `$id` ($nprobes probe(s)).",
        "",
        "## Layer 2 — BRM design (stub)",
        "",
        "stub.",
        "",
        "## Layer 3 — `@rkppl` surface (stub)",
        "",
        "stub.",
        "",
        "## Boundary (stub)",
        "",
        "stub.",
        "",
        "## Layer 4 — emitted RK kernel (stub)",
        "",
        "stub.",
        "",
        "## Verification (stub)",
        "",
        "stub.",
        "",
        "## Layer SB (stub)",
        "",
        "stub.",
        "",
    ], "\n")
    write(joinpath(out_dir, "sections.md"), sections)
    probe_toml(i; grad = "PASS", with_grad = true) = begin
        post = -(100.0 + (i - 1))
        lines = ["[[probe]]", "posterior = $(repr(post))"]
        with_grad && push!(lines, "grad = \"$grad\"")
        push!(lines, "grad_maxdiff = $(repr(1.0e-9))")
        push!(lines, "sb_value = $(repr(post + 1.0e-12))")
        push!(lines, "sb_grad_maxdiff = $(repr(2.0e-12))")
        push!(lines, "oracle = $(repr(post))")
        join(lines, "\n")
    end
    blocks = if bad == "missing-key"
        [probe_toml(1; with_grad = false);
            [probe_toml(i) for i in 2:nprobes]]
    elseif bad == "bad-grad"
        [probe_toml(1; grad = "MAYBE");
            [probe_toml(i) for i in 2:nprobes]]
    else
        [probe_toml(i) for i in 1:nprobes]
    end
    numbers = join(blocks, "\n\n") * "\n\n[pins]\nworker = \"stub\"\n"
    write(joinpath(out_dir, "numbers.toml"), numbers)
    write(joinpath(out_dir, "artifact.jls"), "STUB-V2-ARTIFACT:$id")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(_stub_main(copy(ARGS)))
end
