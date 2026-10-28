# Stub BRM worker for sweep_appends tests (no BRM, no Stan — stdlib only).
#
# Speaks the worker contract (see report/sweep_appends.jl header):
#   julia --startup-file=no stub_worker.jl --spec <in.toml> --out <case-outdir>
# Reads the spec and writes canned-but-spec-derived sections.md +
# numbers.toml + artifact.jls. numbers.toml has the shape the real BRM
# worker's `_write_numbers` (BRM test/rk_append_worker.jl) writes: every
# probe key the driver reads, including the `[probe.oracle]` table.
# `--numbers-from <path>` writes that file verbatim as numbers.toml instead
# (a real worker's output). Failure-path fixtures: `--fail` exits 1;
# `--bad-numbers <kind>` writes a malformed numbers.toml with kind in
# `missing-key`, `bad-grad`, `count`, `unknown-key`, `bad-maxdiff`.
# `--no-token` is accepted and ignored (mirrors the real worker flag the
# driver passes when wrapping).

using TOML

function _stub_args(argv)
    spec, out, fail, bad, numbers_from = nothing, nothing, false, nothing, nothing
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
        elseif a == "--no-token"
            nothing  # driver-wrapped invocation; the stub never gates
        elseif a == "--bad-numbers"
            i += 1; i > length(argv) && error("stub worker: --bad-numbers needs a kind")
            bad = argv[i]
        elseif a == "--numbers-from"
            i += 1; i > length(argv) && error("stub worker: --numbers-from needs a path")
            numbers_from = argv[i]
        else
            error("stub worker: unknown arg $a")
        end
        i += 1
    end
    (spec === nothing || out === nothing) && error("stub worker: need --spec and --out")
    return spec, out, fail, bad, numbers_from
end

function _stub_main(argv)
    spec_path, out_dir, fail, bad, numbers_from = _stub_args(argv)
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
    probe_toml(i; grad = "PASS", with_grad = true, extra = String[],
            fd = repr(3.0e-8)) = begin
        post = -(100.0 + (i - 1))
        lines = ["[[probe]]", "posterior = $(repr(post))"]
        with_grad && push!(lines, "grad = \"$grad\"")
        push!(lines, "grad_maxdiff = $(repr(1.0e-9))")
        push!(lines, "sb_value = $(repr(post + 1.0e-12))")
        push!(lines, "sb_grad_maxdiff = $(repr(2.0e-12))")
        push!(lines, "sb_fd_maxdiff = $fd")
        append!(lines, extra)
        push!(lines, "[probe.oracle]")
        push!(lines, "value = $(repr(post))")
        push!(lines, "diff_vs_sb = $(repr(1.0e-12))")
        join(lines, "\n")
    end
    probe1 = if bad == "missing-key"
        probe_toml(1; with_grad = false)
    elseif bad == "bad-grad"
        probe_toml(1; grad = "MAYBE")
    elseif bad == "unknown-key"
        probe_toml(1; extra = ["sb_new_check = 1.0"])
    elseif bad == "bad-maxdiff"
        probe_toml(1; fd = "\"small\"")
    else
        probe_toml(1)
    end
    blocks = [probe1; [probe_toml(i) for i in 2:nprobes]]
    numbers = "# stub numbers.toml — case $id\n" * join(blocks, "\n") *
        "\n[pins]\nworker = \"stub\"\n"
    numbers_path = joinpath(out_dir, "numbers.toml")
    if numbers_from === nothing
        write(numbers_path, numbers)
    else
        cp(numbers_from, numbers_path; force = true)
    end
    write(joinpath(out_dir, "artifact.jls"), "STUB-V2-ARTIFACT:$id")
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(_stub_main(copy(ARGS)))
end
