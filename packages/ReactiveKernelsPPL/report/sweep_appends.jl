# Fusion sweep driver: corpus manifest in, per-case closeout files out.
#
# Machine-rendered Layer-3/Boundary/Layer-4/Verification append machinery
# for the SB-parity closeout. Reads a TOML fusion manifest listing cases of
# two kinds; writes into the output dir:
#
#   <id>.jls         sweep artifact — surface-kind: v1 `(; ast, data, meta)`;
#                    brm-probe-kind: the worker's v2 artifact, copied opaque
#                    (never deserialized here — no BRM dependency)
#   <id>.md          case closeout file — the fusion runner files each as a
#                    NEW brief with `based_on` at the covering pair verdict
#   <id>.worker.log brm-probe-kind only: captured worker stdout/stderr
#   manifest.toml    machine-readable index (case id → file, generator
#                    version, corpus pins)
#   verdict-table.md re-sweep verdict table (one row per probe)
#
# Case ids form ONE namespace across kinds (the driver errors on collision).
# Pair convention: brm-probe ids are bare stable tags (`escs`), surface ids
# are `corpus/NN_name` — disjoint by construction (slash vs bare). Ids may
# carry `/` separators; outputs mirror the hierarchy under the output dir
# (`..` and escapes fail closed).
#
# Manifest schema (relative paths resolve against the manifest file;
# absolute paths are used as-is):
#
#   [meta]
#   pins = { rk = "<sha>", brm = "<sha>" }   # optional, recorded verbatim
#
#   [[case]]
#   id = "corpus/01_gaussian"         # unique, `/`-separated segments
#   kind = "surface"                  # REQUIRED: "surface" | "brm-probe"
#   surface = "demo_gaussian.jl"      # surface-kind: `@rkppl` block file
#   data = "demo_gaussian_data.jl"    # surface-kind: defines `DATA`
#   brm_inputs = "docs/examples/escs.jl"  # brm-probe-kind: opaque locator
#   worker = ["julia", "--startup-file=no", "--project=/abs/env",
#             "/abs/worker.jl"]       # brm-probe-kind: argv prefix; the driver
#                                     # appends `--spec <in.toml> --out <dir>`
#   u_probes = [[0.5, -0.25, 0.1]]    # optional (absent = single origin probe;
#                                     # TOML has no null). brm-probe-kind: the
#                                     # worker expands the absent case
#                                     # post-translate (dim is its own).
#   expected = [-15.886646898631646]  # optional re-sweep oracle, parallel to
#                                     # u_probes (length mismatches fail loudly)
#   supersedes_ref = "2026-09-18T20-40-56-130-61rfdz"  # optional overlapped
#                                                      # Bambi brief
#   based_on = "ReactiveKernels:brm:parity-fam-gaussian/briefs/<bid>"
#                                     # optional covering pair verdict ref
#
# Surface-kind cases run the REAL pipeline in-process (surface → lower →
# bind → build → query → call) and render Layer-3/Boundary/Layer-4/
# Verification here. The bare lower→bind route is correct for surface-kind
# (no BRM plan exists to patch); it must never touch v2/brm-probe
# artifacts, whose binding goes through the BRM translate entry point.
#
# Brm-probe-kind cases shell out to the BRM worker script (no BRM
# dependency in the driver's env; exact BRM-test-env fidelity inside the
# worker; failure isolation per case; no shell — argv array only). The
# driver writes `<out>/.worker/<id>.in.toml`
# (`{id, brm_inputs, u_probes?, manifest, pins}`) and runs
# `worker --spec <in.toml> --out <case-outdir>`. The worker MUST write:
# `sections.md` (rendered Layer 1/2/3, Boundary, Layer 4, Verification,
# Layer SB — spliced verbatim under the closeout header), `numbers.toml`
# (`[[probe]]` parallel to the expanded probes: `posterior` (required),
# `grad` in PASS|FAIL|not run (required), `grad_maxdiff?`, `sb_value?`,
# `sb_grad_maxdiff?`, `oracle?`; optional `[pins]` String table), and
# `artifact.jls` (v2, opaque). Missing outputs, bad numbers, a probe-count
# mismatch, or a nonzero exit become an ERROR row (never a silent skip).
# The worker gates its own heavy Stan legs on a compute token; to bound a
# hung worker, prefix its argv with `timeout` (the driver's argv is yours).
#
# Exit codes: 0 iff every case reaches VERIFY_OK. A case that throws is
# caught into an ERROR row; the driver still writes every good case plus
# the table, then exits 1. Manifest-level problems throw before anything
# is written. No wall-clock time enters any output: same manifest + same
# pins + same worker outputs → byte-identical text outputs (the .jls
# artifacts are load-equal, not byte-compared).
#
# Fusion-runner shape (PPL test env, Enzyme backend):
#
#   include("packages/ReactiveKernelsPPL/report/sweep_appends.jl")
#   using Enzyme, DifferentiationInterface
#   sweep_appends("fusion_manifest.toml", "fusion_out";
#       backend = AutoEnzyme(; mode = Enzyme.Reverse))

include(joinpath(@__DIR__, "transpile_report.jl"))
using TOML

const _SWEEP_NAME = "sweep_appends"
const _SWEEP_VERSION = 1

const _SWEEP_ID_RE = r"^[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)*$"
const _SWEEP_REF_RE = r"^\S+$"
const _SWEEP_GRADS = ("PASS", "FAIL", "not run")

# ---------------------------------------------------------------- manifest

struct SweepCase
    id::String
    kind::Symbol
    surface::Union{Nothing,String}
    data::Union{Nothing,String}
    brm_inputs::Union{Nothing,String}
    worker::Union{Nothing,Vector{String}}
    uprobes::Union{Nothing,Vector{Vector{Float64}}}
    expected::Union{Nothing,Vector{Float64}}
    supersedes::Union{Nothing,String}
    based_on::Union{Nothing,String}
end

struct SweepManifest
    path::String
    pins::Dict{String,String}
    cases::Vector{SweepCase}
end

function _manifest_error(path, msg)
    throw(ArgumentError("sweep manifest $path: $msg"))
end

_manifest_resolve(dir, p) =
    (s = string(p); isabspath(s) ? s : joinpath(dir, s))

function _manifest_ref(raw, key, id, path)
    haskey(raw, key) || return nothing
    v = raw[key]
    v isa AbstractString && occursin(_SWEEP_REF_RE, v) ||
        _manifest_error(path, "case \"$id\" $key must be a single token " *
                              "(no whitespace), got $(repr(v))")
    return String(v)
end

function _manifest_probes(raw, id, path)
    haskey(raw, "u_probes") || return nothing
    praw = raw["u_probes"]
    praw isa AbstractVector && !isempty(praw) ||
        _manifest_error(path, "case \"$id\" u_probes must be a non-empty " *
                              "list of probes")
    out = Vector{Float64}[]
    for (i, up) in enumerate(praw)
        up isa AbstractVector && all(x -> x isa Real, up) ||
            _manifest_error(path, "case \"$id\" probe $i must be a vector " *
                                  "of numbers, got $(repr(up))")
        uf = Float64.(up)
        all(isfinite, uf) ||
            _manifest_error(path, "case \"$id\" probe $i must be finite")
        push!(out, uf)
    end
    return out
end

function _manifest_expected(raw, id, path, nprobes_known)
    haskey(raw, "expected") || return nothing
    eraw = raw["expected"]
    eraw isa AbstractVector && all(x -> x isa Real, eraw) ||
        _manifest_error(path, "case \"$id\" expected must be a list of " *
                              "numbers parallel to u_probes, got $(repr(eraw))")
    ef = Float64.(eraw)
    all(isfinite, ef) ||
        _manifest_error(path, "case \"$id\" expected must be finite")
    nprobes_known !== nothing && length(ef) != nprobes_known &&
        _manifest_error(path, "case \"$id\" expected has length " *
                              "$(length(ef)) but $(nprobes_known) probe(s)")
    return ef
end

function _manifest_case(raw, i::Int, path::String, dir::String)
    raw isa AbstractDict ||
        _manifest_error(path, "case $i must be a table, got $(typeof(raw))")
    for k in keys(raw)
        k in ("id", "kind", "surface", "data", "brm_inputs", "worker",
            "u_probes", "expected", "supersedes_ref", "based_on") ||
            _manifest_error(path, "case $i has unknown key \"$k\"")
    end
    for k in ("id", "kind")
        haskey(raw, k) || _manifest_error(path, "case $i is missing \"$k\"")
    end
    id = raw["id"]
    id isa AbstractString && occursin(_SWEEP_ID_RE, id) ||
        _manifest_error(path, "case $i has a bad id $(repr(id)) " *
                              "(want `/`-separated `[A-Za-z0-9_.-]` segments)")
    id = String(id)
    kind = raw["kind"]
    kind in ("surface", "brm-probe") ||
        _manifest_error(path, "case \"$id\" kind must be \"surface\" or " *
                              "\"brm-probe\", got $(repr(kind))")
    uprobes = _manifest_probes(raw, id, path)
    if kind == "surface"
        for k in ("surface", "data")
            haskey(raw, k) ||
                _manifest_error(path, "surface case \"$id\" is missing \"$k\"")
        end
        for k in ("brm_inputs", "worker")
            haskey(raw, k) && _manifest_error(path,
                "surface case \"$id\" takes no \"$k\"")
        end
        surface = _manifest_resolve(dir, raw["surface"])
        isfile(surface) ||
            _manifest_error(path, "case \"$id\" surface not found: $surface")
        data = _manifest_resolve(dir, raw["data"])
        isfile(data) ||
            _manifest_error(path, "case \"$id\" data not found: $data")
        n = uprobes === nothing ? 1 : length(uprobes)
        expected = _manifest_expected(raw, id, path, n)
        return SweepCase(id, :surface, surface, data, nothing, nothing,
            uprobes, expected, _manifest_ref(raw, "supersedes_ref", id, path),
            _manifest_ref(raw, "based_on", id, path))
    else
        for k in ("brm_inputs", "worker")
            haskey(raw, k) ||
                _manifest_error(path, "brm-probe case \"$id\" is missing \"$k\"")
        end
        for k in ("surface", "data")
            haskey(raw, k) && _manifest_error(path,
                "brm-probe case \"$id\" takes no \"$k\"")
        end
        brm_inputs = raw["brm_inputs"]
        brm_inputs isa AbstractString && !isempty(brm_inputs) ||
            _manifest_error(path, "case \"$id\" brm_inputs must be a " *
                                  "non-empty locator string")
        wraw = raw["worker"]
        wraw isa AbstractVector && !isempty(wraw) &&
            all(x -> x isa AbstractString, wraw) ||
            _manifest_error(path, "case \"$id\" worker must be a non-empty " *
                                  "argv array of strings")
        worker = String.(wraw)
        occursin("/", worker[1]) && !isfile(worker[1]) &&
            _manifest_error(path, "case \"$id\" worker not found: $(worker[1])")
        # Probe count is the worker's (it expands absent u_probes
        # post-translate), so the expected-length check runs at consume time.
        expected = _manifest_expected(raw, id, path, nothing)
        return SweepCase(id, :brm, nothing, nothing, String(brm_inputs),
            worker, uprobes, expected,
            _manifest_ref(raw, "supersedes_ref", id, path),
            _manifest_ref(raw, "based_on", id, path))
    end
end

"""
    load_manifest(path) -> SweepManifest

Parse and validate a sweep manifest (see the file header for the schema).
Manifest-level problems throw `ArgumentError` before anything is written.
"""
function load_manifest(path::AbstractString)
    isfile(path) || throw(ArgumentError("sweep manifest not found: $path"))
    raw = try
        TOML.parsefile(path)
    catch err
        throw(ArgumentError("sweep manifest $path is not valid TOML " *
                            "($(sprint(showerror, err)))"))
    end
    raw isa AbstractDict ||
        _manifest_error(path, "top level must be a table")
    for k in keys(raw)
        k in ("meta", "case") ||
            _manifest_error(path, "unknown top-level key \"$k\"")
    end
    pins = Dict{String,String}()
    if haskey(raw, "meta")
        meta = raw["meta"]
        meta isa AbstractDict ||
            _manifest_error(path, "[meta] must be a table")
        for k in keys(meta)
            k == "pins" || _manifest_error(path, "[meta] has unknown key \"$k\"")
        end
        if haskey(meta, "pins")
            praw = meta["pins"]
            praw isa AbstractDict ||
                _manifest_error(path, "[meta] pins must be a table")
            for (k, v) in praw
                v isa AbstractString || _manifest_error(path,
                    "[meta] pin \"$k\" must be a string, got $(repr(v))")
                pins[String(k)] = String(v)
            end
        end
    end
    haskey(raw, "case") || _manifest_error(path, "no [[case]] entries")
    craw = raw["case"]
    craw isa AbstractVector && !isempty(craw) ||
        _manifest_error(path, "[[case]] must be a non-empty list")
    dir = dirname(abspath(path))
    cases = [_manifest_case(c, i, path, dir)
        for (i, c) in enumerate(craw)]
    ids = [c.id for c in cases]
    length(unique(ids)) == length(ids) ||
        _manifest_error(path, "duplicate case ids (one namespace across kinds)")
    return SweepManifest(abspath(path), pins, cases)
end

# ------------------------------------------------------------------- sweep

struct SweepProbeResult
    u_show::String
    posterior::Float64
    expected::Union{Nothing,Float64}
    delta::Float64
    grad::String
    grad_maxdiff::Union{Nothing,Float64}
    sb_delta::Union{Nothing,Float64}
    verdict::String
end

struct SweepResult
    id::String
    kind::Symbol
    probes::Vector{SweepProbeResult}
    verdict::String
    note::String
end

function _case_path(out_dir, id, ext)
    p = normpath(joinpath(out_dir, id * ext))
    startswith(p, normpath(out_dir) * "/") ||
        throw(ArgumentError("case id escapes output dir: $(repr(id))"))
    return p
end

function _probe_verdict(val::Float64, expected, grad::String)
    !isfinite(val) && return "NON_FINITE"
    expected !== nothing && val != expected && return "MISMATCH"
    grad == "FAIL" && return "GRADIENT_FAIL"
    return "VERIFY_OK"
end

const _VERDICT_WORST = Dict(
    "ERROR" => 4, "NON_FINITE" => 3, "MISMATCH" => 2, "GRADIENT_FAIL" => 1,
    "VERIFY_OK" => 0)

_case_verdict(probes) = isempty(probes) ? "ERROR" :
    argmax(p -> _VERDICT_WORST[p.verdict], probes).verdict

function _closeout_head(id, kind, pins, manifest_path, supersedes, based_on,
        extra::Vector{String} = String[])
    head = [
        "# Closeout: $id",
        "",
        "generator = \"$(_SWEEP_NAME)\"",
        "generator_version = $(_SWEEP_VERSION)",
        "manifest = \"$manifest_path\"",
        "case = \"$id\"",
        "kind = \"$(kind === :brm ? "brm-probe" : "surface")\"",
    ]
    for k in sort!(collect(keys(pins)))
        push!(head, "$k = \"$(pins[k])\"")
    end
    supersedes !== nothing && push!(head, "Supersedes: $supersedes")
    based_on !== nothing && push!(head, "Based-on: $based_on")
    append!(head, extra)
    push!(head, "")
    return head
end

_show_sweep_expected(::Nothing) = "none"
_show_sweep_expected(x::Float64) = repr(x)
_show_sweep_opt(::Nothing) = "—"
_show_sweep_opt(x::Float64) = repr(x)

function _resweep_block(probes, verdict)
    lines = ["## Re-sweep verdict", "", "- probes: $(length(probes))"]
    for (i, pr) in enumerate(probes)
        expbit = pr.expected === nothing ? "no oracle" :
            pr.delta == 0.0 ? "Δ = 0.0 MATCH" : "Δ = $(repr(pr.delta)) MISMATCH"
        push!(lines, "- probe $i (`u = $(pr.u_show)`): " *
            "`posterior(u) = $(repr(pr.posterior))`; expected " *
            "$(_show_sweep_expected(pr.expected)) ($expbit); " *
            "gradient $(pr.grad); SB Δ $(_show_sweep_opt(pr.sb_delta)); " *
            "$(pr.verdict)")
    end
    push!(lines, "- case verdict: $verdict")
    push!(lines, "")
    return lines
end

function _sweep_error_md(id, kind, pins, manifest_path, supersedes, based_on,
        note)
    return join([
        _closeout_head(id, kind, pins, manifest_path, supersedes, based_on)...,
        "## Sweep error",
        "",
        "```",
        note,
        "```",
        "",
        "- case verdict: ERROR",
        "",
    ], "\n")
end

# ------------------------------------------------------- surface-kind path

function _run_surface_case(c::SweepCase, mani::SweepManifest, out_dir;
        backend = nothing)
    sf = load_surface_files(c.surface, c.data)
    ast = _surface_block(sf.surface)
    cols = _norm_cols(sf.data)
    # Bare lower→bind is correct here: surface-kind cases carry no BRM
    # plan, so no patches exist to apply.
    bound = bind_data(lower_rkppl(ast, keys(cols)), cols)
    prep = _prepare_bound_report(bound)
    expanded = c.uprobes === nothing ? [nothing] : c.uprobes
    prs = [_run_report_probe(prep, up; backend = backend) for up in expanded]
    layer3 = string(strip(sf.surface))
    sections = join([
        _layer3_block(layer3,
            "parsed surface block lowered directly (no separate AST input)")...,
        _boundary_block(bound)...,
        _layer4_block(kernel_expr(bound, prep.built.layout))...,
        _verification_multi_block(prs)...,
    ], "\n")
    probes = SweepProbeResult[]
    for (i, pr) in enumerate(prs)
        exp = c.expected === nothing ? nothing : c.expected[i]
        push!(probes, SweepProbeResult(repr(pr.u), pr.val, exp,
            exp === nothing ? NaN : pr.val - exp,
            pr.grad_ok === nothing ? "not run" : pr.grad_ok ? "PASS" : "FAIL",
            pr.grad_maxdiff, nothing,
            _probe_verdict(pr.val, exp,
                pr.grad_ok === nothing ? "not run" :
                    pr.grad_ok ? "PASS" : "FAIL")))
    end
    verdict = _case_verdict(probes)
    md = join([
        _closeout_head(c.id, c.kind, mani.pins, mani.path, c.supersedes,
            c.based_on)...,
        sections,
        _resweep_block(probes, verdict)...,
    ], "\n")
    md_path = _case_path(out_dir, c.id, ".md")
    mkpath(dirname(md_path))
    write(md_path, md)
    meta = (; model = c.id, source = "sweep $(basename(mani.path))",
        generator = _SWEEP_NAME, generator_version = _SWEEP_VERSION,
        kind = "surface")
    Serialization.serialize(_case_path(out_dir, c.id, ".jls"),
        (; ast, data = sf.data, meta))
    return SweepResult(c.id, c.kind, probes, verdict, "")
end

# ------------------------------------------------------ brm-probe-kind path

function _write_worker_spec(c::SweepCase, mani::SweepManifest, out_dir)
    spec = Dict{String,Any}(
        "id" => c.id, "brm_inputs" => c.brm_inputs,
        "manifest" => mani.path, "pins" => Dict{String,String}(mani.pins))
    c.uprobes !== nothing && (spec["u_probes"] = c.uprobes)
    spec_path = _case_path(joinpath(out_dir, ".worker"), c.id, ".in.toml")
    mkpath(dirname(spec_path))
    open(spec_path, "w") do io
        TOML.print(io, spec)
    end
    return spec_path
end

function _run_worker(c::SweepCase, spec_path, case_out)
    mkpath(case_out)
    cmd = Cmd(vcat(c.worker, ["--spec", spec_path, "--out", case_out]))
    outbuf, errbuf = IOBuffer(), IOBuffer()
    run_ok = try
        run(pipeline(cmd; stdout = outbuf, stderr = errbuf))
        true
    catch err
        err isa InterruptException && rethrow()
        false
    end
    log = join([
        "command: $(join(cmd.exec, " "))",
        "exit: $(run_ok ? "0" : "nonzero (see stderr)")",
        "--- stdout ---",
        String(take!(outbuf)),
        "--- stderr ---",
        String(take!(errbuf)),
        "",
    ], "\n")
    return run_ok, log
end

_worker_case_error(id, what) = ErrorException("worker case \"$id\": $what")

function _read_worker_numbers(case_out, c::SweepCase)
    sec_path = joinpath(case_out, "sections.md")
    isfile(sec_path) ||
        throw(_worker_case_error(c.id, "worker produced no sections.md"))
    num_path = joinpath(case_out, "numbers.toml")
    isfile(num_path) ||
        throw(_worker_case_error(c.id, "worker produced no numbers.toml"))
    art_path = joinpath(case_out, "artifact.jls")
    isfile(art_path) ||
        throw(_worker_case_error(c.id, "worker produced no artifact.jls"))
    numbers = try
        TOML.parsefile(num_path)
    catch err
        throw(_worker_case_error(c.id,
            "numbers.toml is not valid TOML ($(sprint(showerror, err)))"))
    end
    for k in keys(numbers)
        k in ("probe", "pins") ||
            throw(_worker_case_error(c.id, "numbers.toml has unknown key \"$k\""))
    end
    haskey(numbers, "probe") ||
        throw(_worker_case_error(c.id, "numbers.toml has no [[probe]] entries"))
    praw = numbers["probe"]
    praw isa AbstractVector && !isempty(praw) ||
        throw(_worker_case_error(c.id, "numbers.toml [[probe]] must be non-empty"))
    if c.uprobes !== nothing && length(praw) != length(c.uprobes)
        throw(_worker_case_error(c.id, "worker returned $(length(praw)) " *
            "probe(s) for $(length(c.uprobes)) u_probes"))
    end
    if c.expected !== nothing && length(praw) != length(c.expected)
        throw(_worker_case_error(c.id, "worker returned $(length(praw)) " *
            "probe(s) for $(length(c.expected)) expected value(s)"))
    end
    wpins = Dict{String,String}()
    if haskey(numbers, "pins")
        numbers["pins"] isa AbstractDict ||
            throw(_worker_case_error(c.id, "numbers.toml [pins] must be a table"))
        for (k, v) in numbers["pins"]
            v isa AbstractString ||
                throw(_worker_case_error(c.id, "worker pin \"$k\" must be a string"))
            wpins[String(k)] = String(v)
        end
    end
    probes = SweepProbeResult[]
    for (i, p) in enumerate(praw)
        p isa AbstractDict ||
            throw(_worker_case_error(c.id, "probe $i must be a table"))
        for k in keys(p)
            k in ("posterior", "grad", "grad_maxdiff", "sb_value",
                "sb_grad_maxdiff", "oracle") ||
                throw(_worker_case_error(c.id, "probe $i has unknown key \"$k\""))
        end
        for k in ("posterior", "grad")
            haskey(p, k) || throw(_worker_case_error(c.id,
                "probe $i is missing \"$k\""))
        end
        val = p["posterior"]
        val isa Real ||
            throw(_worker_case_error(c.id, "probe $i posterior must be a number"))
        val = Float64(val)
        grad = p["grad"]
        grad in _SWEEP_GRADS ||
            throw(_worker_case_error(c.id, "probe $i grad must be one of " *
                "$(_SWEEP_GRADS), got $(repr(grad))"))
        gmd = haskey(p, "grad_maxdiff") ? Float64(p["grad_maxdiff"]) : nothing
        gmd !== nothing && !(p["grad_maxdiff"] isa Real) &&
            throw(_worker_case_error(c.id, "probe $i grad_maxdiff must be a number"))
        sb = nothing
        if haskey(p, "sb_value")
            p["sb_value"] isa Real && isfinite(Float64(p["sb_value"])) ||
                throw(_worker_case_error(c.id, "probe $i sb_value must be finite"))
            sb = val - Float64(p["sb_value"])
        end
        exp = c.expected === nothing ? nothing : c.expected[i]
        ushow = c.uprobes === nothing ? "origin (worker-expanded)" : repr(c.uprobes[i])
        push!(probes, SweepProbeResult(ushow, val, exp,
            exp === nothing ? NaN : val - exp, String(grad), gmd, sb,
            _probe_verdict(val, exp, String(grad))))
    end
    return read(sec_path, String), probes, art_path, wpins
end

function _run_brm_case(c::SweepCase, mani::SweepManifest, out_dir)
    spec_path = _write_worker_spec(c, mani, out_dir)
    case_out = _case_path(joinpath(out_dir, ".worker"), c.id, ".out")
    run_ok, log = _run_worker(c, spec_path, case_out)
    log_path = _case_path(out_dir, c.id, ".worker.log")
    mkpath(dirname(log_path))
    write(log_path, log)
    run_ok || throw(_worker_case_error(c.id,
        "worker exited nonzero (see $(c.id).worker.log)"))
    sections, probes, art_path, worker_pins = _read_worker_numbers(case_out, c)
    verdict = _case_verdict(probes)
    extra = ["brm_inputs = \"$(c.brm_inputs)\"",
        "worker = \"$(join(c.worker, " "))\""]
    md = join([
        _closeout_head(c.id, c.kind, mani.pins, mani.path, c.supersedes,
            c.based_on, extra)...,
        sections,
        _resweep_block(probes, verdict)...,
    ], "\n")
    md_path = _case_path(out_dir, c.id, ".md")
    mkpath(dirname(md_path))
    write(md_path, md)
    cp(art_path, _case_path(out_dir, c.id, ".jls"); force = true)
    return SweepResult(c.id, c.kind, probes, verdict, ""), worker_pins
end

# ------------------------------------------------------------------ outputs

function _probe_index_dict(pr::SweepProbeResult)
    d = Dict{String,Any}(
        "u" => pr.u_show, "posterior" => pr.posterior,
        "grad" => pr.grad, "verdict" => pr.verdict)
    pr.expected !== nothing && (d["expected"] = pr.expected;
        d["delta"] = pr.delta)
    pr.grad_maxdiff !== nothing && (d["grad_maxdiff"] = pr.grad_maxdiff)
    pr.sb_delta !== nothing && (d["sb_delta"] = pr.sb_delta)
    return d
end

function _write_sweep_index(mani::SweepManifest, results, worker_pins, out_dir)
    cases_out = Dict{String,Any}[]
    for r in results
        d = Dict{String,Any}(
            "id" => r.id,
            "kind" => r.kind === :brm ? "brm-probe" : "surface",
            "file" => "$(r.id).md", "verdict" => r.verdict)
        r.verdict != "ERROR" && (d["artifact"] = "$(r.id).jls")
        r.note != "" && (d["note"] = r.note)
        c = findfirst(x -> x.id == r.id, mani.cases)
        if c !== nothing
            mc = mani.cases[c]
            mc.supersedes !== nothing && (d["supersedes"] = mc.supersedes)
            mc.based_on !== nothing && (d["based_on"] = mc.based_on)
            mc.brm_inputs !== nothing && (d["brm_inputs"] = mc.brm_inputs)
        end
        haskey(worker_pins, r.id) && (d["worker_pins"] = worker_pins[r.id])
        d["probe"] = [_probe_index_dict(p) for p in r.probes]
        push!(cases_out, d)
    end
    index = Dict{String,Any}(
        "generator" => Dict{String,Any}(
            "name" => _SWEEP_NAME,
            "version" => _SWEEP_VERSION,
            "manifest" => mani.path),
        "pins" => Dict{String,String}(mani.pins),
        "case" => cases_out)
    open(joinpath(out_dir, "manifest.toml"), "w") do io
        TOML.print(io, index)
    end
end

function _write_sweep_table(mani::SweepManifest, results, out_dir)
    n_ok = count(r -> r.verdict == "VERIFY_OK", results)
    n_probes = sum(length(r.probes) for r in results)
    lines = [
        "# Sweep verdict table",
        "",
        "generator = \"$(_SWEEP_NAME)\" v$(_SWEEP_VERSION); " *
            "manifest = \"$(mani.path)\"; " *
            "cases = $(length(results)) (VERIFY_OK $n_ok), probes = $n_probes",
        "",
        "| case | probe | kind | posterior | expected | Δ | gradient | SB Δ | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results
        kind = r.kind === :brm ? "brm-probe" : "surface"
        if r.verdict == "ERROR"
            push!(lines, "| $(r.id) | — | $kind | ERROR | — | — | ERROR | — | ERROR |")
        else
            for (i, pr) in enumerate(r.probes)
                exp = pr.expected === nothing ? "—" : repr(pr.expected)
                d = pr.expected === nothing ? "—" : repr(pr.delta)
                push!(lines, "| $(r.id) | $i | $kind | $(repr(pr.posterior)) " *
                    "| $exp | $d | $(pr.grad) | $(_show_sweep_opt(pr.sb_delta)) " *
                    "| $(pr.verdict) |")
            end
        end
    end
    push!(lines, "")
    write(joinpath(out_dir, "verdict-table.md"), join(lines, "\n"))
end

"""
    sweep_appends(manifest_path, out_dir; backend=nothing) -> (; results, ok)

Run the sweep: one real pipeline + machine render per manifest case, plus
`<id>.jls` artifacts, `manifest.toml`, and `verdict-table.md` in `out_dir`
(created when missing). Surface-kind cases run in-process; brm-probe-kind
cases shell out to their manifest `worker` script (argv array, no shell).
`backend` (any `DifferentiationInterface` AD type) enables the per-case
gradient cross-check for surface-kind cases. Per-case failures are caught
into ERROR rows; `ok` is true iff every case reaches VERIFY_OK.
Manifest-level problems throw before anything is written.
"""
function sweep_appends(manifest_path::AbstractString, out_dir::AbstractString;
        backend = nothing)
    mani = load_manifest(manifest_path)
    mkpath(out_dir)
    results = SweepResult[]
    worker_pins = Dict{String,Dict{String,String}}()
    for c in mani.cases
        r = try
            if c.kind === :brm
                rr, wpins = _run_brm_case(c, mani, out_dir)
                worker_pins[rr.id] = wpins
                rr
            else
                _run_surface_case(c, mani, out_dir; backend = backend)
            end
        catch err
            err isa InterruptException && rethrow()
            note = sprint(showerror, err)
            md_path = _case_path(out_dir, c.id, ".md")
            mkpath(dirname(md_path))
            write(md_path, _sweep_error_md(c.id, c.kind, mani.pins, mani.path,
                c.supersedes, c.based_on, note))
            SweepResult(c.id, c.kind, SweepProbeResult[], "ERROR", note)
        end
        push!(results, r)
    end
    _write_sweep_index(mani, results, worker_pins, out_dir)
    _write_sweep_table(mani, results, out_dir)
    return (; results, ok = all(r -> r.verdict == "VERIFY_OK", results))
end

# ------------------------------------------------------------------- CLI

function sweep_main(argv::Vector{String} = ARGS)
    manifest_path = nothing
    out_dir = nothing
    i = 1
    while i <= length(argv)
        flag = argv[i]
        _need() = i + 1 <= length(argv) ||
            throw(ArgumentError("flag $flag needs a value"))
        if flag == "--manifest"
            _need(); i += 1; manifest_path = argv[i]
        elseif flag == "--out"
            _need(); i += 1; out_dir = argv[i]
        else
            throw(ArgumentError("unknown flag $flag (want --manifest PATH --out DIR)"))
        end
        i += 1
    end
    manifest_path === nothing &&
        throw(ArgumentError("sweep needs --manifest PATH --out DIR"))
    out_dir === nothing &&
        throw(ArgumentError("sweep needs --manifest PATH --out DIR"))
    return sweep_appends(manifest_path, out_dir).ok ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(sweep_main(copy(ARGS)))
end
