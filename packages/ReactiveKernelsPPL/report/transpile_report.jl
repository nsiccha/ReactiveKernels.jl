# Transpile reports: surface/AST + data in, markdown out.
#
# Machine-generated Layer 3 → Layer 4 evidence for BRM→RK transpilation.
# Two input modes:
#   --surface model.jl --data data.jl   hand-written/demo path: model.jl holds
#                                       the `@rkppl begin ... end` block (or a
#                                       bare `begin ... end` block), data.jl
#                                       defines `DATA` (NamedTuple or Dict).
#   --artifact model.jls                BRM path: `Serialization.serialize`d
#                                       `(; ast::Expr, data, meta)` as emitted
#                                       by the BRM RK backend (`_rk_emit_ast`
#                                       plus the raw columns). `meta` is any
#                                       NamedTuple/Dict (shown verbatim).
# Output: markdown to `--out PATH` or stdout.
#
# Every report runs the REAL pipeline (`lower_rkppl` → `bind_data` →
# `build_kernel` → `prepare_query` → call) and states the posterior value at
# the probe point. A gradient cross-check (AD vs central differences) runs
# only when `transpile_report` gets a `backend` (the CLI env has no AD
# backend; the test env passes Enzyme).

using ReactiveKernelsPPL
using Serialization

const _REPORT_VERSION = 1

# ---------------------------------------------------------------- data input

function _norm_cols(data)
    if data isa NamedTuple
        return Dict{Symbol,AbstractVector}(k => _norm_col(k, v) for (k, v) in pairs(data))
    elseif data isa AbstractDict
        return Dict{Symbol,AbstractVector}(
            _norm_key(k) => _norm_col(k, v) for (k, v) in data)
    end
    throw(ArgumentError(
        "report data must be a NamedTuple or Dict of columns, got $(typeof(data))"))
end

_norm_key(k::Symbol) = k
_norm_key(k::AbstractString) = Symbol(k)
_norm_key(k) = throw(ArgumentError("report data keys must be Symbols, got $(repr(k))"))

_norm_col(k, v::AbstractVector) = v
_norm_col(k, v) = throw(ArgumentError(
    "report data column $k must be an AbstractVector, got $(typeof(v))"))

# ------------------------------------------------------------- AST fidelity

_strip_lines(x) = x
_strip_lines(s::Symbol) = s
function _strip_lines(ex::Expr)
    return Expr(ex.head, (a isa LineNumberNode ? nothing : _strip_lines(a)
        for a in ex.args if !(a isa LineNumberNode))...)
end

function _assert_same_shape(parsed::Expr, given::Expr, what::String)
    _strip_lines(parsed) == _strip_lines(given) && return nothing
    throw(ArgumentError("$what does not match the input AST " *
                        "(modulo line numbers) — refusing to report a " *
                        "Layer 3 the pipeline did not run"))
end

"""Extract the `begin ... end` block from surface text (`@rkppl ...` or bare)."""
function _surface_block(text::AbstractString)
    ex = Meta.parseall(text)
    # A file parses to `:toplevel` (line nodes included); unwrap the
    # single form it holds.
    if Meta.isexpr(ex, :toplevel)
        forms = [a for a in ex.args if a isa Expr]
        length(forms) == 1 || throw(ArgumentError(
            "surface text must hold exactly one form, got $(length(forms))"))
        ex = forms[1]
    end
    Meta.isexpr(ex, :macrocall) && ex.args[1] === Symbol("@rkppl") || return _as_block(ex)
    length(ex.args) >= 3 || throw(ArgumentError(
        "surface text is an @rkppl call without a block"))
    return _as_block(ex.args[3])
end

_as_block(ex::Expr) = ex.head === :block ? ex :
    throw(ArgumentError("surface text must be a `begin ... end` block, " *
                        "got `$(ex.head)`"))
_as_block(x) = throw(ArgumentError(
    "surface text must be a `begin ... end` block, got $(typeof(x))"))

_render_ast(ast::Expr) = "@rkppl " * sprint(Base.show_unquoted, ast)

# ------------------------------------------------------------ plan summary

_show_cols(cols) = "[" * join(string.(cols), ", ") * "]"
_show_expr(e::Expr) = sprint(Base.show_unquoted, e)
_show_expr(e) = repr(e)

function _show_plan(plan::StructuralPlan)
    rs = ["($(r.family) $(r.link), response $(r.response), " *
          "predictor $(r.predictor), scale $(_show_scale(r.scale)), " *
          "weights $(r.weights === nothing ? "nothing" : string(r.weights)), " *
          "evidence ($(r.evidence.kind), $(_show_ev(r.evidence.lower)), " *
          "$(_show_ev(r.evidence.upper))), " *
          "range $(r.range === nothing ? "eachindex" : string(r.range)))" for r in plan.responses]
    ps = ["($(p.name), $(p.link), terms [$([string(t.kind) * " " *
          _show_cols(t.columns) for t in p.terms] |> x -> join(x, ", "))])"
        for p in plan.predictors]
    prs = ["($(p.predictor), $(p.addressee), $(p.location), $(p.scale))"
        for p in plan.population_priors]
    smps = ["($(p.name), $(p.family), $(_show_args(p.args)))" for p in plan.parameters]
    as = ["($(a.name) = $(_show_expr(a.expr)))" for a in plan.assignments]
    ds = ["($(d.name) = $(_show_expr(d.expr)))" for d in plan.derived]
    return join([
        "n_obs = $(plan.n_obs)",
        "responses   = [$(join(rs, ", "))]",
        "predictors  = [$(join(ps, ", "))]",
        "priors      = [$(join(prs, ", "))]",
        "parameters  = [$(join(smps, ", "))]",
        "assignments = [$(join(as, ", "))]",
        "derived     = [$(join(ds, ", "))]",
        "roles       = $(plan.roles)",
    ], "\n")
end

_show_scale(s::Nothing) = "nothing"
_show_scale(s::Real) = repr(s)
_show_scale(s::Symbol) = ":$s"
_show_ev(::Nothing) = "nothing"
_show_ev(x) = repr(x)
_show_args(nt::NamedTuple) = "(" * join(["$k=$(repr(v))" for (k, v) in pairs(nt)], ", ") * ")"

# --------------------------------------------------------------- pipeline

function _report_findiff(f, u)
    h = cbrt(eps(Float64))
    g = similar(u, Float64)
    for i in eachindex(u)
        up, dn = copy(u), copy(u)
        up[i] += h
        dn[i] -= h
        g[i] = (f(up) - f(dn)) / (2h)
    end
    return g
end

function _run_pipeline(ast::Expr, cols::Dict{Symbol,AbstractVector}, u_probe;
        backend = nothing)
    plan = lower_rkppl(ast, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    n = built.layout.total
    u = u_probe === nothing ? zeros(Float64, n) : Vector{Float64}(u_probe)
    length(u) == n || throw(ArgumentError(
        "probe point has length $(length(u)), layout needs $n"))
    kern = prepare_query(built, bound, :sampler)
    val = Base.invokelatest(kern, u)
    grad_line = if backend === nothing
        "gradient cross-check: not run (no AD backend in this environment)"
    else
        q = prepare_sampler(built, bound, u; backend = backend)
        v2, g = sampler_value_and_gradient!(q, similar(u), u)
        ref = _report_findiff(w -> Base.invokelatest(kern, w), u)
        ok = all(isfinite, g) && isapprox(g, ref; rtol = 1e-5, atol = 1e-7)
        err = maximum(abs.(g .- ref))
        "gradient cross-check: AD vs central differences, max|Δ| = $err " *
            (ok ? "PASS" : "FAIL")
    end
    kex = kernel_expr(bound, built.layout)
    return (; plan = bound, kernel = kex, u, val, grad_line)
end

# ----------------------------------------------------------------- report

_show_meta(meta::NamedTuple) = join(["$k = $(repr(v))" for (k, v) in pairs(meta)], "\n")
_show_meta(meta::AbstractDict) =
    join(["$k = $(repr(v))" for (k, v) in sort!(collect(meta); by = first)], "\n")
_show_meta(meta) = repr(meta)

"""
    transpile_report(surface, data; meta, u, backend) -> String

Run the full thin-layer pipeline on surface text (or an `Expr` AST) plus
data columns and render the markdown Layer 3 → Layer 4 report. `meta`
labels the run; `u` overrides the default zero probe point; `backend`
(any `DifferentiationInterface` AD type) enables the gradient cross-check.
"""
function transpile_report end

function transpile_report(surface::AbstractString, data;
        meta = (; source = "surface text"), u = nothing, backend = nothing)
    ast = _surface_block(surface)
    # The report prints the input text verbatim, so fidelity is by
    # construction; still confirm the text parses to one block (done above).
    return _report(ast, _norm_cols(data), string(strip(surface)), meta, u, backend,
        "parsed surface block lowered directly (no separate AST input)")
end

function transpile_report(ast::Expr, data;
        meta = (; source = "emitter AST"), u = nothing, backend = nothing)
    layer3 = _render_ast(ast)
    # The printed Layer 3 must re-parse to the AST the pipeline ran.
    _assert_same_shape(_surface_block(layer3), ast, "rendered Layer 3")
    return _report(ast, _norm_cols(data), layer3, meta, u, backend,
        "rendered Layer 3 re-parses to the input AST (modulo line numbers)")
end

function _report(ast, cols, layer3, meta, u, backend, fidelity)
    out = _run_pipeline(ast, cols, u; backend = backend)
    val_note = isfinite(out.val) ? "finite" : "NON-FINITE"
    return join([
        "# Transpile report",
        "",
        _show_meta(meta),
        "report_version = $(_REPORT_VERSION)",
        "",
        "## Layer 3 — `@rkppl` surface",
        "",
        "```julia",
        layer3,
        "```",
        "",
        "Fidelity: $fidelity.",
        "",
        "## Boundary — `lower_rkppl` → `bind_data`",
        "",
        "```",
        _show_plan(out.plan),
        "```",
        "",
        "## Layer 4 — emitted RK kernel (verbatim)",
        "",
        "```julia",
        sprint(Base.show_unquoted, out.kernel),
        "```",
        "",
        "## Verification",
        "",
        "- probe point: `u = $(repr(out.u))`",
        "- `posterior(u) = $(repr(out.val))` ($val_note)",
        "- $(out.grad_line)",
        "",
    ], "\n")
end

# ------------------------------------------------------------------- CLI

function load_artifact(path::AbstractString)
    payload = Serialization.deserialize(path)
    keys_of(p) = p isa NamedTuple ? keys(p) : p isa AbstractDict ? keys(p) : ()
    get_of(p, k, d) = p isa NamedTuple ? get(p, k, d) :
        p isa AbstractDict ? get(p, k, d) : d
    ks = keys_of(payload)
    if !(:ast in ks && :data in ks)
        throw(ArgumentError("artifact $path must serialize `(; ast::Expr, data, meta)`, " *
                            "got keys $ks"))
    end
    ast = get_of(payload, :ast, nothing)
    ast isa Expr || throw(ArgumentError(
        "artifact $path field `ast` must be an Expr, got $(typeof(ast))"))
    return (; ast, data = get_of(payload, :data, nothing),
        meta = get_of(payload, :meta, (; source = "artifact $path")))
end

function load_surface_files(model_path::AbstractString, data_path::AbstractString)
    surface = read(model_path, String)
    mod = Module(:ReportData)
    Base.include(mod, data_path)
    isdefined(mod, :DATA) || throw(ArgumentError(
        "data file $data_path must define `DATA`"))
    return (; surface, data = mod.DATA)
end

function _parse_u(s::AbstractString)
    return Float64[parse(Float64, strip(x)) for x in split(s, ",")]
end

function main(argv::Vector{String} = ARGS)
    artifact = nothing
    surface_path = nothing
    data_path = nothing
    out_path = nothing
    u = nothing
    model = nothing
    i = 1
    while i <= length(argv)
        flag = argv[i]
        _need() = i + 1 <= length(argv) ||
            throw(ArgumentError("flag $flag needs a value"))
        if flag == "--artifact"
            _need(); i += 1; artifact = argv[i]
        elseif flag == "--surface"
            _need(); i += 1; surface_path = argv[i]
        elseif flag == "--data"
            _need(); i += 1; data_path = argv[i]
        elseif flag == "--out"
            _need(); i += 1; out_path = argv[i]
        elseif flag == "--u"
            _need(); i += 1; u = _parse_u(argv[i])
        elseif flag == "--model"
            _need(); i += 1; model = argv[i]
        else
            throw(ArgumentError("unknown flag $flag (want --artifact | " *
                                "--surface/--data, [--out], [--u], [--model])"))
        end
        i += 1
    end
    if artifact !== nothing
        (surface_path === nothing && data_path === nothing) ||
            throw(ArgumentError("pass either --artifact or --surface/--data, not both"))
        art = load_artifact(artifact)
        meta = model === nothing ? art.meta : merge((; model), _meta_nt(art.meta))
        md = transpile_report(art.ast, art.data; meta = meta, u = u)
    else
        (surface_path !== nothing && data_path !== nothing) ||
            throw(ArgumentError("surface mode needs --surface MODEL.jl --data DATA.jl"))
        sf = load_surface_files(surface_path, data_path)
        meta = (; source = "surface $(basename(surface_path))",
            model = model === nothing ? basename(surface_path) : model)
        md = transpile_report(sf.surface, sf.data; meta = meta, u = u)
    end
    if out_path === nothing
        print(md)
    else
        write(out_path, md)
    end
    return 0
end

_meta_nt(m::NamedTuple) = m
_meta_nt(m::AbstractDict) = NamedTuple{Tuple(Symbol.(keys(m)))}(Tuple(values(m)))
_meta_nt(m) = (; source = repr(m))

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(copy(ARGS)))
end
