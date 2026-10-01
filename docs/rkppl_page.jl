# `@rkppl` hand-authoring page (docs/src/rkppl.md). Every panel lowers,
# builds, and runs its displayed program at docs-build time. Canonical
# programs come verbatim from ReactiveKernelsPPL's corpus
# (`packages/ReactiveKernelsPPL/test/corpus/`, the surface→plan drift guard),
# so the page cannot drift from the spelling the package pins.

using ReactiveKernelsPPL: ReactiveKernelsPPL, build_kernel, kernel_expr,
    lower_rkppl, prepare_query

const _RKPPL_CORPUS = joinpath(pkgdir(ReactiveKernelsPPL), "test", "corpus")

# Small deterministic data columns, shown in each displayed program.
const _RKPPL_DATA = Dict(
    :y => "[1.0, 2.0, 1.5, 2.5, 3.0, 2.0]",
    :x => "[0.5, -1.0, 1.5, 0.0, -0.5, 1.0]",
    :x1 => "[0.5, -1.0, 1.5, 0.0, -0.5, 1.0]",
    :x2 => "[1.0, 0.5, -0.5, 2.0, 0.0, -1.0]",
    :g => "[1, 2, 1, 3, 2, 3]",
    :o => "[0.1, -0.2, 0.0, 0.3, -0.1, 0.2]",
)

"""Return `(data_names, block_source)` of one corpus program."""
function _rkppl_corpus_program(file::AbstractString)
    src = replace(read(joinpath(_RKPPL_CORPUS, file), String), "\r\n" => "\n")
    lines = split(chomp(src), '\n')
    header = first(lines)
    startswith(header, "# data:") ||
        error("corpus program $file has no `# data:` header")
    names = Symbol.(split(header[length("# data:")+1:end]))
    (names, join(lines[2:end], '\n'))
end

function _rkppl_displayed(name::Symbol, origin::AbstractString,
                          names::Vector{Symbol}, block::AbstractString;
                          preamble::AbstractString = "")
    data = join(("$n = $(_RKPPL_DATA[n])" for n in names), "\n")
    kw = join(string.(names), ", ")
    string(
        isempty(preamble) ? "" : preamble * "\n\n",
        data, "\n\n",
        "model = @rkppl ", block, "\n\n",
        "plan = model(; ", kw, ")\n",
        "built = build_kernel(plan)\n",
        "kernel = prepare_query(built, plan, :sampler)\n",
        "u = collect(range(-0.5, 0.5; length = built.layout.total))\n",
        "docs_example = (; name = :", name, ", origin = \"", origin, "\",\n",
        "    inputs = (u,), kernel, output = kernel(u))",
    )
end

function _rkppl_module()
    mod = Module(gensym(:RKPPLDocs))
    Core.eval(mod, :(using ReactiveKernelsPPL))
    mod
end

"""
    render_rkppl_corpus_example(file, name; preamble = "") -> Markdown.MD

Execute one canonical corpus program exactly as displayed (data, model,
lowering/binding, build, sampler cut, call) and render the standard
source / generated-kernel / DAG panel.
"""
function render_rkppl_corpus_example(file::AbstractString, name::Symbol;
                                     preamble::AbstractString = "")
    names, block = _rkppl_corpus_program(file)
    origin = joinpath("packages", "ReactiveKernelsPPL", "test", "corpus", file)
    execute_example(_rkppl_module(),
        _rkppl_displayed(name, origin, names, block; preamble))
end

"""
    render_rkppl_kernel_program(file) -> Markdown.MD

Show the `@kernel` program `build_kernel` generates for one corpus program —
the model the layer actually compiles (`kernel_expr(plan, layout)`).
"""
function render_rkppl_kernel_program(file::AbstractString)
    names, block = _rkppl_corpus_program(file)
    ast = Meta.parse(block)
    cols = Dict{Symbol,AbstractVector}(n => eval(Meta.parse(_RKPPL_DATA[n]))
        for n in names)
    plan = ReactiveKernelsPPL.bind_data(lower_rkppl(ast, names), cols)
    built = build_kernel(plan)
    program = sprint(Base.show_unquoted,
        Base.remove_linenums!(kernel_expr(plan, built.layout));
        context = :limit => false)
    Markdown.MD(Any[Markdown.Code("julia", "@kernel " * program)])
end

const _RKPPL_UNDECLARED = """
model = @rkppl begin
    a ~ Normal(0, 1)
    mu = a .+ bb .* x        # `bb` was never declared
    y .~ Normal.(mu, 1.0)
end"""

"""
    render_rkppl_strict_error() -> Markdown.MD

Run a program with an undeclared name and show the exact error it raises.
"""
function render_rkppl_strict_error()
    mod = _rkppl_module()
    _evaluate_source(mod, _RKPPL_UNDECLARED)
    data = (; y = eval(Meta.parse(_RKPPL_DATA[:y])),
        x = eval(Meta.parse(_RKPPL_DATA[:x])))
    model = Core.eval(mod, :model)
    err = try
        Base.invokelatest(model; data...)
        nothing
    catch e
        e
    end
    err isa ReactiveKernelsPPL.SurfaceLoweringError || error(
        "the undeclared-name example did not raise a SurfaceLoweringError " *
        "(got $(repr(err)))")
    Markdown.MD(Any[
        Markdown.Code("julia", _RKPPL_UNDECLARED * "\n\nmodel(; y, x)"),
        Markdown.Code("text", sprint(showerror, err)),
    ])
end

const _RKPPL_SUBMODEL_PREAMBLE = """
using ReactiveKernelsPPL

# A reusable submodel: a positive scale with its own prior.
@rkppl half_scale(rate) = begin
    s ~ Exponential(rate)
    s
end"""

const _RKPPL_SUBMODEL_BLOCK = """
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    sigma ~ half_scale(1.0)
    mu = a .+ b .* x
    y .~ Normal.(mu, sigma)
end"""

"""Execute and render the docs-owned submodel example."""
function render_rkppl_submodel_example()
    execute_example(_rkppl_module(),
        _rkppl_displayed(:rkppl_submodel, "docs/rkppl_page.jl", [:y, :x],
            _RKPPL_SUBMODEL_BLOCK; preamble = _RKPPL_SUBMODEL_PREAMBLE))
end
