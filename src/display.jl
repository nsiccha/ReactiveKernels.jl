# Human-readable display of graphs, plans and generated code.
#
# Every view here is a cold-path rendering of an existing graph, plan or
# kernel: nothing is planned differently, lowered again for execution, or
# evaluated. `code_expr` stays the exact executable AST; `readable_code` is its
# display counterpart, and the `Graph` text listing is the structural view.

# Line-number annotations are source metadata, not code. Removing them from
# `:block`/`:quote` arguments and blanking a macro call's line slot matches
# `Base.remove_linenums!`, but builds a fresh expression tree so the caller's
# expression (for example a retained `kernel_expr` or a recipe source) is never
# mutated. Leaves are shared, never written.
_display_expr(x) = x
function _display_expr(ex::Expr)
    args = Any[]
    if ex.head === :macrocall
        for (i, arg) in enumerate(ex.args)
            push!(args, i == 2 && arg isa LineNumberNode ? nothing : _display_expr(arg))
        end
    else
        drop = ex.head === :block || ex.head === :quote
        for arg in ex.args
            drop && arg isa LineNumberNode && continue
            push!(args, _display_expr(arg))
        end
    end
    # Base prints a short definition `f(x) = y` inline only while its body
    # block still carries the line node; keep that one-statement body inline.
    if ex.head === :(=) && length(args) == 2 && args[2] isa Expr &&
            args[2].head === :block && length(args[2].args) == 1
        args[2] = only(args[2].args)
    end
    Expr(ex.head, args...)
end

# Operations spliced into a source as objects (`$(QuoteNode(op))`, as composed
# endpoint calls are) print as their parameterized type by default. Show a
# named function by its name and a captured `@kernel` source by its retained
# source as an anonymous function; every other quoted value prints as before.
_display_expr(node::QuoteNode) = _display_quoted(node, node.value)
_display_quoted(node::QuoteNode, value) = node
_display_quoted(::QuoteNode,
                op::Union{Function,_KernelSourceOp,_KernelSourceFunction,_KernelBranch}) =
    _display_object(op)
_display_quoted(::QuoteNode, child::_KernelPreparedChild) = child.name

# A global the reader knows by its bare Base name is shown by that name when
# the binding is the same object; any other global keeps its module path.
function _display_expr(ref::GlobalRef)
    name = ref.name
    if Base.isexported(Base, name) && isdefined(Base, name) && isdefined(ref.mod, name) &&
            getglobal(ref.mod, name) === getglobal(Base, name)
        return name
    end
    ref
end

# Any other global prints by its bare name, as its source spelled it, when the
# displayed program spells that name no other way: no global of another module
# and no local, argument or field. The display then names the global's module
# among the modules it was evaluated in (`ReadableCode.modules`).
function _display_globals(ex, modules::Vector{Module})
    refs = _display_global_refs!(Tuple{Module,Symbol}[], ex)
    isempty(refs) && return ex, modules
    spelled = _display_plain_symbols!(Set{Symbol}(), ex)
    owners = Dict{Symbol,Set{Module}}()
    for (mod, name) in refs
        push!(get!(Set{Module}, owners, name), mod)
    end
    bare = Set{Tuple{Module,Symbol}}(r for r in refs
        if length(owners[r[2]]) == 1 && !(r[2] in spelled))
    isempty(bare) && return ex, modules
    out = copy(modules)
    for (mod, name) in refs
        (mod, name) in bare && !(mod in out) && push!(out, mod)
    end
    return _display_bare_globals(ex, bare), out
end

_display_global_refs!(out, x) = out
_display_global_refs!(out, ref::GlobalRef) =
    ((ref.mod, ref.name) in out || push!(out, (ref.mod, ref.name)); out)
_display_global_refs!(out, ex::Expr) =
    (foreach(a -> _display_global_refs!(out, a), ex.args); out)

_display_plain_symbols!(out, x) = out
_display_plain_symbols!(out, s::Symbol) = push!(out, s)
_display_plain_symbols!(out, ex::Expr) =
    (foreach(a -> _display_plain_symbols!(out, a), ex.args); out)

_display_bare_globals(x, bare) = x
_display_bare_globals(ref::GlobalRef, bare) =
    (ref.mod, ref.name) in bare ? ref.name : ref
_display_bare_globals(ex::Expr, bare) =
    Expr(ex.head, Any[_display_bare_globals(a, bare) for a in ex.args]...)

# Captured values in a displayed source binding: literals as themselves,
# functions and sources as above, anything else by its type name in brackets.
_display_object(x::Union{Number,AbstractString,Char,Symbol,Nothing}) = x isa Symbol ? QuoteNode(x) : x
_display_object(op::_KernelSourceOp) = _display_object(op.f)
_display_object(branch::_KernelBranch) = _display_object(branch.call)
function _display_object(f::Function)
    name = _readable_callee(f)
    name === :operation ? _opaque_object(f) : name
end
_display_object(x) = _opaque_object(x)
_opaque_object(x) = Symbol("<", nameof(typeof(x)), ">")

# `#x#2` is RuntimeGeneratedFunctions' renaming of the authored argument `x`.
_source_argument(name::Symbol) =
    (m = match(r"^#([^#].*)#\d+$", string(name))) === nothing ? name : Symbol(m.captures[1])
_rename_arguments(x, names) = x
_rename_arguments(x::Symbol, names) = get(names, x, x)
_rename_arguments(ex::Expr, names) =
    Expr(ex.head, (_rename_arguments(arg, names) for arg in ex.args)...)

function _display_object(f::_KernelSourceFunction)
    f.runtime_f isa RuntimeGeneratedFunctions.RuntimeGeneratedFunction ||
        return _opaque_object(f)
    lambda = RuntimeGeneratedFunctions.get_expression(f.runtime_f)
    lambda isa Expr && lambda.head === :-> && lambda.args[1] isa Expr &&
        !isempty(lambda.args[1].args) || return _opaque_object(f)
    environment, params... = lambda.args[1].args
    names = Dict{Symbol,Symbol}(p => _source_argument(p) for p in params if p isa Symbol)
    body = _rename_arguments(lambda.args[2], names)
    # The retained body is `let <captured names> = getfield(env, :name); body end`.
    # Show each captured value through this same renderer.
    if body isa Expr && body.head === :let && f.captures isa NamedTuple
        bindings = Any[Expr(:(=), name, _display_object(value))
                       for (name, value) in pairs(f.captures)]
        body = isempty(bindings) ? body.args[2] :
            Expr(:let, Expr(:block, bindings...), body.args[2])
    end
    Expr(:->, Expr(:tuple, (get(names, p, p) for p in params)...), _display_expr(body))
end

_display_text(x) = sprint(print, _display_expr(x); context = :limit => false)

# One line of authored source for listings and graph labels. A multi-line
# rendering (an `if`/`let`/`begin` block) is joined with `; `, which Julia
# parses back to the same block; nothing is shortened.
function _display_line(x)
    lines = split(_display_text(x), '\n')
    join((strip(line) for line in lines if !isempty(strip(line))), "; ")
end

_has_source(recipe::Recipe) = !(recipe.source isa _NoKernelSource)

# --- structural inventory ----------------------------------------------------------

"""
    recipe_kind(recipe::Recipe) -> Symbol

The structural kind of `recipe`: `:plate` for an authored `plate(...) do`
recipe, `:scan` for an authored `scan(...) do` recipe, and `:ordinary` for
every other recipe. It never throws. The body of a `:plate` recipe is
[`plate_body`](@ref)`(recipe)` and the body of a `:scan` recipe is
[`scan_body`](@ref)`(recipe)`; an `:ordinary` recipe has no body. These
symbols are the supported classification; the operation types behind them are
internal.
"""
recipe_kind(recipe::Recipe) = _recipe_kind(recipe.op)
_recipe_kind(::_AuthoredPlateOp) = :plate
_recipe_kind(::_AuthoredScanOp) = :scan
_recipe_kind(op) = :ordinary

_is_body_op(op) = _recipe_kind(op) !== :ordinary
# The body plan `plate_body`/`scan_body` return for a plate or scan recipe.
_body_plan(op::_AuthoredPlateOp) = op.kernel.plan
_body_plan(op::_AuthoredScanOp) = op.kernel.plan

"""
    recipe_inventory(spec::KernelSpec)
    recipe_inventory(graph::Graph)
    recipe_inventory(plan::Plan)
    recipe_inventory(kernel::PreparedKernel)

Every recipe of a program and, recursively, of each plate and scan body, as a
`Vector` of `(; kind, depth, parent, recipe)` entries in depth-first order: a
plate or scan entry is followed by the entries of its body. `kind` is
[`recipe_kind`](@ref)`(recipe)`; `depth` is `0` for a top-level recipe and one
more for each enclosing plate or scan; `parent` is the index in the returned
vector of the enclosing plate or scan entry, or `0` at top level.

A `KernelSpec` is inventoried through its graph ([`kernel_graph`](@ref)), so
every registered recipe is listed. A `Plan` lists the recipes it selected. A
`PreparedKernel` lists the recipes of the plan it was prepared from; with
`bound=` data that is the residual plan, in which each folded value is an
ordinary recipe. A body is always its plate's [`plate_body`](@ref) or its
scan's [`scan_body`](@ref) plan. The inventory only reads its argument; nothing
is planned, prepared or evaluated.

For example, the plate and scan structure of a program as `(kind, depth)` pairs:

```julia
[(entry.kind, entry.depth) for entry in recipe_inventory(spec) if entry.kind !== :ordinary]
```
"""
recipe_inventory(spec::KernelSpec) = recipe_inventory(kernel_graph(spec))
recipe_inventory(g::Graph) = _recipe_inventory(g.recipes)
recipe_inventory(p::Plan) = _recipe_inventory(p.recipes)
recipe_inventory(k::PreparedKernel) = recipe_inventory(k.plan)

const _InventoryEntry = NamedTuple{(:kind, :depth, :parent, :recipe),
                                   Tuple{Symbol,Int,Int,Recipe}}

_recipe_inventory(recipes) = _recipe_inventory!(_InventoryEntry[], recipes, 0, 0)

function _recipe_inventory!(entries::Vector{_InventoryEntry}, recipes, depth::Int,
                            parent::Int)
    for recipe in recipes
        kind = recipe_kind(recipe)
        push!(entries, (; kind, depth, parent, recipe))
        kind === :ordinary && continue
        _recipe_inventory!(entries, _body_plan(recipe.op).recipes, depth + 1,
                           length(entries))
    end
    entries
end

# The label of a recipe in `explain`, the graph listing and DAG node labels. A
# recipe synthesized from captured `@kernel` source shows that source as a
# function of the recipe's own inputs; every other operation keeps its name.
function _recipe_label(recipe::Recipe)
    _is_body_op(recipe.op) && return _opname(recipe.op)
    recipe.op isa _KernelSourceOp && _has_source(recipe) || return _opname(recipe.op)
    ins = join((string(name) for name in _recipe_source_names(recipe)), ", ")
    "($ins) -> " * _display_line(recipe.source)
end

# --- modules that define authored sources ------------------------------------

_source_module(op::_KernelSourceOp) = _source_module(op.f)
_source_module(f::_KernelSourceFunction) = _source_module(f.f)
_source_module(branch::_KernelBranch) = _source_module(branch.call)
_source_module(f::Function) = parentmodule(f)
_source_module(::Any) = nothing

function _collect_source_modules!(modules::Vector{Module}, recipes)
    for (; recipe) in _recipe_inventory(recipes)
        recipe.op isa _KernelSourceOp || continue
        mod = _source_module(recipe.op)
        mod isa Module && !(mod in modules) && push!(modules, mod)
    end
    modules
end

# --- readable code ---------------------------------------------------------------

"""
    ReadableCode

Display-only Julia source for a ReactiveKernels program, returned by
[`readable_code`](@ref). It prints as ordinary indented Julia without
line-number annotations; `string(code)` returns that text and `code.expr` the
displayed expression. `code.modules` lists the modules whose bindings the
authored `@kernel` sources were evaluated in, in order of first use, and the
module of each global the program shows by its bare name; the text starts
with a comment naming them.

The displayed program is an explanation, not executable authority:
[`code_expr`](@ref) remains the exact compiled AST.
"""
struct ReadableCode
    expr::Any
    modules::Vector{Module}
end

ReadableCode(expr) = ReadableCode(expr, Module[])

"""
    readable_code(spec::KernelSpec; have, want) -> ReadableCode
    readable_code(plan::Plan) -> ReadableCode
    readable_code(kernel::PreparedKernel) -> ReadableCode
    readable_code(expr::Expr; modules = Module[]) -> ReadableCode

Readable Julia for an existing program. A `KernelSpec` or `Plan` shows the
planned program over its HAVE inputs, with every operation slot replaced by its
authored source or operation name; a spec is planned at its default boundary
unless `have`/`want` are given, and nothing is prepared or evaluated. A
`PreparedKernel` shows its prepared program, including any `bound=` data folded
into it as literal constants. An `Expr`, such as a generated `@kernel`
definition, is shown as written. Line-number annotations are removed from a
copy; the argument is never mutated. A global reference (`M.f`) shows by its
bare name, as source spells it, unless the program also spells that name
another way; the comment then names `M`.
"""
function readable_code(p::Plan)
    modules = _collect_source_modules!(Module[], p.recipes)
    ReadableCode(_display_globals(_display_expr(
        _readable_inline_generated(_readable_expr(code_expr(p), p), p)), modules)...)
end

function readable_code(k::PreparedKernel)
    modules = _collect_source_modules!(Module[], k.lowered_recipes)
    ReadableCode(_display_globals(_display_expr(_readable_expr(code_expr(k), k)),
        modules)...)
end

readable_code(spec::KernelSpec; kwargs...) = readable_code(plan(spec; kwargs...))
readable_code(g::Graph; kwargs...) = readable_code(plan(g; kwargs...))
readable_code(ex::Expr; modules = Module[]) =
    ReadableCode(_display_globals(_display_expr(ex), collect(Module, modules))...)

# A definition with a block body (`function (args…) … end`, `f(args…) = begin
# … end`, optionally under a macro such as `@kernel`) prints its statements
# one level (four spaces) inside its header; Base's printer would indent the
# body of an `=` definition under a macro by twelve. Each statement keeps Base's
# own rendering, so nested blocks keep their relative indentation.
_is_block_definition(ex) = ex isa Expr && ex.head in (:function, :(=)) &&
    length(ex.args) == 2 && ex.args[2] isa Expr && ex.args[2].head === :block &&
    (ex.head === :function || ex.args[1] isa Expr && ex.args[1].head in (:call, :where))

function _print_code(io::IO, ex)
    if ex isa Expr && ex.head === :macrocall && length(ex.args) == 3 &&
            ex.args[2] === nothing && _is_block_definition(ex.args[3])
        print(io, ex.args[1], " ")
        return _print_code(io, ex.args[3])
    end
    _is_block_definition(ex) || return print(io, ex)
    signature, body = ex.args
    ex.head === :function ? print(io, "function ", signature) :
                            print(io, signature, " = begin")
    for statement in body.args
        text = sprint(print, statement; context = :limit => false)
        print(io, "\n    ", replace(text, "\n" => "\n    "))
    end
    print(io, "\nend")
end

function _print_readable(io::IO, code::ReadableCode)
    if !isempty(code.modules)
        print(io, "# authored sources evaluated in: ",
              join(string.(code.modules), ", "), "\n")
    end
    _print_code(IOContext(io, :limit => false), code.expr)
end

Base.print(io::IO, code::ReadableCode) = _print_readable(io, code)
Base.show(io::IO, code::ReadableCode) = _print_readable(io, code)
Base.show(io::IO, ::MIME"text/plain", code::ReadableCode) = _print_readable(io, code)

function _html_escape(s::AbstractString)
    replace(s, '&' => "&amp;", '<' => "&lt;", '>' => "&gt;", '"' => "&quot;")
end

function Base.show(io::IO, ::MIME"text/html", code::ReadableCode)
    print(io, "<pre class=\"rk-readable-code\"><code class=\"language-julia\">",
          _html_escape(sprint(_print_readable, code)), "</code></pre>")
end

# --- structural graph listing ------------------------------------------------------

function _graph_value_lines(g::Graph)
    lines = String[]
    for (id, aliases) in sort!(collect(_value_groups(g)); by = first)
        names = join(unique(string(v.name) for v in aliases), " ≡ ")
        types = join(unique(string(valtype(v)) for v in aliases), " ≡ ")
        push!(lines, "$names::$types")
    end
    lines
end

function _write_recipe_listing(io::IO, recipes, indent::String)
    for (; kind, depth, recipe) in _recipe_inventory(recipes)
        nested = indent * "      "^depth
        println(io, nested, "[", recipe.id, "] ", _recipe_line(recipe))
        kind === :ordinary && continue
        body = _body_plan(recipe.op)
        println(io, nested, "    ", _opname(recipe.op), " body: have (",
                join((string(v.name) for v in body.have), ", "), ") → want (",
                join((string(v.name) for v in body.want), ", "), ")")
    end
end

function _write_graph_listing(io::IO, g::Graph)
    values_text = _graph_value_lines(g)
    println(io, "Graph with ", length(values_text), " values and ",
            length(g.recipes), " recipes")
    println(io, "Values:")
    for line in values_text
        println(io, "  ", line)
    end
    print(io, "Recipes:")
    isempty(g.recipes) && return print(io, " (none)")
    println(io)
    buffer = IOBuffer()
    _write_recipe_listing(buffer, g.recipes, "  ")
    print(io, rstrip(String(take!(buffer))))
end

Base.show(io::IO, g::Graph) =
    print(io, "Graph(", length(_value_groups(g)), " values, ", length(g.recipes), " recipes)")
Base.show(io::IO, ::MIME"text/plain", g::Graph) = _write_graph_listing(io, g)
