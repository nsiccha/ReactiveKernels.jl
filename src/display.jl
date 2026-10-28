# Structural displays of graphs and plans. Generated code is exposed directly by `code_expr`.

code_expr(spec::KernelSpec; kwargs...) = code_expr(plan(spec; kwargs...))
code_expr(g::Graph; kwargs...) = code_expr(plan(g; kwargs...))

_display_expr(x) = x
function _display_expr(ex::Expr)
    applied = _display_applied_source(ex)
    applied === nothing || return applied
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

# Captured values in a displayed source binding: literals as themselves,
# functions and sources as above, anything else by its type name in brackets.
_display_object(x::Union{Number,AbstractString,Char,Symbol,Nothing}) = x isa Symbol ? QuoteNode(x) : x
_display_object(op::_KernelSourceOp) = _display_object(op.f)
_display_object(branch::_KernelBranch) = _display_object(branch.call)
_display_object(split::_KernelSplit) = _display_object(split.call)
_display_object(apply::_KernelSplitApply) = _display_object(apply.split)
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

function _display_capture_names!(renamed, x, captured)
    if x isa Symbol
        name = _source_argument(x)
        name !== x && name in captured && (renamed[x] = name)
    elseif x isa Expr && !(x.head in (:quote, :inert))
        foreach(arg -> _display_capture_names!(renamed, arg, captured), x.args)
    end
    renamed
end

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
        # The body may spell a capture as RuntimeGeneratedFunctions renamed it
        # (`#name#2`); show it under the captured name its binding uses.
        captured = Set(keys(f.captures))
        renamed = Dict{Symbol,Symbol}()
        _display_capture_names!(renamed, body.args[2], captured)
        isempty(renamed) || (body = _rename_arguments(body, renamed))
        bindings = Any[Expr(:(=), name, _display_object(value))
                       for (name, value) in pairs(f.captures)]
        body = isempty(bindings) ? body.args[2] :
            Expr(:let, Expr(:block, bindings...), body.args[2])
    end
    Expr(:->, Expr(:tuple, (get(names, p, p) for p in params)...), _display_expr(body))
end

# An operation spliced into a source and called in place (an object endpoint
# evaluated inside a branch arm) shows as its authored body with the call's
# arguments in place of its parameters, `(x - 0.0) / s` rather than
# `((x, location, scale) -> (x - location) / scale)(x, 0.0, s)`. An argument
# that is not a name or literal and is read more than once, or one a
# substitution could capture, stays a `let` binding (`_readable_inline_values`).
function _display_applied_source(ex::Expr)
    ex.head === :call && !isempty(ex.args) && ex.args[1] isa QuoteNode ||
        return nothing
    lambda = _display_expr(ex.args[1])
    lambda isa Expr && lambda.head === :-> && lambda.args[1] isa Expr &&
        lambda.args[1].head === :tuple || return nothing
    params = lambda.args[1].args
    length(params) == length(ex.args) - 1 && all(p -> p isa Symbol, params) ||
        return nothing
    body = lambda.args[2]
    body isa Expr && body.head === :block && length(body.args) == 1 &&
        (body = only(body.args))
    pairs = Pair{Symbol,Any}[param => _display_expr(arg)
                             for (param, arg) in zip(params, ex.args[2:end])
                             if param !== arg]
    rhs, kept = _readable_inline_values(body, pairs)
    isempty(kept) && return rhs
    bindings = Any[Expr(:(=), name, value) for (name, value) in kept]
    Expr(:let, length(bindings) == 1 ? only(bindings) : Expr(:block, bindings...),
         Expr(:block, rhs))
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

# HTML escaping for structural views that include text.
function _html_escape(s::AbstractString)
    replace(s, '&' => "&amp;", '<' => "&lt;", '>' => "&gt;", '"' => "&quot;")
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
