"""
    RKPPLDocumentation

An authored Markdown docstring, separate from a model's numerical program.
`path` is a tuple of literal author names (never private lowering identifiers).
`owner` is `:declaration`, `:instance`, or `:definition`. `definition` names the
reusable definition that supplied the text, or is `nothing` for caller text;
`origin` is its path within that definition. `file` and `line` retain the Julia
source location when available (`nothing`, `0` otherwise).

Definition records have an empty path on a captured submodel and the use-site
path on a lowered plan. An instance can carry both its own use-site docstring
and its reusable definition's docstring. Markdown is retained without parsing,
interpolation, or rendering.
"""
struct RKPPLDocumentation
    path::Tuple{Vararg{Symbol}}
    owner::Symbol
    markdown::String
    definition::Union{Nothing,Symbol}
    origin::Tuple{Vararg{Symbol}}
    file::Union{Nothing,Symbol}
    line::Int
end

"""
    rkppl_documentation(model_or_plan; path=nothing)

Return a fresh vector of immutable [`RKPPLDocumentation`](@ref) records.
Captured models/submodels expose their own authored text; lowered plans also
include documentation instantiated from reusable components. Binding, pins and
unannotated replacements retain it; an annotated replacement replaces the text
at that declaration. A per-cell declaration contributes one record, independent
of observation count. Records follow source order within each captured scope,
with expanded scopes appended in depth-first use order.

`path` optionally selects an exact tuple of author names, such as `(:fit, :b)`.
Querying documentation never lowers a model, resolves a component, evaluates
data, or runs a renderer. In particular it does not inspect Julia's doc registry.
"""
function rkppl_documentation end

function _documentation_records(records; path=nothing)
    path === nothing && return copy(records)
    path isa Tuple && all(x -> x isa Symbol, path) ||
        throw(ArgumentError("documentation path must be a tuple of Symbols"))
    return filter(d -> d.path == path, records)
end

_doc_record(d::RKPPLDocumentation; path=d.path, owner=d.owner,
        definition=d.definition, origin=d.origin) =
    RKPPLDocumentation(path, owner, d.markdown, definition, origin, d.file, d.line)

# Inspect statement positions only. A quoted value, function body or arbitrary
# expression on a RHS is ordinary Julia code, not an RKPPL declaration.
function _documentation_source(ast; definition=nothing)
    docs = RKPPLDocumentation[]
    stripped = _documentation_statement(ast, docs, nothing, 0, definition)
    return stripped, docs
end

function _documentation_path(lhs)
    lhs isa Symbol && return (lhs,)
    lhs isa Expr || return nothing
    lhs.head === :ref && return _documentation_path(first(lhs.args))
    if lhs.head === :. && length(lhs.args) == 2 && lhs.args[2] isa QuoteNode
        parent = _documentation_path(lhs.args[1])
        name = lhs.args[2].value
        parent === nothing || name isa Symbol || return nothing
        return parent === nothing ? nothing : (parent..., name)
    end
    if lhs.head === :call && length(lhs.args) == 2 &&
            first(lhs.args) in (:eachrow, :eachcol)
        return _documentation_path(lhs.args[2])
    end
    return nothing
end

function _documentation_statement(ex, docs, file, line, definition)
    ex isa Expr || return ex
    if ex.head === :block
        args = Any[]
        i = 1
        while i <= length(ex.args)
            a = ex.args[i]
            if a isa LineNumberNode
                file, line = a.file, a.line
                push!(args, a)
            else
                # Julia emits Core.@doc at block level, but leaves the same
                # literal-string spelling as two statements inside a for
                # body. Pair it only with a following model declaration.
                if a isa String
                    j = findnext(x -> !(x isa LineNumberNode), ex.args, i + 1)
                    if j !== nothing && _documentation_lhs(ex.args[j]) !== nothing
                        append!(args, ex.args[i+1:j-1])
                        wrapper = Expr(:macrocall, GlobalRef(Core, Symbol("@doc")),
                            LineNumberNode(line, file), a, ex.args[j])
                        push!(args, _documentation_statement(wrapper, docs, file, line, definition))
                        i = j + 1
                        continue
                    end
                end
                push!(args, _documentation_statement(a, docs, file, line, definition))
            end
            i += 1
        end
        return Expr(:block, args...)
    elseif ex.head === :macrocall && !isempty(ex.args) && _is_doc_macro(ex.args[1])
        st = _unwrap_trivia(ex) # same literal-string and declaration checks as lowering
        if length(ex.args) >= 2 && ex.args[2] isa LineNumberNode
            file, line = ex.args[2].file, ex.args[2].line
        end
        lhs = _documentation_lhs(st)
        names = lhs isa Expr && lhs.head in (:vect, :tuple) ? lhs.args : (lhs,)
        for name in names
            path = _documentation_path(name)
            path === nothing && continue
            push!(docs, RKPPLDocumentation(path, :declaration, ex.args[3],
                definition, path, file, line))
        end
        return _documentation_statement(st, docs, file, line, definition)
    elseif ex.head === :for
        return Expr(:for, ex.args[1],
            _documentation_statement(ex.args[2], docs, file, line, definition))
    elseif _is_block_macro(ex)
        return Expr(:macrocall, ex.args[1:end-1]...,
            _documentation_statement(last(ex.args), docs, file, line, definition))
    elseif Meta.isexpr(ex, :(=), 2) && _is_block_macro(last(ex.args))
        return Expr(:(=), first(ex.args),
            _documentation_statement(last(ex.args), docs, file, line, definition))
    end
    return ex
end

_documentation_lhs(st) = st isa Expr && (_is_sample(st) ||
    _is_broadcast_sample(st) || Meta.isexpr(st, :(=), 2)) ? _stmt_lhs(st) : nothing

function _merge_documentation(old, new)
    out = copy(old)
    for d in new
        i = findfirst(x -> x.path == d.path && x.owner !== :definition, out)
        i === nothing ? push!(out, d) : (out[i] = d)
    end
    return out
end

# Snapshot Julia's raw literal binding doc only at definition time. This does
# not call getdoc/rendering hooks, and plans never retain modules or DocStrs.
function _binding_documentation(mod::Module, name::Symbol)
    metadata = Base.Docs.meta(mod; autoinit=false)
    metadata === nothing && return nothing
    multi = get(metadata, Base.Docs.Binding(mod, name), nothing)
    multi === nothing && return nothing
    return get(multi.docs, Union{}, nothing)
end
