# A dotted cell latent, `v[i] .~ D.(args…)` in a `@plate` cell with `v` not
# data, states one iteration of its Julia loop: `v[i]` holds the broadcast of
# `D` over the arguments' values at index `i`. That is a number when they are
# all numbers, and otherwise an array of their broadcast shape, whose size
# may vary by index. The coordinates of every index are one ordinary array
# parameter, in plate order, under the declaration's prior with each
# argument's entries flattened in that order. The prior, transform and
# Jacobian are therefore the array prior's, and each cell reads its own block.
# The shapes must follow from data at binding. They are read from data per
# index, from scalar parameters, from model values read at an index (as
# Julia indexes them) and from the dotted arithmetic between them.

struct _CellLatentScope
    data::Set{Symbol}               # data and data-only definitions
    definitions::Dict{Symbol,Any}   # model-level definitions
    scalars::Set{Symbol}            # scalar parameters declared at model level
    numbers::Set{Symbol}            # sampled model values holding numbers
end

# Model-level `x ~ D(…)` with a scalar prior: a number wherever it is read.
function _scalar_parameter_names(args)
    out = Set{Symbol}()
    for st in args
        Meta.isexpr(st, :call, 3) && st.args[1] === :~ && st.args[2] isa Symbol ||
            continue
        _scalar_prior_object(st.args[3]) && push!(out, st.args[2])
    end
    return out
end
_scalar_prior_object(ex) = Meta.isexpr(ex, :call) && !isempty(ex.args) &&
    (ex.args[1] in _SCALAR_PRIOR_HEADS || ex.args[1] in (:truncated, :restricted) &&
        length(ex.args) >= 2 && _scalar_prior_object(ex.args[2]))

# Sampled model values that hold numbers, so that reading one at an index
# gives a number (or an array of numbers for an array index): every
# model-level declaration (a number, or an array of numbers whatever its
# prior), the scalar latents and multivariate rows of a latent `@plate`, and
# the carried arrays of a `@scan`. Data are observations, not parameters.
function _number_value_names(args, data::Set{Symbol})
    out = Set{Symbol}()
    add!(name) = name isa Symbol && !(name in data) && push!(out, name)
    for st in args
        st isa Expr || continue
        if Meta.isexpr(st, :call, 3) && st.args[1] in (:~, :.~)
            add!(_declared_value_name(st.args[2]))
        elseif Meta.isexpr(st, :macrocall, 3) && st.args[1] === Symbol("@plate") &&
                Meta.isexpr(st.args[3], :for) && Meta.isexpr(st.args[3].args[1], :(=), 2)
            ivar = st.args[3].args[1].args[1]
            for c in st.args[3].args[2].args
                Meta.isexpr(c, :call, 3) && c.args[1] === :~ &&
                    Meta.isexpr(c.args[2], :ref) && length(c.args[2].args) >= 2 &&
                    c.args[2].args[2] === ivar || continue
                rhs = c.args[3]
                (length(c.args[2].args) == 2 ? _scalar_prior_object(rhs) :
                    Meta.isexpr(rhs, :call) && rhs.args[1] in (:MvNormal, :MvNormalCholesky)) ||
                    continue
                add!(c.args[2].args[1])
            end
        elseif Meta.isexpr(st, :macrocall) && st.args[1] === Symbol("@scan") &&
                Meta.isexpr(st.args[end], :block)
            for c in st.args[end].args
                lhs = Meta.isexpr(c, :(=), 2) ? c.args[1] :
                    Meta.isexpr(c, :call, 3) && c.args[1] === :~ ? c.args[2] : nothing
                Meta.isexpr(lhs, :ref) && add!(lhs.args[1])
            end
        end
    end
    return out
end

# The value a declaration's left-hand side names: `x`, `x[…]`, `eachrow(B[…])`.
_declared_value_name(lhs::Symbol) = lhs
_declared_value_name(lhs::Expr) = Meta.isexpr(lhs, :ref) ? _declared_value_name(lhs.args[1]) :
    Meta.isexpr(lhs, :call, 2) && lhs.args[1] in (:eachrow, :eachcol) ?
        _declared_value_name(lhs.args[2]) : nothing
_declared_value_name(lhs) = nothing

_is_cell_array_latent(c, ivar::Symbol, data::Set{Symbol}) =
    Meta.isexpr(c, :call, 3) && c.args[1] === :.~ &&
    Meta.isexpr(c.args[2], :ref, 2) && c.args[2].args[1] isa Symbol &&
    c.args[2].args[2] === ivar && !(c.args[2].args[1] in data) &&
    _is_cell_latent_object(c.args[3])

_is_cell_latent_object(ex) = _is_dotted_call(ex) && ex.args[1] isa Symbol &&
    _cell_object_call(ex.args[1])

# Rebuild a dotted constructor tree (`truncated.(Normal.(m, s), lo, hi)`),
# replacing each argument that is not itself a constructor by `f(argument)`.
_cell_latent_object(f, ex) = _is_cell_latent_object(ex) ?
    Expr(:., ex.args[1], Expr(:tuple, (_cell_latent_object(f, a) for a in ex.args[2].args)...)) :
    f(ex)

function _cell_latent_leaves(ex)
    out = Any[]
    _cell_latent_object(ex) do leaf
        push!(out, leaf)
        leaf
    end
    return out
end

# What one plate's cells say about the shapes of the values they read.
struct _CellShapes
    ivar::Symbol
    scope::_CellLatentScope
    locals::Vector{Pair{Symbol,Any}}
    dataonly::Set{Symbol}                       # cell locals reading only data
    scalars::Set{Symbol}                        # scalar latents of the cell
    arrays::Dict{Symbol,Tuple{Symbol,Symbol}}   # array latent => (dims, position)
end

_cell_reads_only_data(ex, cx::_CellShapes) = all(s -> s in cx.scope.data ||
    s === cx.ivar || s in cx.dataonly, _value_symbols(ex))

function _push_cell_local!(cx::_CellShapes, name::Symbol, rhs)
    push!(cx.locals, name => rhs)
    _cell_reads_only_data(rhs, cx) && push!(cx.dataonly, name)
    return cx
end

# The shape of a cell expression: `:scalar` when it is a number before any
# data is bound, an expression computing its axes from data at each index,
# or `(nothing, why)` when the shape is not known before sampling.
const _CELL_SCALAR_CALLS = (:+, :-, :*, :/, :^, :exp, :log, :log1p, :expm1,
    :sqrt, :abs, :abs2, :tanh, :logistic, :logit, :log10, :sin, :cos,
    :min, :max, :inv)

function _cell_latent_shape(ex, cx::_CellShapes, active::Set{Symbol} = Set{Symbol}())
    ex isa Number && return :scalar
    ex isa Union{Symbol,Expr} || return (nothing, "it is not a value")
    _cell_reads_only_data(ex, cx) &&
        return Expr(:call, GlobalRef(Base, :axes), ex)
    if ex isa Symbol
        k = findlast(l -> first(l) === ex, cx.locals)
        k === nothing || return _cell_latent_shape(last(cx.locals[k]), cx, active)
        ex in cx.scope.scalars && return :scalar
        # A free name reads its module constant, as `Inf` does.
        !haskey(cx.scope.definitions, ex) && isdefined(Base, ex) &&
            getfield(Base, ex) isa Number && return :scalar
        if haskey(cx.scope.definitions, ex) && !(ex in active)
            push!(active, ex)
            shape = _cell_latent_shape(cx.scope.definitions[ex], cx, active)
            delete!(active, ex)
            return shape
        end
        return (nothing, "`$ex` is not data, a scalar parameter or a cell value")
    end
    if Meta.isexpr(ex, :ref, 2) && ex.args[1] isa Symbol && ex.args[2] === cx.ivar
        ex.args[1] in cx.scalars && return :scalar
        if haskey(cx.arrays, ex.args[1])
            dims, pos = cx.arrays[ex.args[1]]
            return Expr(:call, GlobalRef(@__MODULE__, :_ppl_cell_latent_axes), dims, pos)
        end
    end
    if ex.head === :ref && all(i -> i isa Integer, ex.args[2:end])
        return :scalar
    end
    # A model value read at an index (`m[i]`, `m[g[i]]`, `B[i, 1]`).
    if ex.head === :ref && ex.args[1] isa Symbol &&
            !any(l -> first(l) === ex.args[1], cx.locals) &&
            !(ex.args[1] in cx.scalars) && !haskey(cx.arrays, ex.args[1])
        return _cell_read_shape(ex.args[1], ex.args[2:end], cx, active)
    end
    dotted = _is_dotted_call(ex) ? ex.args[2].args :
        Meta.isexpr(ex, :call) && ex.args[1] isa Symbol &&
            startswith(string(ex.args[1]), ".") ? ex.args[2:end] : nothing
    if dotted !== nothing
        return _cell_broadcast_shape(Any[Meta.isexpr(a, :call, 2) && a.args[1] === :Ref ?
            :scalar : _cell_latent_shape(a, cx, active) for a in dotted])
    end
    if Meta.isexpr(ex, :call) && ex.args[1] isa Symbol
        f = ex.args[1]
        f in REDUCTION_FNS && length(ex.args) == 2 && return :scalar
        f in _CELL_SCALAR_CALLS && return _cell_call_shape(f,
            Any[_cell_latent_shape(a, cx, active) for a in ex.args[2:end]])
    end
    return (nothing, "`$ex` computes from sampled values, and its shape is " *
        "not known before sampling")
end

# The shape of an undotted scalar call over its arguments' shapes.
function _cell_call_shape(f::Symbol, shapes)
    for s in shapes
        s isa Tuple && return s
    end
    all(s -> s === :scalar, shapes) && return :scalar
    f in (:+, :-, :*, :/) || return (nothing, "`$f` of an array is not elementwise")
    return Expr(:call, GlobalRef(@__MODULE__, :_ppl_cell_latent_call_axes),
        QuoteNode(f), (s === :scalar ? :(()) : s for s in shapes)...)
end

# The shape of `name[idx...]`, a model value read at an index in a cell:
# Julia indexes an array of numbers to a number at scalar indices, and to
# the indices' shapes otherwise. Data a definition computes from must hold
# numbers too, which binding checks.
function _cell_read_shape(name::Symbol, idx, cx::_CellShapes, active::Set{Symbol})
    name in cx.scope.data && return (nothing, "`$name` is indexed by a sampled value")
    checks = _cell_number_checks!(Any[], name, cx, active)
    checks isa Tuple && return checks
    index = _cell_index_shape(idx, cx, active)
    (index isa Tuple || isempty(checks)) && return index
    return Expr(:call, GlobalRef(@__MODULE__, :_ppl_cell_number_entries),
        Expr(:tuple, (QuoteNode(c) for c in checks)...), Expr(:tuple, checks...),
        index === :scalar ? :(()) : index)
end

# Does a model-level expression hold numbers? The data it reads are checked
# at binding (`checks`); sampled values hold numbers; arithmetic, scalar
# functions, concatenations and reads of numbers hold numbers.
function _cell_number_checks!(checks, ex, cx::_CellShapes, active::Set{Symbol})
    ex isa Number && return checks
    if _cell_reads_only_data(ex, cx)
        push!(checks, ex)
        return checks
    end
    if ex isa Symbol
        (ex in cx.scope.numbers || ex in cx.scope.scalars) && return checks
        if haskey(cx.scope.definitions, ex) && !(ex in active)
            push!(active, ex)
            out = _cell_number_checks!(checks, cx.scope.definitions[ex], cx, active)
            delete!(active, ex)
            return out
        end
        return (nothing, "`$ex` is not data, a declared parameter or a model definition")
    end
    unknown = (nothing, "`$ex` computes from sampled values, and its shape is " *
        "not known before sampling")
    ex isa Expr || return unknown
    Meta.isexpr(ex, :ref) && return _cell_number_checks!(checks, ex.args[1], cx, active)
    Meta.isexpr(ex, Symbol("'"), 1) && return _cell_number_checks!(checks, ex.args[1], cx, active)
    # `[a, b]` holds its elements, so they must be numbers; `[a; b]` and
    # `[a b]` concatenate theirs.
    if Meta.isexpr(ex, :vect)
        all(a -> _cell_latent_shape(a, cx, active) === :scalar, ex.args) || return unknown
        return checks
    end
    operands = ex.head in (:vcat, :hcat, :row) ? ex.args :
        _is_dotted_call(ex) && ex.args[1] isa Symbol && ex.args[1] in _CELL_SCALAR_CALLS ?
            ex.args[2].args :
        Meta.isexpr(ex, :call) && ex.args[1] isa Symbol &&
            (ex.args[1] in _CELL_SCALAR_CALLS || ex.args[1] in REDUCTION_FNS ||
             ex.args[1] in (:adjoint, :transpose, :vec, :vcat, :hcat) ||
             startswith(string(ex.args[1]), ".") &&
                Symbol(string(ex.args[1])[2:end]) in _CELL_SCALAR_CALLS) ?
            filter(a -> !Meta.isexpr(a, (:parameters, :kw)), ex.args[2:end]) : nothing
    operands === nothing && return unknown
    for a in operands
        checks = _cell_number_checks!(checks, a, cx, active)
        checks isa Tuple && return checks
    end
    return checks
end

# The shape of indexing an array of numbers: a number at scalar indices
# (the loop index, positions, `end`, integers read per index), otherwise
# the indices' shapes in order, as Julia's indexing gives.
function _cell_index_shape(idx, cx::_CellShapes, active::Set{Symbol})
    shapes = Any[]
    for j in idx
        j === :(:) && return (nothing, "a `:` read of a model value is not built yet; " *
            "read its entries by position")
        s = j === cx.ivar || j isa Integer || _is_endpoint_position(j) ? :scalar :
            _cell_latent_shape(j, cx, active)
        s isa Tuple && return s
        push!(shapes, s)
    end
    all(s -> s === :scalar, shapes) && return :scalar
    return Expr(:call, GlobalRef(@__MODULE__, :_ppl_cell_index_axes),
        (s === :scalar ? :(()) : s for s in shapes)...)
end

function _cell_broadcast_shape(shapes)
    for s in shapes
        s isa Tuple && return s
    end
    all(s -> s === :scalar, shapes) && return :scalar
    return Expr(:call, GlobalRef(Base.Broadcast, :broadcast_shape),
        (s === :scalar ? :(()) : s for s in shapes)...)
end

# Rewrite one plate's dotted array latents. Each becomes a model-level packed
# array with an ordinary array prior, read per cell as its block. Returns
# the model-level statements, the plate, and each packed array's authored
# name and dimensions.
function _expand_cell_array_latents(st, taken, scope::_CellLatentScope)
    sites = Dict{Symbol,Tuple{Symbol,Symbol}}()
    loop = st.args[3]
    (Meta.isexpr(loop, :for) && Meta.isexpr(loop.args[1], :(=), 2) &&
        Meta.isexpr(loop.args[2], :block)) || return Any[], st, sites
    ivar, domain = loop.args[1].args
    ivar isa Symbol || return Any[], st, sites
    any(c -> _is_cell_array_latent(c, ivar, scope.data), loop.args[2].args) ||
        return Any[], st, sites
    Meta.isexpr(domain, :call) && domain.args[1] === :levels && _sfail("cell " *
        "latents `v[k] .~ D.(…)` holding arrays over `levels(…)` are not built " *
        "yet; iterate `eachindex`, `axes` or a range")
    fresh(base) = _cell_scan_fresh!(taken, base)
    iterator = deepcopy(domain)
    if Meta.isexpr(iterator, :call) && iterator.args[1] in (:eachindex, :axes)
        iterator.args[1] = GlobalRef(Base, iterator.args[1])
    end
    indices = Expr(:call, GlobalRef(Base, :collect), iterator)
    position = Expr(:call, GlobalRef(@__MODULE__, :_ppl_cell_scan_position), domain, ivar)
    cx = _CellShapes(ivar, scope, Pair{Symbol,Any}[], Set{Symbol}(), Set{Symbol}(),
        Dict{Symbol,Tuple{Symbol,Symbol}}())
    prelude, cells = Any[], Any[]
    column(rhs) = _plate_column_call(ivar, _needed_cell_locals(cx.locals, rhs), rhs,
        nothing; indices)
    for c in loop.args[2].args
        if Meta.isexpr(c, :(=), 2) && c.args[1] isa Symbol
            _push_cell_local!(cx, c.args[1], c.args[2])
        elseif _is_cell_latent_block(c) && Meta.isexpr(c.args[1], :ref, 2) &&
                c.args[1].args[2] === ivar &&
                c.args[2].args[1] == GlobalRef(@__MODULE__, :_ppl_cell_joint_constrain)
            # A joint latent's value has its geometry's shape at the index.
            _, _, _, shapes, pos = filter(a -> !Meta.isexpr(a, :parameters), c.args[2].args[2:end])
            cx.arrays[c.args[1].args[1]] = (shapes, pos)
        elseif Meta.isexpr(c, :call, 3) && c.args[1] === :~ &&
                Meta.isexpr(c.args[2], :ref, 2) && c.args[2].args[2] === ivar &&
                c.args[2].args[1] isa Symbol && !(c.args[2].args[1] in scope.data)
            push!(cx.scalars, c.args[2].args[1])
        end
        _is_cell_array_latent(c, ivar, scope.data) || (push!(cells, c); continue)
        name = c.args[2].args[1]
        leaves = _cell_latent_leaves(c.args[3])
        shapes = Any[]
        for leaf in leaves
            shape = _cell_latent_shape(leaf, cx)
            shape isa Tuple && _sfail("cell latent `$name[$ivar] .~ …`: at each " *
                "index `$name[$ivar]` holds the broadcast of its arguments, so " *
                "their shapes must follow from data; the shape of " *
                "`$(repr(leaf))` does not: $(last(shape))")
            push!(shapes, shape)
        end
        if all(s -> s === :scalar, shapes)
            # Numbers at every index: one scalar draw per index.
            push!(cx.scalars, name)
            push!(cells, Expr(:call, :~, c.args[2], c.args[3]))
            continue
        end
        prefix = Symbol(:_rkppl_latent_, name)
        site, dims = fresh(prefix), fresh(Symbol(prefix, :_dims))
        offsets, count = fresh(Symbol(prefix, :_offsets)), fresh(Symbol(prefix, :_size))
        shape = _cell_broadcast_shape(shapes)
        dimscells = fresh(Symbol(prefix, :_shapes))
        push!(prelude, Expr(:(=), dimscells, column(Expr(:call,
            GlobalRef(@__MODULE__, :_ppl_cell_latent_dims), shape))))
        push!(prelude, Expr(:(=), dims, Expr(:call,
            GlobalRef(@__MODULE__, :_ppl_cell_latent_shapes), QuoteNode(name), dimscells)))
        push!(prelude, Expr(:(=), offsets,
            Expr(:call, GlobalRef(@__MODULE__, :_ppl_cell_latent_offsets), dims)))
        push!(prelude, Expr(:(=), count,
            Expr(:call, GlobalRef(@__MODULE__, :_ppl_cell_latent_size), offsets)))
        # A shared number broadcasts against the packed array as written;
        # every other argument contributes its entries at each index.
        k = 0
        object = _cell_latent_object(c.args[3]) do leaf
            k += 1
            shapes[k] === :scalar && !_expr_has_sym(leaf, ivar) &&
                !any(l -> _expr_has_sym(leaf, first(l)), cx.locals) && return leaf
            entries = fresh(Symbol(prefix, :_arg, k, :_cells))
            flat = fresh(Symbol(prefix, :_arg, k))
            push!(prelude, Expr(:(=), entries, column(Expr(:call,
                GlobalRef(@__MODULE__, :_ppl_cell_latent_entries), leaf,
                Expr(:ref, dims, position)))))
            push!(prelude, Expr(:(=), flat, Expr(:call,
                GlobalRef(@__MODULE__, :_ppl_cell_latent_flatten), entries)))
            return flat
        end
        push!(prelude, Expr(:call, :.~, Expr(:ref, site, Expr(:call, :(:), 1, count)), object))
        pos = fresh(Symbol(prefix, :_cell))
        push!(cells, Expr(:(=), pos, position))
        push!(cells, Expr(:(=), c.args[2], Expr(:call,
            GlobalRef(@__MODULE__, :_ppl_cell_latent_block), site, offsets, dims, pos)))
        _push_cell_local!(cx, pos, position)
        cx.arrays[name] = (dims, pos)
        sites[site] = (name, dims)
    end
    return prelude, Expr(st.head, st.args[1:2]...,
        Expr(:for, loop.args[1], Expr(:block, cells...))), sites
end

# A cell reading its array latent's block, or constraining its joint latent's
# block. Such a plate is retained, one iteration per index, as a cell `@scan`
# is.
_is_cell_latent_block(c) = Meta.isexpr(c, :(=), 2) && Meta.isexpr(c.args[2], :call) &&
    (c.args[2].args[1] == GlobalRef(@__MODULE__, :_ppl_cell_latent_block) ||
     c.args[2].args[1] == GlobalRef(@__MODULE__, :_ppl_cell_joint_constrain))

# --- Joint cell latents: `c[i] ~ F(args…)` with a caller-owned geometry ---
#
# An undotted cell latent whose caller-owned RHS has a geometry other than one
# scalar coordinate states one draw of `F(args…)` at each index: a value of
# the geometry's shape there, under ONE density of the whole value. The
# geometry's shape and packed dimension are data-only expressions of the
# authored arguments, evaluated at each index at binding. The coordinates of
# every index are one flat array parameter in plate order, with no density or
# transform of its own. Each cell constrains its block through the geometry
# and adds its density and log-Jacobian, so the iteration stays in the
# retained plate.

# Names the program gives values: data, every left-hand side (in plates and
# scans too) and every loop variable. They shadow module bindings, as Julia's
# local names do.
function _program_value_names!(out::Set{Symbol}, ex)
    ex isa Expr || return out
    if Meta.isexpr(ex, :(=), 2) || Meta.isexpr(ex, :call, 3) && ex.args[1] in (:~, :.~)
        lhs = Meta.isexpr(ex, :(=)) ? ex.args[1] : ex.args[2]
        name = _declared_value_name(lhs)
        name isa Symbol && push!(out, name)
        Meta.isexpr(ex, :(=)) && _program_value_names!(out, ex.args[2])
        return out
    end
    if Meta.isexpr(ex, :for) && Meta.isexpr(ex.args[1], :(=), 2) && ex.args[1].args[1] isa Symbol
        push!(out, ex.args[1].args[1])
    end
    foreach(a -> _program_value_names!(out, a), ex.args)
    return out
end

_is_cell_joint_candidate(c, ivar::Symbol, data::Set{Symbol}) =
    Meta.isexpr(c, :call, 3) && c.args[1] === :~ &&
    Meta.isexpr(c.args[2], :ref, 2) && c.args[2].args[1] isa Symbol &&
    c.args[2].args[2] === ivar && !(c.args[2].args[1] in data) &&
    Meta.isexpr(c.args[3], :call) && _external_rhs(c.args[3])

# The resolved RHS and its geometry, when that geometry is not one scalar
# coordinate; `nothing` leaves the statement to the scalar cell path, which
# reports any resolution or geometry error itself.
function _cell_joint_geometry(rhs::Expr, mod::Module, names::Set{Symbol})
    head = rhs.args[1]
    head isa Symbol && (head in names || !isdefined(mod, head)) && return nothing
    call = try
        _resolve_module_calls(rhs, mod, names, "sampling RHS")
    catch e
        e isa SurfaceLoweringError || rethrow()
        return nothing
    end
    Meta.isexpr(call, :call) && call.args[1] isa GlobalRef || return nothing
    constructor = getglobal(call.args[1].mod, call.args[1].name)
    geometry = try
        sampling_geometry(constructor, Tuple(deepcopy(call.args[2:end])), ())
    catch e
        e isa Union{ArgumentError,MethodError} || rethrow()
        return nothing
    end
    geometry isa ParameterGeometry || return nothing
    geometry.shape == () && geometry.unconstrained == 1 && return nothing
    return call, geometry
end

# Rewrite one plate's joint cell latents. Returns the model-level statements
# before the plate, the plate, the geometry of each packed array, the cells'
# prior sums and the statements after the plate that sum the cells.
function _expand_cell_joint_latents(st, taken, data::Set{Symbol}, mod::Module,
        names::Set{Symbol})
    sites = Dict{Symbol,CellGeometry}()
    loop = st.args[3]
    (Meta.isexpr(loop, :for) && Meta.isexpr(loop.args[1], :(=), 2) &&
        Meta.isexpr(loop.args[2], :block)) || return Any[], st, sites, Symbol[], Any[]
    ivar, domain = loop.args[1].args
    ivar isa Symbol || return Any[], st, sites, Symbol[], Any[]
    any(c -> _is_cell_joint_candidate(c, ivar, data), loop.args[2].args) ||
        return Any[], st, sites, Symbol[], Any[]
    cellnames = _program_value_names!(union(names, Set([ivar])), loop)
    fresh(base) = _cell_scan_fresh!(taken, base)
    iterator = deepcopy(domain)
    if Meta.isexpr(iterator, :call) && iterator.args[1] in (:eachindex, :axes)
        iterator.args[1] = GlobalRef(Base, iterator.args[1])
    end
    indices = Expr(:call, GlobalRef(Base, :collect), iterator)
    locals = Pair{Symbol,Any}[]
    column(rhs) = _plate_column_call(ivar, _needed_cell_locals(locals, rhs), rhs,
        nothing; indices)
    prelude, cells, priors, post = Any[], Any[], Symbol[], Any[]
    ref(f) = GlobalRef(@__MODULE__, f)
    for c in loop.args[2].args
        Meta.isexpr(c, :(=), 2) && c.args[1] isa Symbol && push!(locals, c.args[1] => c.args[2])
        found = _is_cell_joint_candidate(c, ivar, data) ?
            _cell_joint_geometry(c.args[3], mod, cellnames) : nothing
        found === nothing && (push!(cells, c); continue)
        call, g = deepcopy(found[1]), found[2]
        name = c.args[2].args[1]
        Meta.isexpr(domain, :call) && domain.args[1] === :levels && _sfail("cell " *
            "latent `$name[$ivar] ~ …` with a joint geometry over `levels(…)` is " *
            "not built yet; iterate `eachindex`, `axes` or a range")
        prefix = Symbol(:_rkppl_latent_, name)
        site, geometry = fresh(prefix), fresh(Symbol(prefix, :_geometry))
        shapecells, shapes = fresh(Symbol(prefix, :_shape_cells)), fresh(Symbol(prefix, :_shapes))
        sizecells, offsets = fresh(Symbol(prefix, :_size_cells)), fresh(Symbol(prefix, :_offsets))
        count, pos = fresh(Symbol(prefix, :_size)), fresh(Symbol(prefix, :_cell))
        prior, logjac = fresh(Symbol(prefix, :_prior)), fresh(Symbol(prefix, :_logjac))
        for s in (prior, logjac)
            Symbol(s, :_cells) in taken && _sfail("cell latent `$name`: internal " *
                "name `$(Symbol(s, :_cells))` is already used by the model")
            push!(taken, Symbol(s, :_cells))
        end
        positional = Any[a for a in call.args[2:end] if !Meta.isexpr(a, :parameters)]
        keywords = Any[k for a in call.args[2:end] if Meta.isexpr(a, :parameters) for k in a.args]
        for k in keywords
            Meta.isexpr(k, :kw, 2) && k.args[1] isa Symbol || _sfail("cell latent " *
                "`$name[$ivar] ~ …`: keyword argument $(repr(k)) is not built yet; " *
                "write `keyword = value`")
        end
        # How the host reads an argument at an index: a value that reads the
        # loop index or a cell local is a column of its per-index values.
        cellvalue(a) = !(a isa Union{Number,QuoteNode,GlobalRef}) &&
            !isdisjoint(_cell_free_syms!(Set{Symbol}(), a),
                union(Set([ivar]), Set(first.(locals))))
        function hostvalue(a, label)
            cellvalue(a) || return :shared => a
            col = fresh(Symbol(prefix, :_, label))
            push!(prelude, Expr(:(=), col, column(a)))
            return :cells => col
        end
        arguments = Pair{Symbol,Any}[hostvalue(a, Symbol(:arg, k)) for (k, a) in enumerate(positional)]
        hostkeywords = Pair{Symbol,Pair{Symbol,Any}}[k.args[1] => hostvalue(k.args[2], k.args[1])
            for k in keywords]
        push!(prelude, Expr(:(=), geometry, Expr(:call, ref(:sampling_geometry),
            call.args[1], Expr(:tuple, (_sampling_ast_literal(a) for a in call.args[2:end])...),
            Expr(:tuple))))
        push!(prelude, Expr(:(=), shapecells, column(Expr(:tuple, deepcopy.(g.shape)...))))
        push!(prelude, Expr(:(=), shapes, Expr(:call, ref(:_ppl_cell_joint_shapes),
            QuoteNode(name), shapecells)))
        push!(prelude, Expr(:(=), sizecells, column(deepcopy(g.unconstrained))))
        push!(prelude, Expr(:(=), offsets, Expr(:call, ref(:_ppl_cell_joint_offsets),
            QuoteNode(name), sizecells)))
        push!(prelude, Expr(:(=), count, Expr(:call, ref(:_ppl_cell_latent_size), offsets)))
        push!(prelude, Expr(:call, :.~, Expr(:ref, site, Expr(:call, :(:), 1, count)),
            :(Flat.())))
        endpoint(f) = _external_endpoint_call(ref(f),
            Any[geometry, site, offsets, shapes, pos, positional...], keywords)
        push!(cells, Expr(:(=), pos, Expr(:call, ref(:_ppl_cell_scan_position), domain, ivar)))
        push!(locals, pos => Expr(:call, ref(:_ppl_cell_scan_position), domain, ivar))
        push!(cells, Expr(:(=), c.args[2], endpoint(:_ppl_cell_joint_constrain)))
        push!(cells, Expr(:(=), Expr(:ref, Symbol(prior, :_cells), ivar),
            Expr(:call, ref(:sampling_logdensity), deepcopy(call), c.args[2])))
        push!(cells, Expr(:(=), Expr(:ref, Symbol(logjac, :_cells), ivar),
            endpoint(:_ppl_cell_joint_logjac)))
        push!(priors, prior)
        push!(post, :($prior = sum($(Symbol(prior, :_cells)))))
        push!(post, :($logjac = sum($(Symbol(logjac, :_cells)))))
        sites[site] = CellGeometry(name, g, shapes, offsets, arguments, hostkeywords, logjac)
    end
    return prelude, Expr(st.head, st.args[1:2]...,
        Expr(:for, loop.args[1], Expr(:block, cells...))), sites, priors, post
end

# --- Values the rewritten program computes -------------------------------

# One index's dimensions: the broadcast axes of the arguments there.
_ppl_cell_latent_dims(axes::Tuple) = map(length, axes)

# Every index's dimensions. With one rank at every index each cell's block
# has one concrete type. Different ranks are valid Julia but make the cells
# type unstable, so they are warned about rather than refused.
function _ppl_cell_latent_shapes(name::Symbol, cells::AbstractVector)
    isempty(cells) && return Tuple{}[]
    N = length(first(cells))
    all(d -> d isa NTuple{N,Int}, cells) && return NTuple{N,Int}[d for d in cells]
    @warn "cell latent `$name`: its arguments broadcast to values of " *
        "different dimensions at different indices " *
        "($(join(sort!(unique(length.(cells))), ", ")) dimensions), so its cells " *
        "read values of different types and the prepared program is not type " *
        "stable; give every index's `$name[i]` the same number of dimensions"
    return Tuple{Vararg{Int}}[d for d in cells]
end

function _ppl_cell_latent_offsets(dims::AbstractVector{<:Tuple})
    offsets = Vector{Int}(undef, length(dims) + 1)
    offsets[1] = 0
    for (k, d) in enumerate(dims)
        offsets[k + 1] = offsets[k] + prod(d)
    end
    return offsets
end
_ppl_cell_latent_size(offsets::AbstractVector{Int}) = last(offsets)

# The axes of an earlier array latent of the cell at its position.
_ppl_cell_latent_axes(dims, pos) = map(Base.OneTo, dims[pos])

# Julia's result shape of an undotted `+`, `-`, `*` or `/` with an array
# operand: equal shapes for a sum, one array beside numbers for a product.
function _ppl_cell_latent_call_axes(f::Symbol, axes::Tuple...)
    arrays = filter(!isempty, collect(axes))
    isempty(arrays) && return ()
    f in (:+, :-) && return Base.Broadcast.broadcast_shape(axes...)
    length(arrays) == 1 && return only(arrays)
    throw(ArgumentError("a cell latent argument applies `$f` to several arrays; " *
        "write its elementwise form (`.$f`)"))
end

# The axes of an array read at indices: the indices' axes in order.
_ppl_cell_index_axes(axes::Tuple...) = (Iterators.flatten(axes)...,)

# A model value read at an index is an array of numbers only when the data
# it computes from hold numbers.
function _ppl_cell_number_entries(names::Tuple, values::Tuple, axes::Tuple)
    for (name, x) in zip(names, values)
        x isa Number || x isa AbstractArray &&
            (eltype(x) <: Number || all(v -> v isa Number, x)) ||
            throw(ArgumentError("a cell latent argument reads a model value at an " *
                "index, which is a number only when `$name` holds numbers; it " *
                "holds a $(typeof(x))"))
    end
    return axes
end

# One index's entries of an argument, broadcast to that index's dimensions
# (column-major), in one fresh vector.
_ppl_cell_latent_entries(x::Number, dims::Tuple) = fill(x, prod(dims))
function _ppl_cell_latent_entries(x::AbstractArray, dims::Tuple)
    out = Vector{eltype(x)}(undef, prod(dims))
    for (k, I) in enumerate(CartesianIndices(dims))
        # A singleton dimension of `x` stretches, as broadcasting does.
        out[k] = x[ntuple(d -> size(x, d) == 1 ? 1 : I[d], Val(ndims(x)))...]
    end
    return out
end

# Every index's entries, in plate order, in one fresh vector.
function _ppl_cell_latent_flatten(cells::AbstractVector)
    n = 0
    for c in cells
        n += length(c)
    end
    T = isempty(cells) ? Float64 : mapreduce(eltype, promote_type, cells)
    out = Vector{T}(undef, n)
    k = 0
    for c in cells, x in c
        out[k += 1] = x
    end
    return out
end

_ppl_cell_joint_extent(n) = n isa Integer && !(n isa Bool) && n >= 0

# Each index's constrained shape, from the joint geometry's extents there.
function _ppl_cell_joint_shapes(name::Symbol, cells::AbstractVector)
    for (k, extents) in enumerate(cells)
        extents isa Tuple && all(_ppl_cell_joint_extent, extents) || throw(ArgumentError(
            "cell latent `$name`: its geometry's shape at index $k must be a tuple " *
            "of nonnegative integers, got $(repr(extents))"))
    end
    return _ppl_cell_latent_shapes(name, [map(Int, extents) for extents in cells])
end

# Each index's packed block is `(offsets[k] + 1):offsets[k + 1]`.
function _ppl_cell_joint_offsets(name::Symbol, sizes::AbstractVector)
    offsets = Vector{Int}(undef, length(sizes) + 1)
    offsets[1] = 0
    for (k, n) in enumerate(sizes)
        _ppl_cell_joint_extent(n) || throw(ArgumentError("cell latent `$name`: its " *
            "geometry's packed dimension at index $k must be a nonnegative integer, " *
            "got $(repr(n))"))
        offsets[k + 1] = offsets[k] + n
    end
    return offsets
end

_ppl_cell_joint_block(site, offsets, pos::Int) =
    view(site, (offsets[pos] + 1):offsets[pos + 1])

# A cell's constrained value of its joint latent: the geometry's `constrain`
# on its packed block, read in place, with the cell's argument values.
function _ppl_cell_joint_constrain(g::ParameterGeometry, site, offsets, shapes,
        pos::Int, args...; kwargs...)
    shape = shapes[pos]
    value = g.constrain(_ppl_cell_joint_block(site, offsets, pos), shape, args...; kwargs...)
    _ppl_cell_joint_check(value, shape, pos)
    return value
end
_ppl_cell_joint_logjac(g::ParameterGeometry, site, offsets, shapes, pos::Int,
        args...; kwargs...) =
    g.logjac(_ppl_cell_joint_block(site, offsets, pos), shapes[pos], args...; kwargs...)

function _ppl_cell_joint_check(value, shape::Tuple, pos::Int)
    (isempty(shape) ? value isa Number : value isa AbstractArray && size(value) == shape) ||
        throw(ArgumentError("a cell latent's geometry constrained index $pos to " *
            "$(summary(value)); its shape there is $shape"))
    return nothing
end

# A cell's block of the packed array: a number, a vector, or an array of
# its dimensions. The packed values are read in place.
_ppl_cell_latent_block(site, offsets, dims::AbstractVector{<:Tuple}, pos::Int) =
    _ppl_cell_latent_value(site, offsets, dims[pos], pos)
_ppl_cell_latent_value(site, offsets, ::Tuple{}, pos::Int) = site[offsets[pos] + 1]
_ppl_cell_latent_value(site, offsets, ::Tuple{Int}, pos::Int) =
    view(site, (offsets[pos] + 1):offsets[pos + 1])
_ppl_cell_latent_value(site, offsets, dims::Tuple, pos::Int) =
    reshape(view(site, (offsets[pos] + 1):offsets[pos + 1]), dims)

# --- Draws: one array per index under the authored name ------------------

function _cell_latent_dims(layout, site::Symbol)
    _, dims = layout.cell_latents[site]
    haskey(layout.bound_values, dims) || throw(ContractValidationError(
        "[layout] cell latent $(first(layout.cell_latents[site])) needs bound data"))
    return layout.bound_values[dims]
end

function _cell_latent_nested(layout, site::Symbol, v::AbstractVector)
    dims = _cell_latent_dims(layout, site)
    offsets = _ppl_cell_latent_offsets(dims)
    return [isempty(d) ? v[offsets[k] + 1] :
        reshape(v[(offsets[k] + 1):offsets[k + 1]], d) for (k, d) in enumerate(dims)]
end

function _cell_latent_flat(layout, site::Symbol, value)
    name = first(layout.cell_latents[site])
    dims = _cell_latent_dims(layout, site)
    value isa AbstractVector && length(value) == length(dims) || throw(
        ContractValidationError("[layout] cell latent $name needs one value per " *
            "index ($(length(dims)))"))
    for (k, (x, d)) in enumerate(zip(value, dims))
        size(x) == d || throw(ContractValidationError("[layout] cell latent " *
            "$name[$k] has size $(size(x)), want $d"))
    end
    return reduce(vcat, (vec(collect(x)) for x in value); init = Float64[])
end

function _cell_latent_coordinates(layout, site::Symbol)
    shown = _draw_name(layout, site)
    names = Symbol[]
    for (k, d) in enumerate(_cell_latent_dims(layout, site))
        if isempty(d)
            push!(names, Symbol(shown, ".", k))
        else
            for j in CartesianIndices(d)
                push!(names, Symbol(shown, ".", k, ".", join(Tuple(j), ".")))
            end
        end
    end
    return names
end

# --- Draws of a joint cell latent: its geometry at each index -------------

# Each index's shape and packed block, and a reader of the arguments the
# geometry's endpoints take there. `values` holds the constrained values of
# the other parameters, which the arguments may read.
function _cell_geometry_frame(layout, site::Symbol, values)
    g = layout.cell_geometries[site]
    lookup = _layout_lookup(layout, values)
    shapes, offsets = lookup(g.shapes), lookup(g.offsets)
    read((kind, v)) = kind === :cells ? lookup(v) : _eval_value_expr(v, lookup, g.name)
    args = map(read, g.arguments)
    keywords = map(k -> first(k) => read(last(k)), g.keywords)
    at(k) = (Tuple(kind === :cells ? x[k] : x for ((kind, _), x) in zip(g.arguments, args)),
        (; (first(kw) => (first(last(kw)) === :cells ? last(v)[k] : last(v))
            for (kw, v) in zip(g.keywords, keywords))...))
    return g, shapes, offsets, at
end

function _cell_geometry_constrain(layout, site::Symbol, seg::AbstractVector, values)
    g, shapes, offsets, at = _cell_geometry_frame(layout, site, values)
    last(offsets) == length(seg) || throw(ContractValidationError(
        "[layout] cell latent $(g.name) packs $(last(offsets)) coordinates, got $(length(seg))"))
    out = Any[]
    for k in eachindex(shapes)
        args, kwargs = at(k)
        v = g.geometry.constrain(_ppl_cell_joint_block(seg, offsets, k), shapes[k], args...; kwargs...)
        _ppl_cell_joint_check(v, shapes[k], k)
        push!(out, v)
    end
    return isempty(out) ? out : [x for x in out]
end

function _cell_geometry_unconstrain(layout, site::Symbol, value, values)
    g, shapes, offsets, at = _cell_geometry_frame(layout, site, values)
    value isa AbstractVector && length(value) == length(shapes) || throw(
        ContractValidationError("[layout] cell latent $(g.name) needs one value per " *
            "index ($(length(shapes)))"))
    u = Vector{_host_value_type(value)}(undef, last(offsets))
    for k in eachindex(shapes)
        _ppl_cell_joint_check(value[k], shapes[k], k)
        args, kwargs = at(k)
        block = g.geometry.unconstrain(value[k], shapes[k], args...; kwargs...)
        n = offsets[k + 1] - offsets[k]
        block isa AbstractVector && length(block) == n || throw(ContractValidationError(
            "[layout] cell latent $(g.name)[$k]: its geometry's inverse returned " *
            "$(summary(block)), want $n packed coordinates"))
        u[(offsets[k] + 1):offsets[k + 1]] .= block
    end
    return u
end

function _cell_geometry_logjac(layout, site::Symbol, seg::AbstractVector, values)
    g, shapes, offsets, at = _cell_geometry_frame(layout, site, values)
    total = zero(_host_eltype(seg))
    for k in eachindex(shapes)
        args, kwargs = at(k)
        total += g.geometry.logjac(_ppl_cell_joint_block(seg, offsets, k), shapes[k],
            args...; kwargs...)
    end
    return total
end

# `c.k.j` for packed coordinate `j` of index `k`, or `c.k.<label>` with the
# geometry's coordinate names.
function _cell_geometry_coordinates(layout, site::Symbol)
    g = layout.cell_geometries[site]
    lookup = _layout_lookup(layout, Dict{Symbol,Any}())
    offsets = lookup(g.offsets)
    shown = _draw_name(layout, site)
    labels = g.geometry.coordinate_names
    names = Symbol[]
    for k in 1:(length(offsets) - 1)
        n = offsets[k + 1] - offsets[k]
        if labels === nothing
            append!(names, (Symbol(shown, ".", k, ".", j) for j in 1:n))
        else
            length(labels) == n || throw(ContractValidationError("[layout] cell latent " *
                "$(g.name)[$k]: its geometry names $(length(labels)) coordinates, it packs $n"))
            append!(names, (Symbol(shown, ".", k, ".", label) for label in labels))
        end
    end
    return names
end
