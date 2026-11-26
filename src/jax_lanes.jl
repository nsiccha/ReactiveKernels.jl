# Lane partitions of a plate by its data-bound branches.
#
# A plate is one `jax.vmap` of its cell. Under `vmap` a `jax.lax.cond` whose
# condition differs between cells evaluates both arms on every cell and
# selects, so a branch that only the data decide (a censored row's tail,
# `y <= lower ? logcdf(lower) : logpdf(y)`) runs its expensive arm on every
# row. CONTRIBUTING, "Lazy branches": data-bound branch partitioning must
# evaluate only the taken arm for each lane.
#
# With `data`, the conditions that read only the plate's data are known once
# per dataset. `prepare_data` evaluates them for every cell and returns, for
# each pattern of decisions, the positions of the cells that take it and the
# permutation that restores the plate's order. The functions run one plate per
# pattern over its gathered cells, translated with those conditions decided,
# so each cell emits only the arms it takes (`_jax_conditional` of a decided
# condition translates one arm), and concatenate the parts in the plate's
# order. Native partial evaluation splits a plate by a data-bound branch the
# same way (`_partition_plate_recipe`); there the cell's top-level branches
# are split, here also branches nested in an arm of a parameter-dependent one
# (RKPPL keeps a law's parameter-support guard outside its evidence branches).
#
# The patterns follow the cell's source alone: every pattern has its plate,
# also one that no cell takes, so the module text does not depend on the data
# or on which arms its cells take (CONTRIBUTING, "Runtime values and sizes").

# The branch nodes (`if`, `&&`, `||` expressions, by identity) whose condition
# the plate being emitted decides, with each one's value: a stack, innermost
# last, active while one pattern's plate is translated.
const _JAX_LANE_DECISIONS = IdDict{Any,Bool}[]

function _jax_with_lane_decisions(f, decisions::IdDict{Any,Bool})
    push!(_JAX_LANE_DECISIONS, decisions)
    try
        return f()
    finally
        pop!(_JAX_LANE_DECISIONS)
    end
end

# The decided value of branch node `node`'s condition as a static literal, or
# `nothing` when no partitioned plate decides it.
function _jax_lane_decision(ctx::_JaxContext, node)
    ctx.host === nothing && !isempty(_JAX_LANE_DECISIONS) || return nothing
    for d in Iterators.reverse(_JAX_LANE_DECISIONS)
        haskey(d, node) && return _PyVal(d[node] ? "True" : "False", Bool, true)
    end
    nothing
end

# Expression `ex` with every branch node a partitioned plate decides replaced
# by the arm it takes (`ex` itself when it holds none). Pruning comes before
# the stable rewrites, which rebuild the expression they rewrite: a decided
# node is found by identity, as the partition recorded it.
function _jax_lane_prune(ex)
    isempty(_JAX_LANE_DECISIONS) && return ex
    _jax_lane_prune_walk(ex)
end
function _jax_lane_prune_walk(ex)
    ex isa Expr || return ex
    for d in Iterators.reverse(_JAX_LANE_DECISIONS)
        haskey(d, ex) || continue
        v = d[ex]
        ex.head in (:if, :elseif) &&
            return v ? _jax_lane_prune_walk(ex.args[2]) :
                   length(ex.args) >= 3 ? _jax_lane_prune_walk(ex.args[3]) : nothing
        ex.head === :&& && return v ? _jax_lane_prune_walk(ex.args[2]) : false
        ex.head === :|| && return v ? true : _jax_lane_prune_walk(ex.args[2])
    end
    ex.head in _JAX_LANE_OPAQUE && return ex
    args = Any[_jax_lane_prune_walk(a) for a in ex.args]
    all(i -> args[i] === ex.args[i], eachindex(args)) ? ex : Expr(ex.head, args...)
end

# Recipe `r` with its source pruned (`_jax_lane_prune`), or `r` itself.
function _jax_lane_pruned(r::Recipe)
    isempty(_JAX_LANE_DECISIONS) && return r
    source = _jax_lane_prune(r.source)
    source === r.source && return r
    Recipe(r.id, r.inputs, r.outputs, r.op, r.cost, r.cse_key, r.effectful, source)
end

# One data condition: branch node `node`'s condition, read through the
# recipe's source names `names`, each the cell's HAVE port at `ports[i]`
# (`0`: a name the condition does not read); `mod` and `captures` resolve its
# globals as the recipe's source does.
struct _JaxLaneCondition
    node::Any
    names::Vector{Symbol}
    ports::Vector{Int}
    mod::Module
    captures::Any
end

# `prepare_data`'s recipe of a plate's lane partition: the conditions, the
# patterns (each the conditions it decides, by index, with their values), and
# the plate inputs the conditions read (`inputs`, positions in the plate's
# inputs, with `atomic[i]` for one passed whole to every cell).
struct _JaxLanePrefix
    conditions::Vector{_JaxLaneCondition}
    patterns::Vector{Vector{Pair{Int,Bool}}}
    inputs::Vector{Int}
    atomic::Vector{Bool}
    plate::Int
    ragged::Bool
end

struct _JaxRaggedSum
    ports::Vector{Int} # inner HAVE positions in the outer plate
    bound::Vector{Int} # outer axes included in the prepared domain
end

# The plate partitioned: the original recipe and its operation, the branch
# nodes the patterns decide and the patterns. Its inputs are the original
# plate's, then each pattern's cell positions, then the restoring order.
struct _JaxLanePlate{O}
    recipe::Recipe
    plate::O
    nodes::Vector{Any}
    patterns::Vector{Vector{Pair{Int,Bool}}}
    nested::Union{Nothing,_JaxRaggedSum}
end

const _JAX_LANE_PREFIX_BASE = -(Int(1) << 41)
_jax_lane_prefix_id(plate::Int) = _JAX_LANE_PREFIX_BASE - plate

# More patterns than this keep the plate whole: each pattern is one plate.
const _JAX_LANE_PATTERNS_MAX = 8

# Calls a data condition may make: total on numbers, so evaluating one on a
# cell whose enclosing branch does not reach it raises nothing.
const _JAX_LANE_TOTAL = Set{Symbol}([:<, :<=, :>, :>=, :(==), :!=, :≠, :≤, :≥, :!,
    :isfinite, :isinf, :isnan, :iszero, :isone, :signbit, :isequal, :isless,
    :&, :|, :xor, :⊻, :+, :-, :*, :/, :abs, :floor, :ceil, :(===), :(!==)])

_jax_lane_unwrap(ex) = _kernel_branch_unwrap(ex)

# Constructs whose branches are not searched: a loop or comprehension variable,
# a `let` or a closure argument can shadow the data name a condition reads.
const _JAX_LANE_OPAQUE = (:quote, :function, :->, :macrocall, :for, :while, :let, :do,
                          :generator, :comprehension, :typed_comprehension, :flatten)

# Whether `c` reads only `data` names (and at least one), through total calls
# on numbers and literals.
function _jax_lane_data_condition(c, data::Set{Symbol}, names::Set{Symbol}, mod::Module)
    reads = Ref(false)
    ok = _jax_lane_data_walk(_jax_lane_unwrap(c), data, names, mod, reads)
    ok && reads[]
end
function _jax_lane_data_walk(c, data, names, mod, reads)
    c isa Bool && return true
    c isa Number && return true
    if c isa Symbol
        if c in names
            c in data || return false
            reads[] = true
            return true
        end
        return c in (:Inf, :NaN, :true, :false)
    end
    c isa Expr || return false
    if c.head in (:&&, :||)
        return all(a -> _jax_lane_data_walk(_jax_lane_unwrap(a), data, names, mod, reads), c.args)
    elseif c.head === :comparison
        for (k, a) in enumerate(c.args)
            if iseven(k)
                _jax_lane_total_callee(a, mod) || return false
            else
                _jax_lane_data_walk(a, data, names, mod, reads) || return false
            end
        end
        return true
    elseif c.head === :call && !isempty(c.args)
        _jax_lane_total_callee(c.args[1], mod) || return false
        return all(a -> _jax_lane_data_walk(a, data, names, mod, reads), c.args[2:end])
    end
    false
end
function _jax_lane_total_callee(f, mod::Module)
    f isa Symbol && return f in _JAX_LANE_TOTAL &&
        isdefined(mod, f) && getglobal(mod, f) === getglobal(Base, f)
    f isa GlobalRef && return f.name in _JAX_LANE_TOTAL && f.mod in (Base, Core) &&
        isdefined(Base, f.name)
    false
end

# The partition points of a cell: in emission order, each branch node of
# recipe `r`'s source whose condition reads only data, with that recipe's
# condition record. Arms with their own graph (a leaf arm's `then_spec`) are
# not searched: their branches stay lazy `jax.lax.cond`s.
function _jax_lane_points!(points, r::Recipe, cg::Graph, port_of::Dict{ValueId,Int},
                           data_ports::Set{Int})
    r.op isa _KernelSourceOp && _has_source(r) || return points
    names = _recipe_source_names(r)
    length(names) == length(r.inputs) || return points
    ports = Int[get(port_of, canon_id(cg, v.id), 0) for v in r.inputs]
    data = Set{Symbol}(n for (n, p) in zip(names, ports) if p in data_ports)
    isempty(data) && return points
    mod = _jax_source_module(r.op)
    captures = _jax_source_captures(r.op)
    allnames = Set{Symbol}(names)
    # A name an arm reassigns, or passes to a helper that may write it in
    # place (`_jax_argument_writes!`), is not the input it names everywhere.
    assigned = _jax_assigned_names!(Set{Symbol}(), r.source)
    local_names = union(allnames, assigned)
    _jax_argument_writes!(assigned, r.source, in(data),
                          head -> _jax_static_callee(mod, head, local_names),
                          Dict{Tuple{Any,Int},Bool}())
    setdiff!(data, assigned)
    f = getfield(r.op, :f)
    plan_arms = f isa _KernelBranch ? (f.then_spec isa Plan, f.else_spec isa Plan) : (false, false)
    function visit(ex, top::Bool)
        ex = _jax_lane_unwrap(ex)
        ex isa Expr || return
        if ex.head in (:if, :elseif, :&&, :||) && length(ex.args) >= 2
            if _jax_lane_data_condition(ex.args[1], data, allnames, mod)
                push!(points, (ex, _JaxLaneCondition(ex, names,
                    Int[n in data ? p : 0 for (n, p) in zip(names, ports)], mod, captures)))
            else
                visit(ex.args[1], false)
            end
            for (k, arm) in enumerate(ex.args[2:end])
                top && k <= 2 && plan_arms[k] && continue
                visit(arm, false)
            end
            return
        end
        ex.head in _JAX_LANE_OPAQUE && return
        foreach(a -> visit(a, false), ex.args)
    end
    visit(r.source, f isa _KernelBranch)
    points
end

# The decision patterns of branch nodes `points` over sources `sources`:
# lists of `(point index => value)`, one per reachable combination, in source
# order; `nothing` past `_JAX_LANE_PATTERNS_MAX`.
function _jax_lane_patterns(sources, points)
    index = IdDict{Any,Int}(node => k for (k, (node, _)) in enumerate(points))
    patterns = Vector{Pair{Int,Bool}}[]
    function first_open(ex, decided)
        ex = _jax_lane_unwrap(ex)
        ex isa Expr || return nothing
        if ex.head in (:if, :elseif, :&&, :||) && length(ex.args) >= 2
            k = get(index, ex, 0)
            if k != 0
                haskey(decided, k) || return k
                v = decided[k]
                taken = ex.head in (:if, :elseif) ? (v ? ex.args[2] : get(ex.args, 3, nothing)) :
                    ex.head === :&& ? (v ? ex.args[2] : nothing) : (v ? nothing : ex.args[2])
                return first_open(taken, decided)
            end
        end
        ex.head in _JAX_LANE_OPAQUE && return nothing
        for a in ex.args
            k = first_open(a, decided)
            k === nothing || return k
        end
        nothing
    end
    function grow(decided::Dict{Int,Bool})
        length(patterns) > _JAX_LANE_PATTERNS_MAX && return
        k = nothing
        for s in sources
            k = first_open(s, decided)
            k === nothing || break
        end
        if k === nothing
            push!(patterns, sort!([d for d in decided]; by = first))
            return
        end
        grow(merge(decided, Dict(k => true)))
        grow(merge(decided, Dict(k => false)))
    end
    grow(Dict{Int,Bool}())
    length(patterns) > _JAX_LANE_PATTERNS_MAX && return nothing
    patterns
end

# Plate recipe `r` partitioned by its cell's data conditions: `(; prefix,
# plate)`, the `prepare_data` recipe of the patterns' cell positions and the
# plate reading them, or `nothing` for unsupported shapes or conditions.
# The cell either produces a scalar or sums an inner plate of scalars.
# `dependent` holds the canonical ids in `g` of the values computed from data.
function _jax_lane_partition(g::Graph, r::Recipe, dependent::Set{ValueId})
    op = r.op
    op isa _AuthoredPlateOp || return nothing
    body = op.kernel.plan
    atomic_inputs = typeof(op).parameters[2]
    length(body.have) == length(r.inputs) && length(r.outputs) == 1 &&
        length(body.want) == 1 || return nothing
    # Graph types may be unknown (`Any`); emission checks the argument
    # and cell types when they are known.
    T = valtype(only(r.outputs))
    T === Any || T isa Type && T <: AbstractVector && eltype(T) <: Number || return nothing
    cg = body.graph
    port_of = Dict{ValueId,Int}(canon_id(cg, v.id) => i for (i, v) in enumerate(body.have))
    nested = _jax_ragged_sum_ports(body, op, port_of)
    if nested !== nothing
        body = first(body.recipes).op.kernel.plan
        cg = body.graph
        port_of = Dict(canon_id(cg, v.id) => i for (v, i) in zip(body.have, nested))
    end
    # Each argument is shared (passed whole: a `Ref`, or a number), a lane
    # vector (its entries are the cells, which a pattern gathers), or of a
    # type the graph does not know, which the functions check.
    data_ports = Set{Int}()
    lane_ports = Int[]
    for (i, v) in enumerate(r.inputs)
        V = valtype(v)
        shared = i in atomic_inputs || V isa Type && V <: Number
        if !shared && V !== Any
            supported = nested === nothing ? V isa Type && V <: AbstractVector && eltype(V) <: Number :
                                             _jax_lane_row_type(V)
            supported || return nothing
        end
        canon_id(g, v.id) in dependent || continue
        push!(data_ports, i)
        shared || push!(lane_ports, i)
    end
    isempty(lane_ports) && return nothing
    points = Tuple{Any,_JaxLaneCondition}[]
    sources = Any[]
    _jax_lane_sources!(points, sources, body, port_of, data_ports, nested !== nothing) || return nothing
    isempty(points) && return nothing
    # A condition read in two places would be decided in both.
    length(unique(objectid ∘ first, points)) == length(points) || return nothing
    patterns = _jax_lane_patterns(sources, points)
    patterns === nothing && return nothing
    # Conditions no pattern decides are not computed.
    used = sort!(unique(k for pat in patterns for (k, _) in pat))
    renumber = Dict(k => j for (j, k) in enumerate(used))
    conditions = [last(points[k]) for k in used]
    patterns = [[renumber[k] => v for (k, v) in pat] for pat in patterns]
    # The plate inputs the conditions read, and at least one mapped one, whose
    # entries are the cells.
    read = sort!(unique(p for c in conditions for p in c.ports if p != 0))
    if nested !== nothing
        read = sort!(union(read, lane_ports))
    elseif !any(in(lane_ports), read)
        push!(read, first(lane_ports))
        sort!(read)
    end
    stem = _jax_stable_name(first(r.outputs).name)
    lanes = [Value(Symbol(stem, :_lanes_, j), Vector{Int}) for j in eachindex(patterns)]
    layout = nested === nothing ? (:_lane_order,) : (:_lane_rows, :_lane_columns, :_row_lengths)
    extra = [Value(Symbol(stem, suffix), Vector{Int}) for suffix in layout]
    prefix = Recipe(_jax_lane_prefix_id(r.id), Tuple(r.inputs[i] for i in read),
                    Tuple([lanes; extra]),
                    _JaxLanePrefix(conditions, patterns, read, Bool[i in atomic_inputs for i in read],
                                   r.id, nested !== nothing),
                    0.0, nothing, false)
    plate = Recipe(r.id, (r.inputs..., lanes..., extra...), r.outputs,
                   _JaxLanePlate(r, op, Any[c.node for c in conditions], patterns,
                                 nested === nothing ? nothing : _JaxRaggedSum(nested, lane_ports)),
                   r.cost, r.cse_key, r.effectful, r.source)
    (; prefix, plate)
end

# `prepare_data`: each cell's condition values, then each pattern's cell
# positions (0-based, for the functions' gathers) and the order that restores
# the plate's cells from the patterns' concatenated values.
function _jax_recipe(ctx::_JaxContext, p::Plan, recipe::Recipe, op::_JaxLanePrefix, inputs)
    ctx.host === nothing && _jax_error("a plate's lane partition outside `prepare_data`",
                                       _jax_where(ctx))
    # A cell reads a mapped input's entry, and any other input whole.
    mapped = Bool[!op.atomic[i] && _jax_is_array(v.T) for (i, v) in enumerate(inputs)]
    any(mapped) && all(i -> !mapped[i] || _jax_rank(inputs[i].T) == 1, eachindex(inputs)) ||
        _jax_error("a lane partition reads no vector of the plate's cells", _jax_where(ctx))
    params = Symbol[Symbol(:lane_input_, i) for i in eachindex(inputs)]
    if op.ragged
        all(i -> mapped[i] ? _jax_lane_row_type(inputs[i].T) : _jax_unref(inputs[i].T) <: Number,
            eachindex(inputs)) || _jax_error("ragged lane conditions need scalar entries", _jax_where(ctx))
    end
    types = Any[mapped[i] ? (op.ragged ? eltype(eltype(v.T)) : eltype(v.T)) :
                            _jax_unref(v.T) for (i, v) in enumerate(inputs)]
    position = Dict(port => k for (k, port) in enumerate(op.inputs))
    fname, _ = _host_nested_function!(ctx, :lane_conditions, params, types, () -> begin
        codes = String[]
        for c in op.conditions
            names = Symbol[n for (n, port) in zip(c.names, c.ports) if port != 0]
            values = _PyVal[ctx.locals[params[position[port]]] for port in c.ports if port != 0]
            v = _jax_source_expr(ctx, _jax_lane_unwrap(c.node.args[1]), names, values,
                                 c.mod, c.captures)
            push!(codes, "bool(" * _host_read(ctx, v).code * ")")
        end
        _PyVal("(" * join(codes, ", ") * ",)", NTuple{length(codes),Bool}, true)
    end)
    op.ragged && return _jax_ragged_lane_prefix(ctx, op, inputs, mapped, fname)
    targets, args, sources = String[], String[], String[]
    for (i, v) in enumerate(inputs)
        if !mapped[i]
            push!(args, v.code)
        else
            t = _jax_fresh(ctx, :entry)
            push!(targets, t); push!(args, t); push!(sources, v.code)
        end
    end
    iterable = length(sources) == 1 ? only(sources) : "zip(" * join(sources, ", ") * ")"
    k = length(op.conditions)
    rows = _jax_bind!(ctx, :lane_conditions, _PyVal("np.asarray([" * fname * "(" *
        join(args, ", ") * ") for " * join(targets, ", ") * " in " * iterable *
        "], dtype=bool).reshape((-1, " * string(k) * "))", Matrix{Bool}, true))
    lanes = _jax_lane_indices(ctx, rows, op.patterns)
    order = _PyVal("np.argsort(np.concatenate((" * join((l.code for l in lanes), ", ") *
                   ",)), kind=\"stable\")", Vector{Int}, true)
    [lanes; order]
end

function _jax_lane_indices(ctx, rows, patterns)
    map(patterns) do pattern
        mask = join([(v ? "" : "~") * rows.code * "[:, " * string(c - 1) * "]" for (c, v) in pattern],
                    " & ")
        _jax_bind!(ctx, :lanes, _PyVal("np.flatnonzero(" * mask * ")", Vector{Int}, true))
    end
end

# The functions: one plate per pattern over its cells, each translated with
# the pattern's conditions decided, concatenated in the plate's order.
#
# The plate is partitioned only where its arguments and cells have the forms
# this assembly handles: every mapped argument an unpadded vector of numbers,
# and every cell a number. Otherwise the lines a pattern emitted are taken
# back and the plate is one `jax.vmap`, as without the partition (its cell
# positions from `prepare_data` then go unread).
_jax_lane_number_vector(v::_PyVal) = _jax_is_array(v.T) && _jax_rank(v.T) == 1 &&
    eltype(v.T) isa Type && eltype(v.T) <: Number && isempty(v.extent)
function _jax_recipe(ctx::_JaxContext, p::Plan, recipe::Recipe, op::_JaxLanePlate, inputs)
    ctx.host === nothing || _jax_error("a partitioned plate in `prepare_data`", _jax_where(ctx))
    op.nested === nothing || return _jax_ragged_lane_sum(ctx, p, op, inputs)
    k = length(op.patterns)
    n = length(inputs) - k - 1
    base, lanes, order = inputs[1:n], inputs[n+1:n+k], inputs[end]
    atomic_inputs = typeof(op.plate).parameters[2]
    whole() = _jax_recipe(ctx, p, op.recipe, op.plate, base)
    # The arguments the plate maps, as its own emission decides (a number or
    # a `Ref` operand is passed whole to every cell).
    mapped = Bool[!(i in atomic_inputs) && _jax_is_array(v.T) for (i, v) in enumerate(base)]
    any(mapped) && all(_jax_lane_number_vector(v) for (i, v) in enumerate(base) if mapped[i]) ||
        return whole()
    mark, effects = length(ctx.scope.lines), _JAX_EFFECTS[]
    parts = String[]
    E = Union{}
    for (j, pattern) in enumerate(op.patterns)
        gathered = _PyVal[mapped[i] ?
            _jax_bind!(ctx, :lane_entries,
                       _PyVal("jnp.asarray(" * v.code * ")[" * lanes[j].code * "]", v.T)) : v
            for (i, v) in enumerate(base)]
        decisions = IdDict{Any,Bool}(op.nodes[c] => b for (c, b) in pattern)
        part = only(_jax_with_lane_decisions(decisions) do
            _jax_recipe(ctx, p, op.recipe, op.plate, gathered)
        end)
        if !_jax_lane_number_vector(part)
            resize!(ctx.scope.lines, mark)
            _JAX_EFFECTS[] = effects
            return whole()
        end
        push!(parts, _jax_bind!(ctx, :plate_part, part).code)
        E = _host_join(E, _jax_eltype(part.T))
    end
    [_PyVal("jnp.concatenate((" * join(parts, ", ") * ",))[" * order.code * "]", Vector{E})]
end

include("jax_ragged_lanes.jl")
