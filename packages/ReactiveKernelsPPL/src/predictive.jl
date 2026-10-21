# ── Posterior-predictive draws ─────────────────────────────────────────
#
# The graph computes, for every observation, the arguments of the law it is
# drawn from: a value of the parameters and the design like any other, with
# no randomness in it. The draw happens at the query boundary, in ordinary
# Julia: the Distributions.jl extension builds each observation's law from
# those arguments and `rand(rng, law)` draws it (user decision
# `ReactiveKernels:snag.predictive-rng-f-f51c6561/decisions/2026-10-21T10-08-19-848-05ameuz`,
# option 1; generation may reuse Distributions.jl, principle `0vu7yfd`).
# No RNG enters an RK kernel, and the sampler's coordinates, density and
# gradients are untouched: the law node is a separate WANT.

_law_name(label::Symbol) = Symbol(:_ppl_law_, label)

_is_logpdf_call(ex) = Meta.isexpr(ex, :call) && ex.args[1] isa Expr &&
    ex.args[1].head === :. && ex.args[1].args[2] == QuoteNode(:logpdf)
_has_logpdf_call(ex) = ex isa Expr && ex.head !== :(->) &&
    (_is_logpdf_call(ex) || any(_has_logpdf_call, ex.args))
_is_minus_infinity(ex) = ex == -Inf || ex == :(-Inf)

# The RK distribution object whose `logpdf` IS the cell's density at the
# response `y`, or `nothing`. The density may sit inside support guards
# (`cond ? density : -Inf`), the clamp law's tail arms (`y <= lo ?
# log(cdf(lo)) : density`) and, for truncation, minus the log mass inside the
# bounds: the evidence algebra of `evidence.jl`. A density that computes
# anything else with the object's `logpdf` (a hurdle, a mixture, a weight)
# has no such object, so it is not drawn from one.
function _endpoint_density(ex, y::Symbol, kind::Symbol)
    ex isa Expr || return nothing
    if Meta.isexpr(ex, :block)
        body = filter(a -> !(a isa LineNumberNode), ex.args)
        length(body) == 1 || return nothing
        return _endpoint_density(only(body), y, kind)
    elseif _is_logpdf_call(ex)
        length(ex.args) == 2 && y in _statement_value_reads!(Set{Symbol}(), ex.args[2]) ||
            return nothing
        return ex.args[1].args[1]
    elseif Meta.isexpr(ex, :if, 3)
        a, b = ex.args[2], ex.args[3]
        # One arm is the density; the other ends the cell without it (an
        # impossible observation, or a censored tail's mass).
        terminal(arm) = _is_minus_infinity(arm) || arm == 0.0 || !_has_logpdf_call(arm)
        terminal(b) && return _endpoint_density(a, y, kind)
        terminal(a) && return _endpoint_density(b, y, kind)
        return nothing
    elseif kind === :truncated && Meta.isexpr(ex, :call, 3) && ex.args[1] === :- &&
            !_has_logpdf_call(ex.args[3])
        return _endpoint_density(ex.args[2], y, kind)
    end
    return nothing
end

# An RK distribution object's construction, as a named tuple of its
# arguments tagged with the object's name: `normal(loc, s)` gives
# `(law = Val{:normal}(), _1 = loc, _2 = s)`, `poisson(; log_rate = eta)`
# gives `(law = Val{:poisson}(), log_rate = eta)`. A value the extension maps
# to a Distributions.jl law.
function _law_fields(X)
    Meta.isexpr(X, :call) || return nothing
    ctor = X.args[1]
    name = ctor isa GlobalRef ? ctor.name : ctor isa Symbol ? ctor : return nothing
    tag = Expr(:call, Expr(:curly, GlobalRef(Base, :Val), QuoteNode(name)))
    fields = Any[Expr(:(=), :law, tag)]
    position = 0
    for a in X.args[2:end]
        if Meta.isexpr(a, :parameters)
            for kw in a.args
                Meta.isexpr(kw, :kw, 2) || return nothing
                push!(fields, Expr(:(=), kw.args[1], kw.args[2]))
            end
        elseif Meta.isexpr(a, :kw, 2)
            push!(fields, Expr(:(=), a.args[1], a.args[2]))
        else
            position += 1
            push!(fields, Expr(:(=), Symbol(:_, position), a))
        end
    end
    return fields
end

# The statements of `stmts` the last one needs, in order, and every name
# they read.
function _needed_statements(stmts::Vector{Expr})
    target = stmts[end]
    needed = _statement_value_reads!(Set{Symbol}(), target.args[2])
    keep = Expr[target]
    for st in reverse(stmts[1:end-1])
        names = _assigned_names(st)
        (isempty(names) || isempty(intersect(names, needed))) && continue
        pushfirst!(keep, st)
        _statement_value_reads!(needed, st.args[2])
    end
    return keep, needed
end

# A response whose observations have no single law to draw from.
_law_refused(r::LikelihoodSpec) = r.weights !== nothing || r.mi_jobs !== nothing ||
    r.family === MixtureFam || r.evidence.kind === :interval_censored

# The law plate of a response, read off its pointwise density plate: the same
# plate without the response's own input, whose cell keeps the cell
# statements the distribution object's arguments need and returns those
# arguments (`_law_fields`), with the evidence bounds. Returns `nothing` when
# the density is not one RK distribution object of the response.
function _law_statement(r::LikelihoodSpec, stmts::Vector{Expr})
    _law_refused(r) && return nothing
    pw = _pw_name(r.label)
    i = findlast(st -> Meta.isexpr(st, :(=), 2) && st.args[1] === pw, stmts)
    i === nothing && return nothing
    rhs = stmts[i].args[2]
    _is_plate_do(rhs) || return nothing
    lanes = rhs.args[1].args[2:end]
    lambda = rhs.args[2]
    params = Meta.isexpr(lambda.args[1], :tuple) ? lambda.args[1].args : Any[lambda.args[1]]
    length(params) == length(lanes) || return nothing
    body = Expr[a for a in lambda.args[2].args if !(a isa LineNumberNode)]
    isempty(body) && return nothing
    locals, final = body[1:end-1], body[end]
    response = params[1]
    response isa Symbol || return nothing
    endpoint = _endpoint_density(final, response, r.evidence.kind)
    endpoint === nothing && return nothing
    fields = _law_fields(endpoint)
    fields === nothing && return nothing
    ev = r.evidence
    if ev.kind !== :none
        push!(fields, Expr(:(=), :evidence,
            Expr(:call, Expr(:curly, GlobalRef(Base, :Val), QuoteNode(ev.kind)))))
        for (field, bound) in ((:lower, ev.lower), (:upper, ev.upper))
            bound === nothing && continue
            if bound isa Real
                push!(fields, Expr(:(=), field, Float64(bound)))
            else
                k = findfirst(==(bound), lanes)
                k === nothing && return nothing
                push!(fields, Expr(:(=), field, params[k]))
            end
        end
    end
    value = Expr(:tuple, fields...)
    keep, needed = _needed_statements(Expr[locals..., :(_ppl_law_cell = $value)])
    response in needed && return nothing
    used = [k for k in eachindex(params) if params[k] in needed]
    cell = Expr(:block, LineNumberNode(0, :generator), keep[1:end-1]..., value)
    # The response's input is gone, and the arguments left may all be numbers
    # shared by every entry. A one-entry anchor keeps the plate batched; the
    # query broadcasts the laws over the response's entries, as Julia does.
    anchor = Symbol(:_ppl_law_anchor_, r.label)
    plate_call = Expr(:do, Expr(:call, :plate, :(fill(false, 1)), lanes[used]...),
        Expr(:(->), Expr(:tuple, anchor, params[used]...), cell))
    return Expr(:(=), _law_name(r.label), plate_call)
end

# The predictive entry of a response: its law plate's cells as columns
# (`_ppl_law_columns`), and the response entries a ranged observation's cells
# belong to.
function _law_entry(r::LikelihoodSpec, plan::StructuralPlan, rs::Vector{Expr})
    name = _law_name(r.label)
    any(st -> _assigned_name(st) === name, rs) || return nothing
    columns = GlobalRef(@__MODULE__, :_ppl_law_columns)
    indices = Symbol(:_ppl_range_indices_, r.label)
    if r.range isa Expr && !_cell_broadcast_response(plan, r) &&
            any(st -> _assigned_name(st) === indices, rs)
        return :((; $columns($name)..., indices = $indices))
    end
    return :($columns($name))
end

# The law plate returns one record per cell, `(law = Val{:normal}(), _1 = …,
# _2 = …, evidence = Val{:censored}(), lower = …)`. RK batches a value over
# positions (`vectorize`) only when its leaves are numbers or numeric arrays,
# so the predictive node holds those records as columns, one array per
# argument, keyed by the law's and the evidence's names:
# `(law = (normal = (_1 = A, _2 = B),), evidence = (censored = (lower = L,),))`.
# Every column has the law plate's shape. A response holding one array per
# index concatenates its indices' cells, in index order, and records each
# index's plate shape as a column of `sizes`. An empty response has no law.
function _ppl_law_columns(cells::AbstractArray{T}) where {T<:NamedTuple}
    isconcretetype(T) && return _law_record_columns(cells, T)
    return isempty(cells) ? (law = NamedTuple(),) :
        _law_record_columns(cells, typeof(first(cells)))
end
function _ppl_law_columns(groups::AbstractVector{<:AbstractArray{<:NamedTuple,N}}) where {N}
    sizes = Matrix{Int}(undef, N, length(groups))
    for (i, g) in enumerate(groups)
        sizes[:, i] .= size(g)
    end
    cells = isempty(groups) ? eltype(eltype(groups))[] : reduce(vcat, vec.(groups))
    return (; _ppl_law_columns(cells)..., sizes)
end
# An empty response, or one whose law plate yields no typed cells.
function _ppl_law_columns(cells::Union{Tuple{},AbstractArray})
    isempty(cells) || all(isempty, cells) || throw(ContractValidationError(
        "[query] the observation laws are $(typeof(cells)), not records of " *
        "law arguments"))
    return (law = NamedTuple(),)
end

_law_column(cells, ::Val{field}) where {field} = map(c -> getfield(c, field), cells)
# `(name = (field = column, …),)` for a `Val{name}` tag.
_tagged_columns_expr(::Type{Val{name}}, fields::Tuple) where {name} =
    :(NamedTuple{($(QuoteNode(name)),)}((NamedTuple{$fields}(($((
        :(_law_column(cells, Val($(QuoteNode(f))))) for f in fields)...),)),)))
# `_law_statement` writes a cell as the law, its arguments, then `evidence`
# and the bounds. The field names are the cell type's, so the columns'
# layout is known when this compiles and the node stays concretely typed.
@generated function _law_record_columns(cells::AbstractArray,
        ::Type{NamedTuple{K,V}}) where {K,V}
    i = findfirst(==(:evidence), K)
    args, bounds = i === nothing ? (K[2:end], ()) : (K[2:i-1], K[i+1:end])
    entry = :(law = $(_tagged_columns_expr(V.parameters[1], args)))
    i === nothing && return :(($entry,))
    return :(($entry, evidence = $(_tagged_columns_expr(V.parameters[i], bounds))))
end

function _predictive_statement(points::Vector{Pair{Symbol,Any}},
        laws::Vector{Pair{Symbol,Any}})
    covered = Set(first.(laws))
    missing = Tuple(name for (name, _) in points if !(name in covered))
    isempty(missing) || return Expr(:(=), :_ppl_predictive, Expr(:call,
        GlobalRef(@__MODULE__, :_ppl_predictive_unsupported),
        Expr(:tuple, QuoteNode.(missing)...)))
    values = isempty(laws) ? :(NamedTuple()) :
        Expr(:tuple, (Expr(:(=), name, value) for (name, value) in laws)...)
    return Expr(:(=), :_ppl_predictive, values)
end

_ppl_predictive_unsupported(names::Tuple) = throw(ContractValidationError(
    "[query] predictive draws cover responses whose density is one RK " *
    "distribution object (Normal, LogNormal, Student t, Gamma, Poisson, " *
    "Binomial, …), plain, censored or truncated, observed through a flat, " *
    "ranged or in-cell plate. This program has no law to draw " *
    join(names, ", ") * " from. Not covered yet: densities that are not one " *
    "such object (mixtures, hurdle models, Beta, …), caller-owned sampling " *
    "laws (`LogDensity`), scalar `~` observations and conditioned " *
    "declarations. Weighted, interval-censored and packed (`mi`) responses " *
    "define no single draw"))

# ── Query surface ───────────────────────────────────────────────────────

"""
    PredictiveQuery

A prepared posterior-predictive query (`prepare_predictive`). Call it as
`q(u, rng)` for one draw of every observation at the parameters `u`, or use
[`predictive_draws!`](@ref) and [`predictive_laws`](@ref). Not thread-safe:
prepare one per thread.
"""
struct PredictiveQuery{K,T}
    kernel::K
    targets::T
end

"""
    prepare_predictive(built, plan) -> PredictiveQuery

Prepare posterior-predictive draws of every observation of `plan` on the
graph `built` (`build_kernel` of a binding of the same program). `plan` may
leave its responses absent (a prediction design): the laws read only the
values the observations' distributions receive, and each draw takes the
shape the response would have.

The graph computes each observation's law arguments; drawing happens in
ordinary Julia, so it needs `using Distributions` (the package extension
that builds each law). Every response's family maps to its Distributions.jl
law, with `censored` and `truncated` evidence wrapped as RKPPL observes it
(the clamp law and the renormalized law). Mixtures, weighted,
interval-censored and packed responses, caller-owned sampling laws and
conditioned declarations are refused, naming them.

```julia
using Distributions, Random
pred = prepare_predictive(built, design)
draws = pred(u, Xoshiro(1))            # keyed like :pointwise, shaped like the responses
predictive_draws!(draws, pred, u, rng) # in place
laws = predictive_laws(pred, u)        # lazy: one Distributions law per entry
```
"""
function prepare_predictive(built, plan::StructuralPlan; on_error = nothing)
    kernel = _prepare_wants(built, plan, predictive_node(built); on_error)
    return PredictiveQuery(kernel, predictive_targets(built, plan))
end

"""
    predictive_node(built) -> Symbol

The graph value of `built` (`build_kernel`'s result) that holds every
observation's law arguments: the value `prepare_predictive` evaluates. WANT
it in any cut of `built.spec`, or of a graph composed from it, beside other
values and with any HAVE that determines it (random effects, interventions or
design data as runtime inputs), and `vectorize` that cut over draws like any
other. Then turn its value into laws or draws with [`predictive_laws`](@ref),
[`predictive_draws`](@ref) or [`predictive_draws!`](@ref), given
[`predictive_targets`](@ref).

```julia
node = predictive_node(built)
targets = predictive_targets(built, design)
cut = ReactiveKernels.prepare(ReactiveKernels.compose(built.spec, summary);
    have = (:unconstrained, :b, data_ports...), want = (node, :summary))
reader = ReactiveKernels.vectorize(cut; batched = (:unconstrained, :b))
value, summary = reader(U, B, data...)
predictive_draws(value, targets, rng; position = k)   # draw k's observations
```

The value holds numbers only (one array per law argument), so a vectorized
cut stacks it on its trailing draw axis; its layout is otherwise not part of
the interface.
"""
function predictive_node(built)
    hasproperty(built, :spec) && haskey(built.spec.ports, :_ppl_predictive) ||
        throw(ArgumentError("[query] predictive_node needs a `build_kernel` " *
            "result, whose graph holds the observations' laws"))
    return :_ppl_predictive
end

"""
    predictive_targets(built, plan) -> NamedTuple

The shape each observation's draws take on the binding `plan` of the program
`built` was built from, keyed like `:pointwise`: the bound response's shape,
or, for a response `plan` leaves absent, the shape of the operands it
broadcasts with. A response holding one array per `@plate` index takes each
index's shape (from the law values themselves when the response is absent and
no data the cell reads per index fix it). Pass it to
[`predictive_laws`](@ref) and [`predictive_draws`](@ref) with the value of
[`predictive_node`](@ref) on a cut over the same design.
"""
function predictive_targets(built, plan::StructuralPlan)
    # Binding gives a flat absent response that shape; a response holding one
    # array per index is classified by the program the graph was built from.
    program = hasproperty(built, :program) ? _absent_typed_as(plan, built.program) : plan
    names = Symbol[r.response for r in plan.responses]
    targets = map(plan.responses) do r
        column = plan.columns[r.response]
        if _cell_broadcast_response(program, r)
            dims = column isa _Absent ? nothing :
                Tuple{Vararg{Int}}[size(c) for c in column]
            T = column isa _Absent ? _valueless_response_eltype(r) : eltype(eltype(column))
            return (kind = :cells, dims = dims, eltype = T)
        end
        return (kind = :array, dims = size(column), eltype = eltype(column))
    end
    return NamedTuple{Tuple(names)}(Tuple(targets))
end

"""
    PredictiveLaws

The laws of one response's entries, shaped like the response
([`predictive_laws`](@ref)). Indexing builds that entry's Distributions.jl
law from the arguments the graph computed; nothing is stored per entry.
"""
struct PredictiveLaws{T,N,S,P} <: AbstractArray{T,N}
    source::S
    dims::NTuple{N,Int}
    positions::P
end

# `source` is one response's law columns (`_ppl_law_columns`); `indices`, for
# a ranged response, the response entry each column position belongs to.
function PredictiveLaws(source::NamedTuple, dims::NTuple{N,Int}, indices = nothing) where {N}
    positions = nothing
    if indices !== nothing
        positions = zeros(Int, dims)
        for (k, i) in enumerate(indices)
            positions[i] = k
        end
    elseif !isempty(source.law)
        shape = try
            Base.Broadcast.broadcast_shape(_law_shape(source), dims)
        catch
            nothing
        end
        shape == dims || throw(DimensionMismatch("[query] the observation " *
            "laws have shape $(_law_shape(source)), which does not broadcast to " *
            "the response's shape $dims (a vectorized cut's value needs " *
            "`position = k` to select draw k)"))
    end
    T = isempty(source.law) ? Any :
        Base.promote_op(_entry_law, typeof(source), NTuple{N,Int}, typeof(positions))
    (T === Union{} || !isconcretetype(T)) && (T = Any)
    return PredictiveLaws{T,N,typeof(source),typeof(positions)}(source, dims, positions)
end

# Every column has the law plate's shape.
_law_shape(source::NamedTuple) = size(first(only(values(source.law))))

Base.size(L::PredictiveLaws) = L.dims
Base.IndexStyle(::Type{<:PredictiveLaws}) = IndexCartesian()
function Base.getindex(L::PredictiveLaws{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(L, I...)
    return _entry_law(L.source, I, L.positions)
end

_entry_law(source, I, positions) = _predictive_law(_law_cell(source, I, positions))

# The law plate position entry `I` reads: a ranged response's own position,
# or Julia's broadcast of the plate's shape over the response's.
function _law_position(source, I::Tuple, positions)
    if positions !== nothing
        k = positions[I...]
        k == 0 && throw(ArgumentError("[query] entry $I of the response is " *
            "not one its observation loop covers"))
        return length(first(only(values(source.law)))) == 1 ? (1,) : (k,)
    end
    shape = _law_shape(source)
    return ntuple(d -> shape[d] == 1 ? 1 : I[d], length(shape))
end

# One entry's law cell, as `_law_statement` writes it: the law, its
# arguments, then `evidence` and the bounds.
function _law_cell(source::NamedTuple, I::Tuple, positions)
    J = _law_position(source, I, positions)
    law = source.law
    cell = (; law = Val(only(keys(law))), map(a -> a[J...], only(values(law)))...)
    hasproperty(source, :evidence) || return cell
    ev = source.evidence
    return (; cell..., evidence = Val(only(keys(ev))), map(b -> b[J...], only(values(ev)))...)
end

# Without the Distributions.jl extension there is nothing to build a law with.
_predictive_law(cell) = throw(ArgumentError("predictive draws build each " *
    "observation's law with Distributions.jl: run `using Distributions` " *
    "(it loads ReactiveKernelsPPL's Distributions extension)"))

"""
    predictive_laws(value, targets; position = nothing) -> NamedTuple
    predictive_laws(q::PredictiveQuery, u) -> NamedTuple

The law of every observation, keyed like `:pointwise`: a lazy
[`PredictiveLaws`](@ref) array shaped like each response (a number bound to a
response is one observation, as in `:pointwise`), and a vector of them for a
response holding one array per `@plate` index.

`value` is the value of [`predictive_node`](@ref) in any cut, and `targets`
is [`predictive_targets`](@ref) of the design that cut evaluates. A cut
vectorized over draws returns the value stacked on its trailing axis:
`position = k` takes draw `k`'s laws, without copying. `q(u)`'s laws are
`predictive_laws(q, u)`.
"""
function predictive_laws(value::NamedTuple, targets::NamedTuple; position = nothing)
    names = keys(targets)
    keys(value) == names || throw(ArgumentError("[query] the predictive value " *
        "holds the laws of $(keys(value)), the targets $(names): pass the targets " *
        "of the design the value's cut evaluates"))
    value = position === nothing ? value : _at_position(value, position)
    return NamedTuple{names}(map(k -> _response_laws(value[k], targets[k]), names))
end
predictive_laws(q::PredictiveQuery, u::AbstractVector) =
    predictive_laws(Base.invokelatest(q.kernel, u), q.targets)

# Draw `k` of a vectorized cut's value: every leaf at position `k` of its
# trailing axis.
_at_position(x::Union{Tuple,NamedTuple}, k) = map(v -> _at_position(v, k), x)
function _at_position(x::AbstractArray, k::Integer)
    checkbounds(Bool, axes(x, ndims(x)), k) || throw(BoundsError(x, (k,)))
    return selectdim(x, ndims(x), k)
end

function _response_laws(entry, target)
    if target.kind === :cells
        hasproperty(entry, :sizes) || return [PredictiveLaws(entry, d)
            for d in something(target.dims, Tuple{Vararg{Int}}[])]
        sizes = entry.sizes
        ndims(sizes) == 2 || throw(DimensionMismatch("[query] the observation " *
            "laws hold $(ndims(sizes) - 2) more axes than one design's: a " *
            "vectorized cut's value needs `position = k` to select draw k"))
        dims = target.dims === nothing ?
            [Tuple(view(sizes, :, i)) for i in axes(sizes, 2)] : target.dims
        length(dims) == size(sizes, 2) || throw(DimensionMismatch(
            "[query] the observation laws have $(size(sizes, 2)) indices, the " *
            "response $(length(dims))"))
        laws = PredictiveLaws[]
        offset = 0
        for i in axes(sizes, 2)
            shape = Tuple(view(sizes, :, i))
            n = prod(shape)
            push!(laws, PredictiveLaws(_index_source(entry, offset, shape), dims[i]))
            offset += n
        end
        return laws
    end
    indices = hasproperty(entry, :indices) ? entry.indices : nothing
    return PredictiveLaws(entry, target.dims, indices)
end

# Index `i`'s columns of a response holding one array per index: its plate
# cells, in place.
function _index_source(entry, offset, shape)
    columns(nt) = map(c -> reshape(view(c, offset+1:offset+prod(shape)), shape), nt)
    law = map(columns, entry.law)
    hasproperty(entry, :evidence) || return (; law)
    return (; law, evidence = map(columns, entry.evidence))
end

"""
    predictive_draws(value, targets, rng; position = nothing) -> NamedTuple
    (q::PredictiveQuery)(u, rng) -> NamedTuple

One posterior-predictive draw of every observation, keyed like `:pointwise`
and shaped like the responses, from the laws [`predictive_laws`](@ref) gives
for `value`, `targets` and `position`. `q(u, rng)` draws at the parameters
`u`. Entries are drawn in order (column-major, then index by index for a
response holding one array per index), so a seeded `rng` reproduces the
draws, and passing a copy of one RNG at every posterior draw reuses one
stream.
"""
predictive_draws(value::NamedTuple, targets::NamedTuple, rng; position = nothing) =
    _draws(rng, predictive_laws(value, targets; position), targets)
(q::PredictiveQuery)(u::AbstractVector, rng) =
    _draws(rng, predictive_laws(q, u), q.targets)

_draws(rng, laws::NamedTuple, targets::NamedTuple) =
    NamedTuple{keys(laws)}(map(k -> _draw(rng, laws[k], targets[k].eltype), keys(laws)))

# An empty response draws nothing, and keeps its element type.
_draw(rng, L::PredictiveLaws, T) = isempty(L) ? Array{T}(undef, size(L)) :
    [rand(rng, L[I]) for I in CartesianIndices(L)]
_draw(rng, L::AbstractVector{<:PredictiveLaws}, T) = [_draw(rng, Li, T) for Li in L]

"""
    predictive_draws!(out::NamedTuple, value, targets, rng; position = nothing) -> out
    predictive_draws!(out::NamedTuple, q::PredictiveQuery, u, rng) -> out

Draw every observation into `out` (a result of [`predictive_draws`](@ref) or
`q(u, rng)`, or arrays of the same shapes), reusing its storage.
"""
predictive_draws!(out::NamedTuple, value::NamedTuple, targets::NamedTuple, rng;
        position = nothing) =
    _draws!(rng, out, predictive_laws(value, targets; position))
predictive_draws!(out::NamedTuple, q::PredictiveQuery, u::AbstractVector, rng) =
    _draws!(rng, out, predictive_laws(q, u))

function _draws!(rng, out::NamedTuple, laws::NamedTuple)
    for k in keys(laws)
        _draw!(rng, getproperty(out, k), laws[k])
    end
    return out
end

function _draw!(rng, out::AbstractArray, L::PredictiveLaws)
    size(out) == size(L) || throw(DimensionMismatch("[query] the output has " *
        "shape $(size(out)), the response $(size(L))"))
    for I in CartesianIndices(L)
        out[I] = rand(rng, L[I])
    end
    return out
end
function _draw!(rng, out::AbstractVector, L::AbstractVector{<:PredictiveLaws})
    length(out) == length(L) || throw(DimensionMismatch("[query] the output " *
        "has $(length(out)) indices, the response $(length(L))"))
    foreach((o, Li) -> _draw!(rng, o, Li), out, L)
    return out
end
