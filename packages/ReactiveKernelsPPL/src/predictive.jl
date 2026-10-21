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

# The predictive entry of a response: its law plate, and the response
# entries a ranged observation's cells belong to.
function _law_entry(r::LikelihoodSpec, plan::StructuralPlan, rs::Vector{Expr})
    name = _law_name(r.label)
    any(st -> _assigned_name(st) === name, rs) || return nothing
    indices = Symbol(:_ppl_range_indices_, r.label)
    if r.range isa Expr && !_cell_broadcast_response(plan, r) &&
            any(st -> _assigned_name(st) === indices, rs)
        return :((laws = $name, indices = $indices))
    end
    return :((laws = $name,))
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
    kernel = _prepare_wants(built, plan, :_ppl_predictive; on_error)
    return PredictiveQuery(kernel, _predictive_targets(built, plan))
end

# The shape each response's draws take: the bound response's, or, for a
# response left absent, the shape of the operands it broadcasts with. Binding
# gives a flat absent response that shape; a response holding one array per
# index is classified by the program the graph was built from, and each
# index's draws take the shape of that index's laws (`dims = nothing`).
function _predictive_targets(built, plan::StructuralPlan)
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
struct PredictiveLaws{T,N,C,P} <: AbstractArray{T,N}
    cells::C
    dims::NTuple{N,Int}
    positions::P
end

function PredictiveLaws(cells, dims::NTuple{N,Int}, indices = nothing) where {N}
    positions = nothing
    if indices !== nothing
        positions = zeros(Int, dims)
        for (k, i) in enumerate(indices)
            positions[i] = k
        end
    elseif !(cells isa Tuple)
        shape = try
            Base.Broadcast.broadcast_shape(size(cells), dims)
        catch
            nothing
        end
        shape == dims || throw(DimensionMismatch("[query] the observation " *
            "laws have shape $(size(cells)), which does not broadcast to the " *
            "response's shape $dims"))
    end
    T = cells isa Tuple || isempty(cells) ? Any :
        Base.promote_op(_predictive_law, eltype(cells))
    (T === Union{} || !isconcretetype(T)) && (T = Any)
    return PredictiveLaws{T,N,typeof(cells),typeof(positions)}(cells, dims, positions)
end

Base.size(L::PredictiveLaws) = L.dims
Base.IndexStyle(::Type{<:PredictiveLaws}) = IndexCartesian()
function Base.getindex(L::PredictiveLaws{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(L, I...)
    return _predictive_law(_law_cell(L, I))
end

function _law_cell(L::PredictiveLaws, I::Tuple)
    cells = L.cells
    if L.positions !== nothing
        k = L.positions[I...]
        k == 0 && throw(ArgumentError("[query] entry $I of the response is " *
            "not one its observation loop covers"))
        return length(cells) == 1 ? first(cells) : cells[k]
    end
    nd = ndims(cells)
    nd == 0 && return cells[]
    return cells[ntuple(d -> size(cells, d) == 1 ? 1 : I[d], nd)...]
end

# Without the Distributions.jl extension there is nothing to build a law with.
_predictive_law(cell) = throw(ArgumentError("predictive draws build each " *
    "observation's law with Distributions.jl: run `using Distributions` " *
    "(it loads ReactiveKernelsPPL's Distributions extension)"))

"""
    predictive_laws(q::PredictiveQuery, u) -> NamedTuple

The law of every observation at the parameters `u`, keyed like `:pointwise`:
a lazy [`PredictiveLaws`](@ref) array shaped like each response (a number
bound to a response is one observation, as in `:pointwise`), and a vector of
them for a response holding one array per `@plate` index.
"""
function predictive_laws(q::PredictiveQuery, u::AbstractVector)
    raw = Base.invokelatest(q.kernel, u)
    names = keys(raw)
    return NamedTuple{names}(map(k -> _response_laws(raw[k], q.targets[k]), names))
end

function _response_laws(entry, target)
    indices = hasproperty(entry, :indices) ? entry.indices : nothing
    if target.kind === :cells
        cells = entry.laws
        cells isa Tuple && return PredictiveLaws[]
        dims = target.dims === nothing ? [size(c) for c in cells] : target.dims
        length(cells) == length(dims) || throw(DimensionMismatch(
            "[query] the observation laws have $(length(cells)) indices, the " *
            "response $(length(dims))"))
        return [PredictiveLaws(cells[i], dims[i]) for i in eachindex(dims)]
    end
    return PredictiveLaws(entry.laws, target.dims, indices)
end

"""
    (q::PredictiveQuery)(u, rng) -> NamedTuple

One posterior-predictive draw of every observation at the parameters `u`,
keyed like `:pointwise` and shaped like the responses. Entries are drawn in
order (column-major, then index by index for a response holding one array
per index), so a seeded `rng` reproduces the draws, and passing a copy of
one RNG at every posterior draw reuses one stream.
"""
function (q::PredictiveQuery)(u::AbstractVector, rng)
    laws = predictive_laws(q, u)
    names = keys(laws)
    return NamedTuple{names}(map(k -> _draw(rng, laws[k], q.targets[k].eltype), names))
end

# An empty response draws nothing, and keeps its element type.
_draw(rng, L::PredictiveLaws, T) = isempty(L) ? Array{T}(undef, size(L)) :
    [rand(rng, L[I]) for I in CartesianIndices(L)]
_draw(rng, L::AbstractVector{<:PredictiveLaws}, T) = [_draw(rng, Li, T) for Li in L]

"""
    predictive_draws!(out::NamedTuple, q::PredictiveQuery, u, rng) -> out

Draw every observation into `out` (a result of `q(u, rng)`, or arrays of the
same shapes), reusing its storage.
"""
function predictive_draws!(out::NamedTuple, q::PredictiveQuery, u::AbstractVector,
        rng)
    laws = predictive_laws(q, u)
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
