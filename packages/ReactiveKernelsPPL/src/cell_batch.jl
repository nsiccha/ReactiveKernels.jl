"A callback-scoped packed point; construct with [`with_cell_batch`](@ref)."
mutable struct CellBatch{S}
    const shared::S
    const base::Vector{Float64}
    const generation::UInt
    active::Bool
end

"""
    with_cell_batch(visit, owner, u)

Evaluate cells at one copied packed point. `owner` is a `CellQuery`,
`CellFactors`, or one of its value operators prepared with
`partitioned`. The shared cache is refreshed once before `visit(batch)`.
Inside the callback, value queries and local structural footprints accept
`batch` in place of `u`, without rescanning shared coordinates. No AD backend
is needed. The callback's result is returned.

```julia
fp = partition_footprint_kernel(q; shared = false)
with_cell_batch(q, u) do batch
    [(q(batch, i), fp(batch, i).unconstrained) for i in cells]
end
```

A batch belongs to its owner's cache and expires when the callback exits,
including on error. Changing shared coordinates through another operator
invalidates it. Prepare a separate owner per concurrent caller. Input `u`
may be changed after entry; the batch keeps its own point.
"""
function with_cell_batch(visit::F,
        owner::Union{CellQuery,CellFactors,CellFactorQuery},
        u::AbstractVector{<:Real}) where {F}
    shared = owner.shared
    shared === nothing && throw(ContractValidationError(
        "[query] with_cell_batch requires an owner prepared with partitioned coordinates"))
    base = Vector{Float64}(u)
    Base.invokelatest(_refresh_cell_shared!, shared, base)
    batch = CellBatch(shared, base, shared.generation, true)
    try
        result = visit(batch)
        _check_cell_pass(batch, shared)
        result
    finally
        batch.active = false
    end
end

function _check_cell_pass(p::CellBatch, shared)
    p.active && p.shared === shared && p.generation == shared.generation ||
        throw(ContractValidationError(
            "[query] operator must use this active cell batch's unchanged shared cache"))
    nothing
end

# Value and structural-footprint queries also work inside an existing mixed
# derivative accumulation pass. Neither operation adds a cotangent.
const _CellValuePass = Union{CellBatch,CellPullback}
function (q::CellQuery)(p::_CellValuePass, cell)
    _check_cell_pass(p, q.shared)
    Base.invokelatest(_cell_batch_density, q, p.base, cell)::Float64
end
function _cell_batch_density(q::CellQuery, u, cell)
    q.prefix === nothing && return _cell_density_uncached(q, u, cell)
    values = _cell_shared_values(q.shared)
    residual = _query_residual(q.prefix, (u, cell, values...))
    residual(u, cell, values...)
end
function (q::CellFactorQuery)(p::_CellValuePass, a, b)
    _check_cell_pass(p, q.shared)
    _check_cell_factor_args(q, a)
    args = (p.base, a, b, _cell_shared_values(q.shared)...)
    Base.invokelatest(_cell_factor_residual(q, args), args...)::Float64
end

struct _CellFootprint{Q,K,P,S,A}
    query::Q
    kernel::K
    plan::P
    selection::S
    specializations::Dict{DataType,Any}
    argument_type::Type{A}
end

# The arguments a query's executed residual was typed from: a strict plan is
# prepared again from them, and refuses a value it cannot type without them.
_cell_query_args(q::CellFactorQuery) = q.args
_cell_query_args(q::CellQuery) =
    q.prefix === nothing ? (zeros(q.layout.total), 1) : q.prefix.args

"""
    partition_footprint_kernel(q::Union{CellQuery,CellFactorQuery}; shared=true)

Prepare a structural footprint callable with the same arguments as `q`.
With `shared=true`, return RK's full footprint kernel for `q.kernel`.
With `shared=false`, require a `partitioned` declaration and return only
`(; unconstrained = indices)`, the sorted unique local packed positions.
The local query cuts the shared prefix and projects reads before expanding
them; a large shared block is not enumerated once per cell. Conservative
reads and both arms of retained lazy branches remain conservative locally.

A local footprint accepts either a raw packed vector or an active compatible
[`with_cell_batch`](@ref) / [`cell_pullback!`](@ref) context. Raw calls check
the shared coordinates each time. Outer factors treat their supplied sums as
independent, exactly as their full footprints do.
"""
function ReactiveKernels.partition_footprint_kernel(q::Union{CellQuery,CellFactorQuery};
        shared::Bool = true)
    shared && return ReactiveKernels.partition_footprint_kernel(q.kernel)
    q.shared === nothing && throw(ContractValidationError(
        "[query] a local cell footprint requires partitioned coordinates"))
    residual = q isa CellQuery ? (q.prefix === nothing ? q.kernel : q.prefix.residual) : q.residual
    # Frontier HAVE values are independent at this boundary. Replanning the
    # original graph could otherwise reintroduce their shared producers.
    original = q isa CellQuery && q.prefix !== nothing ? q.prefix.residual_plan : residual.plan
    plan = _conditional_cell_plan(original)
    args = _cell_query_args(q)
    kernel = Base.invokelatest(prepare, plan; exemplars = args)
    selection = :unconstrained => ReactiveKernels._FootprintSelection(q.shared.local_positions)
    footprint = Base.invokelatest(ReactiveKernels._partition_footprint_kernel, kernel, selection)
    _CellFootprint(q, footprint, plan, selection, Dict{DataType,Any}(), typeof(args))
end
function (fp::_CellFootprint)(u::AbstractVector{Float64}, args...)
    Base.invokelatest(_cell_footprint, fp, u, args...)
end
(fp::_CellFootprint)(u::AbstractVector{<:Real}, args...) = fp(Vector{Float64}(u), args...)
function (fp::_CellFootprint)(p::_CellValuePass, args...)
    _check_cell_pass(p, fp.query.shared)
    Base.invokelatest(_cell_footprint_at, fp, (p.base, args..., _cell_shared_values(fp.query.shared)...))
end
function _cell_footprint(fp::_CellFootprint, u, args...)
    q = fp.query
    prefix = q isa CellQuery && q.prefix !== nothing ? _query_prefix_kernel(q.prefix, u) : nothing
    values = _refresh_cell_shared!(q.shared, u, prefix)
    _cell_footprint_at(fp, (u, args..., values...))
end
function _cell_footprint_at(fp::_CellFootprint{<:Any,<:Any,<:Any,<:Any,A}, point) where {A}
    kernel = if point isa A
        fp.kernel
    else
        get!(fp.specializations, typeof(point)) do
            prepared = Base.invokelatest(prepare, fp.plan; exemplars = point)
            Base.invokelatest(ReactiveKernels._partition_footprint_kernel, prepared, fp.selection)
        end
    end
    Base.invokelatest(kernel, point...)
end
