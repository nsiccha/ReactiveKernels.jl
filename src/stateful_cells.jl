# Element-level invalidation of a compiled reactive state (decision `0opsm1w`).
#
# `touch!(state, x, index)` and `mutate!(f, state, x, index)` say that only
# the elements `index` of the HAVE array `x` changed. An authored plate that
# reads `x` only at its own cell (an iterated argument, never `Ref`-shared)
# then reruns just those cells, at once, through a native kernel of one cell
# (the lowering `plate_cell` uses, prepared on the first such change), and a
# `sum` over the plate is updated by
# the cells' difference. Every other dependent of `x` is invalidated as usual,
# and anything this cannot prove falls back to that, so a value read from the
# state is the value a whole recomputation gives: bit for bit for the plate,
# to within rounding for the sum.
#
# The sum's running total is compensated (Neumaier), and recomputed in full
# once as many cells have changed as the plate has, so its error stays near
# one rounding at an amortized O(1) per changed cell.

# One plate whose cells read the HAVE slots `LOCAL` only at their own
# position. `SLOT` is the plate's slot, `INPUTS` the slots of its arguments in
# order, `SUMS` the slots of `sum` over it, and `FREE` whether those sums are
# all it feeds (so its buffer reaches no other value and may be patched in
# place). `kernel[]` is the native kernel of one cell,
# `kernel(position, inputs...)` at a linear position of the plate's domain,
# prepared on the first change that names elements (`_cell_kernel`), so a
# state whose changes never do pays nothing for it. `simplify` and
# `bound_statistics` are the settings the state was prepared with.
struct _CellPlate{SLOT,INPUTS,LOCAL,SUMS,FREE}
    plan::Plan
    recipe::Recipe
    simplify::Bool
    bound_statistics::Bool
    kernel::Base.RefValue{Any}
end

const _CELL_KERNEL_LOCK = ReentrantLock()

# The kernel, or `false` when it could not be prepared: that plate then
# recomputes whole on every change, after one warning naming the error.
@inline function _cell_kernel(plate::_CellPlate)
    kernel = plate.kernel[]
    kernel === nothing ? _prepare_cell_kernel(plate) : kernel
end

@noinline function _prepare_cell_kernel(plate::_CellPlate)
    lock(_CELL_KERNEL_LOCK) do
        prepared = plate.kernel[]
        prepared === nothing || return prepared
        kernel = try
            _with_simplify(plate.simplify) do
                _with_bound_statistics(plate.bound_statistics) do
                    _reactive_cell_kernel(plate.plan, plate.recipe)
                end
            end
        catch err
            err isa InterruptException && rethrow()
            @warn("compiled reactive state: the one-cell kernel of the plate producing " *
                  "$(join((v.name for v in plate.recipe.outputs), ", ")) could not be " *
                  "prepared, so a change to some of its elements recomputes it whole",
                  exception = (err, catch_backtrace()))
            false
        end
        plate.kernel[] = kernel
        kernel
    end
end

# The native kernel of one cell of plate `r`: a one-recipe plan over a copy of
# the graph, as `plate_cell` builds it (the program's own graph keeps its
# version), prepared as `prepare` prepares a cell, so each cell is bitwise the
# plate's. A cell outside the plate's domain throws `BoundsError`.
function _reactive_cell_kernel(plan::Plan, r::Recipe)
    graph = plan.graph
    copied = Graph(copy(graph.values), copy(graph.recipes),
        Dict{ValueId,Vector{Int}}(id => copy(ids) for (id, ids) in graph.producers),
        copy(graph.aliases), graph.version, nothing, copy(graph.checks))
    position = value!(copied, Symbol("#reactive_cell_position"), Int)
    cell = value!(copied, Symbol("#reactive_cell"), valtype(only(r.op.kernel.plan.want)))
    recipe = _add_recipe!(copied, (position, r.inputs...), (cell,),
                          _AuthoredPlateCellOp(r.op, :ignore), r.cost, nothing, false,
                          r.source)
    one = Plan(copied, Value[position, r.inputs...], Value[cell], Recipe[recipe],
               Dict{ValueId,Recipe}(canon_id(copied, cell.id) => recipe), r.cost,
               Recipe[recipe], InputCheck[])
    _prepare_plan(one)
end

_is_reactive_sum(op) = op === sum

# The cell plates of a selected plan: authored plates with one owned output,
# distinct arguments and at least one HAVE argument read only per cell (an
# argument declared as a non-array type cannot be one).
function _reactive_cells(plan::Plan, index, sources::BitVector, dependents)
    graph = plan.graph
    cells = Any[]
    for r in plan.recipes
        r.op isa _AuthoredPlateOp || continue
        length(r.outputs) == 1 || continue
        output = canon_id(graph, only(r.outputs).id)
        _owned_output(plan, r, output) || continue
        inputs = Int[index[canon_id(graph, v.id)] for v in r.inputs]
        allunique(inputs) || continue
        atomic = _plate_atomic(r.op)
        local_slots = Int[]
        for (position, (slot, value)) in enumerate(zip(inputs, r.inputs))
            sources[slot] || continue
            position in atomic && continue
            T = valtype(value)
            (T === Any || T <: AbstractArray) || continue
            push!(local_slots, slot)
        end
        isempty(local_slots) && continue
        slot = index[output]
        sums = Int[]
        for s in plan.recipes
            _is_reactive_sum(s.op) || continue
            length(s.inputs) == 1 && length(s.outputs) == 1 || continue
            index[canon_id(graph, only(s.inputs).id)] == slot || continue
            sum_id = canon_id(graph, only(s.outputs).id)
            _owned_output(plan, s, sum_id) || continue
            push!(sums, index[sum_id])
        end
        free = all(in(sums), dependents[slot])
        push!(cells, _CellPlate{slot,Tuple(inputs),Tuple(local_slots),Tuple(sums),free}(
            plan, r, _simplify(), _bound_statistics(), Ref{Any}(nothing)))
    end
    Tuple(cells)
end

# --- positions ---------------------------------------------------------------

# The elements `index` of the array `source` as positions 1, 2, … of its
# linear order (the plate's cell positions when its axes are the source's),
# in the instance's scratch vector. Bounds are checked here, before anything
# changes.
function _cell_positions!(state::CompiledReactiveState, source, index)
    source isa AbstractArray || throw(ArgumentError(
        "touch! and mutate! at an index need an array HAVE value; got a $(typeof(source))"))
    positions = state.cellstate.positions
    empty!(positions)
    if index isa Union{Integer,CartesianIndex}
        _push_cell_position!(positions, source, index)
    else
        for i in index
            _push_cell_position!(positions, source, i)
        end
    end
    positions
end

function _push_cell_position!(positions, source, i::Integer)
    checkbounds(source, i)
    push!(positions, i - first(LinearIndices(source)) + 1)
end
function _push_cell_position!(positions, source, i::CartesianIndex)
    checkbounds(source, i)
    linear = LinearIndices(source)
    push!(positions, linear[i] - first(linear) + 1)
end
_push_cell_position!(positions, source, i) = throw(ArgumentError(
    "an index of touch! or mutate! is an integer or a CartesianIndex; got $(repr(i))"))

# --- touch! at positions ------------------------------------------------------

function _touch_cells!(state::CompiledReactiveState, ::ReactiveValue{I}, positions) where {I}
    state.valid[I] = true
    cells = state.program.cells
    if isempty(cells)
        _invalidate_dependents!(state, I)
        return state
    end
    keep = state.cellstate.keep
    _mark_cells!(state, cells, Val(I))
    _invalidate_dependents!(state, I, keep)
    # From here `keep` holds only the sums a patched plate updates: a plate
    # patched first may feed a later one, whose edge from it must not be
    # skipped.
    _unkeep_cells!(state, cells, Val(I))
    _patch_cells!(state, cells, Val(I), positions)
    state
end

@inline _unkeep_cells!(state, ::Tuple{}, ::Val) = nothing
@inline function _unkeep_cells!(state, cells::Tuple, source::Val)
    _unkeep_cell!(state, first(cells), source)
    _unkeep_cells!(state, Base.tail(cells), source)
end
@inline function _unkeep_cell!(state, ::_CellPlate{C,INPUTS,LOCAL}, ::Val{I}) where {C,INPUTS,LOCAL,I}
    I in LOCAL && (state.cellstate.keep[C] = false)
    nothing
end

@inline _mark_cells!(state, ::Tuple{}, ::Val) = nothing
@inline function _mark_cells!(state, cells::Tuple, source::Val)
    _mark_cell!(state, first(cells), source)
    _mark_cells!(state, Base.tail(cells), source)
end

# A cell plate of `I` is patched when it is valid, not frozen, and its buffer
# has the source's axes (an argument extruded along an axis is read by every
# cell along it).
@inline function _mark_cell!(state, ::_CellPlate{C,INPUTS,LOCAL}, ::Val{I}) where {C,INPUTS,LOCAL,I}
    I in LOCAL || return nothing
    state.valid[C] && !state.frozen[C] || return nothing
    # A value installed by `assign!` recomputes from its recipe on a change.
    _take_override!(state, C) && return nothing
    _cell_axes_match(_tuple_slot(state.slots, Val(C))[], _tuple_slot(state.slots, Val(I))[]) ||
        return nothing
    state.cellstate.keep[C] = true
    state.cellstate.marked[C] = true
    nothing
end

_cell_axes_match(buffer::AbstractArray, source::AbstractArray) = axes(buffer) == axes(source)
_cell_axes_match(buffer, source) = false

@inline _patch_cells!(state, ::Tuple{}, ::Val, positions) = nothing
@inline function _patch_cells!(state, cells::Tuple, source::Val, positions)
    _patch_cell!(state, first(cells), source, positions)
    _patch_cells!(state, Base.tail(cells), source, positions)
end

function _patch_cell!(state, plate::_CellPlate{C,INPUTS,LOCAL,SUMS,FREE},
                      ::Val{I}, positions) where {C,INPUTS,LOCAL,SUMS,FREE,I}
    I in LOCAL || return nothing
    cellstate = state.cellstate
    keep = cellstate.keep
    cellstate.marked[C] || return nothing
    cellstate.marked[C] = false
    # Invalidated on another path from the source: it recomputes whole.
    state.valid[C] || return nothing
    kernel = _cell_kernel(plate)
    if kernel === false
        state.valid[C] = false
        _invalidate_dependents!(state, C)
        return nothing
    end
    slot = _tuple_slot(state.slots, Val(C))
    buffer = slot[]
    if cellstate.owned[C] !== buffer
        buffer = copy(buffer)
        slot[] = buffer
        cellstate.owned[C] = FREE ? buffer : nothing
    end
    record = buffer isa AbstractArray{Float64} && _cell_sums_live(state, Val(SUMS))
    # A cell that throws leaves the plate to recompute whole on the next read,
    # where the error is raised, exactly as without an index.
    try
        _run_cells!(state, kernel, Val(C), Val(INPUTS), positions, record)
    catch err
        state.valid[C] = false
        _invalidate_dependents!(state, C)
        err isa InterruptException && rethrow()
        return nothing
    end
    _update_cell_sums!(state, Val(SUMS), Val(C), positions, record)
    _invalidate_dependents!(state, C, keep)
    _finish_cell_sums!(state, SUMS)
    nothing
end

@generated function _cell_inputs(slots::Tuple, ::Val{INPUTS}) where {INPUTS}
    Expr(:tuple, (:(getfield(slots, $i)[]) for i in INPUTS)...)
end

_cell_sums_live(state, ::Val{SUMS}) where {SUMS} =
    any(d -> state.valid[d] && !state.frozen[d], SUMS)

# Reruns the cells at `positions` in place; with `record`, the old and new
# values in order. A function barrier on the kernel's type; its arguments are
# all references or singletons, so the call allocates nothing.
function _run_cells!(state, kernel, ::Val{C}, ::Val{INPUTS}, positions,
                     record::Bool) where {C,INPUTS}
    buffer = _tuple_slot(state.slots, Val(C))[]
    inputs = _cell_inputs(state.slots, Val(INPUTS))
    olds, news = state.cellstate.olds, state.cellstate.news
    _run_cells!(buffer, kernel, inputs, positions, olds, news, record)
end

function _run_cells!(buffer, kernel, inputs::Tuple, positions, olds, news, record::Bool)
    places = CartesianIndices(buffer)
    if record
        resize!(olds, length(positions))
        resize!(news, length(positions))
    end
    for (n, position) in enumerate(positions)
        value = kernel(position, inputs...)
        place = places[position]
        old = buffer[place]
        buffer[place] = value
        if record
            olds[n] = old
            news[n] = buffer[place]
        end
    end
    nothing
end

@inline function _neumaier(total, compensation, x)
    next = total + x
    compensation += abs(total) >= abs(x) ? (total - next) + x : (x - next) + total
    next, compensation
end

@generated function _update_cell_sums!(state, ::Val{SUMS}, plate::Val, positions,
                                     record::Bool) where {SUMS}
    Expr(:block, (:(_update_cell_sum!(state, Val($d), plate, positions, record)) for d in SUMS)...,
         nothing)
end

# A valid, unfrozen Float64 sum takes the cells' difference and is kept;
# otherwise it is left to the invalidation from the plate and recomputes whole
# on its next read: a sum of another type, a non-finite total, or the periodic
# full recompute once its incremental updates reach the number of cells.
function _update_cell_sum!(state, ::Val{D}, ::Val{C}, positions, record::Bool) where {D,C}
    (record && state.valid[D] && !state.frozen[D]) || return nothing
    _take_override!(state, D) && return nothing
    cellstate = state.cellstate
    slot = _tuple_slot(state.slots, Val(D))
    value = slot[]
    value isa Float64 || return nothing
    if value !== cellstate.last[D]
        # Computed in full since this instance last updated it.
        cellstate.hi[D] = value
        cellstate.comp[D] = 0.0
        cellstate.updates[D] = 0
    end
    count = length(positions)
    if cellstate.updates[D] + count > length(_tuple_slot(state.slots, Val(C))[])
        cellstate.updates[D] = 0
        return nothing
    end
    total, compensation = cellstate.hi[D], cellstate.comp[D]
    olds, news = cellstate.olds, cellstate.news
    for n in 1:count
        total, compensation = _neumaier(total, compensation, -olds[n])
        total, compensation = _neumaier(total, compensation, news[n])
    end
    result = total + compensation
    isfinite(total) && isfinite(result) || (cellstate.updates[D] = 0; return nothing)
    cellstate.hi[D] = total
    cellstate.comp[D] = compensation
    cellstate.last[D] = result
    cellstate.updates[D] += count
    slot[] = result
    cellstate.keep[D] = true
    nothing
end

function _finish_cell_sums!(state, sums::Tuple)
    keep = state.cellstate.keep
    for d in sums
        keep[d] || continue
        keep[d] = false
        _invalidate_dependents!(state, d)
    end
end

# --- escapes, overrides and copies ------------------------------------------

# `assign!` and `freeze!` with a value install a value that is not the
# recipe's: a later change must recompute the slot, not patch that value.
@inline _note_override!(state::CompiledReactiveState, slot::Int) =
    (isempty(state.cellstate.override) || (state.cellstate.override[slot] = true); nothing)

@inline function _take_override!(state, slot::Int)
    override = state.cellstate.override
    override[slot] || return false
    override[slot] = false
    true
end

@inline _escape_cells!(state, ::Tuple{}, ::Val) = nothing
@inline function _escape_cells!(state, cells::Tuple, slot::Val)
    _escape_cell!(state, first(cells), slot)
    _escape_cells!(state, Base.tail(cells), slot)
end
@inline function _escape_cell!(state, ::_CellPlate{C}, ::Val{I}) where {C,I}
    C == I && (state.cellstate.owned[C] = nothing)
    nothing
end

function _copy_cellstate(cells::_CellState)
    _CellState(falses(length(cells.keep)), falses(length(cells.marked)),
               copy(cells.override), Vector{Any}(nothing, length(cells.owned)),
               copy(cells.hi), copy(cells.comp), copy(cells.last), copy(cells.updates),
               Int[], Float64[], Float64[])
end

function _copyto_cellstate!(destination::_CellState, source::_CellState)
    copyto!(destination.override, source.override)
    fill!(destination.owned, nothing)
    copyto!(destination.hi, source.hi)
    copyto!(destination.comp, source.comp)
    copyto!(destination.last, source.last)
    copyto!(destination.updates, source.updates)
    destination
end
