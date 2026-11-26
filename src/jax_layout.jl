# Layouts of a plate over prepared ragged rows.
#
# A plate is one `jax.vmap` of its cell over every unit, so each unit's rows
# are stored and computed at the longest unit's length. With units of very
# different lengths most of that work, and of what reverse mode stores per
# step, is padding. `plate_layout = :groups` lets the caller of the module choose,
# when it prepares the data, how such a plate runs: its units sorted by length
# and split into a few groups, each group one `jax.vmap` over rows cut to the
# group's longest (or one `jax.lax.map`, its cells one after another), the
# groups' values joined in the plate's order. The cell is the plate's own, so
# the values are the plate's and every layout has ordinary derivatives.
#
# The module text does not depend on the choice or on the data: a plate that
# qualifies calls `_rk_grouped_plate` with the layout the prepared data carry
# (`prepare_plate_layout`), and without one it is the plate's `jax.vmap`. The number
# of groups is the caller's argument, not a size of the data, so a trace
# holds that many copies of the plate whatever the lengths are (CONTRIBUTING,
# "Data-dependent iteration").

const _JAX_LAYOUT_KEY = :reactivekernels_jax_plate_layout
const _JAX_LAYOUTS = (:padded, :groups)
_jax_plate_layout() = get(task_local_storage(), _JAX_LAYOUT_KEY, :padded)::Symbol
# The layout of a `jax_source` call that names none: `:padded`, unless the
# task asks for layouts wherever there are data (the test suite's differential
# run of every module through grouped plates, `test/jax_python.jl`).
const _JAX_LAYOUT_DEFAULT_KEY = :reactivekernels_jax_plate_layout_default
function _jax_with_plate_layout(f, layout, data)
    layout === nothing && (layout = data === nothing ? :padded :
        get(task_local_storage(), _JAX_LAYOUT_DEFAULT_KEY, :padded)::Symbol)
    layout in _JAX_LAYOUTS || throw(ArgumentError(
        "jax_source: `plate_layout` is `:padded` or `:groups`; got `$(repr(layout))`"))
    # Groups follow the prepared lengths of a plate's rows.
    layout === :padded || data !== nothing || throw(ArgumentError(
        "jax_source: `plate_layout = :groups` groups plates by their prepared row lengths, so it needs `data`"))
    task_local_storage(f, _JAX_LAYOUT_KEY, layout)
end

# `prepare_plate_layout`'s partition, on the host. The units are sorted by their
# total length, and a group is a run of that order: its width per argument is
# its longest row there, so the padded size of a split is the sum over groups
# of units times widths, which the split minimizes exactly over the cuts
# between different lengths (thinned to 256 when there are more).
const _JAX_LENGTH_GROUPS = raw"""
class PlateLayout:
    \"\"\"One plate's layout among the prepared data (`prepare_plate_layout`).

    `units` holds each group's units (0-based), `order` restores the plate's
    cells from the groups' joined values, `widths` holds each group's width
    per row argument, and `sequential` says whether a group's units run one
    after another. `jax.jit` treats a layout as static, so the functions jit
    with the prepared data as arguments; another layout traces them again.
    \"\"\"

    def __init__(self, units, order, widths, sequential):
        self.units = tuple(np.asarray(u, dtype=np.int64) for u in units)
        self.order = np.asarray(order, dtype=np.int64)
        self.widths = tuple(tuple(w) for w in widths)
        self.sequential = bool(sequential)
        self._key = (tuple(u.tobytes() for u in self.units), self.widths, self.sequential)

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        return isinstance(other, PlateLayout) and self._key == other._key

    def __repr__(self):
        return "PlateLayout(%s units in groups of %s, widths %s%s)" % (
            self.order.shape[0], [u.shape[0] for u in self.units], list(self.widths),
            ", sequential" if self.sequential else "")


jax.tree_util.register_pytree_node(PlateLayout, lambda layout: ((), layout), lambda layout, _: layout)


def _rk_length_groups(lengths, groups, sequential):
    # One plate's layout, or `None` for the plate as one batch.
    # `None`: a row whose lengths are not prepared data; it keeps its width.
    slots = [n is not None for n in lengths]
    lengths = [np.asarray(n, dtype=np.int64) for n in lengths if n is not None]
    count = lengths[0].shape[0] if lengths else 0
    if count == 0 or (int(groups) <= 1 and not sequential):
        return None
    total = np.sum(lengths, axis=0)
    by_length = np.argsort(-total, kind="stable")
    rows = [n[by_length] for n in lengths]
    cuts = np.concatenate(([0], np.flatnonzero(np.diff(total[by_length]) != 0) + 1, [count]))
    if cuts.shape[0] > 257:
        cuts = np.unique(cuts[np.round(np.linspace(0, cuts.shape[0] - 1, 257)).astype(np.int64)])
    runs = cuts.shape[0] - 1
    cost = np.full((runs + 1, runs + 1), np.inf)
    for a in range(runs):
        width = np.zeros(len(rows), dtype=np.int64)
        for b in range(a + 1, runs + 1):
            for j, n in enumerate(rows):
                width[j] = max(width[j], int(n[cuts[b - 1]:cuts[b]].max()))
            cost[a, b] = (cuts[b] - cuts[a]) * max(int(width.sum()), 1)
    groups = max(1, min(int(groups), runs))
    best = np.full((groups + 1, runs + 1), np.inf)
    back = np.zeros((groups + 1, runs + 1), dtype=np.int64)
    best[0, 0] = 0.0
    for g in range(1, groups + 1):
        for b in range(1, runs + 1):
            options = best[g - 1, :b] + cost[:b, b]
            back[g, b] = int(np.argmin(options))
            best[g, b] = options[back[g, b]]
    g = int(np.argmin(best[1:, runs])) + 1
    bounds = [runs]
    while g > 0:
        bounds.append(int(back[g, bounds[-1]]))
        g -= 1
    bounds.reverse()
    units = tuple(by_length[cuts[a]:cuts[b]] for a, b in zip(bounds[:-1], bounds[1:]))
    # A group of empty rows keeps one stored entry, as an empty column does.
    widths = []
    for u in units:
        known = iter(max(1, int(n[u].max())) for n in lengths)
        widths.append(tuple(next(known) if slot else None for slot in slots))
    order = np.argsort(np.concatenate(units), kind="stable")
    return PlateLayout(units, order, widths, sequential)
"""

const _JAX_GROUPED_PLATE = raw"""
def _rk_grouped_plate(cell, in_axes, args, padded, rows, layout):
    # `jax.vmap(cell, in_axes)(*args)` in the layout the prepared data choose
    # (`prepare_plate_layout`). `padded` holds the positions in `args` of the
    # arguments whose rows are cut to a group's width, and `rows` the position
    # of the widths the cell cuts its own row reads to (`None`: it reads none).
    whole = jax.vmap(cell, in_axes=in_axes)
    if layout is None:
        return whole(*args)
    # The plate's own shapes: every layout returns them, whatever its widths.
    shapes = jax.eval_shape(lambda: whole(*args))
    parts = []
    for members, width in zip(layout.units, layout.widths):
        group = [a if axis is None else jax.tree_util.tree_map(lambda leaf: leaf[members], a)
                 for a, axis in zip(args, in_axes)]
        for position, w in zip(padded, width):
            group[position] = group[position][:, :w]
        if rows is not None:
            group[rows] = tuple(width[len(padded):])
        if layout.sequential:
            def one(entries, group=group):
                entries = iter(entries)
                return cell(*(a if axis is None else next(entries)
                              for a, axis in zip(group, in_axes)))
            parts.append(jax.lax.map(one, tuple(a for a, axis in zip(group, in_axes)
                                                if axis is not None)))
        else:
            parts.append(whole(*group))

    def join(shape, *leaves):
        # Past a group's width a row repeats its last stored entry, as padded
        # data do; every such entry lies past the row's length.
        full = []
        for leaf in leaves:
            pad = [(0, 0)] + [(0, n - m) for n, m in zip(shape.shape[1:], leaf.shape[1:])]
            if any(after for _, after in pad):
                leaf = jnp.pad(leaf, pad, mode="edge" if 0 not in leaf.shape[1:] else "constant")
            full.append(leaf)
        return jnp.concatenate(full)[layout.order]

    return jax.tree_util.tree_map(join, shapes, *parts)
"""

const _JAX_PREPARE_PLATE_LAYOUT = raw"""
def prepare_plate_layout(prepared: dict, groups: int = 1, sequential: bool = False) -> dict:
    \"\"\"The prepared data with a layout for the plates over ragged rows.

    `groups = 1` is each plate as one batch over all of its units, every row
    stored and computed at the longest unit's length. With more groups the
    units are sorted by length and split into at most that many groups, each
    run at its own longest length, so the work and what reverse mode stores
    follow the units' real lengths. `sequential = True` runs a group's units
    one after another instead of side by side. Every layout returns the same
    values and derivatives; which is fastest depends on the device, the model
    and the number of chains, so time the candidates there. `GROUPED_PLATES`
    names the plates and the row lengths that group them: the lengths of a
    mapped argument, or of the rows a cell reads at its unit's index.

    Call the other functions with the result: `f(u, **prepare_plate_layout(prepared, 4))`.
    \"\"\"
    def lengths(entry):
        # A mapped argument's row lengths, or those of the rows a cell reads
        # at its unit's index (1-based), or none that are prepared data.
        if entry is None or isinstance(entry, str):
            return None if entry is None else prepared[entry]
        rows, index = entry
        return np.asarray(prepared[rows])[np.asarray(prepared[index]) - 1]

    layout = {plate: _rk_length_groups([lengths(entry) for entry in entries], groups, sequential)
              for plate, entries in GROUPED_PLATES.items()}
    return {**prepared, "_rk_plate_layout": layout}
"""

# The reference tuner: which layout is fastest is a property of the device,
# the model and the chain count, so it is measured where the module runs. The
# caller supplies the work to time (its own batched step), so a sampler's
# memory and cost are part of the measurement.
const _JAX_SELECT_PLATE_LAYOUT = raw"""
def select_plate_layout(prepared: dict, make_step: Any, chains: Any = None,
                        candidates: Any = None, repeats: int = 3) -> tuple:
    \"\"\"Time candidate plate layouts on this device and return the fastest.

    `make_step(prepared)` returns a function of no arguments that runs the work
    to time once, for example one batched gradient or one sampler step over
    all chains. With `chains` (an increasing sequence of chain counts) it is
    called as `make_step(prepared, count)` and candidates are compared by time
    per chain, so the result also names the chain count to run. `candidates`
    are `(groups, sequential)` pairs, by default 1 to 16 groups, side by side
    and one after another. Each candidate is run once to compile and then
    `repeats` times; its time is the shortest.

    A candidate the device has no memory for is recorded with its error and,
    with `chains`, its larger counts are skipped. Any other error is raised.

    Returns `(prepared, report)`: the prepared data with the fastest layout,
    and one `dict` per run with `groups`, `sequential`, `chains`, `seconds`
    and `error`, the fastest first.
    \"\"\"
    import time
    if candidates is None:
        candidates = [(groups, sequential) for groups in (1, 2, 4, 8, 16)
                      for sequential in (False, True)]
    report = []
    seen = []
    for groups, sequential in candidates:
        chosen = prepare_plate_layout(prepared, groups, sequential)
        # Fewer distinct lengths than groups give the layout of a smaller count.
        key = list(chosen["_rk_plate_layout"].values())
        if key in seen:
            continue
        seen.append(key)
        for count in (None,) if chains is None else chains:
            row = {"groups": groups, "sequential": sequential, "chains": count,
                   "seconds": float("inf"), "error": None}
            report.append(row)
            try:
                step = make_step(chosen) if count is None else make_step(chosen, count)
                jax.block_until_ready(step())
                for _ in range(repeats):
                    start = time.perf_counter()
                    jax.block_until_ready(step())
                    row["seconds"] = min(row["seconds"], time.perf_counter() - start)
            except Exception as error:
                if "RESOURCE_EXHAUSTED" not in str(error) and not isinstance(error, MemoryError):
                    raise
                row["error"] = str(error)
                break
    cost = lambda row: row["seconds"] / (row["chains"] or 1)
    report.sort(key=cost)
    if not report or report[0]["error"] is not None:
        raise RuntimeError("select_plate_layout: no candidate ran on this device")
    return prepare_plate_layout(prepared, report[0]["groups"], report[0]["sequential"]), report
"""

# The plate arguments whose rows a group cuts to its width: each mapped ragged
# vector of numeric rows whose lengths are prepared data, as its position
# among the plate call's arguments and the prepared key of its lengths.
function _jax_layout_arguments(ctx::_JaxContext, inputs, mapped, ragged)
    found = Tuple{Int,String}[]
    position = 0
    for (i, v) in enumerate(inputs)
        if ragged[i]
            key = get(ctx.scope.prepared_keys, v.extent, nothing)
            key !== nothing && mapped[i] && _jax_ragged(v.T) && !_jax_tuple_valued(v.T) &&
                eltype(eltype(v.T)) <: Number && push!(found, (position, key))
            position += 2
        else
            position += 1
        end
    end
    found
end

# The cell's row reads (`_JaxScope.row_widths`) as the prepared data name them:
# for a read of a shared prepared ragged array at the plate's own mapped
# prepared index, the keys of the rows' lengths and of that index; `nothing`
# for any other read, which keeps its stored width.
function _jax_layout_rows(ctx::_JaxContext, cell_scope::_JaxScope, params, args, axes)
    map(cell_scope.row_widths) do (lengths, selector)
        i = findfirst(==(lengths), params)
        j = findfirst(==(selector), params)
        (i === nothing || j === nothing || max(i, j) > length(args)) && return nothing
        axes[i] == "None" && axes[j] == "0" || return nothing
        rows = get(ctx.scope.prepared_keys, args[i], nothing)
        index = get(ctx.scope.prepared_keys, args[j], nothing)
        rows === nothing || index === nothing ? nothing : (rows, index)
    end
end

# The plate's call in the layout the prepared data choose, or `nothing` for a
# plate that does not qualify (no prepared ragged rows, mapped or read at its
# index; the module is emitted without layouts).
function _jax_layout_call(ctx::_JaxContext, cell::AbstractString, axes_text::AbstractString,
                          args, padded, rows_at, rows)
    _jax_plate_layout() === :groups && (!isempty(padded) || any(!isnothing, rows)) ||
        return nothing
    plates = get!(() -> Pair{String,Vector{Any}}[], ctx.mod.helper_names, :layout_plates)
    entries = Any[[key for (_, key) in padded]; rows]
    # One name per distinct plate: a plate emitted for several functions keeps
    # its entry, and another plate whose cell has the same name gets its own.
    name, k = String(cell), 1
    while any(p -> first(p) == name && last(p) != entries, plates)
        name = cell * "_" * string(k += 1)
    end
    any(p -> first(p) == name, plates) || push!(plates, name => entries)
    helper = _jax_helper!(ctx, "_rk_grouped_plate", _JAX_GROUPED_PLATE)
    positions = "(" * join((string(position) * "," for (position, _) in padded), " ") * ")"
    helper * "(" * cell * ", " * axes_text * ", (" * join(args, ", ") * ",), " * positions *
        ", " * (rows_at === nothing ? "None" : string(rows_at)) *
        ", _rk_plate_layout.get(" * repr(name) * "))"
end

_jax_layout_plates(mod::_JaxModule) =
    get(mod.helper_names, :layout_plates, Pair{String,Vector{Any}}[])
