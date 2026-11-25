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
function _jax_with_plate_layout(f, layout, data)
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
def _rk_length_groups(lengths, groups, sequential):
    # One plate's layout: each group's units (0-based), the order restoring
    # the plate's cells from the groups' joined values, each group's width per
    # padded argument, and whether a group's cells run one after another.
    lengths = [np.asarray(n, dtype=np.int64) for n in lengths]
    count = lengths[0].shape[0]
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
    widths = tuple(tuple(max(1, int(n[u].max())) for n in lengths) for u in units)
    order = np.argsort(np.concatenate(units), kind="stable")
    return units, order, widths, bool(sequential)
"""

const _JAX_GROUPED_PLATE = raw"""
def _rk_grouped_plate(cell, in_axes, args, padded, layout):
    # `jax.vmap(cell, in_axes)(*args)` in the layout the prepared data choose
    # (`prepare_plate_layout`). `padded` holds the positions in `args` of the
    # arguments whose rows are cut to a group's width.
    whole = jax.vmap(cell, in_axes=in_axes)
    if layout is None:
        return whole(*args)
    units, order, widths, sequential = layout
    # The plate's own shapes: every layout returns them, whatever its widths.
    shapes = jax.eval_shape(lambda: whole(*args))
    parts = []
    for members, width in zip(units, widths):
        group = [a if axis is None else jax.tree_util.tree_map(lambda leaf: leaf[members], a)
                 for a, axis in zip(args, in_axes)]
        for position, w in zip(padded, width):
            group[position] = group[position][:, :w]
        if sequential:
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
        return jnp.concatenate(full)[order]

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
    names the plates and the row lengths that group them.

    Call the other functions with the result: `f(u, **prepare_plate_layout(prepared, 4))`.
    \"\"\"
    layout = {plate: _rk_length_groups([prepared[key] for key in keys], groups, sequential)
              for plate, keys in GROUPED_PLATES.items()}
    return {**prepared, "_rk_plate_layout": layout}
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

# The plate's call in the layout the prepared data choose, or `nothing` for a
# plate that does not qualify (no mapped prepared ragged rows; the module is
# emitted without layouts).
function _jax_layout_call(ctx::_JaxContext, cell::AbstractString, axes_text::AbstractString, args, padded)
    _jax_plate_layout() === :groups && !isempty(padded) || return nothing
    plates = get!(() -> Pair{String,Vector{String}}[], ctx.mod.helper_names, :layout_plates)
    keys = String[key for (_, key) in padded]
    # One name per distinct plate: a plate emitted for several functions keeps
    # its entry, and another plate whose cell has the same name gets its own.
    name, k = cell, 1
    while any(p -> first(p) == name && last(p) != keys, plates)
        name = cell * "_" * string(k += 1)
    end
    any(p -> first(p) == name, plates) || push!(plates, name => keys)
    helper = _jax_helper!(ctx, "_rk_grouped_plate", _JAX_GROUPED_PLATE)
    positions = "(" * join((string(position) * "," for (position, _) in padded), " ") * ")"
    helper * "(" * cell * ", " * axes_text * ", (" * join(args, ", ") * ",), " * positions *
        ", _rk_plate_layout.get(" * repr(name) * "))"
end

_jax_layout_plates(mod::_JaxModule) =
    get(mod.helper_names, :layout_plates, Pair{String,Vector{String}}[])
