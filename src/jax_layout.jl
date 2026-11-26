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
const _JAX_LAYOUTS = (:padded, :groups, :chains, :rounds)
_jax_requested_layout() = get(task_local_storage(), _JAX_LAYOUT_KEY, :padded)::Symbol
# Both packed modes use the same data representation and cell lowering.
_jax_plate_layout() = _jax_requested_layout() === :rounds ? :chains : _jax_requested_layout()
# The layout of a `jax_source` call that names none: `:padded`, unless the
# task asks for layouts wherever there are data (the test suite's differential
# run of every module through grouped plates, `test/jax_python.jl`).
const _JAX_LAYOUT_DEFAULT_KEY = :reactivekernels_jax_plate_layout_default
function _jax_with_plate_layout(f, layout, data)
    layout === nothing && (layout = data === nothing ? :padded :
        get(task_local_storage(), _JAX_LAYOUT_DEFAULT_KEY, :padded)::Symbol)
    layout in _JAX_LAYOUTS || throw(ArgumentError(
        "jax_source: `plate_layout` is `:padded`, `:groups`, `:chains`, or `:rounds`; got `$(repr(layout))`"))
    # Groups follow the prepared lengths of a plate's rows.
    layout !== :groups || data !== nothing || throw(ArgumentError(
        "jax_source: `plate_layout = :groups` groups plates by their prepared row lengths, so it needs `data`"))
    task_local_storage(f, _JAX_LAYOUT_KEY, layout)
end

# `prepare_plate_layout`'s partition, on the host. The units are sorted by their
# total length, and a group is a run of that order: its width per argument is
# its longest row there, so the padded size of a split is the sum over groups
# of units times widths, which the split minimizes exactly over the cuts
# between different lengths (thinned to 256 when there are more).
const _JAX_LENGTH_GROUPS = raw"""
@jax.tree_util.register_pytree_node_class
class PlateLayout:
    \"\"\"One plate's layout among the prepared data (`prepare_plate_layout`).

    `units` holds each group's units (0-based), `order` restores the plate's
    cells from the groups' joined values, `widths` holds each group's width
    per row argument, and `sequential` says whether a group's units run one
    after another. The permutations are array arguments to a jitted function.
    Its group shapes and choice of sequential execution determine the trace.
    \"\"\"

    def __init__(self, units, order, widths, sequential):
        self.units = tuple(units)
        self.order = order
        self.widths = tuple(tuple(w) for w in widths)
        self.sequential = bool(sequential)

    def tree_flatten(self):
        return (self.units, self.order), (self.widths, self.sequential)

    @classmethod
    def tree_unflatten(cls, metadata, arrays):
        return cls(*arrays, *metadata)

    def __repr__(self):
        return "PlateLayout(%s units in groups of %s, widths %s%s)" % (
            self.order.shape[0], [u.shape[0] for u in self.units], list(self.widths),
            ", sequential" if self.sequential else "")


def _rk_length_groups(lengths, groups, sequential):
    # One plate's layout, or `None` for the plate as one batch.
    lengths = [np.asarray(n, dtype=np.int64) for n in lengths]
    lengths = np.broadcast_arrays(*lengths) if lengths else []
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
    if len(units) == 1 and not sequential:
        # Units of one length: the plate as one batch.
        return None
    # A group of empty rows keeps one stored entry, as an empty column does.
    widths = [tuple(max(1, int(n[u].max())) for n in lengths) for u in units]
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
_RK_LAYOUT_NAMES = ("groups",)


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
        if isinstance(entry, str):
            return prepared[entry]
        rows, index = entry
        sizes, index = np.asarray(prepared[rows]), np.asarray(prepared[index])
        # Leave an invalid gather to the model's own checked branch.
        if np.any(index < 1) or np.any(index > len(sizes)):
            return None
        return sizes[index - 1]

    layout = {}
    for plate, entries in GROUPED_PLATES.items():
        sizes = [lengths(entry) for entry in entries]
        layout[plate] = (None if any(n is None for n in sizes) else
                         _rk_length_groups(sizes, groups, sequential))
    return {**prepared, "_rk_plate_layout": layout}


def _rk_layout_candidates(prepared, candidates):
    if candidates is None:
        candidates = [(groups, sequential) for groups in (1, 2, 4, 8, 16)
                      for sequential in (False, True)]
    seen = set()
    for groups, sequential in candidates:
        chosen = prepare_plate_layout(prepared, groups, sequential)
        # Group sizes fix the cuts into this dataset's stable subject order.
        key = tuple(None if layout is None else
                    (tuple(u.size for u in layout.units), layout.sequential)
                    for layout in chosen["_rk_plate_layout"].values())
        if key not in seen:
            seen.add(key)
            yield "groups", groups, sequential, chosen
"""

const _JAX_PREPARE_SUBJECT_LAYOUT = raw"""
_RK_LAYOUT_NAMES = ("padded", "events")


def prepare_plate_layout(prepared: dict, *, subject_walk: str = "events") -> dict:
    \"\"\"Choose the padded or real-event schedule for eligible subject scans.\"\"\"
    if subject_walk not in _RK_LAYOUT_NAMES:
        raise ValueError("subject_walk must be 'padded' or 'events'")
    result = dict(prepared)
    for name, inputs in SUBJECT_SCAN_SCHEDULES.items():
        if subject_walk == "padded":
            result[name] = None
        elif result[name] is None:
            result[name] = _rk_subject_scan_schedule(*(prepared[key] for key in inputs))
    return result


def _rk_layout_candidates(prepared, candidates):
    if candidates is None:
        candidates = _RK_LAYOUT_NAMES
    seen = set()
    for walk in candidates:
        chosen = prepare_plate_layout(prepared, subject_walk=walk)
        has_events = any(chosen[name] is not None and chosen[name][0]
                         for name in SUBJECT_SCAN_SCHEDULES)
        effective = "events" if has_events else "padded"
        if effective not in seen:
            seen.add(effective)
            if walk != effective:
                chosen = prepare_plate_layout(prepared, subject_walk=effective)
            yield effective, None, None, chosen
"""

# The reference tuner: which layout is fastest is a property of the device,
# the model and the chain count, so it is measured where the module runs. The
# caller supplies the work to time (its own batched step), so a sampler's
# memory and cost are part of the measurement.
const _JAX_SELECT_PLATE_LAYOUT = raw"""
def select_plate_layout(prepared: dict, make_step: Any, chains: Any = None,
                        candidates: Any = None, repeats: int = 3, *,
                        alternatives: Any = None, calls: Any = None, guarded_reads: Any = None) -> tuple:
    \"\"\"Time layouts on the device and return the chosen prepared data and timings.

    A builder `make_step(data, count)` returns a zero-argument callable for one
    gradient batch or sampler step at that chain count. Omit `chains` for a
    single workload and a builder taking only the data. A grouped module
    tries the `(groups, sequential)` pairs in `candidates`. An eligible
    padded module tries subject walks named "padded" and "events". An optional
    `alternatives` mapping names other emitted layouts. Each value is a pair
    `(prepared_data, builder)`, optionally followed by its `prepare_guarded_reads`.
    Eligible reads try compact and dense capacities. Set `guarded_reads` to
    "compact" or "dense" to hold that choice fixed across layouts.

    The first call warms the executable. With `calls(count)`, rank by the
    median warm time multiplied by the number of calls the fit requires at
    that count. Otherwise rank by the shortest warm time per chain. An
    allocation failure skips the larger chain counts for that layout.

    The first row in the report names the chosen variant and chain count.
    For each row, `setup_seconds` runs from entering the builder through the
    first blocked call. `calibration_seconds` also includes the warm repeats.
    A memory failure ends the interval at the failed attempt. Successful
    blocked warm calls are recorded in `samples_seconds`, with their
    `median_seconds` once all repeats finish. With `calls`, `fit_seconds`
    gives the projected fit duration at the measured warm latency.
    \"\"\"
    import operator
    import time
    if repeats < 1:
        raise ValueError("repeats must be positive")
    counts = (None,) if chains is None else sorted({operator.index(n) for n in chains})
    if chains is not None and any(n <= 0 for n in counts):
        raise ValueError("chain counts must be positive")
    if calls is not None and chains is None:
        raise ValueError("calls(count) requires chain counts")
    if alternatives is None:
        alternatives = {}
    for name in _RK_LAYOUT_NAMES:
        if name in alternatives:
            raise ValueError(f"{name} names a layout of this module")

    if guarded_reads not in (None, "compact", "dense"):
        raise ValueError("guarded_reads must be 'compact', 'dense' or None")

    def layouts():
        for name, groups, sequential, chosen in _rk_layout_candidates(prepared, candidates):
            yield name, groups, sequential, chosen, make_step, @prepare_reads@
        for name, option in alternatives.items():
            chosen, builder, selector = (*option, None) if len(option) == 2 else option
            yield name, None, None, chosen, builder, selector

    def variants():
        for name, groups, sequential, chosen, builder, selector in layouts():
            if selector is None:
                yield name, groups, sequential, None, chosen, builder
            elif guarded_reads is not None:
                yield name, groups, sequential, guarded_reads, selector(chosen, guarded_reads), builder
            else:
                compact = selector(chosen, "compact")
                yield name, groups, sequential, "compact", compact, builder
                dense = selector(compact, "dense")
                if dense is not compact:
                    yield name, groups, sequential, "dense", dense, builder

    def rank(row):
        return row.get("fit_seconds", row["seconds"] / (row["chains"] or 1))

    report = []
    best_score = None
    best_data = None
    work = {} if calls is None else {n: float(calls(n)) for n in counts}
    if any(not np.isfinite(n) or n <= 0 for n in work.values()):
        raise ValueError("the projected number of calls must be finite and positive")
    for name, groups, sequential, mode, chosen, builder in variants():
        for count in counts:
            row = {"variant": name, "groups": groups, "sequential": sequential,
                   "chains": count, "seconds": float("inf"), "error": None,
                   "samples_seconds": [], "median_seconds": None}
            if mode is not None:
                row["guarded_reads"] = mode
            if calls is not None:
                row["fit_seconds"] = float("inf")
            report.append(row)
            step = None
            started = time.perf_counter()
            try:
                step = builder(chosen) if count is None else builder(chosen, count)
                jax.block_until_ready(step())
                row["setup_seconds"] = time.perf_counter() - started
                samples = row["samples_seconds"]
                for _ in range(repeats):
                    start = time.perf_counter()
                    jax.block_until_ready(step())
                    samples.append(time.perf_counter() - start)
                row["median_seconds"] = float(np.median(samples))
                row["seconds"] = min(samples) if calls is None else row["median_seconds"]
                if calls is not None:
                    row["fit_seconds"] = row["seconds"] * work[count]
                score = rank(row)
                if best_score is None or score < best_score:
                    best_score, best_data = score, chosen
            except Exception as error:
                if "RESOURCE_EXHAUSTED" not in str(error) and not isinstance(error, MemoryError):
                    raise
                row["error"] = str(error)
                break
            finally:
                elapsed = time.perf_counter() - started
                row["calibration_seconds"] = elapsed
                row.setdefault("setup_seconds", elapsed)
    report.sort(key=rank)
    if not report or report[0]["error"] is not None:
        raise RuntimeError("select_plate_layout: no candidate ran on this device")
    return best_data, report
"""

# Add the optional API after naming the requested functions and constants:
# existing callers can already have an entry named `select_plate_layout`.
function _jax_layout_functions!(mod, bodies, functions, constants)
    prepared = Set(first(io) for io in first(functions).outputs)
    bounds = filter(b -> first(b) in prepared, _jax_prepared_bounds(mod))
    reads = [(name, keys[1], keys[2], parse(BigInt, keys[3]))
             for (name, kind, keys) in bounds if kind === :match_count]
    read_selector = "None"
    if !isempty(reads)
        read_selector = _jax_identifier(mod.used, :prepare_guarded_reads)
        push!(bodies, rstrip(replace(_JAX_PREPARE_GUARDED_READS,
            "prepare_guarded_reads" => read_selector, "@read_bounds@" => _jax_py_literal(Tuple(reads)))))
        push!(functions, JAXFunction(read_selector,
            Pair{String,Any}["prepared" => Dict, "mode" => String], Pair{String,Any}["prepared" => Dict]))
    end
    if mod.layout === :groups
        plates = _jax_layout_plates(mod)
        isempty(plates) || _jax_module_helper!(mod, "_rk_length_groups", _JAX_LENGTH_GROUPS)
        metadata = "GROUPED_PLATES"
        values = Pair{String,Tuple}[name => Tuple(keys) for (name, keys) in plates]
        source = _JAX_PREPARE_PLATE_LAYOUT
        args = Pair{String,Any}["prepared" => Dict, "groups" => Int, "sequential" => Bool]
    else
        metadata = "SUBJECT_SCAN_SCHEDULES"
        values = [name => keys for (name, kind, keys) in bounds if kind === :subject_scan]
        source = _JAX_PREPARE_SUBJECT_LAYOUT
        args = Pair{String,Any}["prepared" => Dict, "subject_walk" => String]
    end
    if mod.layout === :groups || (!isempty(values) && haskey(mod.helper_names, (:helper, "_rk_subject_scan")))
        name = _jax_identifier(mod.used, metadata)
        push!(constants, name * " = " * _jax_py_literal(values))
        prepare = _jax_identifier(mod.used, :prepare_plate_layout)
        push!(bodies, rstrip(replace(source, metadata => name, "prepare_plate_layout" => prepare)))
        push!(functions, JAXFunction(prepare, args, Pair{String,Any}["prepared" => Dict]))
    elseif !isempty(reads)
        push!(bodies, "_RK_LAYOUT_NAMES = ('module',)\ndef _rk_layout_candidates(prepared, candidates):\n    yield 'module', None, None, prepared")
    else
        return
    end
    select = _jax_identifier(mod.used, :select_plate_layout)
    push!(bodies, rstrip(replace(_JAX_SELECT_PLATE_LAYOUT,
        "select_plate_layout" => select, "@prepare_reads@" => read_selector)))
    push!(functions, JAXFunction(select,
        Pair{String,Any}["prepared" => Dict, "make_step" => Any, "chains" => Any,
                         "candidates" => Any, "repeats" => Int, "alternatives" => Any, "calls" => Any,
                         "guarded_reads" => Any],
        Pair{String,Any}["prepared" => Dict, "report" => Vector{Any}]))
    nothing
end

# The plate arguments whose rows a group cuts to its width: each mapped ragged
# vector of numeric rows, as its argument position and the prepared lengths
# that describe it. A missing key keeps the whole plate at its stored widths.
function _jax_layout_arguments(ctx::_JaxContext, inputs, mapped, ragged)
    found = Tuple{Int,Union{Nothing,String,Tuple{String,String}}}[]
    position = 0
    for (i, v) in enumerate(inputs)
        if ragged[i]
            if mapped[i] && _jax_ragged(v.T) && !_jax_tuple_valued(v.T) && eltype(eltype(v.T)) <: Number
                key = get(ctx.scope.prepared_keys, v.extent, nothing)
                if key === nothing
                    key = get(ctx.mod.helper_names, (:layout_lengths, ctx.scope, v.extent), nothing)
                end
                push!(found, (position, key))
            end
            position += 2
        else
            position += 1
        end
    end
    found
end

# A read of a shared prepared ragged array at a mapped prepared index,
# expressed in the prepared data's keys.
function _jax_layout_row(ctx::_JaxContext, (lengths, selector), params, args, axes)
    i = findfirst(==(lengths), params)
    j = findfirst(==(selector), params)
    (i === nothing || j === nothing || max(i, j) > length(args)) && return nothing
    axes[i] == "None" && axes[j] == "0" || return nothing
    rows = get(ctx.scope.prepared_keys, args[i], nothing)
    index = get(ctx.scope.prepared_keys, args[j], nothing)
    rows === nothing || index === nothing ? nothing : (rows, index)
end

# The plate's call in the layout the prepared data choose, or `nothing` for a
# plate that does not qualify (no prepared ragged rows, mapped or read at its
# index; the module is emitted without layouts).
function _jax_layout_call(ctx::_JaxContext, cell::AbstractString, axes_text::AbstractString,
                          args, padded, rows_at, rows)
    _jax_plate_layout() === :groups && (!isempty(padded) || any(!isnothing, rows)) ||
        return nothing
    # An untracked row may be zipped with a known row by a nested plate.
    # Cutting just its partner would give the inner vmap different widths.
    any(p -> last(p) === nothing, padded) && return nothing
    any(isnothing, rows) && return nothing
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
