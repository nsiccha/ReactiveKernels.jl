# Summed location-scale cells. On positive scales, express the scalar cell
# in the residual (y-m), scale s and log(s). A quadratic in (y-m)/s has a
# weighted least-squares statistic for any separable scale s = sigma*w(data).
# This is a preparation-time proof; the authored plate remains the other arm.
struct _LSUnsupported <: Exception end
_ls_unknown() = throw(_LSUnsupported())
const _LSCellForm = Dict{Symbol,Float64}
_ls_monomial(p) = _LSCellForm(p => 1.0)
_ls_prune!(p) = filter!(kv -> !iszero(last(kv)), p)
function _ls_polyadd(a, b)
    # Never prove independence of an input by cancelling approximate coefficients.
    any(p -> p != :constant && haskey(b, p), keys(a)) && _ls_unknown()
    _ls_prune!(mergewith(+, a, b))
end
function _ls_polymul(a, b)
    # Do not cancel scale powers or chains of large and small coefficients
    # across floating-point operations. The only product of varying terms
    # needed here squares the standardized residual.
    if a == b && (a == _ls_monomial(:standardized) ||
            a == _LSCellForm(:standardized => -1.0))
        return _ls_monomial(:square)
    end
    if !all(==(:constant), keys(a))
        a, b = b, a
    end
    all(==(:constant), keys(a)) || _ls_unknown()
    multiplier = _ls_number(a)
    if abs(multiplier) != 1 && any(!=(:constant), keys(b))
        !iszero(multiplier) && all(kv -> first(kv) == :constant ||
            (first(kv) in (:logscale, :square) && abs(last(kv)) == 1), b) || _ls_unknown()
    end
    _ls_prune!(_LSCellForm(k => multiplier * x for (k, x) in b))
end
_ls_scalar(x) = _LSCellForm(:constant => Float64(x))
_ls_number(p) = all(==(:constant), keys(p)) ? get(p, :constant, 0.0) : _ls_unknown()

# Interpret only pure, supported operations. A guard must be decidable under
# the same positive-scale assumption used at evaluation. Never enter its
# inactive arm, even while analyzing the program.
function _ls_poly(ex, env, mod, depth)
    depth < 100 || _ls_unknown()
    sub(a) = _ls_poly(a, env, mod, depth + 1)
    ex isa Symbol && haskey(env, ex) && return env[ex]
    if ex isa Union{GlobalRef,Symbol,Real}
        x = ex isa Real ? ex : _rw_resolve(ex, mod)
        x isa Union{Int,Float64,Irrational} && isfinite(x) || _ls_unknown()
        return _ls_scalar(x)
    end
    ex isa Expr || _ls_unknown()
    if ex.head === :(=) && first(ex.args) isa Symbol
        return env[first(ex.args)] = sub(last(ex.args))
    elseif ex.head === :block
        value = nothing
        for a in ex.args
            a isa LineNumberNode || (value = sub(a))
        end
        return value
    elseif ex.head === :let
        inner = copy(env)
        _ls_poly(ex.args[1], inner, mod, depth + 1)
        return _ls_poly(ex.args[2], inner, mod, depth + 1)
    elseif ex.head === :if
        flag = sub(ex.args[1])
        flag isa Bool || _ls_unknown()
        return sub(ex.args[flag ? 2 : 3])
    elseif ex.head === :&& || ex.head === :||
        flag = sub(ex.args[1])
        flag isa Bool || _ls_unknown()
        return flag == (ex.head === :||) ? flag : sub(ex.args[2])
    elseif ex.head === :return
        return sub(only(ex.args))
    end
    Meta.isexpr(ex, :call) || _ls_unknown()
    head = first(ex.args)
    f = head isa QuoteNode ? head.value : _rw_resolve(head, mod)
    args = [sub(a) for a in ex.args[2:end]]
    isempty(args) && _ls_unknown()
    if f isa _KernelSourceOp
        rec = _source_record(f.f)
        rec isa _KernelSource && _source_native(rec) === nothing &&
            isempty(_source_captures(rec)) || _ls_unknown()
        prog = _program_expression(_native_generated_function(rec.companion))
        names = _source_params(rec, length(args))
        names === nothing && _ls_unknown()
        body = _pr_tail_value(prog.args[2])
        _inline_contains_return(body) && _ls_unknown()
        return _ls_poly(body, Dict{Symbol,Any}(zip(names, args)),
            _kernel_endpoint_source_module(f), depth + 1)
    end
    if f === (-) && length(args) == 2
        args == [:observation, :location] && return _ls_monomial(:residual)
        args == [:location, :observation] && return _LSCellForm(:residual => -1.0)
    end
    all(a -> a isa _LSCellForm, args) || _ls_unknown()
    f === (+) && return reduce(_ls_polyadd, args)
    if f === (-) && length(args) in (1, 2)
        neg = _ls_polymul(_ls_scalar(-1), last(args))
        return length(args) == 1 ? neg : _ls_polyadd(first(args), neg)
    end
    f in ((*), _native_mul) && return reduce(_ls_polymul, args)
    if f === (/) && length(args) == 2 && length(args[2]) == 1
        p, x = only(args[2])
        if p === :scale && x == 1 && length(args[1]) == 1
            degree, sign = only(args[1])
            degree == :residual && abs(sign) == 1 || _ls_unknown()
            return _LSCellForm(:standardized => sign)
        end
        p == :constant && isfinite(inv(x)) || _ls_unknown()
        return _ls_polymul(_ls_scalar(inv(x)), args[1])
    elseif f === (^) && length(args) == 2 && _ls_number(args[2]) == 2
        return _ls_polymul(args[1], args[1])
    elseif f === log && length(args) == 1
        only(args) == _ls_monomial(:scale) && return _ls_monomial(:logscale)
        x = _ls_number(only(args))
        x > 0 || _ls_unknown()
        return _ls_scalar(log(x))
    elseif f in ((==), (!=), (>), (<), (>=), (<=)) && length(args) == 2
        # Only comparisons whose truth is fixed for every positive scale.
        a, b = args
        scale = _ls_monomial(:scale)
        a == scale && iszero(_ls_number(b)) && return f(1.0, 0.0)
        b == scale && iszero(_ls_number(a)) && return f(0.0, 1.0)
    end
    _ls_unknown()
end

function _ls_scale_law(r::Recipe, y, m, s)
    inner = r.op.kernel
    ast = code_expr(inner)
    length(r.inputs) == 3 && isempty(r.op.axis_checks) || return nothing
    any(q -> q.effectful, inner.plan.recipes) && return nothing
    names = [_pr_param_name(a) for a in ast.args[1].args[3:end]]
    all(a -> a isa Symbol, names) && length(names) == 3 || return nothing
    all(v -> valtype(v) in (Any, Float64), inner.plan.have) || return nothing
    env = Dict{Symbol,Any}(names[y] => :observation, names[m] => :location,
                           names[s] => _ls_monomial(:scale))
    body = _pr_tail_value(ast.args[2])
    _inline_contains_return(body) && return nothing
    poly = try
        _ls_poly(body, env, @__MODULE__, 0)
    catch err
        err isa _LSUnsupported || rethrow()
        return nothing
    end
    poly isa _LSCellForm && haskey(poly, :square) && all(isfinite, values(poly)) &&
        all(p -> p in (:constant, :logscale, :square), keys(poly)) || return nothing
    (; constant = get(poly, :constant, 0.0), logscale = get(poly, :logscale, 0.0),
       square = poly[:square])
end

struct _LSWeightedStatistics{F}
    fit::F
    logsum::Float64
    smallest::Float64
    largest::Float64
    # Conservative bounds for the triangular product and its squared norm.
    factor_norm::Float64
    centre::Float64
end
struct _LSWeightedBind{S,W} end
@generated function (::_LSWeightedBind{S,W})(response, args...) where {S,W}
    missing = _ls_mixed(S) ? :(_ls_mixed_unusable(n)) : :(_ls_unusable(0, 0, Int[], n))
    quote
        data = args
        n = response isa AbstractVector ? length(response) : 0
        scales = $(_ls_column_expr(W))
        if scales === nothing || isempty(scales) || !all(_ls_positive_scale, scales)
            return _LSWeightedStatistics($missing, NaN, NaN, NaN, NaN, NaN)
        end
        fit = _ls_bind(Val($(QuoteNode(S))), response, data, scales)
        _ls_weighted_statistics(fit, scales)
    end
end
_ls_positive_scale(x) = isfinite(x) && x > 0
_ls_rss0(fit::_LSStatistics) = fit.rss0
_ls_rss0(fit::_LSMixedStatistics) = fit.shared.rss0
_ls_factor_bound(fit::_LSStatistics) = norm(fit.factor)*sqrt(fit.k)
_ls_centre_bound(fit) = maximum(abs, fit.hi; init=0.0) + maximum(abs, fit.lo; init=0.0)
_ls_factor_bound(fit::_LSMixedStatistics) = _ls_factor_bound(fit.shared) +
    norm(fit.norms) + sqrt(fit.shared.k) * norm(fit.cross)
function _ls_weighted_statistics(fit, scales)
    lo, hi = extrema(scales)
    centre = _ls_centre_bound(fit)
    fit isa _LSMixedStatistics && (centre = max(centre, _ls_centre_bound(fit.shared)))
    _LSWeightedStatistics(fit, sum(log, scales), lo, hi, _ls_factor_bound(fit), centre)
end

_ls_magnitude(x) = Inf
_ls_magnitude(x::Float64) = abs(x)
_ls_magnitude(x::AbstractVector{Float64}) = maximum(abs, x; init=0.0)
@inline _ls_magnitude(::Tuple{}) = 0.0
@inline _ls_magnitude(xs::Tuple) = max(_ls_magnitude(first(xs)), _ls_magnitude(Base.tail(xs)))
# The ordinary plate handles extreme scales and nonfinite inputs. Bound the
# norms before doing live arithmetic, so reverse never traverses a discarded
# overflowing product. The k bound accounts for all entries in one block.
@generated function _ls_scalar_slots(::Val{I}, xs::Tuple) where {I}
    foldr((i, tail) -> :(xs[$i] isa Float64 && $tail), I; init = true)
end
_ls_product_argument(x) = false
_ls_product_argument(x::Float64) = _ls_product_range(x)
_ls_product_argument(x::AbstractVector{Float64}) = all(_ls_product_range, x)
@generated function _ls_product_arguments(::Val{P}, args::Tuple) where {P}
    foldr((i, tail) -> :(_ls_product_argument(args[$i]) && $tail), P; init = true)
end
struct _LSScaleGuard{I,P}
    maxnorm::Float64
    maxrows::Float64
end
# Inlined: as a call, ordinary reverse paid for the guard's active arguments on
# every evaluation (about 5 ns of a 40 ns gradient), although it returns a Bool.
@inline function (guard::_LSScaleGuard{I,P})(w::_LSWeightedStatistics, sigma::Float64, live::Vararg{Any,N}) where {I,P,N}
    w.fit.usable && sigma > 0 && _ls_scalar_slots(Val(I), live) || return false
    _ls_product_arguments(Val(P), (sigma, live...)) || return false
    w.fit.rows <= guard.maxrows || return false
    small, large = sigma*w.smallest, sigma*w.largest
    floatmin(Float64) <= small && isfinite(large) || return false
    m = _ls_magnitude(live)
    radius = w.factor_norm * (m + w.centre) + sqrt(_ls_rss0(w.fit))
    isfinite(radius * w.largest) && radius < sqrt(floatmax(Float64)) &&
        radius / sigma < guard.maxnorm && _plate_factor_usable(inv(sigma)^2)
end

struct _LSWeightedTotal{S,ND}
    constant::Float64
    logscale::Float64
    square::Float64
end
@inline function (f::_LSWeightedTotal{S,ND})(w::_LSWeightedStatistics, sigma, response,
                                             args::Vararg{Any,N}) where {S,ND,N}
    loc = _LSLocationCall(S, ND, N - ND, ())(w.fit, response, args...)
    f.constant * w.fit.rows + f.logscale * (w.fit.rows * log(sigma) + w.logsum) +
        f.square * (_ls_rss(loc) / sigma / sigma)
end

function _ls_weighted_rewrite(p, plate, total, dataonly, wanted, next_start)
    length(plate.inputs) == 3 && isempty(typeof(plate.op).parameters[2]) || return nothing
    ids = [canon_id(p.graph, v.id) for v in plate.inputs]
    length(unique(ids)) == 3 && count(in(dataonly), ids) == 1 || return nothing
    y = findfirst(in(dataonly), ids)
    response = plate.inputs[y]
    for s in eachindex(ids)
        s == y && continue
        scale = _ls_location(p, plate.inputs[s], dataonly, true)
        scale === nothing && continue
        length(scale.terms) == 1 && isempty(scale.offsets) || continue
        (c, shared, mat, group) = only(scale.terms)
        mat === nothing && group === nothing || continue
        sigma = p.graph.values[shared]
        valtype(sigma) === Float64 || continue
        m = only(setdiff(eachindex(ids), (s, y)))
        loc = _ls_location(p, plate.inputs[m], dataonly, true)
        loc === nothing && continue
        _ls_consistent(loc) || continue
        law = _ls_scale_law(plate, y, m, s)
        law === nothing && continue
        data, live = _ls_inputs(loc)
        append!(data, _ls_data_ids(c))
        unique!(data)
        kept = union(wanted, Set(data), Set(live), Set([shared]))
        drop = union(_ls_exclusive(p, ids[m], plate, kept, dataonly),
                     _ls_exclusive(p, ids[s], plate, kept, dataonly))
        earliest = findfirst(q -> q.id in drop || q === plate, p.recipes)
        start = max(earliest, next_start)
        stop = findfirst(==(total), p.recipes)
        # Begin after every input to the guard and bound fit is available.
        required = Set([shared; ids[y]; live; data])
        lastinput = findlast(q -> any(v -> canon_id(p.graph, v.id) in required, q.outputs), p.recipes)
        # A common scale already factors out of the authored row reduction.
        # If an exclusive producer must run before the guard, the fit would
        # pay for that location twice. Keep the factored reduction in this case.
        lastinput === nothing || (start = max(start, lastinput + 1))
        isempty(_ls_data_ids(c)) && earliest < start && continue
        start <= findfirst(==(plate), p.recipes) || continue
        selected = p.recipes[start:stop]
        # Keep all intervening definitions in their original order in both
        # arms. Escape values (e.g. coefficient tuples shared with a prior)
        # remain ordinary outputs. Other summed regions keep their own guard.
        checked = Set(canon_id(p.graph, a.id) for check in p.checks for a in check.args if a isa Value)
        any(q -> any(v -> canon_id(p.graph,v.id) in checked, q.outputs), selected) && continue
        any(q -> q.effectful || (q !== plate && q.op isa _AuthoredPlateOp &&
            _authored_plate_sum_recipe(p, q) !== nothing), selected) && continue
        scalars = Tuple(i for (i, (_, l, matrix, index)) in enumerate(loc.terms) if
                        matrix === nothing && index === nothing && valtype(p.graph.values[l]) === Any)
        product_slots = Int[i + 1 for (i, t) in enumerate(loc.terms) if _ls_has_live_product(t[1].tree)]
        _ls_has_live_product(c.tree) && pushfirst!(product_slots, 1)
        products = Tuple(product_slots)
        factor = _ls_row_factor(isempty(_ls_data_ids(c)),
            all(t -> t[3] === nothing && t[4] === nothing, loc.terms))
        return (; plate, total, selected, response, sigma, law, data, live, scalars, products,
            factor, S = _ls_structure(loc, data, live), W = _ls_slots(c.tree, data))
    end
    nothing
end

# The statistics alternative of one summed plate, until a binding decides.
#
# The port-level pass leaves the authored plate, its total and their producers
# where they are, so every later pass (index runs, the data-only prefix, class
# narrowing, plate fusion, and per binding the cell caches, branch partitions
# and duplicate-row grouping) treats them as it treats a plate without
# statistics. It adds the bound fit and this marker in front of them. The
# marker reads the fit and one live value, which keeps it in the residual; it
# holds the statistics arm's own recipes, so it is no reader of anything else.
#
# Per binding, the inner-plate pass closes the marker at the plate's total
# (`_ls_pending_close!`): the recipes between them are then that binding's own
# row program. Where the statistics cost less than those rows
# (`_ls_prefers_statistics`), they become the authored arm of a region whose
# other arm is `fast`. Otherwise the marker and the unread fit are removed,
# and what remains is the plan without the rewrite.
struct _LSPendingStatistics
    # The scale guard whose Bool selects the arm; `nothing` where the fit's
    # own `usable` does.
    guard::Union{Nothing,Recipe}
    # The statistics arm: recipes that compute `output` from the fit.
    fast::Vector{Recipe}
    output::ValueId
    # What one authored row costs, as a multiple of a row of a vectorized
    # loop over bound columns (`_ls_row_factor`).
    factor::Int
end
# A marker no binding closed computes nothing the program reads.
(::_LSPendingStatistics)(stats, live) = false

function _ls_weighted_recipes(g, w, id)
    output = only(w.total.outputs)
    stats = value!(g, Symbol(output.name, :_weighted_statistics), _LSWeightedStatistics{_ls_fit_type(w.S)})
    usable = value!(g, Symbol(output.name, :_weighted_usable), Bool)
    data = [g.values[i] for i in w.data]
    live = [g.values[i] for i in w.live]
    a = Recipe(id, (w.response, data...), (stats,), _LSWeightedBind{w.S,w.W}(), 0.0, nothing, false)
    headroom = floatmax(Float64) / 4
    # |log(s)| is at most about 745 on positive finite Float64 scales,
    # including subnormals. Budget this much log-scale contribution per row.
    guard = _LSScaleGuard{w.scalars,w.products}(sqrt(headroom / max(1.0, abs(w.law.square))),
        headroom / max(1.0, abs(w.law.constant), 750 * abs(w.law.logscale)))
    check = Recipe(id-1, (stats, w.sigma, live...), (usable,), guard, 0.0, nothing, false)
    fast = Recipe(id-2, (stats, w.sigma, w.response, data..., live...), (output,),
        _LSWeightedTotal{w.S,length(data)}(w.law...), 0.0, nothing, false)
    pending = _LSPendingStatistics(check, Recipe[fast], canon_id(g, output.id), w.factor)
    b = Recipe(id-1, (stats, w.sigma), (usable,), pending, 0.0, nothing, false)
    (a, b)
end

# The lazy location's rewrite (`_ls_plate_rewrite`) as a pending alternative:
# its statistics arm is the plate reading the fit's location, as
# `_ls_class_residual` builds it for a location with a matrix product or a
# gather. `nothing` where that arm cannot start after its inputs and before
# the plate, or the recipes between hold what a region may not.
function _ls_location_pending(p::Plan, g::Graph, r::Recipe, total::Recipe, rewrite,
                              next_start::Int, id::Int)
    (; S, response, location, position, data, live, drop) = rewrite
    at = findfirst(==(r), p.recipes)
    stop = findfirst(==(total), p.recipes)
    earliest = minimum((i for (i, q) in enumerate(p.recipes) if q.id in drop); init = at)
    required = Set([canon_id(g, response.id); live; data])
    lastinput = findlast(q -> any(v -> canon_id(g, v.id) in required, q.outputs), p.recipes)
    start = max(earliest, next_start, lastinput === nothing ? 1 : lastinput + 1)
    # An exclusive producer in front of the region would run beside the fit.
    earliest < start && return nothing
    start <= at || return nothing
    selected = p.recipes[start:stop]
    checked = Set(canon_id(g, a.id) for check in p.checks for a in check.args if a isa Value)
    any(q -> any(v -> canon_id(g, v.id) in checked, q.outputs), selected) && return nothing
    any(q -> q.effectful || (q !== r && q.op isa _AuthoredPlateOp &&
        _authored_plate_sum_recipe(p, q) !== nothing), selected) && return nothing
    stem = Symbol(location.name)
    stats = value!(g, Symbol(stem, :_least_squares), _ls_fit_type(S))
    loc = value!(g, Symbol(stem, :_least_squares_location), Any)
    flag = value!(g, Symbol(stem, :_least_squares_pending), Bool)
    datavalues = Value[g.values[i] for i in data]
    livevalues = Value[g.values[i] for i in live]
    a = Recipe(id, (response, datavalues...), (stats,), _LSStatisticsOp{S}(), 0.0, nothing, false)
    read = Recipe(id-2, (stats, response, datavalues..., livevalues...), (loc,),
        _LSLocationCall(S, length(data), length(live), ()), 0.0, nothing, false)
    inputs = collect(Value, r.inputs)
    inputs[position] = loc
    plate = Recipe(r.id, Tuple(inputs), r.outputs, r.op, r.cost, r.cse_key, r.effectful, r.source)
    pending = _LSPendingStatistics(nothing, Recipe[read, plate, total],
        canon_id(g, only(total.outputs).id), 1)
    b = Recipe(id-1, (stats, first(livevalues)), (flag,), pending, 0.0, nothing, false)
    (; first = first(selected), recipes = (a, b))
end

# --- one binding's choice -----------------------------------------------------
#
# Measured per value-and-gradient call (ordinary Enzyme reverse, one core;
# `benchmark/near_miss/repeated_rows.jl` repeats the measurement):
#
# - The statistics arm costs about 6 ns for its guard and region, about 2.8 ns
#   per entry of a dense block's triangular factor, and for a grouped
#   coefficient beside a shared design about 1 ns per level plus 0.26 ns per
#   level and shared column.
# - An authored row costs about 0.05 ns plus 0.05 ns per term where the plate
#   is one vectorized loop over bound columns: a common scale, and every term
#   a live number times bound per-row data. Other rows cost a multiple of
#   that: the grouped loop over distinct rows, which weights each cell by its
#   row count, about 3 times (0.46 ns per row at two coefficients); a row
#   that reads a gather or a matrix product at least 6 times; a row with its
#   own scale, which takes a logarithm and a division, 50 to 60 times (7.7 to
#   9.2 ns per row; 40 is used).
#
# The costs below are those, in units of 0.05 ns. A binding runs its
# statistics when they cost less than the rows its authored arm would visit:
# the distinct rows where the duplicate-row grouping applied, else every row.
#
# A row is priced as the loop's vector blocks run it, its cheapest. A loop of
# a few dozen rows or fewer runs mostly its scalar remainder, at about 1.2 ns
# per row: there the statistics would already cost less from 12 rows (one
# coefficient) to 20 rows (three), and those bindings keep their rows. So a
# binding's statistics cost less than its rows however the loop splits them,
# and some that would pay do not run.
const _LS_REGION_COST = 120
const _LS_ENTRY_COST = 56
const _LS_LEVEL_COST = 20
const _LS_LEVEL_COLUMN_COST = 5
const _LS_GROUPED_ROW_FACTOR = 3
const _LS_INDIRECT_ROW_FACTOR = 6
const _LS_SCALED_ROW_FACTOR = 40

# The row factor a plan fixes: `common` scale or one per row, and whether
# every term reads bound `columns`.
_ls_row_factor(common::Bool, columns::Bool) =
    !common ? _LS_SCALED_ROW_FACTOR : columns ? 1 : _LS_INDIRECT_ROW_FACTOR

# Whether one block per index value costs less per call than the `n` gathered
# rows the blocks replace. With few rows per value it does not: two
# coefficients per value pay from about ten rows each (the Rats regression,
# five rows per rat, ran 11 to 20% slower from its blocks). A binding that
# fails this keeps its rows through the region's authored arm, like any
# other unusable fit, so the compiled program does not depend on it.
_ls_blocks_pay(n::Int, k::Int, blocks::Int) =
    n * (1 + k) * _LS_INDIRECT_ROW_FACTOR >=
        _LS_REGION_COST + _LS_ENTRY_COST * blocks * (k * (k + 1) ÷ 2)

_ls_fit(w::_LSWeightedStatistics) = w.fit
_ls_fit(fit::Union{_LSStatistics,_LSMixedStatistics}) = fit
_ls_fit(x) = nothing
_ls_statistics_cost(fit::_LSStatistics) =
    _LS_ENTRY_COST * fit.blocks * (fit.k * (fit.k + 1) ÷ 2)
_ls_statistics_cost(fit::_LSMixedStatistics) = _ls_statistics_cost(fit.shared) +
    length(fit.norms) * (_LS_LEVEL_COST + _LS_LEVEL_COLUMN_COST * size(fit.cross, 2))
_ls_row_terms(fit::_LSStatistics) = fit.k
_ls_row_terms(fit::_LSMixedStatistics) = fit.shared.k + 1

# Whether a binding's statistics arm costs less than its `rows` authored
# rows. A fit that can never be used (`usable = false`) never does.
function _ls_prefers_statistics(fit, rows::Int, grouped::Bool, factor::Int)
    fit.usable || return false
    grouped && (factor = max(factor, _LS_GROUPED_ROW_FACTOR))
    rows * (1 + _ls_row_terms(fit)) * factor >= _LS_REGION_COST + _ls_statistics_cost(fit)
end

# Whether the duplicate-row grouping rewrote this plate: its cell ends in the
# row-count weight (`_cell_group_kernel`).
_ls_grouped_plate(r::Recipe) = any(q -> q.op === _cell_multiplicity_weighted,
    (_plate_cell_kernel(r.op).plan::Plan).recipes)

# Rows the authored arm visits at this binding, and whether a grouped plate
# visits them: the longest bound vector of each plate between the marker and
# the total. A grouped plate reads its compacted vectors, one entry per
# distinct row.
function _ls_pending_rows(g::Graph, cold::Vector{Recipe}, known)
    rows, grouped = 0, false
    for r in cold
        r.op isa _AuthoredPlateOp || continue
        longest = 0
        for v in r.inputs
            data = get(known, canon_id(g, v.id), nothing)
            data isa AbstractVector && (longest = max(longest, length(data)))
        end
        rows += longest
        grouped |= _ls_grouped_plate(r)
    end
    rows, grouped
end

# Remove an open marker and the fit only it read: the plan without the rewrite.
function _ls_pending_abandon!(g::Graph, recipes::Vector{Recipe}, pending::Vector{Recipe},
                              want, marker::Recipe)
    at = findfirst(r -> r === marker, recipes)
    at === nothing || deleteat!(recipes, at)
    stale = Set{ValueId}((canon_id(g, first(marker.inputs).id),))
    _cell_group_drop_unread!(g, recipes, pending, want, stale)
    :rows
end

# Whether recipe `r` produces the total the open `marker` waits for.
_ls_pending_closes(g::Graph, marker::Recipe, r::Recipe) =
    any(v -> canon_id(g, v.id) == (marker.op::_LSPendingStatistics).output, r.outputs)

# Settle the open `marker`: close it against the total just emitted (`found`),
# or abandon it. The binding's choice joins the pass's signature, since the
# two choices are different plans.
function _ls_pending_settle!(g::Graph, p::Plan, recipes::Vector{Recipe}, pending::Vector{Recipe},
                             known, signature::Vector{Any}, marker::Recipe, opened::Int,
                             rewrites, fresh, found::Bool)
    at = findfirst(r -> r === marker, recipes)
    at === nothing && (found = false)
    # The plates rewritten since the marker fix the region up to its bind-time
    # constants, as they fix the plan.
    key = (:pending_statistics, marker.id, Tuple(signature[opened+1:end]))
    choice = found ?
        _ls_pending_close!(g, p, recipes, pending, known, at::Int, rewrites, key, fresh) :
        _ls_pending_abandon!(g, recipes, pending, p.want, marker)
    push!(signature, (:pending_statistics, marker.id, choice))
    nothing
end

# Close the marker at `recipes[at]` against the total just emitted, the last
# of `recipes`. Returns `:statistics` or `:rows`, this binding's choice.
function _ls_pending_close!(g::Graph, p::Plan, recipes::Vector{Recipe}, pending::Vector{Recipe},
                            known, at::Int, rewrites, @nospecialize(key), fresh)
    marker = recipes[at]
    op = marker.op::_LSPendingStatistics
    abandon() = _ls_pending_abandon!(g, recipes, pending, p.want, marker)
    fit = _ls_fit(get(known, canon_id(g, first(marker.inputs).id), nothing))
    fit === nothing && return abandon()
    inside = recipes[at+1:end]
    constants = Recipe[r for r in inside if r.op isa _BoundConstant]
    cold = Recipe[r for r in inside if !(r.op isa _BoundConstant)]
    any(r -> r.effectful || r.op isa Union{_LSRegion,_LSPendingStatistics}, cold) && return abandon()
    rows, grouped = _ls_pending_rows(g, cold, known)
    _ls_prefers_statistics(fit, rows, grouped, op.factor) || return abandon()
    # Every value the statistics arm reads is ready in front of the marker: a
    # HAVE port, or the output of an earlier recipe. A bound vector the
    # grouping replaced is bound again from this binding's data.
    ready = Set{ValueId}(canon_id(g, v.id) for v in p.have)
    for r in view(recipes, 1:at-1), v in r.outputs
        push!(ready, canon_id(g, v.id))
    end
    arm = op.guard === nothing ? op.fast : Recipe[op.guard; op.fast]
    made = Set(canon_id(g, v.id) for r in arm for v in r.outputs)
    restored = Recipe[]
    for r in arm, v in r.inputs
        id = canon_id(g, v.id)
        (id in made || id in ready) && continue
        haskey(known, id) || return abandon()
        push!(restored, Recipe(fresh(), (), (g.values[id],), _BoundConstant(known[id]),
                               0.0, nothing, false))
        push!(ready, id)
    end
    region = rewrites === nothing ? nothing : _plate_rewrite(rewrites, key)
    if region === nothing
        region = _ls_pending_region(g, p, pending, marker, cold)
        rewrites === nothing || (region = _plate_rewrite!(rewrites, key, region))
    end
    resize!(recipes, at - 1)
    append!(recipes, restored)
    append!(recipes, constants)
    op.guard === nothing || push!(recipes, op.guard)
    push!(recipes, region)
    :statistics
end

# The two-arm region of one closed marker. `cold` is the binding's row program
# between the marker and the total, in order. The statistics arm keeps what
# something outside the region reads and replaces the rest, the plate and its
# exclusive producers, by the marker's own recipes.
function _ls_pending_region(g::Graph, p::Plan, pending::Vector{Recipe}, marker::Recipe,
                            cold::Vector{Recipe})
    op = marker.op::_LSPendingStatistics
    statistics = Value[op.guard === nothing ? first(marker.inputs) : only(op.guard.outputs)]
    made = Set(canon_id(g, v.id) for r in op.fast for v in r.outputs)
    produced = Set(canon_id(g, v.id) for r in cold for v in r.outputs)
    have = unique(v -> canon_id(g, v.id), Value[statistics;
        (v for r in op.fast for v in r.inputs if !(canon_id(g, v.id) in made))...;
        (v for r in cold for v in r.inputs if !(canon_id(g, v.id) in produced))...])
    outside = Set(canon_id(g, v.id) for r in pending for v in r.inputs)
    union!(outside, (canon_id(g, v.id) for v in p.want))
    union!(outside, (canon_id(g, a.id) for c in p.checks for a in c.args if a isa Value))
    push!(outside, op.output)
    want = unique(v -> canon_id(g, v.id),
                  Value[v for r in cold for v in r.outputs if canon_id(g, v.id) in outside])
    # What the statistics arm still computes of the rows: the producers of
    # every region output other than the total.
    needed = Set(canon_id(g, v.id) for v in want)
    delete!(needed, op.output)
    kept = falses(length(cold))
    for index in reverse(eachindex(cold))
        r = cold[index]
        any(v -> canon_id(g, v.id) in needed, r.outputs) || continue
        kept[index] = true
        union!(needed, (canon_id(g, v.id) for v in r.inputs))
    end
    hot = Recipe[cold[index] for index in eachindex(cold) if kept[index]]
    append!(hot, op.fast)
    template = Plan(g, p.have, p.want, p.recipes, p.producer, p.cost, p.candidates)
    literal = _lowering_plan(_partial_subplan(template, have, want, cold))
    quick = _lowering_plan(_partial_subplan(template, have, want, hot))
    _ls_make_region(last(cold).id, quick, literal, statistics,
                    sum(r.cost for r in cold; init = 0.0), false)
end
