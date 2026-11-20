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
function (guard::_LSScaleGuard{I,P})(w::_LSWeightedStatistics, sigma::Float64, live::Vararg{Any,N}) where {I,P,N}
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
        return (; plate, total, selected, drop, response, sigma, law, data, live, scalars, products,
            S = _ls_structure(loc, data, live), W = _ls_slots(c.tree, data))
    end
    nothing
end

function _ls_weighted_recipes(p, g, w, id)
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
    b = Recipe(id-1, (stats, w.sigma, live...), (usable,), guard, 0.0, nothing, false)
    cold = w.selected
    produced = Set(canon_id(g,v.id) for r in cold for v in r.outputs)
    fast = Recipe(id-2, (stats, w.sigma, w.response, data..., live...), (output,),
        _LSWeightedTotal{w.S,length(data)}(w.law...), 0.0, nothing, false)
    have = unique(v -> canon_id(g, v.id), Value[usable; fast.inputs...;
        (v for r in cold for v in r.inputs if !(canon_id(g,v.id) in produced))...])
    selected_ids = Set(r.id for r in cold)
    outside = Set(canon_id(g, v.id) for r in p.recipes if !(r.id in selected_ids) for v in r.inputs)
    union!(outside, (canon_id(g, v.id) for v in p.want))
    want = Value[v for r in cold for v in r.outputs if canon_id(g, v.id) in outside]
    hot = Recipe[r for r in cold if !(r.id in w.drop) && r !== w.plate && r !== w.total]
    push!(hot, fast)
    template = Plan(g, p.have, p.want, p.recipes, p.producer, p.cost, p.candidates)
    literal = _lowering_plan(_partial_subplan(template, have, want, cold))
    quick = _lowering_plan(_partial_subplan(template, have, want, hot))
    region = _ls_make_region(w.total.id, quick, literal, Value[usable], sum(r.cost for r in cold), false)
    (a, b, region)
end
