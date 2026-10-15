# A stochastic cell scan uses the ordinary array layout for its draw sites.
# The cell reads those values and states its conditional densities as RK
# operations. Recurrence/history lowering remains owned by RK's @scan.

function _cell_scan_fresh!(taken, base)
    name = base
    while name in taken
        name = Symbol(name, :_)
    end
    push!(taken, name)
    return name
end

function _ppl_cell_scan_nsteps(T, m)
    T isa Integer || throw(ArgumentError(
        "a cell scan's upper bound must be an integer; got $T"))
    # As in the retained Julia loop, a bound below the first step gives
    # an empty range. The authored seeds still exist in that cell.
    return T <= m ? 0 : Int(T - m)
end

# The parameter coordinates follow the authored plate domain, including a
# selected range whose first index is not one.
_ppl_cell_scan_position(r::AbstractRange{<:Integer}, i) =
    Int(div(i - first(r), step(r)) + 1)
_ppl_cell_scan_position(r, i) = findfirst(isequal(i), r)

function _cell_scan_geometry(st)
    args = NamedTuple{Tuple(Symbol(:arg, i) for i in eachindex(st.args))}(Tuple(st.args))
    tr, lo, hi = _entry_transform(st.family, st.support, args)
    tr === :identity && return :(Flat.())
    tr === :exp && return :(restricted.(Flat.(), 0.0, Inf))
    tr === :logistic && return :(restricted.(Flat.(), 0.0, 1.0))
    tr === :floored && return :(restricted.(Flat.(), $lo, Inf))
    tr === :upper && return :(restricted.(Flat.(), -Inf, $hi))
    tr === :interval && return :(restricted.(Flat.(), $lo, $hi))
    _scan_fail("unsupported cell draw transform $tr")
end

_cell_scan_has_sample(ex) = ex isa Expr &&
    (Meta.isexpr(ex, :call, 3) && ex.args[1] === :~ ||
        any(_cell_scan_has_sample, ex.args))

function _expand_stochastic_cell_scans(st, taken)
    loop = st.args[3]
    (Meta.isexpr(loop, :for) && Meta.isexpr(loop.args[1], :(=), 2) &&
        Meta.isexpr(loop.args[2], :block)) || return Any[], st, Symbol[]
    ivar, domain = loop.args[1].args
    ivar isa Symbol || return Any[], st, Symbol[]
    prelude, cells, priors = Any[], Any[], Symbol[]
    locals = Pair{Symbol,Any}[]
    for c in loop.args[2].args
        if _is_cell_scan_block(c) && _cell_scan_has_sample(c)
            before, body, prior = _stochastic_cell_scan(c, ivar, domain, locals, taken)
            append!(prelude, before)
            append!(cells, body)
            push!(priors, prior)
        else
            push!(cells, c)
            Meta.isexpr(c, :(=), 2) && c.args[1] isa Symbol &&
                push!(locals, c.args[1] => c.args[2])
        end
    end
    return prelude, Expr(st.head, st.args[1:2]...,
        Expr(:for, loop.args[1], Expr(:block, cells...))), priors
end

function _stochastic_cell_scan(c, ivar, domain, locals, taken)
    fresh(base) = _cell_scan_fresh!(taken, base)
    block = c.args[end]
    # The shared parser checks draw families, seed dependencies, body order,
    # lags and history reads. A cell's upper bound may be a data expression
    # (for example length(y[g])); its value is checked by the packed extent.
    parsed = deepcopy(block)
    items = Any[x for x in parsed.args if !(x isa LineNumberNode)]
    forx = last(items)
    Meta.isexpr(forx, :for) || _scan_fail("the block must end with the recurrence loop")
    range = forx.args[1].args[2]
    Meta.isexpr(range, :call, 3) && range.args[1] === :(:) ||
        _scan_fail("a cell scan takes the range lo:hi")
    hi = range.args[3]
    range.args[3] = fresh(:_ppl_cell_scan_bound)
    spec = parse_scan_block(parsed)
    m, t = spec.lo - 1, spec.loopvar
    prefix = Symbol(:_rkppl_scan_, first(spec.states))
    count_values = fresh(Symbol(prefix, :_count_values))
    counts, offsets = fresh(Symbol(prefix, :_counts)), fresh(Symbol(prefix, :_offsets))
    pos = fresh(Symbol(prefix, :_cell))
    prior = fresh(Symbol(prefix, :_prior))
    while Symbol(prior, :_cells) in taken
        prior = fresh(Symbol(prior, :_))
    end
    push!(taken, Symbol(prior, :_cells))
    lp = fresh(Symbol(prefix, :_density))
    count = Expr(:call, GlobalRef(@__MODULE__, :_ppl_cell_scan_nsteps), hi, m)
    iterator = deepcopy(domain)
    if Meta.isexpr(iterator, :call) && iterator.args[1] in (:eachindex, :axes)
        iterator.args[1] = GlobalRef(Base, iterator.args[1])
    end
    indices = Expr(:call, GlobalRef(Base, :collect), iterator)
    prelude = Any[
        Expr(:(=), count_values, _plate_column_call(ivar,
            _needed_cell_locals(locals, count), count, nothing; indices)),
        # A bound empty cell column has no first value from which to infer
        # its element type. Extents and offsets are integer-valued anyway.
        :($counts = Int.($count_values)),
        :($offsets = cumsum(vcat(0, $counts))),
    ]
    cells = Any[Expr(:(=), pos,
        Expr(:call, GlobalRef(@__MODULE__, :_ppl_cell_scan_position), domain, ivar))]
    seeds, body = Any[], Any[]
    seednames = Dict{Tuple{Symbol,Int},Symbol}()
    seedread(ex) = Meta.isexpr(ex, :ref, 2) && ex.args[1] in spec.states ?
        seednames[(ex.args[1], ex.args[2])] :
        ex isa Expr ? Expr(ex.head, map(seedread, ex.args)...) : ex
    seedterms = Any[]
    allocate(st, name, extent) = begin
        geom = _cell_scan_geometry(st)
        push!(prelude, Expr(:call, :.~, Expr(:ref, name, :(1:$extent)), geom))
        name
    end
    for f in spec.setup
        value = fresh(Symbol(prefix, :_init_, f.target, :_, f.index))
        if f.kind === :sample
            name = allocate(f, fresh(Symbol(prefix, :_seed_, f.target, :_, f.index)),
                :(length($counts)))
            push!(cells, :($value = $name[$pos]))
            pre = Expr[]
            density = _scan_density!(pre, f, seedread, value, fresh(Symbol(value, :_lp)))
            append!(cells, map(_untyped_branch_locals, pre))
            push!(seedterms, density)
        else
            push!(cells, Expr(:(=), value, seedread(f.expr)))
        end
        seednames[(f.target, f.index)] = value
        push!(seeds, Expr(:(=), Expr(:ref, f.target, f.index), value))
    end
    seedlp = isempty(seedterms) ? 0.0 : Expr(:call, :+, seedterms...)
    for k in 1:m
        push!(seeds, Expr(:(=), Expr(:ref, lp, k), k == m ? seedlp : 0.0))
    end
    stepterms = Any[]
    for st in spec.step
        lhs = st.indexed ? Expr(:ref, st.target, t) : st.target
        if st.kind === :sample
            name = allocate(st, fresh(Symbol(prefix, :_draw_, st.target)), :(sum($counts; init = 0)))
            index = :($offsets[$pos] + $t - $m)
            push!(body, Expr(:(=), lhs, Expr(:ref, name, index)))
            pre = Expr[]
            density = _scan_density!(pre, st, identity, lhs,
                fresh(Symbol(prefix, :_lp_, st.target)))
            append!(body, map(_untyped_branch_locals, pre))
            # Bind at the statement's position: later local assignments must
            # not change the conditional density's arguments.
            term = fresh(Symbol(prefix, :_term_, st.target))
            push!(body, Expr(:(=), term, density))
            push!(stepterms, term)
        else
            push!(body, Expr(:(=), lhs, st.expr))
        end
    end
    stepterm = isempty(stepterms) ? 0.0 : Expr(:call, :+, stepterms...)
    push!(body, :($lp[$t] = $lp[$t - 1] + $stepterm))
    push!(seeds, Expr(:for, Expr(:(=), t, :($(spec.lo):$hi)), Expr(:block, body...)))
    push!(cells, Expr(:macrocall, Symbol("@scan"), c.args[2], Expr(:block, seeds...)))
    push!(cells, Expr(:(=), Expr(:ref, Symbol(prior, :_cells), ivar), :(last($lp))))
    return prelude, cells, prior
end
