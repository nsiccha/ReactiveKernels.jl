module ReactiveKernelsPPLFilterExt

using ReactiveKernelsPPL, Random
import ReactiveKernels as RK
import ReactiveKernelsPPL as PPL
import Distributions as D
import SequentialMonteCarlo as SMC

struct FilterTarget{F,P,D,S,C,O}
    advance::F
    parameters::P
    data::D
    signature::S
    prototype::C
    program::Expr
    operations::O
end

_fail(message) = throw(ArgumentError("[filter] " * message))
_macro(ex, name) = Meta.isexpr(ex, :macrocall) &&
    (ex.args[1] == Symbol("@", name) ||
     ex.args[1] isa GlobalRef && ex.args[1].name == Symbol("@", name))
_stmts(ex) = filter(x -> !(x isa LineNumberNode), ex.args)
_field(x, name) = Expr(:., x, QuoteNode(name))
_ref(x, indices...) = Expr(:ref, x, indices...)
_call(f, args...) = Expr(:call, f, args...)
_global(name) = GlobalRef(@__MODULE__, name)

function PPL.filter_parameter_names(model::RKPPLModel; observations=(:y,),inputs=(),data::NamedTuple=(;))
    supplied = Set{Symbol}((observations...,inputs...,keys(data)...,keys(model.fixed)...))
    ast,scopes = PPL._expand_submodels(model.ast,Set(n for n in supplied if !occursin('.',String(n))),model.mod;
        rewrites=model.rewrites,with_scopes=true)
    paths = PPL._scope_name_paths(scopes)
    found = Symbol[]
    function visit(block)
        for st in _stmts(block)
            if _macro(st,:scan)
                continue
            elseif _macro(st,:plate)
                visit(st.args[end].args[2])
            elseif Meta.isexpr(st,:call,3) && st.args[1] in (:~,Symbol(".~"))
                lhs = st.args[2]
                name = lhs isa Symbol ? lhs : Meta.isexpr(lhs,:ref) ? lhs.args[1] : nothing
                name isa Symbol || _fail("cannot identify static declaration $lhs")
                public_name = haskey(paths,name) ? Symbol(join(paths[name],".")) : name
                public_name in supplied || push!(found,public_name)
            end
        end
    end
    visit(ast)
    return Tuple(unique(found))
end

_hasvalue(values::NamedTuple,name) = haskey(values,name)
_hasvalue(values::AbstractDict,name) = haskey(values,name)
_getvalue(values::NamedTuple,name) = getproperty(values,name)
_getvalue(values::AbstractDict,name) = values[name]
function _parameter_value(values,name)
    _hasvalue(values,name) && return _getvalue(values,name)
    value = values
    for part in Symbol.(split(String(name),'.'))
        value isa Union{NamedTuple,AbstractDict} && _hasvalue(value,part) ||
            _fail("constrained draw has no static parameter $name")
        value = _getvalue(value,part)
    end
    return value
end
function PPL.filter_parameters(model::RKPPLModel, values::Union{NamedTuple,AbstractDict};kwargs...)
    names = filter_parameter_names(model;kwargs...)
    return NamedTuple{names}(Tuple(deepcopy(_parameter_value(values,n)) for n in names))
end

# This compiler runs only at preparation. Hot transitions are concrete ordinary
# Julia code; random generation is at the host boundary, never in an RK kernel.
mutable struct FilterCompiler
    observations::Tuple
    inputs::Tuple
    parameters::Tuple
    data::NamedTuple
    groups::NamedTuple
    states::Dict{Symbol,Tuple{Symbol,Int,Any}}
    shapes::Vector{Pair{Symbol,Tuple{Int,Int}}}
    sampled::Set{Symbol}
    observed::Set{Symbol}
    mod::Module
    operations::Vector{Any}
    provider_count::Int
    authored_names::Dict{Symbol,Symbol}
    series::Set{Symbol}
    rows::Set{Symbol}
end

_bare_series(c,ex::Symbol) = ex in c.series
_bare_series(c,ex) = ex isa Expr && ex.head !== :ref && any(a -> _bare_series(c,a),ex.args)

function _group_local(ex, g)
    g === nothing && return ex
    if Meta.isexpr(ex, :ref, 2) && Meta.isexpr(ex.args[1], :ref, 2) &&
            ex.args[1].args[2] == g
        return Expr(:ref, ex.args[1].args[1], _group_local(ex.args[2], g))
    end
    ex isa Expr || return ex
    return Expr(ex.head, (_group_local(a, g) for a in ex.args)...)
end

function _group_domain_value(c,ex,axis)
    ex isa Integer && return ex
    ex isa AbstractRange && return ex
    ex isa Symbol && haskey(c.data,ex) && return getproperty(c.data,ex)
    if Meta.isexpr(ex,:call)
        head,args = ex.args[1],ex.args[2:end]
        if head in (:eachindex,:length) && length(args)==1
            n = only(args)
            if n in (c.inputs...,c.observations...)
                return head === :eachindex ? axis : length(axis)
            end
            value = _group_domain_value(c,n,axis)
            value === nothing || return head === :eachindex ? eachindex(value) : length(value)
        elseif head === :(:) && length(args)==2
            lo,hi = map(a -> _group_domain_value(c,a,axis),args)
            lo isa Integer && hi isa Integer && return lo:hi
        end
    end
    return nothing
end

# Single-time-slice projection. Every carried read is either current or a
# statically bounded lag. Whole-trajectory work is not silently truncated.
function _value(c, ex; time=nothing, seed=false, group=nothing, scalarize=false)
    if ex isa Symbol
        if haskey(c.states, ex)
            scalarize || _fail("whole trajectory $ex is not a finite-state expression")
            key, depth, lane = c.states[ex]
            return _ref(_field(:_filter_next, key), depth, lane)
        end
        return ex
    end
    ex isa Expr || return ex
    if Meta.isexpr(ex,:call)
        head = ex.args[1]
        provider = head isa GlobalRef ? getglobal(head.mod,head.name) :
            head isa Symbol && isdefined(c.mod,head) ? getglobal(c.mod,head) : nothing
        if provider isa RK.KernelSpec
            c.provider_count += 1
            prefix = Symbol("kernel",c.provider_count)
            args = [_value(c,a;time,seed,group,scalarize) for a in ex.args[2:end]]
            operation! = op -> begin
                push!(c.operations,op)
                _ref(:_filter_operations,length(c.operations))
            end
            carry! = (i,initial) -> begin
                key = Symbol(group === nothing ? "" : string(group)*".",prefix,".",i)
                lanes = group === nothing ? 1 : length(getproperty(c.groups,group))
                push!(c.shapes,key=>(1,lanes))
                lane = something(group,1)
                old = _ref(_field(:_filter_carry,key),1,lane)
                dest = _ref(_field(:_filter_next,key),1,lane)
                (:(_filter_time == 1 ? $initial : $old),value -> Expr(:(=),dest,value))
            end
            return RK._stream_point_expression(provider,args,
                [_bare_series(c,a) for a in ex.args[2:end]],operation!,carry!)
        end
        if scalarize && _bare_series(c,ex)
            projected = RK._stream_design_expression(ex,a -> _bare_series(c,a),
                a -> RK._stream_is_row(a,c.rows),
                a -> _value(c,a;time,seed,group,scalarize))
            projected === nothing || return projected
            dotted = head isa Symbol && startswith(String(head),".")
            (dotted || head === :identity) || _fail(
                "whole-trajectory call $head needs an authored pointwise broadcast or finite scan")
        end
    end
    if ex.head in (:generator, :comprehension, :for, :while)
        _fail("iteration inside a filter expression needs a bounded authored state representation")
    end
    if Meta.isexpr(ex, :ref, 2) && ex.args[1] isa Symbol && haskey(c.states, ex.args[1])
        key, depth, lane = c.states[ex.args[1]]
        idx = ex.args[2]
        if seed
            idx isa Int || _fail("seed reads need literal indices")
            # Before this seed, the last entries hold the earlier seed values.
            old = _ref(_field(:_filter_carry, key), :($depth - _filter_time + 1 + $idx), lane)
            current = _ref(_field(:_filter_next,key),depth,lane)
            return :(_filter_time == $idx ? $current : $old)
        elseif idx == time
            return _ref(_field(:_filter_next, key), depth, lane)
        elseif Meta.isexpr(idx, :call, 3) && idx.args[1] == :- && idx.args[2] == time && idx.args[3] isa Int
            lag = idx.args[3]
            1 <= lag <= depth || _fail("lag exceeds retained seed depth")
            return _ref(_field(:_filter_carry, key), depth-lag+1, lane)
        end
        _fail("history read $ex has no fixed finite lag")
    end
    if Meta.isexpr(ex, :ref, 2) && ex.args[1] in (c.inputs..., c.observations...) && ex.args[2] == time
        return ex.args[1]
    end
    # Broadcast laws/pointwise arithmetic become one observation's scalar work.
    if scalarize && Meta.isexpr(ex, :., 2) && Meta.isexpr(ex.args[2], :tuple)
        return _call(_value(c, ex.args[1]; time, seed, group, scalarize),
            (_value(c, a; time, seed, group, scalarize) for a in ex.args[2].args)...)
    end
    if scalarize && Meta.isexpr(ex, :call) && ex.args[1] isa Symbol && startswith(String(ex.args[1]), ".")
        return _call(Symbol(String(ex.args[1])[2:end]),
            (_value(c, a; time, seed, group, scalarize) for a in ex.args[2:end])...)
    end
    return Expr(ex.head, (_value(c, a; time, seed, group, scalarize) for a in ex.args)...)
end

_student(nu, mu, sigma) = D.LocationScale(mu, sigma, D.TDist(nu))
_half(::Val{:normal}, sigma) = D.truncated(D.Normal(0, sigma); lower=0)
_half(::Val{:cauchy}, sigma) = D.truncated(D.Cauchy(0, sigma); lower=0)

function _law(c, rhs; kwargs...)
    rhs = _value(c, rhs; scalarize=true, kwargs...)
    Meta.isexpr(rhs, :call) || _fail("expected a distribution law, got $rhs")
    head = rhs.args[1]
    head === :Flat && _fail("an improper Flat law cannot generate a filtering transition")
    head === :restricted && _fail("restricted transitions are not normalized; use truncated for a transition law")
    head === :StudentT && return _call(_global(:_student), rhs.args[2:end]...)
    head in (:HalfNormal, :HalfCauchy) && return _call(_global(:_half),
        QuoteNode(Val(head === :HalfNormal ? :normal : :cauchy)), rhs.args[2:end]...)
    if head in (:truncated, :censored)
        return _call(GlobalRef(D, head), _law(c, rhs.args[2]; kwargs...), rhs.args[3:end]...)
    end
    if head isa Symbol && isdefined(D, head)
        return _call(GlobalRef(D, head), rhs.args[2:end]...)
    end
    return rhs # caller-owned normalized laws use the ordinary public protocol
end

function _site(c, lhs, rhs, sampled; group=nothing, time=nothing, seed=false)
    nm = lhs isa Symbol ? lhs : lhs.args[1]
    observed = nm in c.observations
    value = if sampled
        law = _law(c, rhs; time, seed, group)
        if observed
            push!(c.observed, nm)
            obs = _field(:_filter_observation, nm)
            group === nothing || (obs = _ref(obs, group))
            absent = haskey(c.states,nm) ? _call(GlobalRef(Random,:rand),:_filter_rng,law) : nothing
            return quote
                if hasproperty(_filter_observation,$(QuoteNode(nm))) && !ismissing($obs)
                    _filter_score += $(GlobalRef(PPL,:sampling_logdensity))($law,$obs)
                    $lhs = $obs
                else
                    $lhs = $absent
                end
            end
        else
            _call(GlobalRef(Random, :rand), :_filter_rng, law)
        end
    else
        _value(c, rhs; time, seed, group, scalarize=true)
    end
    return Expr(:(=), lhs, value)
end

function _scan!(c, ex, group)
    raw = _group_local(ex.args[end], group)
    scan = PPL.parse_scan_block(raw)
    PPL._scan_has_history(scan) && _fail("scan $(scan.label) reads arbitrary history; use full-history update_smc")
    any(k -> !(k isa Int), Iterators.flatten(values(PPL._scan_lags(scan.step, scan.states, scan.loopvar)))) &&
        _fail("filter lags must be literal integers")
    depth = scan.lo-1
    for name in scan.states
        public_name = get(c.authored_names,name,name)
        key = group === nothing ? public_name : Symbol(group, ".", public_name)
        any(p -> first(p) == key, c.shapes) && _fail("duplicate carried state $key")
        lanes = group === nothing ? 1 : length(getproperty(c.groups, group))
        push!(c.shapes, key => (depth, lanes))
        c.states[name] = (key, depth, something(group, 1))
        any(s -> s.target == name && s.kind === :sample, scan.step) && push!(c.sampled, key)
    end
    # Keep the original statement ordering, while each state owns only a lag window.
    setup = _stmts(raw)[1:end-1]
    branches = Any[]
    for index in 1:depth
        body = Any[]
        for st in setup
            sampled = Meta.isexpr(st, :call) && st.args[1] == :~
            lhs, rhs = sampled ? st.args[2:3] : st.args
            lhs.args[2] == index || continue
            key, _, lane = c.states[lhs.args[1]]
            dest = _ref(_field(:_filter_next, key), depth, lane)
            site = _site(c, lhs.args[1], rhs, sampled; group, time=scan.loopvar, seed=true)
            push!(body, site, Expr(:(=), dest, lhs.args[1]))
        end
        push!(branches, :(_filter_time == $index && begin $(body...); nothing end))
    end
    body = Any[]
    loop = last(_stmts(raw))
    for st in _stmts(loop.args[2])
        sampled = Meta.isexpr(st, :call) && st.args[1] == :~
        lhs, rhs = sampled ? st.args[2:3] : st.args
        if lhs isa Symbol
            push!(body, _site(c, lhs, rhs, sampled; group, time=scan.loopvar))
        else
            key, _, lane = c.states[lhs.args[1]]
            dest = _ref(_field(:_filter_next, key), depth, lane)
            push!(body, _site(c, lhs.args[1], rhs, sampled; group, time=scan.loopvar),
                Expr(:(=), dest, lhs.args[1]))
        end
    end
    return quote
        $(Expr(:block, [:( $(_field(:_filter_next,c.states[n][1]))[1:$(depth-1),$(c.states[n][3])] .=
            $(_field(:_filter_carry,c.states[n][1]))[2:$depth,$(c.states[n][3])]) for n in scan.states]...))
        $(scan.loopvar) = _filter_time
        if _filter_time < $(scan.lo)
            $(branches...)
        else
            $(body...)
        end
    end
end

function _compile_block!(c, block; group=nothing)
    stmts = _stmts(block)
    assignments = [st for st in stmts if Meta.isexpr(st,:(=),2) && st.args[1] isa Symbol]
    needed = Set{Symbol}()
    for st in stmts
        st in assignments && continue
        if Meta.isexpr(st,:call,3) && st.args[1] in (:~,Symbol(".~"))
            name = st.args[2] isa Symbol ? st.args[2] : st.args[2].args[1]
            name in c.observations || continue # fixed priors are not scored
        end
        union!(needed,PPL._expr_value_symbols(st))
    end
    changed = true
    while changed
        old = length(needed)
        for st in assignments
            st.args[1] in needed && !haskey(c.data,st.args[1]) &&
                union!(needed,PPL._expr_value_symbols(st.args[2]))
        end
        changed = old != length(needed)
    end
    filter!(st -> st ∉ assignments || st.args[1] in needed,stmts)
    # Batch horizon definitions (e.g. T=length(y[g])) are not stream inputs.
    horizons = Set{Symbol}()
    for st in stmts
        if _macro(st, :scan)
            s = PPL.parse_scan_block(_group_local(st.args[end], group))
            s.hi isa Symbol && push!(horizons, s.hi)
        end
    end
    result = Any[]
    # Ordinary RKPPL definitions are declarative, including hyperparameters
    # authored after a scan that consumes them. Hoist state-independent ones.
    scanned = Set{Symbol}()
    for st in stmts
        _macro(st, :scan) && union!(scanned, PPL.parse_scan_block(_group_local(st.args[end], group)).states)
    end
    union!(c.series,scanned)
    changed = true
    while changed
        changed = false
        for st in stmts
            if Meta.isexpr(st,:(=),2) && st.args[1] isa Symbol && !haskey(c.data,st.args[1]) &&
                    st.args[1] ∉ c.series && _bare_series(c,st.args[2])
                push!(c.series,st.args[1]); changed = true
            end
        end
    end
    changed = true
    while changed
        changed = false
        for st in assignments
            if st.args[1] in c.series && st.args[1] ∉ c.rows &&
                    RK._stream_is_row(st.args[2],c.rows)
                push!(c.rows,st.args[1]); changed = true
            end
        end
    end
    definitions = [st for st in stmts if Meta.isexpr(st, :(=), 2) && st.args[1] isa Symbol &&
        st.args[1] ∉ horizons && !haskey(c.data,st.args[1]) &&
        st.args[1] ∉ c.series]
    # Stable dependency ordering of invariant definitions, including chains
    # authored after their consumers. Streaming definitions stay with scans.
    pending = copy(definitions)
    empty!(definitions)
    while !isempty(pending)
        names = Set(st.args[1] for st in pending)
        index = findfirst(st -> isempty(intersect(PPL._all_symbols!(Set{Symbol}(),st.args[2]),names)),pending)
        index === nothing && _fail("cyclic invariant definitions")
        push!(definitions,popat!(pending,index))
    end
    for st in definitions
        push!(result, Expr(:(=), st.args[1], _value(c, st.args[2]; group, scalarize=true)))
    end
    for st in stmts
        st in definitions && continue
        if _macro(st, :scan)
            push!(result, _scan!(c, st, group))
        elseif _macro(st, :plate)
            group === nothing || _fail("nested plates are not supported by finite-state filtering")
            loop = st.args[end]
            Meta.isexpr(loop, :for) || _fail("expected @plate for")
            g = loop.args[1].args[1]
            hasproperty(c.groups, g) || _fail("supply fixed groups=(; $g=1:G) for this plate")
            getproperty(c.groups, g) == collect(1:length(getproperty(c.groups,g))) ||
                _fail("group indices must be contiguous 1:G; retain external labels in frozen data")
            axis = getproperty(c.groups,g)
            domain = _group_domain_value(c,loop.args[1].args[2],axis)
            domain !== nothing && collect(domain)==collect(axis) || _fail(
                "authored group domain must match groups.$g; use eachindex of a stream port or a fixed bound range")
            oldstates = copy(c.states)
            # A grouped alias loc[g] is one local value in this cell. Its
            # readers use that same local, not indexing into a scalar.
            localnames = Set{Symbol}()
            for item in _stmts(loop.args[2])
                Meta.isexpr(item,:(=),2) && Meta.isexpr(item.args[1],:ref,2) &&
                    item.args[1].args[2] == g && push!(localnames,item.args[1].args[1])
            end
            function localize(ex)
                Meta.isexpr(ex,:ref,2) && ex.args[1] in localnames && ex.args[2] == g && return ex.args[1]
                ex isa Expr ? Expr(ex.head,map(localize,ex.args)...) : ex
            end
            body = _compile_block!(c, localize(loop.args[2]); group=g)
            c.states = oldstates
            inputbindings = [Expr(:(=),n,_ref(_field(:_filter_inputs,n),g)) for n in c.inputs]
            push!(result, quote
                for $g in $(_field(:_filter_active,g))
                    _filter_time = $(_field(:_filter_times,g))[$g]+1
                    $(inputbindings...)
                    $body
                end
                _filter_time = _filter_global_time
            end)
        elseif Meta.isexpr(st, :(=), 2)
            (st.args[1] in horizons || haskey(c.data,st.args[1])) && continue
            lhs = st.args[1]
            if group !== nothing && Meta.isexpr(lhs, :ref, 2) && lhs.args[2] == group
                lhs = lhs.args[1]
            end
            lhs isa Symbol || _fail("only scalar definitions are supported outside scans")
            push!(result, Expr(:(=), lhs, _value(c, st.args[2]; group, scalarize=true)))
        elseif Meta.isexpr(st, :call, 3) && st.args[1] in (:~, Symbol(".~"))
            lhs, rhs = st.args[2:3]
            name = lhs isa Symbol ? lhs : Meta.isexpr(lhs, :ref, 2) ? lhs.args[1] : nothing
            if name in c.parameters || haskey(c.data,name)
                continue # calibrated parameters: neither draw nor multiply their prior
            end
            name in c.observations || _fail("static declaration $lhs needs a calibrated parameter value")
            push!(result, _site(c, name, rhs, true; group))
        else
            _fail("unsupported finite-state model statement: $st")
        end
    end
    return Expr(:block, result...)
end

function PPL.prepare_filter(model::RKPPLModel, bound::PPL.StructuralPlan;
        observations=(:y,), inputs=(), parameters::NamedTuple=(;), data::NamedTuple=(;),
        frozen::PPL.FrozenValues=bound.frozen, kwargs...)
    PPL.isbound(bound) || _fail("prepare_filter requires a bound plan")
    definitions,_ = PPL._definition_exprs(bound)
    paths = PPL._scope_name_paths(bound.submodel_scopes)
    value_inputs = PPL._bound_value_inputs(bound)
    function public_name(n)
        n in value_inputs && (n=Symbol(chopprefix(String(n),"_rkppl_value_")))
        return haskey(paths,n) ? Symbol(join(paths[n],".")) : n
    end
    excluded = Set{Symbol}((observations...,inputs...,keys(parameters)...,keys(model.fixed)...))
    # Recompute authored covariate definitions from new inputs. Only explicitly
    # frozen definitions substitute their fitted value; never freeze a design
    # merely because binding happened to evaluate it on training rows.
    fixed = (; (public_name(n)=>v for (n,v) in bound.columns
        if !haskey(definitions,n) && public_name(n) ∉ excluded)...)
    frozen_values = (; (public_name(n)=>v for (n,v) in pairs(frozen.values))...)
    return prepare_filter(model;observations,inputs,parameters,
        data=merge(fixed,data),frozen=PPL.FrozenValues(frozen_values,(;)),kwargs...)
end

function PPL.prepare_filter(model::RKPPLModel; model_id::AbstractString,
        observations=(:y,), inputs=(), parameters::NamedTuple=(;), data::NamedTuple=(;),
        drifting=(), groups::NamedTuple=(;), frozen::PPL.FrozenValues=PPL.FrozenValues())
    isempty(model_id) && _fail("model_id must identify source, captured code and frozen transforms")
    isempty(model.conditioned) || _fail("pass stream observations to filter_step, not a conditioned model")
    data = merge((; model.fixed...), data, frozen.values)
    allnames = (Tuple(observations)..., Tuple(inputs)..., keys(parameters)..., keys(data)...)
    length(unique(allnames)) == length(allnames) || _fail("observation, input, parameter and fixed data names must be disjoint")
    names = PPL._all_symbols!(Set{Symbol}(), model.ast)
    any(n -> startswith(String(n), "_filter_"), names) && _fail("model uses reserved _filter_ names")
    ast,scopes = PPL._expand_submodels(model.ast, Set{Symbol}(n for n in allnames if !occursin('.',String(n))),
        model.mod; rewrites=model.rewrites,with_scopes=true)
    paths = Dict(n => Symbol(join(path,".")) for (n,path) in PPL._scope_name_paths(scopes))
    private_names = Dict(v=>k for (k,v) in paths)
    parameter_names = Tuple(get(private_names,n,n) for n in keys(parameters))
    compiler_data = NamedTuple{Tuple(get(private_names,n,n) for n in keys(data))}(values(data))
    c = FilterCompiler(Tuple(observations), Tuple(inputs), parameter_names, compiler_data, groups,
        Dict{Symbol,Tuple{Symbol,Int,Any}}(), Pair{Symbol,Tuple{Int,Int}}[], Set{Symbol}(), Set{Symbol}(),
        model.mod,Any[],0,paths,Set{Symbol}((inputs...,observations...)),Set{Symbol}())
    body = _compile_block!(c, ast)
    # Inspect the actual projected code, after horizon removal and frozen
    # substitution. Training observations, input columns and dead constants
    # never enter the retained target merely because the binding held them.
    reads = PPL._expr_value_symbols(body)
    setdiff!(reads,PPL._lazy_value_locals!(Set{Symbol}(),body))
    data = (; (n=>v for (n,v) in pairs(data) if get(private_names,n,n) in reads)...)
    Set(observations) == c.observed || _fail("every observation must have an authored law")
    all(n -> n in c.sampled, drifting) || _fail("drifting parameters must name sampled carried states")
    bindings = Any[]
    append!(bindings,[Expr(:(=),get(private_names,n,n),_field(:_filter_parameters,n)) for n in keys(parameters)])
    append!(bindings,[Expr(:(=),get(private_names,n,n),_field(:_filter_data,n)) for n in keys(data)])
    append!(bindings,[Expr(:(=),n,_field(:_filter_inputs,n)) for n in inputs])
    program = :((_filter_rng, _filter_carry, _filter_parameters, _filter_data,
                 _filter_inputs, _filter_observation, _filter_global_time, _filter_operations,
                 _filter_times, _filter_active) -> begin
        $(bindings...)
        _filter_time = _filter_global_time
        _filter_next = $(GlobalRef(Base, :deepcopy))(_filter_carry)
        _filter_score = 0.0
        $body
        (_filter_next, Float64(_filter_score))
    end)
    fn = Core.eval(model.mod, program)
    prototype = NamedTuple{Tuple(first.(c.shapes))}(Tuple(zeros(last(s)...) for s in c.shapes))
    signature = deepcopy((; model_id=String(model_id), ast=PPL._program_identity(ast), observations=Tuple(observations),
        inputs=Tuple(inputs), parameters, data, drifting=Tuple(drifting), groups,
        shapes=c.shapes))
    return FilterTarget(fn, deepcopy(parameters), deepcopy(data), signature, prototype, program,Tuple(c.operations))
end

struct FilterStage
    time::Int
    predictive_logscore::Float64
    ess::Float64
    unique_parents::Int
    surviving_ancestors::Int
    parameter_ess::Float64
    condition_state_only::Bool
end

struct FilterState{T,C,P}
    target::T
    carries::Vector{C}
    ensemble::Vector{P}
    parameter_mass::Vector{Float64}
    log_parameter_mass::Vector{Float64}
    weights::Vector{Float64}
    particles::Int # per ensemble member
    rng::Xoshiro
    time::Int
    predictive_logscore::Float64
    log_normalizer::Float64
    ancestors::Vector{Int}
    lineage::Vector{Int}
    history::Vector{FilterStage}
    history_limit::Int
    group_times::NamedTuple
end

function _mass(weights, n)
    p = weights === nothing ? fill(1/n,n) : Vector{Float64}(weights)
    length(p) == n && all(x -> isfinite(x) && x >= 0, p) && isfinite(sum(p)) && sum(p) > 0 ||
        _fail("weights must be finite nonnegative masses with a positive total")
    return p ./ sum(p)
end

function PPL.initialize_filter(target::FilterTarget; rng::Xoshiro, particles::Integer=1000,
        ensemble=nothing, weights=nothing, history_limit::Integer=32)
    particles >= 1 || _fail("at least one state particle per parameter member is required")
    history_limit >= 0 || _fail("history_limit must be nonnegative")
    ps = ensemble === nothing ? [deepcopy(target.parameters)] : deepcopy(collect(ensemble))
    !isempty(ps) && all(p -> typeof(p) == typeof(target.parameters), ps) ||
        _fail("ensemble members need the same parameter names and types as target.parameters")
    mass = _mass(weights, length(ps))
    n = Int(particles)
    total = n * length(ps)
    carries = [deepcopy(target.prototype) for _ in 1:total]
    return FilterState(target, carries, ps, mass, log.(mass), repeat(mass ./ n; inner=n), n,
        copy(rng), 0, 0.0, 0.0, collect(1:total), collect(1:total), FilterStage[], Int(history_limit),
        map(g -> zeros(Int,length(g)),target.signature.groups))
end

mutable struct FilterIndex
    index::Int
    logweight::Float64
end
FilterIndex() = FilterIndex(0,0.0)

# All engine arrays have length bounded by N or 2, regardless of stream time.
# Static ensemble members retain separate conditional clouds, permitting a
# state-only warm-up without multiplying already calibrated parameter masses.
function _resample(logweights, seed)
    all(x -> isfinite(x) || x == -Inf, logweights) || _fail("NaN or +Inf observation log weight")
    maximum(logweights) > -Inf || _fail("all state particles have zero observation weight")
    n = length(logweights)
    starts, finishes = Ref(0), Ref(0)
    ancestors = Vector{Int}(undef,n)
    mutation! = function (out, rng, step, parent, scratch)
        if step == 1
            i = starts[] += 1
            out.index = i
            out.logweight = logweights[i]
        else
            i = finishes[] += 1
            ancestors[i] = parent.index
            out.index = i
            out.logweight = 0.0
        end
        nothing
    end
    model = SMC.SMCModel(mutation!, (t,p,s) -> p.logweight, 2, FilterIndex, Nothing)
    output = SMC.SMCIO{FilterIndex,Nothing}(n,2,1,false)
    task = @task begin
        Random.seed!(seed)
        SMC.smc!(model,output)
    end
    schedule(task)
    fetch(task)
    return ancestors, output.logZhats[end], n*output.esses[1]
end

function _logsum(x)
    m = maximum(x)
    m == -Inf && _fail("all parameter members have zero predictive weight")
    return m + log(sum(v -> exp(v-m), x))
end

function PPL.filter_step(state::FilterState, observation::NamedTuple;
        inputs::NamedTuple=(;), condition_state_only::Bool=false, groups=nothing)
    all(n -> n in state.target.signature.observations,keys(observation)) || _fail("unknown observation name")
    keys(inputs) == state.target.signature.inputs || _fail("input names differ from target")
    known = state.target.signature.groups
    if groups !== nothing && !isempty(known)
        globalcarry = any(p -> !any(g -> startswith(String(first(p)),string(g)*"."),keys(known)),
            state.target.signature.shapes)
        globalcarry && _fail("a model with global and grouped recurrences needs synchronized slices; " *
            "asynchronous local clocks would change its authored global-history indexing")
    end
    groups === nothing || all(g -> haskey(known,g),keys(groups)) || _fail("unknown group axis")
    active = NamedTuple{keys(known)}(Tuple(groups === nothing ? collect(v) :
        collect(Int,get(groups,g,Int[])) for (g,v) in pairs(known)))
    for (g,indices) in pairs(active)
        length(unique(indices)) == length(indices) && all(i -> i in getproperty(known,g),indices) ||
            _fail("duplicate or unknown active group")
    end
    rng = copy(state.rng)
    n, k = state.particles, length(state.ensemble)
    predicted = similar(state.carries)
    carries = similar(state.carries)
    ancestors = similar(state.ancestors)
    increments = Vector{Float64}(undef,k)
    esses = Vector{Float64}(undef,k)
    for member in 1:k
        offset = (member-1)*n
        scores = Vector{Float64}(undef,n)
        for j in 1:n
            i = offset+j
            carry, score = Base.invokelatest(state.target.advance, rng, state.carries[i],
                state.ensemble[member], state.target.data, inputs, observation, state.time+1,
                state.target.operations,state.group_times,active)
            all(a -> all(isfinite,a), carry) || _fail("non-finite recurrence state")
            predicted[i] = carry
            scores[j] = score
        end
        all(x -> isfinite(x) || x == -Inf,scores) || _fail("NaN or +Inf observation log weight")
        parents, increment, ess = if maximum(scores) == -Inf
            condition_state_only && state.parameter_mass[member] > 0 &&
                _fail("state-only warm-up has zero conditional likelihood for a calibrated member")
            (collect(1:n),-Inf,0.0)
        else
            _resample(scores,rand(rng,UInt64))
        end
        increments[member], esses[member] = increment, ess
        for j in 1:n
            i, parent = offset+j, offset+parents[j]
            ancestors[i] = parent
            carries[i] = deepcopy(predicted[parent])
        end
    end
    logmass = state.log_parameter_mass .+ increments
    score = _logsum(logmass)
    isfinite(score) || _fail("non-finite predictive log score")
    mass = condition_state_only ? copy(state.parameter_mass) : exp.(logmass .- score)
    nextlogmass = condition_state_only ? copy(state.log_parameter_mass) : logmass .- score
    total = state.log_normalizer + (condition_state_only ? 0.0 : score)
    isfinite(total) || _fail("log normalizer overflow")
    lineage = state.lineage[ancestors]
    stage = FilterStage(state.time+1, score, sum(state.parameter_mass .* esses),
        length(Set(ancestors)), length(Set(lineage)), inv(sum(abs2,mass)), condition_state_only)
    history = [state.history; stage]
    length(history) > state.history_limit && deleteat!(history,1:length(history)-state.history_limit)
    times = deepcopy(state.group_times)
    for (g,indices) in pairs(active)
        getproperty(times,g)[indices] .+= 1
    end
    return FilterState(state.target, carries, state.ensemble, mass, nextlogmass, repeat(mass ./ n; inner=n),
        n, rng, state.time+1, score, total, ancestors, lineage, history, state.history_limit,times)
end

function PPL.filter_stream(state::FilterState, records; steps::Integer=typemax(Int),
        stop=state->false, condition_state_only::Bool=false)
    steps >= 0 || _fail("steps must be nonnegative")
    iterstate = nothing
    first_record = true
    for _ in 1:steps
        stop(state) && break
        item = first_record ? iterate(records) : iterate(records, iterstate)
        item === nothing && break
        record, iterstate = item
        first_record = false
        state = filter_step(state, record.observation; inputs=get(record,:inputs,(;)),
            groups=get(record,:groups,nothing),condition_state_only)
    end
    return state
end

_versions() = (; julia=VERSION,rk=Base.pkgversion(RK),ppl=Base.pkgversion(PPL),smc=Base.pkgversion(SMC),distributions=Base.pkgversion(D))
_same(a,b) = PPL._same_program(a,b)
_same(a::Tuple,b::Tuple) = length(a)==length(b) && all(_same(x,y) for (x,y) in zip(a,b))
_same(a::NamedTuple,b::NamedTuple) = keys(a)==keys(b) && _same(values(a),values(b))
_same(a::AbstractVector,b::AbstractVector) = axes(a)==axes(b) && all(_same(x,y) for (x,y) in zip(a,b))
function PPL.filter_checkpoint(state::FilterState)
    return deepcopy((; schema_version=1, versions=_versions(), signature=state.target.signature,
        carries=state.carries, ensemble=state.ensemble, parameter_mass=state.parameter_mass,
        log_parameter_mass=state.log_parameter_mass,
        weights=state.weights, particles=state.particles, rng=state.rng, time=state.time,
        predictive_logscore=state.predictive_logscore, log_normalizer=state.log_normalizer,
        ancestors=state.ancestors, lineage=state.lineage, history=state.history,
        history_limit=state.history_limit,group_times=state.group_times))
end

function PPL.resume_filter(target::FilterTarget, checkpoint)
    c = deepcopy(checkpoint)
    c.schema_version == 1 && c.versions == _versions() || _fail("checkpoint schema or versions differ")
    _same(c.signature,target.signature) || _fail("checkpoint model, fixed data or configuration differs")
    n, k = c.particles, length(c.ensemble)
    n >= 1 && k >= 1 && length(c.carries) == n*k || _fail("invalid checkpoint cloud size")
    for carry in c.carries
        keys(carry) == keys(target.prototype) || _fail("checkpoint carry names differ")
        all(size(a) == size(b) && all(isfinite,a) for (a,b) in zip(carry,target.prototype)) ||
            _fail("invalid checkpoint carry shape/value")
    end
    all(p -> typeof(p) == typeof(target.parameters), c.ensemble) || _fail("checkpoint parameter schema differs")
    mass = _mass(c.parameter_mass,k)
    length(c.log_parameter_mass)==k && all(x -> isfinite(x) || x == -Inf,c.log_parameter_mass) &&
        isapprox(_logsum(c.log_parameter_mass),0.0;atol=1e-12) &&
        all(isapprox.(exp.(c.log_parameter_mass),c.parameter_mass;atol=1e-15)) ||
        _fail("invalid checkpoint log parameter masses")
    isapprox(sum(c.parameter_mass),1.0) && c.weights == repeat(c.parameter_mass ./ n;inner=n) ||
        _fail("invalid checkpoint weights")
    c.time >= 0 && isfinite(c.log_normalizer) && isfinite(c.predictive_logscore) || _fail("invalid checkpoint time/score")
    c.history_limit >= 0 && length(c.history) <= c.history_limit || _fail("unbounded checkpoint history")
    length(c.ancestors) == length(c.lineage) == n*k && all(i -> 1 <= i <= n*k,c.ancestors) &&
        all(i -> 1 <= i <= n*k,c.lineage) || _fail("invalid checkpoint ancestry")
    c.rng isa Xoshiro || _fail("invalid continuation RNG")
    keys(c.group_times) == keys(target.signature.groups) &&
        all(length(t) == length(g) && all(x -> 0 <= x <= c.time,t)
            for (t,g) in zip(c.group_times,target.signature.groups)) || _fail("invalid group clocks")
    return FilterState(target,c.carries,c.ensemble,c.parameter_mass,c.log_parameter_mass,c.weights,n,c.rng,c.time,
        c.predictive_logscore,c.log_normalizer,c.ancestors,c.lineage,c.history,c.history_limit,c.group_times)
end

end
