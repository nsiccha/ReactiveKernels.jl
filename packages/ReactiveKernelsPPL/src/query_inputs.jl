# Runtime inputs: `prepare_query(…; inputs = (:x,))` prepares once and takes
# the named data values as call arguments, `q(u, x)`.
#
# A binding passes most data to the graph as it was given, but it also
# evaluates data definitions and value matrices from it (`w = f(x)`,
# `X = hcat(…)`). The graph reads those as inputs of their own, so a call
# that changes `x` evaluates them again from the call's `x`, exactly as
# binding does, and passes them beside it. Everything else the preparation
# holds stays the binding's: the call checks that what it changes keeps the
# type and size the binding gave the program.

# The per-call evaluation of what a binding computes from runtime inputs.
struct _DerivedInputs
    plan::StructuralPlan
    # The ports the call's values bind to, in call order.
    ports::Vector{Symbol}
    # Value matrices reading a runtime input, in binding order.
    matrices::Vector{DesignMatrix}
    # Data definitions reading a runtime input that binding evaluates, in
    # binding order; `columns` are those it binds as graph inputs.
    names::Vector{Symbol}
    columns::Set{Symbol}
    # The graph inputs evaluated per call, in the kernel's argument order.
    passed::Vector{Symbol}
    # The binding's value of each matrix and definition above.
    templates::Dict{Symbol,Any}
    # Every other value the definitions read: the binding's.
    base::Dict{Symbol,Any}
    exprs::Dict{Symbol,Any}
    vector_defs::Set{Symbol}
end

struct _InputQuery{K}
    kernel::K
    inputs::Vector{Symbol}
    # The binding's value of each input: its type and size are the program's.
    templates::Vector{Any}
    derived::_DerivedInputs
end

function Base.show(io::IO, q::_InputQuery)
    print(io, "prepared query with runtime inputs (",
        join(string.(q.inputs), ", "), "): call it as q(u, ",
        join(string.(q.inputs), ", "), ")")
end

function (q::_InputQuery)(u, args::Vararg{Any,N}) where {N}
    N == length(q.inputs) || throw(ArgumentError("[query] this query takes " *
        "the parameter vector and $(length(q.inputs)) runtime input" *
        (length(q.inputs) == 1 ? "" : "s") * " (" * _quoted_names(q.inputs) *
        "), got $N"))
    for i in 1:N
        _same_shape(q.templates[i], args[i]) || _input_mismatch(q.inputs[i],
            q.templates[i], args[i])
    end
    return q.kernel(u, args..., _derived_values(q.derived, args)...)
end

_quoted_names(names) = join(("`$n`" for n in names), ", ", " and ")

_value_summary(v::AbstractArray) = string(typeof(v), " of size ", size(v))
_value_summary(v) = string(typeof(v), " ", repr(v))

function _input_mismatch(name, template, v)
    throw(ArgumentError("[query] runtime input `$name` must have the type " *
        "and size of the value it was bound with, " * _value_summary(template) *
        "; got " * _value_summary(v) * ". The preparation holds the program " *
        "and every extent of that binding: bind the other type or size with " *
        "`bind_data` and prepare again."))
end

# The call may change floating-point values; everything else in a value (its
# type, the size of a dense array, integers, Booleans, labels, ranges) is
# what it was bound as.
function _same_shape(a, b)
    typeof(a) === typeof(b) || return false
    a isa AbstractFloat && return true
    if a isa Array
        size(a) == size(b) || return false
        eltype(a) <: AbstractFloat && return true
        return all(_same_shape(x, y) for (x, y) in zip(a, b))
    end
    a isa Union{Tuple,NamedTuple} &&
        return all(_same_shape(x, y) for (x, y) in zip(a, b))
    return isequal(a, b)
end

# What binding computes from the inputs `args`, evaluated in the binding's own
# order: value matrices first, then each data definition, every column one in
# its bound storage. Returns every value by name.
function _derived_memo(d::_DerivedInputs, args::Tuple)
    memo = Dict{Symbol,Any}()
    for (p, v) in zip(d.ports, args)
        memo[p] = v
    end
    for m in d.matrices
        memo[m.name] = _value_matrix(m, c -> haskey(memo, c) ? memo[c] : d.base[c])
    end
    isempty(d.names) && return memo
    lookup = _definition_lookup(d.plan, d.base, d.exprs, d.vector_defs, memo)
    for nm in d.names
        v = try
            lookup(nm)
        catch e
            e isa ContractValidationError && rethrow()
            throw(ContractValidationError("[query] data definition `$nm` " *
                "failed to evaluate on the call's runtime inputs: " *
                sprint(showerror, e)))
        end
        nm in d.columns && (memo[nm] = _bound_column_value(v))
    end
    return memo
end

# The graph inputs a call evaluates, in the kernel's argument order, after
# checking that everything evaluated keeps what the preparation holds.
function _derived_values(d::_DerivedInputs, args::Tuple)
    isempty(d.templates) && return ()
    memo = _derived_memo(d, args)
    for (nm, template) in d.templates
        _same_shape(template, memo[nm]) || throw(ArgumentError(
            "[query] `$nm`, which binding computes from the runtime " *
            "input" * (length(d.ports) == 1 ? "" : "s") * ", is " *
            _value_summary(memo[nm]) * " for these values, but this " *
            "preparation was made for " * _value_summary(template) *
            ". Only floating-point values may change between calls: " *
            "what indexes, counts or sizes the program stays as bound. " *
            "Bind these values with `bind_data` and prepare again."))
    end
    return Tuple(memo[nm] for nm in d.passed)
end

# ── Preparation ──────────────────────────────────────────────────────────

# The port a data value of `plan` binds to: a number the model reads as a
# model-level value travels under its internal input (`_bound_value_input`).
function _runtime_input_port(plan::StructuralPlan, name::Symbol)
    input = _bound_value_input(name)
    return input in _bound_value_inputs(plan) ? input : name
end

# The caller's data values of a binding, by the names the caller gave them:
# supplied, and neither computed by binding nor internal.
function _runtime_input_candidates(plan::StructuralPlan)
    computed = union(_defined_names(plan), (m.name for m in plan.matrices))
    names = Symbol[]
    for (k, v) in plan.columns
        _supplied(v) || continue
        s = String(k)
        if startswith(s, "_rkppl_value_")
            push!(names, Symbol(chopprefix(s, "_rkppl_value_")))
        elseif !(k in computed) && !startswith(s, "_rkppl_") && !startswith(s, "_ppl_")
            push!(names, k)
        end
    end
    return sort!(names)
end

# Names whose values binding checks as observations, or against them: every
# value the program observes (its `pointwise` fields: graph observations and
# conditioned declarations), the observations of statistical responses
# (conditioned values, caller-owned observations, derived responses), and the
# response slots read as data (weights, trials, scales, censoring bounds,
# cutpoints).
function _observation_checked_names(built, plan::StructuralPlan)
    names = Set{Symbol}(first(f) for f in last(_program_pointwise(built, plan)))
    slot!(s) = s isa Symbol ? push!(names, s) :
        s isa ScalePredictorRef ? push!(names, s.predictor) : names
    for r in plan.responses
        push!(names, r.response)
        union!(names, r.count_columns, r.extra_responses)
        foreach(slot!, (r.weights, r.trials, r.scale, r.nu, r.zi, r.discrimination,
            r.mi_jobs, r.evidence.lower, r.evidence.upper))
        foreach(slot!, r.mixture_scales)
        foreach(slot!, r.mixture_trials)
        r.interval === nothing || foreach(slot!, r.interval)
        r.thresholds === nothing || push!(names, _threshold_name(r.thresholds))
    end
    for p in plan.external_observations
        push!(names, p.name)
    end
    for n in plan.conditioned
        push!(names, n, _conditioned_input(n))
    end
    union!(names, plan.indexed_observations, _derived_response_names(plan))
    for s in _observed_selections(plan, plan.columns)
        push!(names, s.array)
    end
    for (k, role) in plan.roles
        role in (:data, :predictor) || push!(names, k)
    end
    return names
end

# Names binding reads as factor labels and maps to level codes.
function _label_names(plan::StructuralPlan)
    names = Set{Symbol}(_label_only_columns(plan))
    for m in plan.levelmaps
        push!(names, m.column, m.source)
    end
    return names
end

# A value the call may change: a floating-point number, or a dense array whose
# entries are such values.
_float_valued(v) = v isa AbstractFloat ||
    (v isa Array && (eltype(v) <: AbstractFloat ||
        (!isempty(v) && all(_float_valued, v))))

function _refuse_input(name::Symbol, why::AbstractString)
    throw(ContractValidationError("[query] runtime input `$name`: " * why))
end

# Validate the `inputs` of `plan` and plan what each call evaluates.
function _derived_inputs(built, plan::StructuralPlan, inputs)
    isempty(inputs) && throw(ContractValidationError(
        "[query] `inputs` names no data value"))
    for name in inputs
        name isa Symbol || throw(ContractValidationError("[query] `inputs` " *
            "holds names of data values (Symbols), got $(repr(name))"))
    end
    allunique(inputs) || throw(ContractValidationError(
        "[query] `inputs` names a data value twice: " * _quoted_names(inputs)))
    candidates = _runtime_input_candidates(plan)
    observed = _observation_checked_names(built, plan)
    labels = _label_names(plan)
    ports = Symbol[]
    for name in inputs
        port = _runtime_input_port(plan, name)
        (name in observed || port in observed) && _refuse_input(name,
            "binding checks it as an observation or against one (its " *
            "support, missing entries, weights, trials or bounds), and a call " *
            "would skip those checks. Rebind to change it.")
        name in candidates || _refuse_input(name, "this binding has no data " *
            "value `$name` that a call can supply. " * (isempty(candidates) ?
            "It binds no data values." :
            "Its data values are " * _quoted_names(candidates) * ".") *
            (haskey(plan.columns, name) || name in _defined_names(plan) ?
            " `$name` is computed by binding from them: name the data values it reads." : ""))
        (name in labels || port in labels) && _refuse_input(name,
            "binding maps its labels to level codes. Rebind to change it.")
        value = plan.columns[port]
        _float_valued(value) || _refuse_input(name, "it is bound as " *
            _value_summary(value) * ". A runtime input is a floating-point " *
            "number or a dense `Array` of them: binding checks other data value by " *
            "value (indices, counts, extents, labels). Bind a value read only " *
            "as a number as floating-point (`Float64.(x)`) to supply it per call.")
        push!(ports, port)
    end
    # What binding computes from the inputs: value matrices over them, and
    # every data definition reading one of those, through other definitions.
    exprs, vector_defs = _definition_exprs(plan)
    frozen = _sorted_names(keys(plan.frozen.values))
    matrices = DesignMatrix[m for m in plan.matrices
        if haskey(plan.columns, m.name) && any(c -> c in ports, m.columns)]
    roots = Set{Symbol}(ports)
    union!(roots, (m.name for m in matrices))
    affected = Set{Symbol}(nm for (nm, order) in _definition_read_orders(plan, frozen)
        if !(nm in frozen) && any(in(roots), @view order[2:end]))
    # Of those, binding evaluates the ones it binds as graph inputs and the
    # module data it evaluates for them. The graph computes the rest from
    # its inputs, as it does for every binding.
    columns = Set{Symbol}(nm for nm in affected
        if haskey(plan.columns, nm) && _supplied(plan.columns[nm]))
    names = sort!(collect(union(columns, intersect(affected, _module_data_names(plan)))))
    reached = sort!(collect(intersect(union(names, (m.name for m in matrices)), observed)))
    isempty(reached) || _refuse_input(first(inputs), "binding computes " *
        _quoted_names(reached) * " from the runtime inputs and checks " *
        (length(reached) == 1 ? "it" : "them") * " as an observation or " *
        "against one. Rebind to change " * (length(inputs) == 1 ? "it" : "them") * ".")
    replaced = union(Set{Symbol}(ports), affected, (m.name for m in matrices))
    base = Dict{Symbol,Any}(k => v for (k, v) in plan.columns if !(k in replaced))
    data = plan.data_values
    if data !== nothing
        lock(data.lock) do
            for (k, v) in data.values
                k in replaced || (base[k] = v)
            end
        end
    end
    passed = Symbol[m.name for m in matrices]
    append!(passed, (nm for nm in names if nm in columns))
    derived = _DerivedInputs(plan, ports, matrices, names, columns, passed,
        Dict{Symbol,Any}(), base, exprs, vector_defs)
    # The binding's own values, evaluated again: the per-call evaluation must
    # reproduce what binding bound before it may replace it.
    memo = try
        _derived_memo(derived, Tuple(plan.columns[p] for p in ports))
    catch e
        _refuse_input(first(inputs), "binding computes values from it that " *
            "cannot be evaluated per call: " * sprint(showerror, e))
    end
    for nm in passed
        v, bound = memo[nm], plan.columns[nm]
        isequal(v, bound) && typeof(v) === typeof(bound) || _refuse_input(
            first(inputs), "evaluating `$nm` again from the bound values gives " *
            _value_summary(v) * ", not the bound " * _value_summary(bound) *
            ", so a call could not reproduce what binding computes from it.")
    end
    for nm in Iterators.flatten((passed, names))
        derived.templates[nm] = memo[nm]
    end
    return derived
end

# Prepare `want` with the `inputs` of `plan` left as call arguments.
function _prepare_wants_inputs(built, plan::StructuralPlan, want, inputs;
        on_error = nothing, exemplars = nothing)
    derived = Base.invokelatest(_derived_inputs, built, plan, Any[x for x in inputs])
    inputs = Symbol[x for x in inputs]
    runtime = (derived.ports..., derived.passed...)
    have = (:unconstrained, runtime...,
        (n for n in _query_have(plan)[2:end] if !(n in runtime))...)
    full = _query_bound(plan)
    names = Tuple(k for k in keys(full) if !(k in runtime))
    bound = NamedTuple{names}(Tuple(full[k] for k in names))
    examples = exemplars === nothing ? nothing : (exemplars...,
        (plan.columns[p] for p in derived.ports)...,
        (derived.templates[nm] for nm in derived.passed)...)
    kernel = _absent_refusal(plan, built.spec, have, want) do
        Base.invokelatest(_prepare_cached, plan, built.spec; have,
            want, bound, on_error, exemplars = examples)
    end
    return _InputQuery(kernel, inputs,
        Any[plan.columns[p] for p in derived.ports], derived)
end
