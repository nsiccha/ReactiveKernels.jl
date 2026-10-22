# ── Data-only definitions frozen by value ─────────────────────────────
#
# A model fits its preprocessing in the graph, from a fit-role copy of its
# data (`x_mean = fit_mean(x_fit)`), and a later design freezes that
# preprocessing by value: `frozen_values` reads the definitions' values off
# the fit's binding, and `bind_data(plan, design; frozen)` takes them in place
# of evaluating them. Freezing is partial evaluation: the program stays the
# one a binding with the fit-role data generates. A definition binding
# evaluates binds its frozen value as data; a definition RK preparation folds
# stays in the graph and preparation receives its frozen value as a HAVE
# value, so RK computes neither it nor anything only it reads. A data input
# that only frozen definitions read binds absent, typed as the program
# declares it, and nothing reads it.

"""
    frozen_values(bound, names) -> FrozenValues

The values of the data-only definitions `names` (a `Symbol` or a collection
of them) in the binding `bound`, as [`bind_data`](@ref)'s `frozen` keyword
takes them. A definition is data-only when it reads data alone, never a
parameter: one binding evaluates (a fitted mean, a level set) or one RK
preparation folds (a whole value a parameter-dependent call reads, such as an
HSGP interval), which `bound.columns` does not hold. Each is evaluated once,
from `bound`'s data, as binding evaluates it; a definition `bound` itself
froze keeps its frozen value. The result also records the type and size of
every data input those definitions read (never its values), so a binding
that omits such an input still generates the same program.

```julia
fit = bind_data(plan, (; x, x_fit = x, y))
state = frozen_values(fit, (:x_mean, :x_sd))
design = bind_data(plan, (; x = xnew); frozen = state)   # no x_fit
```
"""
function frozen_values(plan::StructuralPlan, names)
    isbound(plan) || throw(ContractValidationError(
        "[frozen] frozen_values reads a bound plan (bind_data first)"))
    want = sort!(unique!(names isa Symbol ? Symbol[names] : collect(Symbol, names)))
    exprs, vector_defs = _definition_exprs(plan)
    for nm in want
        _check_frozen_definition(plan, exprs, nm)
    end
    lookup = _definition_lookup(plan, plan.columns, exprs, vector_defs,
        Dict{Symbol,Any}(pairs(plan.frozen.values)))
    values = map(want) do nm
        v = try
            lookup(nm)
        catch e
            e isa ContractValidationError && rethrow()
            throw(ContractValidationError("[frozen] data definition $nm " *
                "failed to evaluate: " * sprint(showerror, e)))
        end
        v isa _Absent && throw(ContractValidationError("[frozen] data " *
            "definition $nm reads `$(_absent_source(v))`, which this binding " *
            "leaves absent, so it has no value to freeze"))
        v
    end
    inputs = Dict{Symbol,Any}()
    seen = Set{Symbol}()
    function reach(nm)
        nm in seen && return nothing
        push!(seen, nm)
        if haskey(exprs, nm)
            foreach(reach, _definition_reads(exprs[nm]))
        elseif haskey(plan.columns, nm)
            inputs[nm] = _frozen_input_info(plan.columns[nm])
        end
        return nothing
    end
    foreach(reach, want)
    order = sort!(collect(keys(inputs)))
    return FrozenValues(NamedTuple{Tuple(want)}(Tuple(values)),
        NamedTuple{Tuple(order)}(Tuple(inputs[k] for k in order)))
end

# Only a definition reading data alone can be frozen: one reading a
# parameter has no value apart from the parameter.
function _check_frozen_definition(plan::StructuralPlan, exprs::AbstractDict, nm::Symbol)
    haskey(exprs, nm) || _fail(:frozen, "the model defines no `$nm`, so " *
        "there is no definition to freeze")
    _reads_data_only(plan, nm, _all_name_set(plan), Set{Symbol}()) ||
        _fail(:frozen, "`$nm` reads a parameter: only a definition that " *
            "reads data alone can be frozen")
    return nothing
end

# The type a program declares for a data input (`_data_arg`) and its size.
_frozen_input_info(v::_AbsentNumber{T}) where {T} = (type = T, size = ())
_frozen_input_info(v::Number) = (type = typeof(v), size = ())
_frozen_input_info(v::AbstractArray{T,N}) where {T,N} = (type = Array{T,N}, size = size(v))

# An omitted frozen input's absent stand-in: the type and size `info`
# records, no values.
function _frozen_input_stub(name::Symbol, info)
    T, dims = info.type, Tuple(info.size)
    T isa Type && T <: AbstractArray && length(dims) == ndims(T) &&
        return _AbsentArray{eltype(T),ndims(T)}(name, dims)
    T isa Type && T <: Real && isempty(dims) && return _AbsentNumber{T}(name)
    _fail(:bind, "frozen input `$name` records type $(repr(T)) and size " *
        "$(repr(dims)), which no data value has")
end

# Validate `frozen` against `plan` and bind every input only frozen
# definitions read, and that `columns` omits, absent. Returns those inputs.
function _bind_frozen!(plan::StructuralPlan, columns::AbstractDict{Symbol},
        frozen::FrozenValues)
    stubs = Set{Symbol}()
    isempty(frozen) && return stubs
    exprs, _ = _definition_exprs(plan)
    names = Set{Symbol}(keys(frozen.values))
    for nm in sort!(collect(names))
        haskey(columns, nm) && _fail(:bind, "`$nm` is frozen and also " *
            "supplied as data — drop it from the data")
        _check_frozen_definition(plan, exprs, nm)
    end
    # The inputs the frozen definitions read, through definitions they read.
    reached = Set{Symbol}()
    function reach(nm)
        nm in reached && return nothing
        push!(reached, nm)
        haskey(exprs, nm) && foreach(reach, _definition_reads(exprs[nm]))
        return nothing
    end
    foreach(reach, names)
    live = _frozen_live_names(plan, names, Set{Symbol}(keys(frozen.inputs)))
    for (input, info) in pairs(frozen.inputs)
        # An input another value reads stays required, and an omitted one
        # fails exactly as it does without `frozen`.
        (input in reached && input ∉ live && !haskey(columns, input)) || continue
        columns[input] = _frozen_input_stub(input, info)
        push!(stubs, input)
    end
    return stubs
end

# The definitions and `inputs` some value other than the `frozen`
# definitions reads: every plan slot but the definition tables, then each
# definition a read value is, except a frozen one, whose value is given.
_frozen_live_names(plan::StructuralPlan, frozen::Set{Symbol}, inputs::Set{Symbol}) =
    _structural(plan, (:frozen_live, _sorted_names(frozen), _sorted_names(inputs));
            columns = :none) do
        exprs, _ = _definition_exprs(plan)
        candidates = union(Set{Symbol}(keys(exprs)), inputs)
        free = copy(candidates)
        for f in fieldnames(StructuralPlan)
            f in (:assignments, :derived, :columns, :roles, :n_obs,
                :submodel_scopes, :frozen) && continue
            _drop_held_names!(free, getfield(plan, f))
        end
        live = setdiff(candidates, free)
        queue = collect(live)
        while !isempty(queue)
            nm = pop!(queue)
            (haskey(exprs, nm) && nm ∉ frozen) || continue
            for s in _definition_reads(exprs[nm])
                s in live && continue
                push!(live, s)
                push!(queue, s)
            end
        end
        live
    end

# Frozen definitions the graph computes (RK preparation folds them): each is
# a HAVE value of every preparation, with its frozen value bound.
_frozen_graph_names(plan::StructuralPlan) =
    sort!([k for k in keys(plan.frozen.values) if !haskey(plan.columns, k)])

_frozen_graph_values(plan::StructuralPlan) = let names = _frozen_graph_names(plan)
    NamedTuple{Tuple(names)}(Tuple(plan.frozen.values[k] for k in names))
end
