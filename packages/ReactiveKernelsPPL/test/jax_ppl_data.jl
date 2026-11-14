# The raw data of a bound plan, as a `jax_source` module's `prepare_data`
# takes it. Shared by every RKPPL test file that checks JAX parity.
using ReactiveKernelsPPL

# The data the caller bound, in `prepare_data`'s order: every input is a
# name the caller wrote (a column, a scalar value), never a value only the
# Julia binder computes.
function jax_raw_data(plan, src)
    names = first.(only(f for f in src.functions if f.name == "prepare_data").inputs)
    # A value the program reads only through its bound copy, and the observed
    # value of a conditioned declaration, are passed by their authored names;
    # the plan holds them as `_rkppl_value_<name>` and
    # `_rkppl_conditioned_<name>`. A definition `prepare_data` takes as an
    # input (`UNTRANSLATED`) is the binding's own value of it (a submodel's
    # `r.g` is the input `r_g`).
    held = plan.data_values === nothing ? Symbol[] : collect(keys(plan.data_values.values))
    # A response with a presence mask is passed as the caller had it, its
    # absent entries `missing`: the plan stores it filled.
    function raw(n, v)
        mask = ReactiveKernelsPPL._observed_mask_name(Symbol(n))
        haskey(plan.columns, mask) && v isa AbstractArray || return v
        return map((x, p) -> p ? x : missing, v, plan.columns[mask])
    end
    function value(n)
        for k in (Symbol(n), Symbol("_rkppl_value_", n), Symbol("_rkppl_conditioned_", n))
            haskey(plan.columns, k) && return raw(n, plan.columns[k])
        end
        ReactiveKernelsPPL._bound_value(plan, plan.columns,
            only(h for h in held if replace(String(h), "." => "_") == n))
    end
    Tuple(value(n) for n in names)
end
