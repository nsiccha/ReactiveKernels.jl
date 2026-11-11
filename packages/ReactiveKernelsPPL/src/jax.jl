# JAX modules of built `@rkppl` programs (todo `1l3nni7`). The program is
# translated over the sampler boundary, the packed unconstrained parameters
# followed by the bound data columns (`_query_have`), so the Python module's
# functions take the same inputs as the native queries. Data stay inputs:
# one module serves every binding of the same program.

"""
    jax_source(built, plan; values = (), doc = nothing) -> JAXSource

A Python module for the `build_kernel` output `built`, evaluating the program
with `jax.numpy` over the sampler boundary: the packed unconstrained
parameters, then the prepared data. Its functions:

- `prepare_data(data...)`: the functions' data inputs, computed in host NumPy
  from the data the model is bound with, by the names the model reads them
  (sorted, or by name), including the data-only definitions `bind_data`
  evaluates; frozen definitions are inputs too;

- `log_density(unconstrained, data...)`: the `:sampler` value (prior,
  likelihood and log Jacobian), the log density a NumPyro or BlackJAX
  sampler targets as `-log_density` potential;
- `log_prior`, `log_likelihood` and `log_jacobian`: its parts;
- `constrain(unconstrained, data...)`: a `dict` of every declared
  parameter's constrained value (an empty parameter has none), plus each value
  named in `values` (a definition such as `:mu`).

Module constants describe the inputs: `DIMENSION` (the length of
`unconstrained`), `PARAMETERS` (each parameter's 0-based `start`/`stop` in
`unconstrained`, its transform and dimensions), `INPUTS` (the parameter names
in order), `DATA` (each `prepare_data` parameter's dtype and rank) and
`UNTRANSLATED` (each data-only definition `prepare_data` takes as an input
instead of computing it, with the reason; `{}` when none).

`plan` is any binding that generates the same program as `built`, as for
[`prepare_query`](@ref); its values only type the inputs. Data arrays keep
Julia's dimension order and integer index data stay 1-based. See
`ReactiveKernels.jax_source` for the translation.
"""
function ReactiveKernels.jax_source(built::NamedTuple, plan::StructuralPlan;
                                    values = (), doc = nothing)
    built = _binding_build(built, plan)
    isbound(plan) || throw(ContractValidationError(
        "[jax] jax_source requires a bound plan (bind_data first)"))
    _check_built_program(built, plan)
    have = _query_have(plan)
    data = _query_bound(plan)
    types = merge((; unconstrained = Vector{Float64}),
                  NamedTuple{keys(data)}(map(typeof, Tuple(data))))
    # Every parameter the program has a value of, also an empty one whose size
    # reads data (`b[axes(X, 2)]` with no columns), so one module serves every
    # binding; a structurally empty one (the thresholds of a two-category
    # ordinal response) has no value in the program.
    parameters = [entry.name for entry in built.layout.entries
                  if entry.size > 0 || haskey(built.spec.ports, entry.name)]
    returned = Symbol[parameters; collect(Symbol, values)]
    # `constrain` returns what native `constrain(layout, u)` does: a value in
    # a submodel's scope (`lv.sigma`) nested under the submodel (`lv`).
    paths = [get(built.layout.name_paths, name, (name,)) for name in returned]
    constrained = _jax_nested_want(paths, returned)
    functions = (log_density = workflow_wants(:sampler),
                 log_prior = workflow_wants(:prior),
                 log_likelihood = workflow_wants(:likelihood),
                 log_jacobian = workflow_wants(:log_jacobian),
                 constrain = constrained)
    # The layout's data-derived positions and axis lengths (`_layout_reads`):
    # `prepare_data` returns them, and `dimension`/`parameters` read them, so
    # the module states no binding's sizes. A layout that reads no data is
    # also stated as the constants `DIMENSION` and `PARAMETERS`.
    sizes = _jax_layout_names(built.layout)
    data_meta = (DATA = [String(name) => (dtype = ReactiveKernels._jax_dtype_name(typeof(value)),
                                          ndim = value isa AbstractArray ? ndims(value) : 0,
                                          ragged = ReactiveKernels._jax_ragged(typeof(value)))
                         for (name, value) in pairs(data)],)
    constants = !isempty(sizes) ? data_meta : (
        DIMENSION = built.layout.total,
        PARAMETERS = [String(entry.name) => (
            start = entry.offset - 1, stop = entry.offset - 1 + entry.size,
            transform = String(entry.transform), dims = Tuple(entry.dims))
            for entry in built.layout.entries],
        data_meta...)
    definitions = _jax_data_definitions(plan, keys(data))
    # The authored value behind a bound copy only the program reads
    # (`_rkppl_value_r => :r`): the caller passes it, and `jax_source` types
    # that parameter by this binding's value.
    for (nm, ex) in definitions
        ex isa Symbol && !haskey(data, ex) && haskey(data, nm) &&
            (data = merge(data, NamedTuple{(ex,)}((data[nm],))))
    end
    # A definition that is no data value of the program (`g = prep(labels)`,
    # read as `g.n`) and has no host translation is an input of `prepare_data`
    # in its place: the binding's own value types it.
    for (nm, _) in definitions
        haskey(data, nm) || !_holds_bound_value(plan, plan.columns, nm) ||
            (data = merge(data, NamedTuple{(nm,)}((_bound_value(plan, plan.columns, nm),))))
    end
    defined = Set{Symbol}(first.(definitions))
    # The model's own docstrings (`rkppl_documentation`), by author path.
    authored = rkppl_documentation(plan)
    documented(path) = join((d.markdown for d in authored if d.path == path), "\n")
    entry(path) = (text = documented(path); "- `" * join(String.(path), ".") * "`" *
                   (isempty(text) ? "" : ": " * replace(strip(text), "\n" => "\n  ")))
    model_docs = isempty(authored) ? "" : "\n\nModel documentation:\n\n" *
        join(("- `" * join(String.(d.path), ".") * "`: " * replace(strip(d.markdown), "\n" => "\n  ")
              for d in authored if !isempty(d.path)), "\n")
    module_doc = doc === nothing ?
        "Generated by ReactiveKernelsPPL.jax_source from a built @rkppl program; " *
        "do not edit.\n\nData: " *
        join(string.(_jax_data_name.(k for k in keys(data) if !(k in defined))), ", ") * "." *
        model_docs : doc
    over = " at `unconstrained`, the packed unconstrained parameters (length " *
        "`dimension(prepared)`), for the data `prepare_data` returned: `f(u, **prepared)`."
    function_docs = (
        log_density = "The log density a sampler targets" * over *
            "\n\nThe sum of `log_prior`, `log_likelihood` and `log_jacobian`.",
        log_prior = "The log prior density of the constrained parameters" * over,
        log_likelihood = "The log likelihood of the observations" * over,
        log_jacobian = "The log Jacobian of the constraining transforms" * over,
        constrain = "The constrained value of each parameter, and of each requested value," *
            over * " Returns a dict, a submodel's values nested under its name:\n\n" *
            join(map(entry, paths), "\n"))
    src = _absent_refusal(plan, built.spec, have, :posterior) do
        jax_source(built.spec; have, functions, types, constants, doc = module_doc, data,
                   definitions, keep = Tuple(sizes), function_docs)
    end
    _jax_with_layout(src, built.layout)
end

# A nested `NamedTuple` of the program's `names`, keyed by their author
# `paths` (`(:lv, :sigma)` under `lv`), in first-appearance order.
function _jax_nested_want(paths, names)
    keys_ = Symbol[]
    groups = Dict{Symbol,Vector{Int}}()
    for (i, p) in enumerate(paths)
        haskey(groups, first(p)) || (push!(keys_, first(p)); groups[first(p)] = Int[])
        push!(groups[first(p)], i)
    end
    vals = map(keys_) do k
        idx = groups[k]
        length(idx) == 1 && length(paths[only(idx)]) == 1 ? names[only(idx)] :
            _jax_nested_want([paths[i][2:end] for i in idx], names[idx])
    end
    NamedTuple{Tuple(keys_)}(Tuple(vals))
end

# The names of the data values a layout reads its positions and axis lengths
# from (`LayoutEntry.positions`, `.axes`; literals need none).
function _jax_layout_names(layout::LayoutTable)
    names = Symbol[]
    for e in layout.entries
        e.positions === nothing || append!(names, (x for x in e.positions if x isa Symbol))
        e.axes === nothing || append!(names, (x for x in e.axes if x isa Symbol))
    end
    unique!(names)
end

# `dimension(prepared)` and `parameters(prepared)`: the length of
# `unconstrained` and each parameter's 0-based `start`/`stop`, transform and
# dimensions, for the data `prepare_data` returned.
function _jax_with_layout(src::ReactiveKernels.JAXSource, layout::LayoutTable)
    # A layout name in a submodel's scope (`_ppl_last_b_g.z`) has another key.
    function read(x)
        x isa Int && return string(x)
        haskey(src.prepared, x) || throw(ContractValidationError(
            "[jax] the layout reads `$x`, which `prepare_data` does not return"))
        "int(prepared[\"" * src.prepared[x] * "\"])"
    end
    start(e) = e.positions === nothing ? string(e.offset - 1) :
        e.positions[1] isa Int ? string(e.positions[1] - 1) : read(e.positions[1]) * " - 1"
    stop(e) = e.positions === nothing ? string(e.offset - 1 + e.size) : read(e.positions[2])
    tuple(xs) = isempty(xs) ? "()" : "(" * join(xs, ", ") * (length(xs) == 1 ? ",)" : ")")
    dims(e) = e.axes === nothing ? tuple(string.(e.dims)) : tuple(map(read, e.axes))
    entries = ["\"" * String(e.name) * "\": {\"start\": " * start(e) * ", \"stop\": " * stop(e) *
               ", \"transform\": \"" * String(e.transform) * "\", \"dims\": " * dims(e) * "}"
               for e in layout.entries]
    total = isempty(layout.entries) ? "0" : stop(last(layout.entries))
    text = """


def dimension(prepared: dict) -> int:
    \"\"\"The length of `unconstrained` for the data `prepared` (what `prepare_data` returned).\"\"\"
    return $total


def parameters(prepared: dict) -> dict:
    \"\"\"Each parameter's 0-based `start`/`stop` in `unconstrained`, its transform and dimensions,
    for the data `prepared` (what `prepare_data` returned).\"\"\"
    return {$(join(entries, ", "))}
"""
    ReactiveKernels.JAXSource(src.text * text,
        [src.functions;
         ReactiveKernels.JAXFunction("dimension", ["prepared" => Dict], ["dimension" => Int]);
         ReactiveKernels.JAXFunction("parameters", ["prepared" => Dict], ["parameters" => Dict])],
        src.prepared)
end

# The data-only definitions `bind_data` evaluated, as `prepare_data`
# computes them on the host from the data the caller binds (decision of the
# downstream JAX handoff: a Python caller rebinds raw data without Julia). A scalar
# data value `s` reaches the program as the definition
# `s = _bound_value(_rkppl_value_s)`: the caller passes `s`, and
# `_rkppl_value_s` is that value; likewise the observed value of a
# conditioned declaration `y`, `_rkppl_conditioned_y`, is the `y` the caller
# passes. A derived definition whose binding guards
# missing observations (a presence mask spliced from the data) is not
# translated: `prepare_data` takes it as an input, listed in `UNTRANSLATED`.
# Frozen definitions (`frozen_values`) stay inputs, as their values come from
# another binding.
function _jax_data_definitions(plan::StructuralPlan, names)
    exprs, derived = _definition_exprs(plan)
    available = Set{Symbol}(names)
    found = Dict{Symbol,Any}()
    function definition(nm)
        ex = exprs[nm]
        nm in derived && _guard_missing_observation_argument(nm, ex, plan, plan.columns) !== ex &&
            return ReactiveKernels.JAXEmissionError(
                "the binding guards the missing observations of `$nm` with a presence " *
                "mask computed from the data: `prepare_data` takes `$nm` as an input")
        ex
    end
    for nm in available
        str = String(nm)
        if startswith(str, "_rkppl_value_")
            # The bound copy of a data value is that value, which the caller
            # passes by its authored name, also when the program reads it
            # only through the copy (`v = a .* r`).
            authored = Symbol(chopprefix(str, "_rkppl_value_"))
            _is_bound_value_call(get(exprs, authored, nothing)) && (found[nm] = authored)
        elseif startswith(str, "_rkppl_conditioned_")
            # The observed value of a conditioned declaration
            # (`y ~ LogDensity(f, …)` bound `| (; y)`), which the program
            # reads as `y = _rkppl_conditioned_y`: the caller passes it by
            # its authored name, as it passes a response.
            authored = Symbol(chopprefix(str, "_rkppl_conditioned_"))
            authored in _conditioned_inputs(plan) && _conditioned_input(authored) === nm &&
                !(authored in available) && (found[nm] = authored)
        elseif haskey(exprs, nm) && !_is_bound_value_call(exprs[nm])
            found[nm] = definition(nm)
        end
    end
    # The data-only values the program computes from a definition and the
    # binding holds no column of (`g = prep(labels)`, whose fields `g.n` and
    # `g.pos` the program reads): when the program's recipe of one has no
    # translation, `jax_source` takes the value from its definition instead
    # (and `prepare_data` drops one nothing reads). A frozen value is never
    # recomputed from its expression.
    held = plan.data_values === nothing ? Symbol[] :
        lock(() -> collect(Symbol, keys(plan.data_values.values)), plan.data_values.lock)
    for nm in held
        nm in available || haskey(found, nm) || haskey(plan.frozen.values, nm) ||
            !haskey(exprs, nm) || _is_bound_value_call(exprs[nm]) ||
            (found[nm] = definition(nm))
    end
    # The definitions those read that are not data values of the program
    # (`angle = x ./ period` read only by a definition `cos.(angle)`): host
    # locals of `prepare_data`, so a name never resolves to a global function.
    queue = collect(keys(found))
    while !isempty(queue)
        ex = found[pop!(queue)]
        ex isa ReactiveKernels.JAXEmissionError && continue
        for r in (ex isa Symbol ? (ex,) : _definition_reads(ex))
            (r in available || haskey(found, r) || !haskey(exprs, r)) && continue
            _is_bound_value_call(exprs[r]) && continue
            found[r] = definition(r)
            push!(queue, r)
        end
    end
    # Dependency order: a definition after the definitions it reads.
    order = Symbol[]
    state = Dict{Symbol,Int}()
    function visit(nm)
        get(state, nm, 0) == 2 && return
        get(state, nm, 0) == 1 && throw(ContractValidationError(
            "[jax] the data definitions of `$nm` read each other"))
        state[nm] = 1
        ex = found[nm]
        reads = ex isa Symbol ? Set{Symbol}([ex]) :
            ex isa ReactiveKernels.JAXEmissionError ? Set{Symbol}() : _definition_reads(ex)
        for r in sort!(collect(reads))
            haskey(found, r) && visit(r)
        end
        state[nm] = 2
        push!(order, nm)
    end
    foreach(visit, sort!(collect(keys(found))))
    Pair{Symbol,Any}[nm => found[nm] for nm in order]
end
_jax_data_name(nm::Symbol) = nm

# --- translations of RKPPL's own functions ---------------------------------------

# A law constructed in a cell (`y[g] ~ MvNormal(mu, Sigma)` lowers to
# `sampling_logdensity(LogDensity(f, args...), y[g])`): a value whose only use
# is its log density, `f(y[g], args...)`, translated where it is read.
struct _JaxLogDensity
    f::Any
    args::Vector{ReactiveKernels._PyVal}
end
ReactiveKernels._jax_call_positional(ctx::ReactiveKernels._JaxContext, ::Type{LogDensity},
        args::Vector{ReactiveKernels._PyVal}) =
    ReactiveKernels._PyVal("None", _JaxLogDensity(ReactiveKernels._jax_function_of(args[1]),
                                                  args[2:end]), true)
function ReactiveKernels._jax_function_call(ctx::ReactiveKernels._JaxContext,
        ::typeof(sampling_logdensity), args::Vector{ReactiveKernels._PyVal})
    d = args[1].T
    d isa _JaxLogDensity && length(args) == 2 ||
        return ReactiveKernels._jax_traceable_or_lowered(ctx, sampling_logdensity, args)
    ReactiveKernels._jax_call(ctx, d.f, ReactiveKernels._PyVal[args[2]; d.args])
end

# A multivariate normal log density through the lower Cholesky factor, with
# a validity flag the caller checks at run time, as native execution checks
# the covariance. `n` is the count of entries a padded (ragged) vector holds,
# or `None`: the padding then reads as an identity block with zero residual,
# which adds nothing to the density.
const _JAX_MVNORMAL = """
def _rkppl_mvnormal_logpdf(x, mu, S, n, factor):
    d = x - mu
    k = d.shape[0]
    if n is not None:
        m = jnp.arange(k) < n
        d = jnp.where(m, d, 0.0)
        S = jnp.where(m[:, None] & m[None, :], S, jnp.eye(k))
    else:
        n = k
    L = jnp.tril(S) if factor else jnp.linalg.cholesky(S)
    diag = jnp.diagonal(L)
    ok = jnp.all(jnp.isfinite(L)) & jnp.all(diag > 0)
    safe = jnp.where(ok, L, jnp.eye(k))
    z = jax.scipy.linalg.solve_triangular(safe, d, lower=True)
    value = (-0.5 * jnp.dot(z, z) - jnp.sum(jnp.log(jnp.diagonal(safe)))
             - 0.5 * n * jnp.log(2 * jnp.pi))
    return value, ok
"""
function _jax_mvnormal(ctx::ReactiveKernels._JaxContext, args, factor::Bool, what)
    length(args) == 3 && all(a -> ReactiveKernels._jax_is_array(a.T), args) ||
        return nothing
    x, mu, S = args
    ctx.host === nothing || ReactiveKernels._jax_error("`", what, "` in host code",
                                                        ReactiveKernels._jax_where(ctx))
    ReactiveKernels._jax_helper!(ctx, "_rkppl_mvnormal_logpdf", _JAX_MVNORMAL)
    n = something(findfirst(!isempty, (x.extent, mu.extent, S.extent)), 0)
    count = n == 0 ? "None" : (x.extent, mu.extent, S.extent)[n]
    pair = ReactiveKernels._jax_bind!(ctx, :mvnormal, ReactiveKernels._PyVal(
        "_rkppl_mvnormal_logpdf(" * x.code * ", " * mu.code * ", " * S.code * ", " * count *
        ", " * (factor ? "True" : "False") * ")", Tuple{Float64,Bool}))
    ReactiveKernels._jax_call(ctx, ReactiveKernels._runtime_check,
        [ReactiveKernels._PyVal(pair.code * "[1]", Bool),
         ReactiveKernels._PyVal(repr(what * ": the " * (factor ? "factor" : "covariance") *
                                     " is not positive definite"), DomainError)])
    ReactiveKernels._PyVal(pair.code * "[0]", Float64)
end
function ReactiveKernels._jax_function_call(ctx::ReactiveKernels._JaxContext,
        ::typeof(_ppl_mvnormal_logdensity), args::Vector{ReactiveKernels._PyVal})
    v = _jax_mvnormal(ctx, args, false, "MvNormal")
    v === nothing ? ReactiveKernels._jax_traceable_or_lowered(ctx, _ppl_mvnormal_logdensity, args) : v
end
function ReactiveKernels._jax_function_call(ctx::ReactiveKernels._JaxContext,
        ::typeof(_ppl_mvnormal_cholesky_logdensity), args::Vector{ReactiveKernels._PyVal})
    v = _jax_mvnormal(ctx, args, true, "MvNormalCholesky")
    v === nothing ?
        ReactiveKernels._jax_traceable_or_lowered(ctx, _ppl_mvnormal_cholesky_logdensity, args) : v
end

ReactiveKernels._jax_function_call(ctx::ReactiveKernels._JaxContext,
        ::Union{typeof(_bound_value),typeof(_bound_array_value)},
        args::Vector{ReactiveKernels._PyVal}) = only(args)

ReactiveKernels._jax_elementwise(::typeof(PPLGeneratedModels._ppl_logistic)) = "jax.nn.sigmoid"
ReactiveKernels._jax_elementwise(::typeof(PPLGeneratedModels._ppl_normcdf)) = "jsp.ndtr"

# The packed-coordinate transforms, as vector expressions (`layout.jl`). `lo`
# and `hi` are 1-based literal bounds of the block.
const _JAX_ORDERED = """
def _rk_ordered_constrain(u, lo, hi):
    # RKPPL ordered transform: y[1] = u[lo], y[i] = y[i - 1] + exp(u[lo + i - 1]).
    v = u[lo - 1:hi]
    return jnp.cumsum(jnp.concatenate([v[:1], jnp.exp(v[1:])]))
"""
const _JAX_SIMPLEX = """
def _rk_simplex_constrain_logjac(u, lo, hi):
    # RKPPL stick-breaking simplex of u[lo:hi] and its log-Jacobian.
    v = u[lo - 1:hi]
    K = hi - lo + 2
    z = 1.0 / (1.0 + jnp.exp(-(v + jnp.log(K - jnp.arange(1, K)))))
    l = jnp.log1p(-z)
    lr = jnp.cumsum(jnp.concatenate([jnp.zeros(1), l]))
    return (jnp.exp(lr) * jnp.concatenate([z, jnp.ones(1)]),
            jnp.sum(lr[:K - 1] + jnp.log(z) + l))
"""
function ReactiveKernels._jax_function_call(ctx::ReactiveKernels._JaxContext,
        ::typeof(_ordered_constrain_kernel), args::Vector{ReactiveKernels._PyVal})
    name = ReactiveKernels._jax_helper!(ctx, "_rk_ordered_constrain", _JAX_ORDERED)
    ReactiveKernels._PyVal(name * "(" * ReactiveKernels._jax_args(args) * ")", Vector{Float64})
end
function ReactiveKernels._jax_function_call(ctx::ReactiveKernels._JaxContext,
        ::typeof(_simplex_constrain_logjac_kernel), args::Vector{ReactiveKernels._PyVal})
    name = ReactiveKernels._jax_helper!(ctx, "_rk_simplex_constrain_logjac", _JAX_SIMPLEX)
    ReactiveKernels._PyVal(name * "(" * ReactiveKernels._jax_args(args) * ")",
                           Tuple{Vector{Float64},Float64})
end

ReactiveKernels._jax_function_call(ctx::ReactiveKernels._JaxContext,
        ::typeof(_ppl_range_values), args::Vector{ReactiveKernels._PyVal}) =
    ReactiveKernels._jax_is_array(args[1].T) ?
    ReactiveKernels._jax_index_values(ctx, args[1], args[2:end]) : args[1]

# The labels of a level axis (`_ppl_levels`, `_ppl_unique`, `_ppl_values`;
# `preprocessing.jl`), as host NumPy over a plain column: a `jax_source`
# module's `prepare_data` computes them from the column it is given, and a
# categorical pool's own order does not reach a JAX module. `_rk_levels` is
# `DataAPI.levels` of a plain vector: the distinct values in sorted order,
# missing (`None`) left out.
const _JAX_LEVELS = """
def _rk_levels(x):
    arr = np.asarray(x)
    values = sorted(set(v for v in arr.tolist() if v is not None))
    return np.array(values, dtype=object if arr.dtype == object else arr.dtype)
"""
const _JAX_UNIQUE = """
def _rk_unique(x):
    arr = np.asarray(x)
    return np.array(list(dict.fromkeys(arr.tolist())),
                    dtype=object if arr.dtype == object else arr.dtype)
"""
for (f, helper, code) in ((_ppl_levels, "_rk_levels", _JAX_LEVELS),
                          (_ppl_unique, "_rk_unique", _JAX_UNIQUE))
    @eval function ReactiveKernels._jax_function_call(ctx::ReactiveKernels._JaxContext,
            ::typeof($f), args::Vector{ReactiveKernels._PyVal})
        name = ReactiveKernels._jax_helper!(ctx, $helper, $code)
        ReactiveKernels._PyVal(name * "(" * ReactiveKernels._jax_args(args) * ")",
                               only(args).T, only(args).static)
    end
end
ReactiveKernels._jax_function_call(ctx::ReactiveKernels._JaxContext,
        ::typeof(_ppl_values), args::Vector{ReactiveKernels._PyVal}) =
    ReactiveKernels._PyVal("np.array(" * only(args).code * ")", only(args).T,
                           only(args).static)

# `_ppl_rows(x, n)`: a number filled over `n` rows, or a vector broadcast
# against them, as Julia's `ones(n) .* x` (a vector of the rows' length is
# itself; a mismatched one fails the broadcast while JAX traces).
function ReactiveKernels._jax_function_call(ctx::ReactiveKernels._JaxContext,
        ::typeof(_ppl_rows), args::Vector{ReactiveKernels._PyVal})
    x, n = args
    np = ctx.host === nothing ? "jnp" : "np"
    code = ReactiveKernels._jax_is_array(x.T) ?
        np * ".ones(" * n.code * ") * " * ReactiveKernels._py_atom(x.code) :
        np * ".full((" * n.code * ",), " * x.code * ", dtype=" * np * ".float64)"
    ReactiveKernels._PyVal(code, Vector{Float64}, false, x.extent)
end

# A cell scan's step count (`_ppl_cell_scan_nsteps`): the steps after its `m`
# seeds, none when the bound is below them. Its bound is an integer by type.
function ReactiveKernels._jax_function_call(ctx::ReactiveKernels._JaxContext,
        ::typeof(_ppl_cell_scan_nsteps), args::Vector{ReactiveKernels._PyVal})
    T, m = args
    T.T isa Type && T.T <: Integer || ReactiveKernels._jax_error(
        "a cell scan's upper bound of type `", T.T, "` has no JAX translation",
        ReactiveKernels._jax_where(ctx))
    steps = ReactiveKernels._py_infix("-", [T.code, m.code])
    ReactiveKernels._PyVal((ctx.host === nothing ? "jnp.maximum(" : "max(") * steps * ", 0)",
                           Int, T.static && m.static)
end

# A cell's position in the plate's domain (`_ppl_cell_scan_position`): in a
# unit range, the index's offset from its first entry, plus one.
function ReactiveKernels._jax_function_call(ctx::ReactiveKernels._JaxContext,
        ::typeof(_ppl_cell_scan_position), args::Vector{ReactiveKernels._PyVal})
    r, i = args
    r.T isa Type && r.T <: AbstractUnitRange || ReactiveKernels._jax_error(
        "a cell position in a domain of type `", r.T, "` has no JAX translation",
        ReactiveKernels._jax_where(ctx))
    first_entry = ReactiveKernels._py_atom(r.code) * "[0]"
    ReactiveKernels._PyVal(ReactiveKernels._py_infix("+",
        [ReactiveKernels._py_infix("-", [i.code, first_entry]), "1"]), Int, false)
end

# The bound of a cell's observed `@scan`: the cell's series must hold one
# entry per scan index, checked when the module runs (a `DimensionMismatch`
# natively, a raising runtime check here); the bound itself is `hi`.
function ReactiveKernels._jax_function_call(ctx::ReactiveKernels._JaxContext,
        ::typeof(_observed_scan_cell_bound), args::Vector{ReactiveKernels._PyVal})
    name, _, value, hi = args
    value.T isa Type && value.T <: _Absent && return hi
    n = ReactiveKernels._jax_call(ctx, length, [value])
    same = ReactiveKernels._jax_call(ctx, ==, [n, hi])
    message = ReactiveKernels._PyVal(repr("observed `@scan` array " * strip(name.code, ['"', '\'']) *
        " has a different number of entries than the scan writes"), DimensionMismatch)
    ReactiveKernels._jax_call(ctx, ReactiveKernels._runtime_check, [same, message])
    hi
end

# The broadcast row count of bound columns: their first-axis lengths are
# shapes, known while JAX traces, and compatible (one, or a common length).
function ReactiveKernels._jax_function_call(ctx::ReactiveKernels._JaxContext,
        ::typeof(_observation_rows), args::Vector{ReactiveKernels._PyVal})
    rows = [ReactiveKernels._jax_is_array(a.T) ?
        ReactiveKernels._py_atom(a.code) * ".shape[0]" : "1" for a in args]
    ReactiveKernels._PyVal(length(rows) == 1 ? only(rows) : "max(" * join(rows, ", ") * ")",
                           Int, true)
end
