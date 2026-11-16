# Host collections are reusable runtime values: Python's one-shot zip would
# silently empty a second Dict(z), unlike Julia's Zip.
const _HOST_ZIP = """
class _RKZip:
    def __init__(self, *sources):
        self.sources = sources

    def __iter__(self):
        return zip(*self.sources) if self.sources else iter(lambda: (), None)

    def __len__(self):
        if not self.sources:
            raise TypeError("zip() has no finite length")
        return min(map(len, self.sources))
"""

function _jax_function_call(ctx::_JaxContext, ::typeof(zip), args::Vector{_PyVal})
    ctx.host === nothing && return _jax_traceable_or_lowered(ctx, zip, args)
    parts = [_host_iterable_value(ctx, a) for a in args]
    name = _jax_module_helper!(ctx.mod, "_RKZip", _HOST_ZIP)
    T = Base.Iterators.Zip{Tuple{(_jax_typeof(a) for a in args)...}}
    _PyVal(name * "(" * join(first.(parts), ", ") * ")", T, true)
end

# Dict uses isequal, not Python's ==: all NaNs are one key, and -0.0 is
# distinct from +0.0. Retain the original keys for keys/pairs iteration.
const _HOST_DICT = """
def _rk_dict_key(x):
    if isinstance(x, tuple):
        return (1, tuple(map(_rk_dict_key, x)))
    if isinstance(x, np.ndarray):
        return (2, x.shape, tuple(map(_rk_dict_key, x.ravel(order="F"))))
    if isinstance(x, (complex, np.complexfloating)):
        if x.imag == 0 and not np.signbit(x.imag):
            return _rk_dict_key(x.real)
        return (3, _rk_dict_key(x.real), _rk_dict_key(x.imag))
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return (4,)
        if x == 0 and np.signbit(x):
            return (5,)
    return (0, x)

class _RKDict:
    def __init__(self, pairs=()):
        self.data = {}
        for key, value in pairs:
            self[key] = value

    def __len__(self):
        return len(self.data)

    def __contains__(self, key):
        return _rk_dict_key(key) in self.data

    def __getitem__(self, key):
        return self.data[_rk_dict_key(key)][1]

    def __setitem__(self, key, value):
        self.data[_rk_dict_key(key)] = (key, value)

    def get(self, key, default):
        item = self.data.get(_rk_dict_key(key))
        return default if item is None else item[1]

    def setdefault(self, key, default):
        return self.data.setdefault(_rk_dict_key(key), (key, default))[1]

    def pop(self, key, default):
        item = self.data.pop(_rk_dict_key(key), None)
        return default if item is None else item[1]

    def items(self):
        return self.data.values()

    def keys(self):
        return (item[0] for item in self.items())

    def values(self):
        return (item[1] for item in self.items())

    def copy(self):
        return _RKDict(self.items())
"""

function _host_dict(ctx::_JaxContext, T::Type, x::_PyVal)
    iterable, E = _host_iterable_value(ctx, x)
    E isa Type && E <: Tuple && fieldcount(E) == 2 ||
        _jax_error("a dictionary constructor needs key/value pairs, got `", E, "`", _jax_where(ctx))
    R = _jax_return_type(T, Any[x.T])
    R isa Type && R <: AbstractDict && isconcretetype(R) ||
        _jax_error("the key and value types of `", T, "` cannot be inferred", _jax_where(ctx))
    key, value = _jax_fresh(ctx, :key), _jax_fresh(ctx, :value)
    function convert_element(code, S, D)
        converted = _host_convert(ctx, code, S, D)
        D <: Number && !(S <: D) ? _host_dtype(D) * "(" * converted * ")" : converted
    end
    k = convert_element(key, fieldtype(E, 1), Base.keytype(R))
    v = convert_element(value, fieldtype(E, 2), Base.valtype(R))
    name = _jax_module_helper!(ctx.mod, "_RKDict", _HOST_DICT)
    _PyVal(name * "((" * k * ", " * v * ") for " * key * ", " * value *
           " in " * iterable * ")", R, true)
end

# A vector search retains one runtime loop, including for a predicate with
# statements or effects. Bool validation is Julia's predicate contract.
function _jax_function_call(ctx::_JaxContext, ::typeof(findall), args::Vector{_PyVal})
    values = isempty(args) ? nothing : last(args)
    if ctx.host === nothing || !(length(args) in (1, 2)) ||
       !(values.T isa Type && values.T <: AbstractVector)
        return _jax_traceable_or_lowered(ctx, findall, args)
    end
    predicate = length(args) == 1 ? identity : _jax_function_of(first(args))
    _host_find(ctx, findall, predicate, values)
end

function _host_find(ctx::_JaxContext, f, predicate, values::_PyVal)
    found, index, element = (_jax_fresh(ctx, hint) for hint in (:found, :position, :element))
    _jax_emit!(ctx, found * (f === findfirst ? " = None" : " = []"))
    _jax_emit!(ctx, "for " * index * ", " * element * " in enumerate(" * values.code * ", 1):")
    ctx.scope.indent += 1
    try
        test = _jax_bind!(ctx, :matched,
                         _jax_call(ctx, predicate, [_PyVal(element, eltype(values.T), true)]))
        _jax_emit!(ctx, "if not isinstance(" * test.code * ", (bool, np.bool_)):")
        _jax_emit!(ctx, "    raise TypeError(\"" * string(f) * " predicate must return Bool\")")
        _jax_emit!(ctx, "if " * test.code * ":")
        if f === findfirst
            _jax_emit!(ctx, "    " * found * " = " * index)
            _jax_emit!(ctx, "    break")
        else
            _jax_emit!(ctx, "    " * found * ".append(" * index * ")")
        end
    finally
        ctx.scope.indent -= 1
    end
    f === findfirst ? _PyVal(found, Union{Nothing,Int}, true) :
        _PyVal("np.asarray(" * found * ", dtype=np.int64)", Vector{Int}, true)
end

function _host_array_equality(ctx::_JaxContext, f, args::Vector{_PyVal})
    # Object scalars compare integers to floats exactly; NumPy's usual
    # promotion can equate distinct integers above 2^53 to one Float64.
    values = ["np.asarray(" * a.code * ", dtype=object)" for a in args]
    code = "np.array_equal(" * join(values, ", ") * ")"
    _PyVal(f === (==) ? code : "(not " * code * ")", Bool, true)
end

# Host predicate reductions stop at the first decisive entry, retaining
# effects, exceptions and the empty-input identity.
function _host_predicate_reduce(ctx::_JaxContext, f, predicate, values::_PyVal)
    iterable, E = _host_iterable_value(ctx, values)
    result, element = _jax_fresh(ctx, :accepted), _jax_fresh(ctx, :element)
    initial = f === all ? "True" : "False"
    _jax_emit!(ctx, result * " = " * initial)
    _jax_emit!(ctx, "for " * element * " in " * iterable * ":")
    ctx.scope.indent += 1
    try
        test = _jax_bind!(ctx, :predicate, _jax_call(ctx, predicate, [_PyVal(element, E, true)]))
        test.T === Bool || _jax_error("host `", f,
            "` currently needs a Bool predicate result; got ", test.T, _jax_where(ctx))
        _jax_emit!(ctx, "if not isinstance(" * test.code * ", (bool, np.bool_)):")
        _jax_emit!(ctx, "    raise TypeError(\"" * string(f) * " predicate must return Bool\")")
        _jax_emit!(ctx, "if " * (f === all ? "not " : "") * test.code * ":")
        _jax_emit!(ctx, "    " * result * " = " * (f === all ? "False" : "True"))
        _jax_emit!(ctx, "    break")
    finally
        ctx.scope.indent -= 1
    end
    _PyVal(result, Bool, true)
end

# Tuple rows remain Python tuples: making an ndarray would promote mixed
# fields and turn a vector of records into an extra array axis.
function _host_tuple_order(ctx::_JaxContext, which::Symbol, x::_PyVal)
    key = _host_order_helper!(ctx, :isless)
    if which === :sort
        return _PyVal("sorted(" * x.code * ", key=" * key * ")", x.T, true)
    elseif which === :sortperm
        helper = _jax_helper!(ctx, "_rk_tuple_sortperm", """
def _rk_tuple_sortperm(x):
    return np.asarray(sorted(range(1, len(x) + 1),
                             key=lambda i: _rk_isless_key_np(x[i - 1])), dtype=np.int64)
""")
        return _PyVal(helper * "(" * x.code * ")", Vector{Int}, true)
    end
    _jax_module_helper!(ctx.mod, "_RKDict", _HOST_DICT)
    helper = _jax_helper!(ctx, "_rk_tuple_unique", """
def _rk_tuple_unique(x):
    seen = set()
    result = []
    for value in x:
        key = _rk_dict_key(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result
""")
    result = _jax_bind!(ctx, :unique_rows, _PyVal(helper * "(" * x.code * ")", x.T, true))
    push!(ctx.host.safe, result.code)
    push!(ctx.host.lists, result.code)
    result
end

# NumPy otherwise interprets the fields of each tuple as a second broadcast
# axis. Box each complete row, without converting its individual fields.
function _host_broadcast_operand(ctx::_JaxContext, x::_PyVal)
    _host_tuple_vector(x.T) || return x
    helper = _jax_helper!(ctx, "_rk_tuple_vector", """
def _rk_tuple_vector(rows):
    result = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        result[i] = row
    return result
""")
    _PyVal(helper * "(" * x.code * ")", x.T, true)
end

function _host_rewrite_value!(ctx::_JaxContext, s, c::_PyVal, value::_PyVal)
    s === nothing && return value
    target = _host_own!(ctx, s)
    _jax_emit!(ctx, target.code * "[:] = " * value.code)
    target
end

# Python sorts cache keys, whereas Julia may call `by` per comparison and
# skip it on singleton inputs. Only total record projections can safely
# cross that difference without changing effects or exceptions.
function _host_sort_projection(ex, parameter::Symbol, T)
    ex === parameter && return T
    ex isa Number && return typeof(ex)
    ex isa Expr || return nothing
    if ex.head === :block
        parts = filter(x -> !(x isa LineNumberNode), ex.args)
        return length(parts) == 1 ? _host_sort_projection(only(parts), parameter, T) : nothing
    elseif ex.head === :tuple
        fields = [_host_sort_projection(x, parameter, T) for x in ex.args]
        return all(x -> x isa Type, fields) ? Tuple{fields...} : nothing
    elseif ex.head === :ref && length(ex.args) == 2
        row = _host_sort_projection(ex.args[1], parameter, T)
        i = ex.args[2]
        row isa Type && row <: Tuple && isconcretetype(row) &&
            i isa Integer && 1 <= i <= fieldcount(row) && return fieldtype(row, i)
    end
    nothing
end

function _host_sort_by(ctx::_JaxContext, f, ex, by::_PyVal)
    s, x = _host_target(ctx, ex)
    x.T isa Type && x.T <: AbstractVector ||
        _jax_error("`", f, "` needs a vector in host preparation", _jax_where(ctx))
    callable = _jax_function_of(by)
    projected = callable === identity ? eltype(x.T) :
        callable isa Union{_HostLambda,_HostLocalFunction} && length(callable.params) == 1 ?
            _host_sort_projection(callable.body, only(callable.params), eltype(x.T)) : nothing
    projected isa Type || _jax_error("host `by` sorting currently needs a total tuple-field ",
        "projection; caching an effectful or throwing key would change Julia's calls", _jax_where(ctx))
    key = _host_order_helper!(ctx, :isless)
    name, _ = _host_nested_function!(ctx, :order_key, [:element], [eltype(x.T)], () -> begin
        value = _jax_call(ctx, callable, [ctx.locals[:element]])
        value.T isa Type && _host_orderable(value.T) ||
            _jax_error("the sort key `", value.T, "` has no host ordering", _jax_where(ctx))
        _PyVal(key * "(" * value.code * ")", Any, true)
    end)
    if f === sortperm
        i = _jax_fresh(ctx, :position)
        code = "sorted(range(1, len(" * x.code * ") + 1), key=lambda " * i *
               ": " * name * "(" * x.code * "[" * i * " - 1]))"
        return _host_collected(code, Int)
    end
    value = _host_collected("sorted(" * x.code * ", key=" * name * ")", eltype(x.T))
    f === sort! ? _host_rewrite_value!(ctx, s, x, value) : value
end

_jax_expr(ctx::_JaxContext, ::Val{:curly}, ex) =
    _PyVal("<type>", _JaxCallable(_jax_resolve(ctx, ex)), true)

# Local named helpers use the same owned-argument analysis as recorded
# helpers. Captures are passed at the call, so rebinding an outer local is
# visible and a write/growth reaches the caller's collection.
struct _HostLocalFunction <: Function
    name::Symbol
    params::Vector{Symbol}
    body::Any
    captures::Vector{Symbol}
    mod::Module
end

function _host_local_function!(ctx::_JaxContext, signature::Expr, body)
    name = first(signature.args)
    name isa Symbol || _jax_error("the local function signature `", signature, "`", _jax_where(ctx))
    params = Symbol[]
    for arg in signature.args[2:end]
        arg isa Symbol || _jax_error("the local function argument `", arg, "`", _jax_where(ctx))
        push!(params, arg)
    end
    haskey(ctx.locals, name) && ctx.locals[name].T isa _JaxCallable &&
        _jax_error("multiple definitions of local function `", name,
                   "` have no host translation", _jax_where(ctx))
    assigned = first(_host_targets([body]))
    captures = unique!(Symbol[s for s in [_jax_symbols_read!(Symbol[], body); collect(assigned)]
                              if haskey(ctx.locals, s) && !(s in params)])
    rebound = intersect(assigned, Set(captures))
    isempty(rebound) || _jax_error("rebinding the captured local(s) ", join(sort!(collect(rebound)), ", "),
                                   " in a local function has no host translation", _jax_where(ctx))
    fn = _HostLocalFunction(name, params, body, captures, ctx.source_module)
    v = _PyVal("<local function>", _JaxCallable(fn), true)
    ctx.locals[name] = v
    v
end
nameof_or(f::_HostLocalFunction) = f.name

function _jax_expr(ctx::_JaxContext, ::Val{:function}, ex)
    ctx.host === nothing && _jax_error("a local function has no device translation", _jax_where(ctx))
    _host_local_function!(ctx, ex.args[1], ex.args[2])
end

function _jax_call_positional(ctx::_JaxContext, f::_HostLocalFunction, args::Vector{_PyVal})
    length(args) == length(f.params) || _jax_error("wrong argument count for local function `", f.name, "`")
    captures = _PyVal[_host_raw(ctx, s) for s in f.captures]
    _host_function_call(ctx, f, [f.params; f.captures], f.body, f.mod, [args; captures])
end
