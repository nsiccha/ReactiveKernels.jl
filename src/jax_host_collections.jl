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
