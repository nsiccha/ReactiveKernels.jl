class PreparedData(collections.abc.Mapping):
    """The prepared data by name: what `prepare_data` and the layout functions return.

    It reads like a dict (`prepared["x"]`, `.get`, `.items()`, `f(u, **prepared)`)
    and `dict(prepared)` gives a plain dict of the same values. It cannot be
    changed in place: build a new one with `PreparedData({**prepared, name: value})`.

    As a JAX pytree its leaves are one flat array per dtype, which hold every
    NumPy array and scalar, followed by any other value as its own leaf (a
    Python number keeps its weak type, a JAX array its device). Passing it to a
    jitted function costs a few leaves however many arrays it holds, and the
    function reads each value as a static slice of its array.
    `jax.device_put` and `jax.tree_util.tree_map` keep it packed.
    """

    __slots__ = ("_buffers", "_others", "_layout", "_values")

    def __init__(self, values=(), **named):
        values = dict(values, **named)
        names = tuple(values)
        leaves, tree = jax.tree_util.tree_flatten(values)
        groups, spec, others = {}, [], []
        for leaf in leaves:
            if isinstance(leaf, (np.ndarray, np.generic)) and leaf.dtype.kind in "biufc":
                array = np.asarray(leaf)
                group = groups.setdefault(array.dtype, [])
                offset = sum(part.size for part in group)
                group.append(array.reshape(-1))
                spec.append((array.dtype, offset, array.shape, isinstance(leaf, np.generic)))
            else:
                spec.append(len(others))
                others.append(leaf)
        dtypes = tuple(groups)
        buffers = []
        for dtype in dtypes:
            buffer = np.concatenate(groups[dtype])
            buffer.flags.writeable = False
            buffers.append(buffer)
        self._buffers, self._others = tuple(buffers), tuple(others)
        self._layout = (names, tree, dtypes, tuple(spec))
        self._values = None

    @classmethod
    def _unflatten(cls, layout, children):
        self = object.__new__(cls)
        count = len(layout[2])
        self._buffers, self._others = tuple(children[:count]), tuple(children[count:])
        self._layout, self._values = layout, None
        return self

    def _flatten(self):
        return (*self._buffers, *self._others), self._layout

    def __reduce__(self):
        return PreparedData._unflatten, (self._layout, (*self._buffers, *self._others))

    def __deepcopy__(self, memo):
        import copy
        buffers = copy.deepcopy(self._buffers, memo)
        for buffer in buffers:
            if isinstance(buffer, np.ndarray):
                buffer.flags.writeable = False
        return PreparedData._unflatten(self._layout, (*buffers, *copy.deepcopy(self._others, memo)))

    def _read(self, entry):
        if isinstance(entry, int):
            return self._others[entry]
        dtype, offset, shape, scalar = entry
        buffer = self._buffers[self._layout[2].index(dtype)]
        size = int(np.prod(shape, dtype=np.int64))
        if isinstance(buffer, np.ndarray):
            return buffer[offset] if scalar else buffer[offset:offset + size].reshape(shape)
        return jax.lax.slice_in_dim(buffer, offset, offset + size).reshape(shape)

    def _dict(self):
        if self._values is None:
            names, tree, _, spec = self._layout
            values = jax.tree_util.tree_unflatten(tree, [self._read(entry) for entry in spec])
            self._values = {name: values[name] for name in names}
        return self._values

    def __getitem__(self, name):
        return self._dict()[name]

    def __iter__(self):
        return iter(self._layout[0])

    def __len__(self):
        return len(self._layout[0])

    def __contains__(self, name):
        return name in self._layout[0]

    def __repr__(self):
        return "PreparedData(" + ", ".join(self._layout[0]) + ")"


jax.tree_util.register_pytree_node(PreparedData, PreparedData._flatten, PreparedData._unflatten)
