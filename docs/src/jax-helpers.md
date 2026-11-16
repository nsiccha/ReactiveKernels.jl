# JAX helpers and submodels

This guide continues [JAX source](jax.md): how emitted modules share submodel functions, and which helpers with loops and branches translate.

## Named submodels and shared entries

Repeated calls to an authored submodel share a Python function when their
selected computations agree. A scoped rewrite that changes the computation
gets a separate function. Each function carries the submodel's documentation.
The native model retains its expanded graph, so individual locals remain
queryable.

The public density functions and `constrain` call a shared model function.
Their output selectors are fixed by the wrappers, allowing JAX to trace only
the requested computation. Call them as before: `log_density(u, **prepared)`.
The submodel functions likewise take a static `outputs` selection; their
arguments and array lengths remain runtime inputs. Consecutive statements
that the same outputs need share one selection block (`if _entry in (...):`,
or `if any(name in outputs for name in (...)):` in a submodel function), and
the checks every public function runs, such as the packed-length check, run
once before the per-function returns.

A plate may use separate functions for a scalar return value and a batched
density. Combined computations, such as grouped scalar priors, retain their
shared graph boundary. Interleaved dependencies split a submodel into named
portions that execute in the original dependency order.

```@example jax_submodels
using ReactiveKernels, ReactiveKernelsPPL

@rkppl effect(loc, scale, spread_rate, n) = begin
    "Mean of the effect."
    mean ~ Normal(loc, scale)
    "Spread of the members around the mean."
    spread ~ Exponential(spread_rate)
    "Standardised member deviations."
    z[1:n] .~ Normal.(0, 1)
    return mean .+ spread .* z
end

model = @rkppl begin
    a ~ effect(a_loc, a_scale, a_rate, n)
    b ~ effect(b_loc, b_scale, b_rate, n)
    y .~ Normal.(a .+ b, noise)
end
bound = model(; n = 3, a_loc = 0.0, a_scale = 1.0, a_rate = 1.0,
               b_loc = 0.0, b_scale = 1.0, b_rate = 1.0, noise = 1.0) |
        (; y = [0.1, 0.2, 0.3])
built = build_kernel(bound)
print(jax_source(built, bound))
```

## Helpers with loops and branches

A helper becomes a named Python function with its method's argument names,
reused by calls with the same argument representation: from recorded source,
or, for a plain helper outside Julia and ReactiveKernels, from its lowered
code. Its Julia docstring accompanies the body; a broadcast vectorizes the
function itself. A leading underscore and ReactiveKernels' own helpers stay
private (`_name`). Arrays, captured numerical values and padded lengths are
normally explicit arguments. A helper that carries loop bounds or reads scan
history stays nested in that scope, preserving the bounds and history access
analysis. Nested plate cells and scan steps are named after the authored
values they compute.

A helper translates from recorded source when available: `@traceable` records it, and a trailing Vararg argument (`args...`, `args::Vararg{T,N}`) reads as the tuple of the trailing arguments, as the method receives them, and `sum(f, t; init)` folds such a tuple over its fixed length; `for` becomes `jax.lax.fori_loop`, `if` becomes `jax.lax.cond`, and `A[i] = v` becomes `A.at[i].set(v)`. A `jax.lax.cond` arm that is one expression is a `lambda`, and a branch between two values already bound or literal (`s == 0 && x == m ? 1.0 : s`) is `jnp.where`, whose value and derivative are the conditional's; an arm that computes anything stays lazy. A condition decided while tracing (a shape or literal test) is Python's own `if`, so a guard `n == k || throw(...)` reads `if not (n == k): raise ValueError(...)`. Plain lowered helpers support straight-line code and terminal forward branches, including size guards. They can create immutable closures and fold fixed argument lists with `Base.afoldl`. Broadcasting `Ref(x)` passes `x` whole to each call. Captures remain runtime inputs; host `prepare_data` uses the same translation.

`@traceable` methods also accept named keywords, including required and typed
keywords. Calls such as `weighted(xs; scale)` bind by name. Omitted defaults
are evaluated in the helper's module, in declaration order, and may read
positional arguments and earlier keywords. Supplied keywords skip their
defaults. Keyword values and data-derived sizes remain runtime arguments in
both host preparation and JAX functions.

A captured `NamedTuple` of data can be read whole, for example passed to a
helper (`metadata = (; weights = w, indices = idx)` captured by a kernel that
calls `weighted(x, metadata)`). It is the Python tuple of its fields, as a
named-tuple HAVE crosses, and the helper's field reads (`metadata.weights[1]`)
index it by position. Each array field is a typed module constant named after
the record and the field (`METADATA_WEIGHTS = jnp.asarray(..., dtype=jnp.float64)`),
with Julia's shape and element type, so a matrix or a rank-3 integer array
keeps its axes and one-based reads. A record holding an array is itself a
module constant (`METADATA = (METADATA_WEIGHTS, METADATA_INDICES)`), so a
call passes that one name; a record of numbers stays a static literal and
can size an allocation. A field that is a function or a type raises
`JAXEmissionError`.

Field names obtained from `keys(record)` or the `names` parameter of
`record::NamedTuple{names}` remain structural through locals and helper calls.
For example, `getproperty(record, first(keys(record)))` selects the first field
in the record's declared order, in host preparation and JAX functions. The
record's numerical fields and array lengths remain runtime values; this does
not specialize a module to the data used when emitting it.

A host helper that writes its argument in place (`rv[k] = x`, `push!(rv, x)`, `sort!(rv)`, or a helper it calls doing so) writes the caller's vector, as Julia's does, whether or not the caller reads the helper's result (RK 0.1.278). The caller passes its own local: a copy of an input or a module constant first, and a list when the helper grows it. A slice or another expression in that position raises `JAXEmissionError`, because Julia's `x[r]` is a copy and `view(x, r)` is not, and the two have one NumPy spelling.

A method that a package generates, such as one its macro writes into the caller's module, cannot carry `@traceable`. The generating package keeps the method's source as a `TraceableSource` (argument names, body, and the module whose globals the body reads) and hands it over through a provider. It registers the provider with `register_source_provider!`, once, from the `__init__` of the package extension that loads with ReactiveKernels. The emitter asks the provider for a call that has no other translation and no `@traceable` source, with the function and the `Tuple` type of the argument types. The provider answers `nothing` or the source of the method those arguments call natively.

The example below stands in for such a package. Its macro defines the method and keeps its source.

```@example jax
using ReactiveKernels

const GENERATED = Dict{Any,TraceableSource}()

macro generate(def)
    call, body = def.args
    names = Tuple(arg.args[1] for arg in call.args[2:end])
    source = TraceableSource(names, body, __module__)
    :($(esc(def)); GENERATED[$(esc(call.args[1]))] = $source)
end

@generate function decay_sum(x::Vector{Float64}, rate::Float64)
    total = zero(rate)
    for i in 1:3
        total = total + x[i] * exp(-i * rate)
    end
    if rate > 0
        total = total / rate
    end
    return total
end

provide_source(f, argument_types) = get(GENERATED, f, nothing)
register_source_provider!(provide_source)

@kernel decayed(x::Vector{Float64}, rate::Float64) = begin
    value = decay_sum(x, rate)
    return value
end

print(jax_source(decayed; have = (:x, :rate), want = :value))
```

Body-level early returns, including `condition || return fallback(...)`, keep the continuation in the falling-through branch. The native method retains its original control flow. A return inside a loop remains unsupported. In device code, `continue` skips the remaining iteration statements and `break` exits the nearest loop. A bounded `for` carries a flag and skips later steps after an exit, preserving reverse differentiation. A loop without a static bound remains a runtime loop; JAX requires an authored derivative rule for reverse differentiation through it. Host `prepare_data` supports the same exits. A provider's source is read only by `jax_source`; native execution runs the method itself.

`@inbounds` and `@simd` preserve the source loop. Chained comparisons evaluate a shared middle operand once and reach later operands only when earlier comparisons succeed. Plain helpers accept acyclic forward joins and static Boolean `Val` parameters. Helpers can destructure numerical arrays and use `firstindex`, `lastindex`, integer parity predicates, and floating-point `copysign`.

A numerical helper may return a scalar or `nothing`; consume that result inside the program with `something` or `isnothing`. Device branches carry a presence flag and the scalar, while host preparation keeps ordinary Python values. A numerical vector prefix `c[1:n]` carries its runtime length and checks bounds. `evalpoly(x, c)` for real vectors retains Horner's loop over that logical length, including a runtime prefix; unused coefficients contribute neither values nor derivatives.

Conditional arrays and tuple fields retain their logical lengths, including
when destructured. A bound such as `min(n, length(row) - 1)` keeps a runtime
range bounded by the row's capacity. Numerical matrix allocation can use a
bounded dimension, such as `Matrix{Float64}(undef, length(row), 2)`, and
several axes may share that logical extent. Independently padded dimensions
remain unsupported. Array lengths and loop bodies are never copied from a
sample dataset into emitted source.

A source `ntuple(j -> expression, Val(N))` records the authored, fixed number of fields with runtime captures. A fixed tuple of numbers of one type supports computed integer reads, such as indexing that basis inside a loop. The device read checks bounds and uses a numerical array view of the tuple; the data-dependent loop remains a runtime loop.

A device helper can allocate `Vector{Vector{Float64}}(undef, k)`, fill its slots with fixed-width numerical rows, and use `isassigned(rows, i)` to skip earlier assignments. A size such as `isempty(ids) ? 0 : maximum(ids)` uses bounds computed from prepared integer data, including ragged cells. Allocation and iteration remain runtime array operations; emitted source does not grow with the data. Assignment state travels with the numerical storage through loops and branches. Missing references and invalid indices raise when read; `isassigned` returns false for either.

This nested-buffer representation accepts fresh rows with an authored fixed shape, such as `[a, b, c]` or `zeros(3)`. Different row shapes require a ragged representation, and mutation through stored row aliases or a helper's buffer or row argument requires reference writeback; these cases raise `JAXEmissionError`. Buffers are consumed inside the program by indexing or plates; direct public buffer outputs are not supported yet.
