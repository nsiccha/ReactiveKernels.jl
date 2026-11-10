# JAX target

`jax_source` translates a planned ReactiveKernels program into a Python module
that computes the same values with `jax.numpy`. The module is plain Python,
with no Julia at run time. `jax.grad` differentiates it, and JAX samplers such
as NumPyro and BlackJAX run on it, so a model written in Julia can sit inside a
standalone Python service.

## A kernel as a JAX function

```@example jax
using ReactiveKernels

@kernel regression(y::Vector{Float64}, x::Vector{Float64}, a::Float64, b::Float64,
                   logsigma::Float64) = begin
    sigma = exp(logsigma)
    lp = plate(y, x) do yi, xi
        if sigma > 0
            -0.5 * ((yi - a - b * xi) / sigma)^2 - log(sigma)
        else
            -Inf
        end
    end
    total = sum(lp)
    return total
end

src = jax_source(regression; have = (:a, :b, :logsigma, :y, :x), want = :total)
print(src)
```

The module defines one function per requested output over the HAVE values, in
order. `functions = (name = want, …)` defines several functions over the same
inputs. A function's WANT is a value name, a tuple of names (it returns a
tuple), or a `NamedTuple` (it returns a `dict`). `constants = (; …)` adds
module-level constants. The types of the HAVE values decide every value's rank.
By default they are the ports' declared types; `types = (; name = T)` supplies
any that are undeclared.

## `@rkppl` models

For a built `@rkppl` program, `jax_source(built, plan)` translates the sampler
boundary, the same one `prepare_query(built, plan, :sampler)` uses: the packed
unconstrained parameters and the data columns.

```@example jax
using ReactiveKernelsPPL

x = collect(range(-1, 1; length = 20))
y = 1 .+ 2 .* x .+ 0.1 .* sin.(7 .* x)
model = @rkppl begin
    a ~ Normal(0, 5)
    b ~ Normal(0, 2)
    sigma ~ Exponential(1.0)
    mu = a .+ b .* x
    y .~ Normal.(mu, sigma)
end
plan = model(; x) | (; y)
built = build_kernel(plan)
module_source = jax_source(built, plan; values = (:mu,))
[f.name => first.(f.inputs) for f in module_source.functions]
```

The module's functions are:
- `prepare_data`, which takes the data the model is bound with, by the
  names the model reads (columns and scalar values, in sorted name order or
  by name), and returns the functions' data inputs as a `dict` (see
  [Data preparation](#Data-preparation)). It computes the data-only
  definitions `bind_data` evaluates, such as `w = cos.(x ./ period)`, itself;
- `log_density`, the sampler's target (prior, likelihood and log Jacobian);
- its parts, `log_prior`, `log_likelihood` and `log_jacobian`;
- `constrain`, a `dict` of every declared parameter's constrained value plus
  each value named in `values`.

The functions take `unconstrained` and the prepared data by keyword,
`f(unconstrained, **prepared)`. Each
has a docstring and type annotations, and the model's own docstrings (see
`rkppl_documentation`) appear in the module docstring and, for each value
`constrain` returns, in its docstring.

Two functions of the prepared data describe the parameters, since their
sizes can follow the data (a level axis, a series length):
- `dimension(prepared)`, the length of `unconstrained`;
- `parameters(prepared)`, each parameter's 0-based `start`/`stop` in
  `unconstrained`, its transform and its dimensions.

When no parameter's size reads data, the constants `DIMENSION` and
`PARAMETERS` state the same. The module's other constants describe the
inputs:
- `DATA`, each of `prepare_data`'s parameters' dtype and rank, and whether
  it is ragged;
- `UNTRANSLATED`, each data-only definition `prepare_data` takes as an input
  instead of computing it, with the reason: one without a host translation,
  or a derived definition whose binding guards missing observations. It is
  `{}` when every definition is computed. A model whose `UNTRANSLATED` is
  not empty is **not supported for standalone preparation from raw data**:
  supplying the Julia binder's values for those inputs runs it, but that is
  not a standalone module, so a consumer's standalone check asserts
  `UNTRANSLATED == {}`. A response with missing values also passes its
  presence mask, `_rkppl_present_<name>`;
- `PREPARED`, the same for each prepared input;
- `INPUTS`, the functions' parameters in order.

BRM models reach the same function through their RKPPL emission:
`jax_source(rk.model, rk_bound_plan(rk))` for an `RKBRMI` backend `rk`.

In Python, a sampler targets `-log_density`, with the prepared data closed
over:

```python
import jax, jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
import model  # the module jax_source wrote

prepared = model.prepare_data(x=x, y=y)  # the bound data, once per dataset
potential = lambda u: -model.log_density(u, **prepared)
mcmc = MCMC(NUTS(potential_fn=potential), num_warmup=500, num_samples=1000)
mcmc.run(jax.random.PRNGKey(0), init_params=jnp.zeros(model.dimension(prepared)))
draws = jax.vmap(lambda u: model.constrain(u, **prepared))(mcmc.get_samples())
```

## Data preparation

With `data`, `jax_source(spec; have, want, data)` splits the program in two,
where binding data splits it (`prepare(...; bound)`): the recipes that read
data alone run once per dataset in `prepare_data`, and the functions compute
the rest from its results. `data` names the HAVE values that are data, or
gives one binding's values (`data = (; g, y)`), whose prepared values then type
the functions' inputs exactly. `jax_source(built, plan)` passes its binding.

`prepare_data` is NumPy on the host, translated from the same source as the
JAX code:

- Its values are concrete, so a condition is a Python `if`, a loop a Python
  `for` or `while` (with `break` and `continue`), and a value's shape may
  follow its data: level codes of string columns, stopping-ratio stage lanes,
  positions of missing values. A loop in the source is a loop in the module;
  nothing is unrolled.
- A vector built with `push!` is a list; dictionaries are `dict`s; an array a
  source writes by index is written in place, after a copy if the source did
  not allocate it.
- A ragged column is a list of arrays. A ragged prepared value reaches the
  functions padded, with its `<name>_lengths` and `<name>_flat` (the
  positions of its entries in the flattened padded array).
- Values that read no data (literals, a prior's scale) stay in the functions,
  where JAX folds them.
- A data-only recipe without a host translation stays in the functions; a
  comment in `prepare_data` names it.
- `definitions = [name => expression, …]` (with `data`) computes data
  values on the host before that prefix, from the other data values;
  `prepare_data` then takes what the expressions read instead of the values
  they define. `jax_source(built, plan)` passes the binding's data-only
  definitions this way. One without a host translation stays an input,
  listed with its reason in the module's `UNTRANSLATED`.

The functions' inputs have static shapes, so they also run under `jax.jit`
with the prepared data as arguments: `jax.jit(model.log_density)(u,
**prepared)`. An integer of the prepared data that sets a shape (a parameter
block's end, a scan's length, a level count) is a `StaticInt`, an `int` that
`jax.jit` treats as static; a new value traces the function again, as a new
array shape does.

## What the translation keeps

The translation follows the [core constraints](constraints.md):

- **Plates** are `jax.vmap` of the cell, with shared operands unmapped.
- **Scans** are `jax.lax.scan` of the step. A scan whose range ends at a
  group's length in a ragged plate cell (`for t in 2:length(y[g])`) runs to
  the longest group's length: past a group's own length its carry keeps its
  value and its outputs are padding, which reductions mask.
- **Lazy branches** are `jax.lax.cond` of their arms. Under `vmap`, JAX
  evaluates both arms of a condition that differs between cells and selects.
  Each lane's untaken arm reads its inputs through `stop_gradient`, so a NaN
  or Inf it computes reaches neither the value nor the gradient.
- **Generators and comprehensions** (`sum(f(i) for i in r; init = x)`,
  `[f(i) for i in r]`) are `jax.vmap` of the element over the range; a filter
  masks the reduction.
- **Loops** written in a recipe or a `@traceable` helper are
  `jax.lax.fori_loop` (`while` loops are `jax.lax.while_loop`). Their carry
  is the locals the body assigns, and an
  indexed write `A[i] = v` is `A.at[i].set(v)`.
- **Static conditions** decide while JAX traces: shapes, literals, and guards
  such as `n == k || throw(...)`. A function of such values is computed with
  NumPy, which JAX does not stage, so it decides too: a guard
  `isfinite(jitter) && jitter >= 0 || throw(...)` over a literal a caller
  passes, or `log(length(x)) >= 0`. A failing guard raises while JAX traces,
  including inside a data-dependent branch the native program would not take
  (the guard is a fact of the shapes, so every run of that branch fails it).
- **Runtime checks** (`ReactiveKernels._runtime_check`, the domain checks in
  helpers such as the multivariate normal densities) run when the module runs,
  through `jax.debug.callback`, and raise on values that fail them. Under a
  data-dependent condition a check is masked by it: JAX evaluates both arms
  of a vmapped condition, and the check holds for lanes that do not take its
  arm.
- **No unrolling.** Nothing iterates in Python, and sizes are read from the
  inputs (`x.shape[0]`). The same module serves every data size of the same
  program.
- **Derivatives** are `jax.grad` of the emitted primal. A derivative rule
  contributes only its primal cut.

The interface conventions:

- **64-bit floats.** The module enables JAX's 64-bit mode on import
  (`jax_enable_x64`), so its arithmetic is Julia's `Float64`. Import it before
  creating arrays that should be 64-bit.

- **Arrays keep Julia's dimension order.** A Julia `n × p` design matrix is an
  `(n, p)` array. Julia's broadcasting is kept by padding axes from known
  ranks, and `reshape` and `vec` stay column-major. A `:` dimension
  (`reshape(v, 1, :)`, also in a dimension tuple) is the remaining length.
- **Integer index data stay 1-based**, as the Julia program reads them; the
  emitted code subtracts one where it indexes.
- **Ragged data** (one vector per group, such as per-subject series) are a
  padded array plus a `<name>_lengths` input; `prepare_data` builds both from
  a list of arrays (without `data`, the module's `pad_ragged` does). Padding
  repeats each row's last entry, so padded slots hold valid values.
  Reductions mask by length, and a loop over a group's entries keeps its carry
  past the group's length. `reduce(vcat, cells)` of a ragged value drops the
  padding; its length, the total, is the length of `<name>_flat` of the
  prepared input whose lengths the cells have (a plate over its rows, or over
  its group index).

## Helpers

A function that a recipe calls is translated if:

- it has a `jax.numpy` counterpart (Base and LogExpFunctions functions, and
  SpecialFunctions through the package extension). `logbeta` is a helper the
  module defines, since `jax.scipy.special.betaln` (JAX 0.12) is accurate to
  about 1e-6 in float64 once an argument reaches 8;
- it is defined with `@traceable`, in which case its authored body is
  translated (in `prepare_data`, as a host function);
- it is a prepared kernel, which becomes a module-level function of its plan;
- or it is a plain function whose lowered code is straight-line.

A package adds a translation for its own function with a method of
`ReactiveKernels._jax_elementwise` (a scalar function's `jax` name) or
`ReactiveKernels._jax_function_call`. A package's plain helper with branches
that native code must keep as written records its source with
`ReactiveKernels.@_record_source` (a leading `@inline` stays).

Shape tests decide while tracing. Shapes are known then, and so are the
prepared integers (`StaticInt`s); `size(x, d)` past `x`'s rank, or of a
number, is 1. So `ndims(x) <= 3 && size(x, 1) in (1, K)` selects its arm in
Python, and a `throw` in the other arm raises then. An index that Julia
requires to be 1 (`v[i, 1, 1]` of a vector, `s[1, 1, 1]` of a number) is
checked and dropped.

## Limits

These shapes have no translation; each raises a `JAXEmissionError` naming the
recipe and the construct:

- a plain helper with loops or branches that is not `@traceable`;
- in the JAX functions, a loop whose trip count is data with no static bound
  (JAX loops and arrays need static shapes), and a `throw` that depends on
  data (a runtime check is the form that has one); `prepare_data` runs both;
- `jax.grad` through a `while` loop (JAX differentiates `lax.while_loop`
  forward only; write a differentiated loop as a `for` over a range);
- filtered comprehensions in the JAX functions (their length is data);
- `reduce(vcat, cells)` of a ragged value whose lengths are no prepared
  input's;
- plates of rank above one over ragged data;
- `reshape` of a ragged value (its padding has no shape);
- `history` scans over a padded (ragged) sequence or with a named-tuple
  history.

Frozen definitions (`frozen_values`) come from another binding, so
`prepare_data` takes their values, as `DATA` lists them.
