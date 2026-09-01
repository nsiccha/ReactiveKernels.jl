"""
    PreparedADKernel

A prepared DifferentiationInterface gradient for a ReactiveKernels scalar
objective. Construct one with [`prepare_ad`](@ref), naming the selected HAVE
port that remains active; every other selected HAVE port is supplied to
DifferentiationInterface as a `DifferentiationInterface.Constant` context.

The stored differentiation preparation is mutable and not thread-safe. Prepare
one `PreparedADKernel` per concurrent caller.
"""
struct PreparedADKernel{I,K,R,F,B,P}
    kernel::K
    resolver::R
    call::F
    backend::B
    preparation::P
end

# DifferentiationInterface differentiates its first argument and requires every
# later argument to be a Context. RK kernels retain authored HAVE order, so this
# callable restores that order before entering the generated kernel. The active
# index is a type parameter and the generated call performs no Symbol lookup or
# runtime permutation.
struct _ADKernelCall{I,K}
    kernel::K
end

# Plated PreparedKernels carry their already-generated native and tensorized
# bodies behind a runtime function-pair selector. Native AD preparation has
# concrete exemplars, so it can bypass only that selector and differentiate the
# exact native callable/operation table used by primal execution. No AD-specific
# kernel or AST is generated.
struct _ADNativeKernelCall{I,F,O}
    native::F
    ops::O
end

@generated function (call::_ADNativeKernelCall{I})(
        active, contexts::Vararg{Any,N}) where {I,N}
    1 <= I <= N + 1 || return :(throw(ArgumentError(
        "invalid active input index $I for an RK AD call with $(N + 1) inputs")))
    positional = Any[]
    context_index = 1
    for input_index in 1:(N + 1)
        if input_index == I
            push!(positional, :active)
        else
            push!(positional, :(getfield(contexts, $context_index)))
            context_index += 1
        end
    end
    :(call.native(call.ops, $(positional...)))
end

function _ad_kernel_call(kernel::PreparedKernel, args::Tuple, ::Val{I}) where {I}
    native_exemplars = _dynamic_tensorized_marker(args) === nothing
    if kernel.f isa Union{
            _ArrayFunctionPair,_EmbeddedFunctionPair,
            _DynamicEmbeddedFunctionPair} && native_exemplars
        return _ADNativeKernelCall{I,typeof(kernel.f.native),typeof(kernel.ops)}(
            kernel.f.native, kernel.ops)
    end
    _ADKernelCall{I,typeof(kernel)}(kernel)
end

@generated function (call::_ADKernelCall{I})(
        active, contexts::Vararg{Any,N}) where {I,N}
    1 <= I <= N + 1 || return :(throw(ArgumentError(
        "invalid active input index $I for an RK AD call with $(N + 1) inputs")))
    arguments = Any[]
    context_index = 1
    for input_index in 1:(N + 1)
        if input_index == I
            push!(arguments, :active)
        else
            push!(arguments, :(getfield(contexts, $context_index)))
            context_index += 1
        end
    end
    :(call.kernel($(arguments...)))
end

function _ad_active_index(kernel::PreparedKernel, active::Symbol)
    matches = findall(input -> input.name === active, inputs(kernel))
    isempty(matches) && throw(ArgumentError(
        "active port :$active is not in the selected HAVE boundary " *
        "$(Tuple(input.name for input in inputs(kernel)))"))
    length(matches) == 1 || throw(ArgumentError(
        "active port name :$active is ambiguous in the selected HAVE boundary"))
    only(matches)
end

function _ad_active_index(kernel::PreparedKernel, active::Value)
    graph = kernel.plan.graph
    owned = get(graph.values, active.id, nothing)
    if owned === nothing || typeof(owned) !== typeof(active) ||
       owned.id != active.id || owned.name !== active.name
        throw(ArgumentError(
            "active Value $(active) does not belong to the selected kernel graph"))
    end
    active_id = canon_id(graph, active.id)
    matches = findall(input -> canon_id(graph, input.id) == active_id,
                      inputs(kernel))
    isempty(matches) && throw(ArgumentError(
        "active Value $(active) is not in the selected HAVE boundary"))
    length(matches) == 1 || throw(ArgumentError(
        "active Value $(active) is ambiguous in the selected HAVE boundary"))
    only(matches)
end

function _ad_active_index(::PreparedKernel, active)
    throw(ArgumentError(
        "active must identify a selected HAVE port by Symbol or Value; got " *
        string(typeof(active))))
end

function _ad_validate_unique_haves(spec::KernelSpec)
    graph = spec.graph
    haves = inputs(spec)
    first_name = Dict{Int,Symbol}()
    aliases = Pair{Symbol,Symbol}[]
    for input in haves
        id = canon_id(graph, input.id)
        if haskey(first_name, id)
            push!(aliases, first_name[id] => input.name)
        else
            first_name[id] = input.name
        end
    end
    isempty(aliases) || throw(ArgumentError(
        "AD preparation requires unique default HAVE ports; aliased or " *
        "duplicate boundaries: " *
        join((":$(left) aliases :$(right)" for (left, right) in aliases), ", ")))
    nothing
end

# A HAVE value reachable from the active value is not constant model data: it
# is an active-derived boundary cut. Marking it Constant would sever a real
# derivative. The caller must instead derive it inside the selected kernel or
# use DifferentiationInterface directly with an explicit Cache contract.
function _ad_validate_constant_boundary(kernel::PreparedKernel, active_index::Int)
    graph = kernel.plan.graph
    active = inputs(kernel)[active_index]
    downstream = Set((canon_id(graph, active.id),))
    changed = true
    while changed
        changed = false
        for recipe in graph.recipes
            any(input -> canon_id(graph, input.id) in downstream,
                recipe.inputs) || continue
            for output in recipe.outputs
                id = canon_id(graph, output.id)
                if !(id in downstream)
                    push!(downstream, id)
                    changed = true
                end
            end
        end
    end

    derived = Symbol[]
    for (index, input) in pairs(inputs(kernel))
        index == active_index && continue
        canon_id(graph, input.id) in downstream && push!(derived, input.name)
    end
    isempty(derived) || throw(ArgumentError(
        "inactive HAVE port$(length(derived) == 1 ? "" : "s") " *
        join((":" * string(name) for name in derived), ", ") *
        " $(length(derived) == 1 ? "is" : "are") transitively downstream of " *
        "active port :$(active.name) and cannot be treated as " *
        "DifferentiationInterface.Constant; derive active-dependent values " *
        "inside the selected kernel, or use an explicit DI Cache boundary"))
    nothing
end

_ad_differentiable_value(::AbstractFloat) = true
_ad_differentiable_value(value::AbstractArray) =
    eltype(typeof(value)) <: AbstractFloat
_ad_differentiable_value(value::Tuple) =
    !isempty(value) && all(_ad_differentiable_value, value)
_ad_differentiable_value(::Any) = false

function _ad_validate_kernel(kernel::PreparedKernel, active_index::Int,
                             args::Tuple)
    length(args) == length(inputs(kernel)) || throw(ArgumentError(
        "selected HAVE boundary expects $(length(inputs(kernel))) values " *
        "$(Tuple(input.name for input in inputs(kernel))); got $(length(args))"))

    length(outputs(kernel)) == 1 || throw(ArgumentError(
        "AD gradient preparation requires exactly one selected WANT port; got " *
        string(Tuple(output.name for output in outputs(kernel)))))
    output = only(outputs(kernel))
    output_type = valtype(output)
    if !(output_type <: Number)
        observed_type = if isconcretetype(output_type)
            output_type
        else
            typeof(kernel(args...))
        end
        observed_type <: Number || throw(ArgumentError(
            "AD gradient preparation requires a scalar Number objective; " *
            ":$(output.name) has declared type $output_type and exemplar " *
            "result type $observed_type"))
    end

    active = inputs(kernel)[active_index]
    _ad_differentiable_value(args[active_index]) || throw(ArgumentError(
        "active HAVE port :$(active.name) received non-differentiable exemplar " *
        "type $(typeof(args[active_index])); expected floating-point scalar, " *
        "array, or tuple storage"))

    _ad_validate_constant_boundary(kernel, active_index)
    nothing
end

@generated function _ad_arguments(::Val{I}, args::A) where {I,A<:Tuple}
    N = fieldcount(A)
    1 <= I <= N || return :(throw(ArgumentError(
        "invalid active input index $I for $N kernel arguments")))
    contexts = [
        :(DifferentiationInterface.Constant(getfield(args, $index)))
        for index in 1:N if index != I
    ]
    :(getfield(args, $I), ($(contexts...),))
end

function _ad_resolve(resolver, args::Tuple, kwargs::NamedTuple)
    resolver(args...; kwargs...)
end

function _ad_call(kernel::PreparedKernel, resolved::Tuple, active)
    active_index = _ad_active_index(kernel, active)
    _ad_validate_kernel(kernel, active_index, resolved)
    call = _ad_kernel_call(kernel, resolved, Val(active_index))
    point, contexts = _ad_arguments(Val(active_index), resolved)
    call, point, contexts, active_index
end

function _ad_spec_kernel(spec::KernelSpec, want)
    _ad_validate_unique_haves(spec)
    selected_wants = _kernel_selection(spec, want, spec.want_names, :want)
    length(selected_wants) == 1 || throw(ArgumentError(
        "AD gradients require exactly one explicit WANT port; got " *
        string(Tuple(output.name for output in selected_wants))))
    prepare(plan(spec; want = only(selected_wants)))
end

function _ad_resolver(spec::KernelSpec)
    _kernel_signature_callable(tuple, spec.call_signature)
end

function _prepare_ad(kernel::PreparedKernel, resolver,
                     backend::DifferentiationInterface.AbstractADType,
                     args::Tuple, kwargs::NamedTuple, active)
    resolved = _ad_resolve(resolver, args, kwargs)
    call, point, contexts, active_index =
        _ad_call(kernel, resolved, active)
    preparation = DifferentiationInterface.prepare_gradient(
        call, backend, point, contexts...)
    PreparedADKernel{active_index,typeof(kernel),typeof(resolver),typeof(call),
                     typeof(backend),typeof(preparation)}(
        kernel, resolver, call, backend, preparation)
end

"""
    prepare_ad(spec, backend, args...; active, want, kwargs...) -> PreparedADKernel

Prepare a reusable DifferentiationInterface gradient of the explicit scalar
`want` port in a [`KernelSpec`](@ref). `active` names the one selected HAVE
port that remains differentiable; all other selected HAVE values are rebound
on every call and passed as `Constant` contexts.

The ordinary authored call surface is preserved: positional defaults and
keyword HAVE ports are resolved exactly as they are by `prepare(spec)`. The
preparation arguments are type/shape exemplars, not frozen data.
"""
function prepare_ad(spec::KernelSpec,
                    backend::DifferentiationInterface.AbstractADType,
                    args...; active, want, kwargs...)
    kernel = _ad_spec_kernel(spec, want)
    _prepare_ad(kernel, _ad_resolver(spec), backend, args,
                NamedTuple(kwargs), active)
end

"""
    prepare_ad(kernel, backend, args...; active) -> PreparedADKernel

Prepare a reusable gradient for a low-level [`PreparedKernel`](@ref) whose
boundary is already fully selected. Such kernels accept only their positional
HAVE values and must expose exactly one scalar WANT.
"""
function prepare_ad(kernel::PreparedKernel,
                    backend::DifferentiationInterface.AbstractADType,
                    args...; active, kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "a low-level PreparedKernel has a positional HAVE boundary and does " *
        "not accept keywords; use a KernelSpec to preserve authored keywords"))
    _prepare_ad(kernel, tuple, backend, args, NamedTuple(), active)
end

"""
    ad_gradient(spec, backend, args...; active, want, kwargs...)
    ad_gradient(kernel, backend, args...; active)
    ad_gradient(prepared, args...; kwargs...)

Compute a gradient with respect to one named active HAVE port. The `KernelSpec`
form selects an explicit scalar `want` and preserves authored defaults and
keywords. The low-level `PreparedKernel` form requires an already selected,
positional, single-scalar boundary. The reusable form uses a
[`PreparedADKernel`](@ref) returned by [`prepare_ad`](@ref).
"""
function ad_gradient(spec::KernelSpec,
                     backend::DifferentiationInterface.AbstractADType,
                     args...; active, want, kwargs...)
    kernel = _ad_spec_kernel(spec, want)
    resolver = _ad_resolver(spec)
    resolved = _ad_resolve(resolver, args, NamedTuple(kwargs))
    call, point, contexts, _ = _ad_call(kernel, resolved, active)
    DifferentiationInterface.gradient(call, backend, point, contexts...)
end

function ad_gradient(kernel::PreparedKernel,
                     backend::DifferentiationInterface.AbstractADType,
                     args...; active, kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "a low-level PreparedKernel has a positional HAVE boundary and does " *
        "not accept keywords; use a KernelSpec to preserve authored keywords"))
    call, point, contexts, _ = _ad_call(kernel, args, active)
    DifferentiationInterface.gradient(call, backend, point, contexts...)
end

function _ad_prepared_arguments(
        prepared::PreparedADKernel{I}, args, kwargs::NamedTuple) where {I}
    resolved = _ad_resolve(prepared.resolver, args, NamedTuple(kwargs))
    length(resolved) == length(inputs(prepared.kernel)) || throw(ArgumentError(
        "selected HAVE boundary expects $(length(inputs(prepared.kernel))) " *
        "values; got $(length(resolved))"))
    _ad_arguments(Val(I), resolved)
end

function ad_gradient(prepared::PreparedADKernel, args...; kwargs...)
    point, contexts = _ad_prepared_arguments(
        prepared, args, NamedTuple(kwargs))
    DifferentiationInterface.gradient(
        prepared.call, prepared.preparation, prepared.backend,
        point, contexts...)
end

"""
    ad_value_and_gradient!(prepared, gradient, args...; kwargs...)

Compute the scalar value and gradient for a reusable [`PreparedADKernel`](@ref),
writing the gradient into `gradient`. Runtime arguments follow the original RK
HAVE boundary; the prepared active port is reordered to DI position one, and
every other current argument is rebound as a fresh
`DifferentiationInterface.Constant` context.

Returns the `(value, gradient)` pair from
`DifferentiationInterface.value_and_gradient!`. The destination must be valid
for the active argument's gradient and is mutated in place. Like
[`ad_gradient`](@ref), this prepared object and its DI preparation are not
thread-safe; use one per concurrent caller.
"""
@generated function ad_value_and_gradient!(
        prepared::PreparedADKernel{I,K,typeof(tuple)}, gradient,
        args::Vararg{Any,N}) where {I,K,N}
    1 <= I <= N || return :(throw(ArgumentError(
        "prepared active input index $I is invalid for $N arguments")))
    contexts = [
        :(DifferentiationInterface.Constant(getfield(args, $index)))
        for index in 1:N if index != I
    ]
    quote
        length(inputs(prepared.kernel)) == $N || throw(ArgumentError(
            "selected HAVE boundary expects " *
            string(length(inputs(prepared.kernel))) *
            " values; got $N"))
        DifferentiationInterface.value_and_gradient!(
            prepared.call, gradient, prepared.preparation, prepared.backend,
            getfield(args, $I), $(contexts...))
    end
end

function ad_value_and_gradient!(
        prepared::PreparedADKernel, gradient, args...; kwargs...)
    point, contexts = _ad_prepared_arguments(
        prepared, args, NamedTuple(kwargs))
    DifferentiationInterface.value_and_gradient!(
        prepared.call, gradient, prepared.preparation, prepared.backend,
        point, contexts...)
end

inputs(prepared::PreparedADKernel) = inputs(prepared.kernel)
outputs(prepared::PreparedADKernel) = outputs(prepared.kernel)
code_expr(prepared::PreparedADKernel) = code_expr(prepared.kernel)

function Base.show(io::IO, prepared::PreparedADKernel{I}) where {I}
    print(io, "PreparedADKernel(active=:", inputs(prepared.kernel)[I].name,
          ", want=:", only(outputs(prepared.kernel)).name, ", kernel=")
    show(io, prepared.kernel)
    print(io, ")")
end

# --- Reactant-compiled automatic differentiation -----------------------------
# The AD analog of the primal Reactant path (`@compile sync=true kernel(args...)`).
# These take a native `PreparedADKernel` — which already owns the scalar-WANT /
# single-active-port validation and the authored-HAVE-order reorder — and compile
# a DifferentiationInterface gradient (or value-and-gradient) through Reactant.
#
# The differentiation engine stays the caller's DifferentiationInterface backend
# (the one passed to `prepare_ad`), so ReactiveKernels imports no concrete AD
# engine here: a caller-configured reverse-mode backend traces through Reactant
# with exact parity against the native reverse pass. The real methods live in
# `ext/ReactiveKernelsReactantExt.jl` and are selected when the active argument is
# a Reactant-traced value; without the Reactant weak dependency loaded (or with a
# host-array active argument) these raise a clear, actionable error instead of a
# bare `MethodError`.

function _reactant_ad_marker(prepared::PreparedADKernel{I}, args::Tuple) where {I}
    length(args) == length(inputs(prepared.kernel)) || throw(ArgumentError(
        "Reactant AD compilation expects the $(length(inputs(prepared.kernel)))-value " *
        "HAVE boundary $(Tuple(input.name for input in inputs(prepared.kernel))); " *
        "got $(length(args)) argument(s)"))
    getfield(args, I)
end

# Selected by the Reactant extension on `marker::Reactant.RArray` /
# `Reactant.RNumber`. This fallback fires when Reactant is not loaded or the
# active argument was not traced.
_reactant_compile_ad(::Val, ::PreparedADKernel, marker, args...; kwargs...) =
    throw(ArgumentError(
        "Reactant-compiled AD requires the Reactant weak dependency (`using Reactant`) " *
        "and a Reactant-traced active argument (e.g. `Reactant.to_rarray(active)`); " *
        "the active argument is a $(typeof(marker))"))

"""
    compile_ad_gradient(prepared::PreparedADKernel, traced_args...; sync = true)

Reactant/XLA-compile the gradient of a [`PreparedADKernel`](@ref) with respect to
its one active HAVE port. `traced_args` are the selected HAVE values in authored
order, already traced with `Reactant.to_rarray` (matching the primal Reactant
path). Returns a compiled callable that accepts the same traced boundary and
returns the gradient with respect to the active port; every inactive HAVE is
held constant, exactly as on the native [`ad_gradient`](@ref) path.

This is the AD analog of compiling the primal kernel with
`@compile sync = true kernel(traced_args...)`. It is only possible where the
primal kernel itself compiles through Reactant; where it does not, the underlying
`@compile` error propagates unchanged.

Requires the Reactant weak dependency to be loaded. The differentiation engine is
the DifferentiationInterface backend passed to [`prepare_ad`](@ref).
"""
function compile_ad_gradient(prepared::PreparedADKernel, args...; sync::Bool = true)
    _reactant_compile_ad(Val(:gradient), prepared,
                         _reactant_ad_marker(prepared, args), args...; sync)
end

"""
    compile_ad_value_and_gradient(prepared::PreparedADKernel, traced_args...; sync = true)

Reactant/XLA-compile the scalar value and gradient of a [`PreparedADKernel`](@ref)
together, mirroring the native [`ad_value_and_gradient!`](@ref) boundary. Returns
a compiled callable that accepts the traced HAVE boundary in authored order and
returns the `(value, gradient)` pair (a traced scalar and a traced gradient with
respect to the active port). This is the sampler-facing surface: one compiled
call yields both the potential and its gradient.

Like [`compile_ad_gradient`](@ref), this requires the Reactant weak dependency,
reuses the DifferentiationInterface backend from [`prepare_ad`](@ref), and only
compiles where the primal kernel itself compiles through Reactant.
"""
function compile_ad_value_and_gradient(prepared::PreparedADKernel, args...; sync::Bool = true)
    _reactant_compile_ad(Val(:value_and_gradient), prepared,
                         _reactant_ad_marker(prepared, args), args...; sync)
end
