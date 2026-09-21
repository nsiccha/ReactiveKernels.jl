"""
    PreparedADKernel

A prepared DifferentiationInterface gradient for a ReactiveKernels scalar
objective. Construct one with [`prepare_ad`](@ref), naming one selected HAVE
port or an ordered tuple of ports that remain active; every other selected
HAVE port is supplied to DifferentiationInterface as a
`DifferentiationInterface.Constant` context. A tuple selector is differentiated
as one structured point, so one reverse pass returns a tuple of gradients in
the selector's order.

The stored differentiation preparation is mutable and not thread-safe. Prepare
one `PreparedADKernel` per concurrent caller.
"""
struct PreparedADKernel{I,K,R,F,B,P,E}
    kernel::K
    resolver::R
    call::F
    backend::B
    preparation::P
    external_values::E
end

"""
    PreparedADPullback

A prepared DifferentiationInterface reverse pullback for one selected
ReactiveKernels WANT port. Construct one with [`prepare_ad_pullback`](@ref),
providing an output cotangent exemplar. Calls may then evaluate vector-Jacobian
products with [`ad_pullback`](@ref) or [`ad_value_and_pullback`](@ref).

Like [`PreparedADKernel`](@ref), the stored differentiation preparation is
mutable and not thread-safe. Prepare one object per concurrent caller.
"""
struct PreparedADPullback{I,K,R,F,B,P,E}
    kernel::K
    resolver::R
    call::F
    backend::B
    preparation::P
    external_values::E
end

# Low-level prepared kernels AD accepts: the allocating dataflow kernel and
# the non-allocating step-program kernel. Both expose the same positional
# HAVE boundary, single-WANT outputs, and plan graph; only the differentiated
# call construction differs (see `_ad_kernel_call`).
const _ADKernel = Union{PreparedKernel,NonAllocatingKernel}

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
# exact native callable and its live operations. No AD-specific kernel or AST
# is generated.
struct _ADNativeKernelCall{I,F,O}
    native::F
    ops::O
end

# The differentiated call for a `NonAllocatingKernel`: the elided unbound step
# program (`_ad_na_program`) with the operation table and cache driver held
# constant. The owned AD caches arrive as the trailing `Cache` context, so
# the backend shadows them itself; the kernel's borrowed primal caches are
# never exposed to differentiation.
struct _ADNonAllocatingKernelCall{I,F,O,A}
    f::F
    ops::O
    cache_apply::A
end

_ad_selector_indices(index::Int) = (index,)
_ad_selector_indices(indices::Tuple) = indices

# DifferentiationInterface's Enzyme extension cannot annotate a structured
# point that mixes duplicated storage (arrays) and active scalars. Represent a
# scalar component as mutable scalar storage so the whole tuple is duplicated,
# then unwrap it at the RK callable boundary and restore its cotangent before
# returning to the caller. Optional compiler extensions may keep their traced
# scalar wrappers unchanged by specializing `_ad_active_point_component`.
_ad_active_point_component(value::Number) = Ref(value)
_ad_active_point_component(value) = value
_ad_active_value(value::Base.RefValue) = value[]
_ad_active_value(value) = value

_ad_restore_cotangent(::Base.RefValue, cotangent::Base.RefValue) = cotangent[]
function _ad_restore_cotangent(point::Tuple, cotangent::Tuple)
    map(_ad_restore_cotangent, point, cotangent)
end
_ad_restore_cotangent(point, cotangent) = cotangent

@generated function (call::_ADNativeKernelCall{I})(
        active, contexts::Vararg{Any,N}) where {I,N}
    indices = _ad_selector_indices(I)
    input_count = N + length(indices)
    all(index -> 1 <= index <= input_count, indices) || return :(throw(
        ArgumentError("invalid active input selector $I for an RK AD call " *
                      "with $input_count inputs")))
    positional = Any[]
    context_index = 1
    for input_index in 1:input_count
        active_position = findfirst(==(input_index), indices)
        if active_position !== nothing
            push!(positional, I isa Int ? :active :
                  :(_ad_active_value(getfield(active, $active_position))))
        else
            push!(positional, :(getfield(contexts, $context_index)))
            context_index += 1
        end
    end
    :(call.native(call.ops, $(positional...)))
end

function _ad_operation_slots!(used, node)
    node === _OPS_ARG && return false
    node isa Expr || return true
    slot = _operation_slot(node)
    if slot !== nothing
        push!(used, slot)
        return true
    end
    all(child -> _ad_operation_slots!(used, child), node.args)
end

function _ad_native_ops(kernel::PreparedKernel)
    ops = kernel.ops
    any(op -> op isa _AuthoredScanOp, ops) || return ops
    native = kernel.f.native
    native isa RuntimeGeneratedFunctions.RuntimeGeneratedFunction || return ops
    # Inspect the compiled callable's cached source, not the separately mutable
    # display AST exposed by code_expr(kernel).
    ast = RuntimeGeneratedFunctions.get_expression(native)
    used = Set{Int}()
    _ad_operation_slots!(used, ast.args[2]) || return ops
    # Native scan steps are inlined, but the shared operation table also holds
    # the tensorized scan's complete prepared kernel. Rebuilding a bound-array
    # table with that unused graph metadata defeats readonly analysis on Julia
    # 1.13. Keep every live slot and the original positional ABI; remove only
    # unused scan metadata from this internal native call. A source transform
    # that accesses the table dynamically conservatively retains the whole table.
    ntuple(length(ops)) do index
        op = ops[index]
        op isa _AuthoredScanOp && !(index in used) ? nothing : op
    end
end

function _ad_kernel_call(kernel::PreparedKernel, args::Tuple, ::Val{I}) where {I}
    native_exemplars = _dynamic_tensorized_marker(args) === nothing
    if kernel.f isa Union{
            _ArrayFunctionPair,_EmbeddedFunctionPair,
            _DynamicEmbeddedFunctionPair} && native_exemplars
        ops = _ad_native_ops(kernel)
        # Bound views cross this Enzyme boundary as owning copies: a
        # `SubArray`-typed `Constant` operand defeats static activity
        # analysis, while identical owning contents differentiate cleanly.
        externalized, values = _externalize_bound_array_call(
            kernel.f.native, ops; materialize_view_copies = true)
        isempty(values) && return (
            _ADNativeKernelCall{I,typeof(kernel.f.native),typeof(ops)}(
                kernel.f.native, ops),
            (),
        )
        return _ADKernelCall{I,typeof(externalized)}(externalized), values
    end
    externalized, values = _externalize_bound_arrays(
        kernel; materialize_view_copies = true)
    _ADKernelCall{I,typeof(externalized)}(externalized), values
end

# Match one non-allocating cache step,
# `__cache_apply__(__caches__[j], __ops__[j], callargs...)`, returning the
# slot index and call arguments. Anything else is not a step.
function _ad_na_step_parts(node, ops_length::Int)
    node isa Expr && node.head === :call && length(node.args) >= 3 ||
        return nothing
    node.args[1] === _CACHE_APPLY_ARG || return nothing
    caches_ref = node.args[2]
    ops_ref = node.args[3]
    caches_ref isa Expr && caches_ref.head === :ref &&
        length(caches_ref.args) == 2 &&
        caches_ref.args[1] === _CACHES_ARG &&
        caches_ref.args[2] isa Int || return nothing
    ops_ref isa Expr && ops_ref.head === :ref &&
        length(ops_ref.args) == 2 && ops_ref.args[1] === _OPS_ARG &&
        ops_ref.args[2] isa Int || return nothing
    step = caches_ref.args[2]
    ops_ref.args[2] == step && 1 <= step <= ops_length || return nothing
    step, node.args[4:end]
end

# Taint-walk one program node, rewriting cache steps whose arguments are all
# free of the active input into plain operation calls. Returns the rewritten
# node and whether the active input reaches it. Unknown shapes stay tainted
# (kept as cache steps): elision is value-preserving either way, but keeping
# an active step preserves its buffer reuse while eliding one only costs it.
function _ad_na_walk(node, tainted::Set{Symbol}, ops::Tuple, caches::Tuple,
                     elided::Set{Int})
    parts = _ad_na_step_parts(node, length(ops))
    if parts !== nothing
        step, callargs = parts
        rewritten = Any[]
        step_tainted = false
        for arg in callargs
            new_arg, arg_tainted =
                _ad_na_walk(arg, tainted, ops, caches, elided)
            push!(rewritten, new_arg)
            step_tainted = step_tainted || arg_tainted
        end
        if !step_tainted && caches[step] isa Base.RefValue
            push!(elided, step)
            return Expr(:call, Expr(:ref, _OPS_ARG, step), rewritten...), false
        end
        return Expr(:call, _CACHE_APPLY_ARG, Expr(:ref, _CACHES_ARG, step),
                    Expr(:ref, _OPS_ARG, step), rewritten...), step_tainted
    end
    node isa Symbol && return node, node in tainted
    node isa GlobalRef && return node, false
    node isa LineNumberNode && return node, false
    node isa Expr || return node, false
    if node.head === :(=) && length(node.args) == 2 &&
            node.args[1] isa Symbol
        new_rhs, rhs_tainted =
            _ad_na_walk(node.args[2], tainted, ops, caches, elided)
        if rhs_tainted
            push!(tainted, node.args[1])
        else
            delete!(tainted, node.args[1])
        end
        return Expr(:(=), node.args[1], new_rhs), false
    end
    node_tainted = false
    new_args = Any[]
    for arg in node.args
        new_arg, arg_tainted = _ad_na_walk(arg, tainted, ops, caches, elided)
        push!(new_args, new_arg)
        node_tainted = node_tainted || arg_tainted
    end
    Expr(node.head, new_args...), node_tainted
end

# Re-wrap one seeded primal slot as a concretely-typed AD-owned slot, copying
# array contents so the AD program never mutates (or aliases) the kernel's
# borrowed primal buffers. Views become owning vectors: the slot only needs a
# same-shaped reusable buffer, not the caller's memory.
function _ad_na_concretize_cache(slot::Base.RefValue, step::Int, op)
    seeded = slot[]
    seeded === nothing && throw(ArgumentError(
        "AD preparation over a non-allocating kernel requires every retained " *
        "cache slot to be seeded by the exemplar primal call, but step " *
        "$step ($(_opname(op))) still holds `nothing`; every step must " *
        "execute once during preparation"))
    value = seeded isa Array ? copy(seeded) :
        seeded isa AbstractArray ? collect(seeded) : seeded
    Base.RefValue{typeof(value)}(value)
end

_ad_na_concretize_cache(::Nothing, step::Int, op) = nothing

# Bound views cross the AD boundary as owning copies with identical contents:
# a `SubArray`-typed constant operand defeats reverse-mode static activity
# analysis, while identical owning contents differentiate cleanly (the same
# freeze `prepare_ad` already applies to externalized dataflow operands).
_ad_na_ad_ops(ops::Tuple) = map(ops) do op
    op isa _BoundConstant && op.value isa AbstractArray ?
        _BoundConstant(_externalize_bound_value(op.value, true)) : op
end

# Build the differentiated program for a `NonAllocatingKernel`: seed the
# primal caches with the preparation exemplars, elide the cache steps that
# cannot see the active input (storing caller-owned constant data into a
# backend-shadowed slot would be a static-activity error), and compile the
# resulting unbound program alongside its owned AD caches and operation
# table.
function _ad_na_program(kernel::NonAllocatingKernel, active_selector,
                        exemplars::Tuple)
    ast = kernel.ast
    ast.head === :function && ast.args[1] isa Expr &&
        ast.args[1].head === :tuple &&
        length(ast.args[1].args) >= 3 &&
        ast.args[1].args[1] === _OPS_ARG &&
        ast.args[1].args[2] === _CACHES_ARG &&
        ast.args[1].args[3] === _CACHE_APPLY_ARG || throw(ArgumentError(
            "AD preparation over a non-allocating kernel requires the " *
            "unbound step-program form; the kernel AST has an unexpected shape"))
    have_syms = map(ast.args[1].args[4:end]) do arg
        arg isa Expr && arg.head === :(::) ? arg.args[1] : arg
    end
    length(have_syms) == length(inputs(kernel)) &&
        all(sym -> sym isa Symbol, have_syms) &&
        all(index -> 1 <= index <= length(have_syms),
            _ad_selector_indices(active_selector)) || throw(ArgumentError(
            "AD preparation over a non-allocating kernel requires the " *
            "active selector to name ports among the " *
            "$(length(inputs(kernel))) positional HAVE arguments"))
    kernel(exemplars...)
    tainted = Set{Symbol}(
        have_syms[index] for index in _ad_selector_indices(active_selector))
    elided = Set{Int}()
    ops, caches = kernel.ops, kernel.caches
    length(ops) == length(caches) || throw(ArgumentError(
        "AD preparation over a non-allocating kernel requires aligned " *
        "operation and cache tables; got $(length(ops)) operations and " *
        "$(length(caches)) caches"))
    new_stmts = Any[]
    for stmt in ast.args[2].args
        new_stmt, _ = _ad_na_walk(stmt, tainted, ops, caches, elided)
        push!(new_stmts, new_stmt)
    end
    ad_caches = ntuple(length(caches)) do index
        index in elided ? nothing :
            _ad_na_concretize_cache(caches[index], index, ops[index])
    end
    f = compile(Expr(:function, deepcopy(ast.args[1]),
                     Expr(:block, new_stmts...)))
    f, ad_caches, _ad_na_ad_ops(ops)
end

function _ad_kernel_call(kernel::NonAllocatingKernel, args::Tuple,
                         ::Val{I}) where {I}
    f, ad_caches, ad_ops = _ad_na_program(kernel, I, args)
    # A cacheless step program still carries the positional cache-table ABI,
    # but asking the backend to shadow `Tuple{Nothing,...}` marks a ghost-only
    # type differentiable. Keep that inert table Constant; any retained slot
    # still uses Cache so the backend owns and shadows its mutable storage.
    cache_context = all(isnothing, ad_caches) ?
        DifferentiationInterface.Constant(ad_caches) :
        DifferentiationInterface.Cache(ad_caches)
    _ADNonAllocatingKernelCall{I,typeof(f),typeof(ad_ops),
                               typeof(kernel.cache_apply)}(
        f, ad_ops, kernel.cache_apply),
    (cache_context,)
end

@generated function (call::_ADKernelCall{I})(
        active, contexts::Vararg{Any,N}) where {I,N}
    indices = _ad_selector_indices(I)
    input_count = N + length(indices)
    all(index -> 1 <= index <= input_count, indices) || return :(throw(
        ArgumentError("invalid active input selector $I for an RK AD call " *
                      "with $input_count inputs")))
    arguments = Any[]
    context_index = 1
    for input_index in 1:input_count
        active_position = findfirst(==(input_index), indices)
        if active_position !== nothing
            push!(arguments, I isa Int ? :active :
                  :(_ad_active_value(getfield(active, $active_position))))
        else
            push!(arguments, :(getfield(contexts, $context_index)))
            context_index += 1
        end
    end
    :(call.kernel($(arguments...)))
end

# The trailing context is the owned AD cache tuple; the leading N - 1 restore
# the inactive HAVE arguments around the active one, exactly as above.
@generated function (call::_ADNonAllocatingKernelCall{I})(
        active, contexts::Vararg{Any,N}) where {I,N}
    indices = _ad_selector_indices(I)
    input_count = N + length(indices) - 1
    all(index -> 1 <= index <= input_count, indices) || return :(throw(
        ArgumentError("invalid active input selector $I for an RK " *
                      "non-allocating AD call with $input_count inputs")))
    arguments = Any[]
    context_index = 1
    for input_index in 1:input_count
        active_position = findfirst(==(input_index), indices)
        if active_position !== nothing
            push!(arguments, I isa Int ? :active :
                  :(_ad_active_value(getfield(active, $active_position))))
        else
            push!(arguments, :(getfield(contexts, $context_index)))
            context_index += 1
        end
    end
    :(call.f(call.ops, getfield(contexts, $N), call.cache_apply,
             $(arguments...)))
end

function _ad_active_index(kernel::_ADKernel, active::Symbol)
    matches = findall(input -> input.name === active, inputs(kernel))
    isempty(matches) && throw(ArgumentError(
        "active port :$active is not in the selected HAVE boundary " *
        "$(Tuple(input.name for input in inputs(kernel)))"))
    length(matches) == 1 || throw(ArgumentError(
        "active port name :$active is ambiguous in the selected HAVE boundary"))
    only(matches)
end

function _ad_active_index(kernel::_ADKernel, active::Value)
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

function _ad_active_index(::_ADKernel, active)
    throw(ArgumentError(
        "active must identify a selected HAVE port by Symbol or Value; got " *
        string(typeof(active))))
end

function _ad_active_selector(kernel::_ADKernel, active::Tuple)
    isempty(active) && throw(ArgumentError(
        "active tuple must identify at least one selected HAVE port"))
    indices = map(identifier -> _ad_active_index(kernel, identifier), active)
    length(unique(indices)) == length(indices) || throw(ArgumentError(
        "active tuple names the same selected HAVE port more than once: " *
        string(active)))
    indices
end

_ad_active_selector(kernel::_ADKernel, active) =
    _ad_active_index(kernel, active)

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
function _ad_validate_constant_boundary(kernel::_ADKernel, active_selector)
    graph = kernel.plan.graph
    active_indices = _ad_selector_indices(active_selector)
    active_inputs = map(index -> inputs(kernel)[index], active_indices)
    downstream = Set(canon_id(graph, active.id) for active in active_inputs)
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
        index in active_indices && continue
        canon_id(graph, input.id) in downstream && push!(derived, input.name)
    end
    active_names = join((":" * string(input.name) for input in active_inputs), ", ")
    isempty(derived) || throw(ArgumentError(
        "inactive HAVE port$(length(derived) == 1 ? "" : "s") " *
        join((":" * string(name) for name in derived), ", ") *
        " $(length(derived) == 1 ? "is" : "are") transitively downstream of " *
        "active port$(length(active_indices) == 1 ? "" : "s") $active_names " *
        "and cannot be treated as " *
        "DifferentiationInterface.Constant; derive active-dependent values " *
        "inside the selected kernel, or use an explicit DI Cache boundary"))
    nothing
end

_ad_differentiable_value(::AbstractFloat) = true
_ad_differentiable_value(value::AbstractArray) =
    eltype(typeof(value)) <: AbstractFloat
_ad_differentiable_value(value::Tuple) =
    !isempty(value) && all(_ad_differentiable_value, value)
_ad_differentiable_value(value::NamedTuple) =
    !isempty(value) && all(_ad_differentiable_value, values(value))
_ad_differentiable_value(::Any) = false

function _ad_validate_kernel(kernel::_ADKernel, active_selector,
                             args::Tuple; scalar_output::Bool = true)
    length(args) == length(inputs(kernel)) || throw(ArgumentError(
        "selected HAVE boundary expects $(length(inputs(kernel))) values " *
        "$(Tuple(input.name for input in inputs(kernel))); got $(length(args))"))

    length(outputs(kernel)) == 1 || throw(ArgumentError(
        "AD preparation requires exactly one selected WANT port; got " *
        string(Tuple(output.name for output in outputs(kernel)))))
    if scalar_output
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
    end

    for active_index in _ad_selector_indices(active_selector)
        active = inputs(kernel)[active_index]
        _ad_differentiable_value(args[active_index]) || throw(ArgumentError(
            "active HAVE port :$(active.name) received non-differentiable " *
            "exemplar type $(typeof(args[active_index])); expected " *
            "floating-point scalar, array, tuple, or NamedTuple storage"))
    end

    _ad_validate_constant_boundary(kernel, active_selector)
    nothing
end

@generated function _ad_arguments(::Val{I}, args::A) where {I,A<:Tuple}
    N = fieldcount(A)
    indices = _ad_selector_indices(I)
    all(index -> 1 <= index <= N, indices) || return :(throw(ArgumentError(
        "invalid active input selector $I for $N kernel arguments")))
    # Values that already carry a DifferentiationInterface context wrapper
    # (the non-allocating path's owned `Cache`) pass through untouched; only
    # raw boundary values become `Constant` contexts.
    contexts = [
        A.parameters[index] <: DifferentiationInterface.Context ?
            :(getfield(args, $index)) :
            :(DifferentiationInterface.Constant(getfield(args, $index)))
        for index in 1:N if !(index in indices)
    ]
    point = I isa Int ? :(getfield(args, $I)) :
        Expr(:tuple, (:(
            _ad_active_point_component(getfield(args, $index)))
            for index in indices)...)
    :($point, ($(contexts...),))
end

# A native (non-traced) prepared-AD call inside a Reactant trace never reaches
# native Enzyme: Reactant's autodiff overlay intercepts the call with all-native
# arguments and the staged derivative comes back a silent zero gradient, while
# the value stays correct. Every native call boundary checks its already-split
# point here; the Reactant extension implements the in-trace refusal, and this
# fallback keeps the core independent of the weak dependency.
_ad_trace_sanity(point, contexts) = nothing

function _ad_resolve(resolver, args::Tuple, kwargs::NamedTuple)
    resolver(args...; kwargs...)
end

function _ad_call(kernel::_ADKernel, resolved::Tuple, active;
                  scalar_output::Bool = true)
    active_selector = _ad_active_selector(kernel, active)
    _ad_validate_kernel(kernel, active_selector, resolved; scalar_output)
    call, external_values =
        _ad_kernel_call(kernel, resolved, Val(active_selector))
    point, contexts = _ad_arguments(
        Val(active_selector), (resolved..., external_values...))
    call, point, contexts, active_selector, external_values
end

function _ad_spec_kernel(spec::KernelSpec, want, bound = NamedTuple())
    _ad_validate_unique_haves(spec)
    selected_wants = _kernel_selection(spec, want, spec.want_names, :want)
    length(selected_wants) == 1 || throw(ArgumentError(
        "AD preparation requires exactly one explicit WANT port; got " *
        string(Tuple(output.name for output in selected_wants))))
    isempty(bound) && return prepare(plan(spec; want = only(selected_wants)))
    prepare(plan(spec; want = only(selected_wants));
            bound = _kernel_bound_pairs(spec, bound))
end

# A partially-evaluated kernel is positional over the remaining HAVE ports;
# the authored keyword/default resolution layer maps the full signature and
# therefore does not apply.
function _ad_reject_bound_keywords(kwargs::NamedTuple)
    isempty(kwargs) || throw(ArgumentError(
        "a bound AD preparation is positional over the remaining HAVE ports; " *
        "authored keyword arguments do not apply"))
    kwargs
end

function _ad_resolver(spec::KernelSpec)
    _kernel_signature_callable(tuple, spec.call_signature)
end

# DifferentiationInterface provides each backend's preparation methods through
# the backend package's own extension (e.g. `using Enzyme` loads
# `DifferentiationInterfaceEnzymeExt`); constructing the backend value alone —
# `AutoEnzyme()` needs only ADTypes — is not enough. Without the extension,
# preparation dies inside DifferentiationInterface with a bare `MethodError`
# on an internal `_prepare_pullback_aux` / `_prepare_pushforward_aux` symbol,
# so every entry point that hands a caller backend to DifferentiationInterface
# checks first and names the missing `using` instead. The backend-type naming
# rule mirrors DifferentiationInterface's own `required_packages`; a backend
# shape this cannot derive fails open and DifferentiationInterface speaks.
function _ad_required_packages(backend::DifferentiationInterface.AbstractADType)
    packages = String[]
    _ad_required_packages!(packages, typeof(backend)) || return nothing
    return packages
end

function _ad_required_packages!(packages, ::Type{B}) where {B}
    parameters = B isa DataType ? B.parameters : ()
    if (nameof(B) === :SecondOrder || nameof(B) === :MixedMode) &&
            length(parameters) == 2 && parameters[1] isa DataType &&
            parameters[2] isa DataType
        return _ad_required_packages!(packages, parameters[1]) &&
            _ad_required_packages!(packages, parameters[2])
    elseif nameof(B) === :AutoSparse && length(parameters) == 1 &&
            parameters[1] isa DataType
        "SparseMatrixColorings" in packages ||
            push!(packages, "SparseMatrixColorings")
        return _ad_required_packages!(packages, parameters[1])
    end
    name = String(nameof(B))
    startswith(name, "Auto") || return false
    package = name[5:end]
    isempty(package) && return false
    package in packages || push!(packages, package)
    return true
end

function _ad_require_backend_packages(
        backend::DifferentiationInterface.AbstractADType)
    required = _ad_required_packages(backend)
    required === nothing && return nothing
    loaded = Set{String}()
    for loaded_module in values(Base.loaded_modules)
        push!(loaded, String(nameof(loaded_module)))
    end
    missing = filter(package -> !(package in loaded), required)
    isempty(missing) && return nothing
    imports = join(missing, ", ")
    throw(ArgumentError(
        "AD preparation with backend `$backend` requires the backend " *
        "package to be loaded (`using $imports`); constructing the " *
        "backend value alone does not load DifferentiationInterface's " *
        "backend extension, which provides the preparation methods"))
end

function _prepare_ad(kernel::_ADKernel, resolver,
                     backend::DifferentiationInterface.AbstractADType,
                     args::Tuple, kwargs::NamedTuple, active)
    _ad_require_backend_packages(backend)
    resolved = _ad_resolve(resolver, args, kwargs)
    call, point, contexts, active_selector, external_values =
        _ad_call(kernel, resolved, active)
    preparation = DifferentiationInterface.prepare_gradient(
        call, backend, point, contexts...)
    PreparedADKernel{active_selector,typeof(kernel),typeof(resolver),typeof(call),
                     typeof(backend),typeof(preparation),
                     typeof(external_values)}(
        kernel, resolver, call, backend, preparation, external_values)
end

function _prepare_ad_pullback(kernel::_ADKernel, resolver,
                              backend::DifferentiationInterface.AbstractADType,
                              seed, args::Tuple, kwargs::NamedTuple, active)
    _ad_require_backend_packages(backend)
    resolved = _ad_resolve(resolver, args, kwargs)
    call, point, contexts, active_selector, external_values =
        _ad_call(kernel, resolved, active; scalar_output = false)
    preparation = DifferentiationInterface.prepare_pullback(
        call, backend, point, (seed,), contexts...)
    PreparedADPullback{
        active_selector,typeof(kernel),typeof(resolver),typeof(call),
        typeof(backend),typeof(preparation),typeof(external_values),
    }(kernel, resolver, call, backend, preparation, external_values)
end

"""
    prepare_ad(spec, backend, args...; active, want, bound=(;), kwargs...) -> PreparedADKernel

Prepare a reusable DifferentiationInterface gradient of the explicit scalar
`want` port in a [`KernelSpec`](@ref). `active` names one selected HAVE port or
an ordered tuple of ports that remain differentiable; all other selected HAVE
values are rebound on every call and passed as `Constant` contexts. A tuple
selector is differentiated as one structured point, and gradients are returned
in the same order.

The ordinary authored call surface is preserved: positional defaults and
keyword HAVE ports are resolved exactly as they are by `prepare(spec)`. The
preparation arguments are type/shape exemplars, not frozen data.

The backend's package must be loaded in the calling session (`using Enzyme`
for `AutoEnzyme`); the backend value alone does not load
DifferentiationInterface's backend extension, and preparation without it
fails loudly naming the missing `using`.

A non-empty `bound` NamedTuple runs the [`partial_evaluation`](@ref) pre-pass
first: the named ports are fixed to the supplied values, their data-only
subgraph executes once during preparation, and both the differentiated call
and its `Constant` contexts cover only the remaining ports. `args` and
`active` then refer to those remaining ports positionally; authored keyword
arguments do not apply to a bound preparation. Array-valued residual constants
are passed to the backend as hidden `Constant` contexts rather than captured in
the differentiated callable; this does not change the public HAVE boundary.
Bound views cross as owning copies with identical contents, since a
prebuilt view operand defeats reverse-mode static activity analysis.
"""
function prepare_ad(spec::KernelSpec,
                    backend::DifferentiationInterface.AbstractADType,
                    args...; active, want, bound = NamedTuple(), kwargs...)
    kernel = _ad_spec_kernel(spec, want, bound)
    if !isempty(bound)
        _ad_reject_bound_keywords(NamedTuple(kwargs))
        return _prepare_ad(kernel, tuple, backend, args, NamedTuple(), active)
    end
    _prepare_ad(kernel, _ad_resolver(spec), backend, args,
                NamedTuple(kwargs), active)
end

"""
    prepare_ad(kernel, backend, args...; active) -> PreparedADKernel

Prepare a reusable gradient for a low-level [`PreparedKernel`](@ref) whose
boundary is already fully selected. Such kernels accept only their positional
HAVE values and must expose exactly one scalar WANT. `active` may be one port
identifier or an ordered tuple of identifiers.

The backend's package must be loaded in the calling session (`using Enzyme`
for `AutoEnzyme`); the backend value alone does not load
DifferentiationInterface's backend extension, and preparation without it
fails loudly naming the missing `using`.
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
    prepare_ad(kernel, backend, args...; active) -> PreparedADKernel

Prepare a reusable gradient for a low-level [`NonAllocatingKernel`](@ref)
whose boundary is already fully selected. Such kernels accept only their
positional HAVE values and must expose exactly one scalar WANT. `active` may be
one port identifier or an ordered tuple of identifiers.

The preparation seeds the kernel's caches once with the exemplar arguments,
then differentiates the same step program through AD-owned concretely-typed
cache copies threaded as a `DifferentiationInterface.Cache` context, so the
backend shadows them itself. Cache steps that cannot see the active input
(bound constants in particular) run as plain operation calls inside the
differentiated program: storing caller-owned constant data into a shadowed
slot would be a static-activity error. The kernel's borrowed primal caches
are never exposed to the backend, and primal calls keep using them
independently of the prepared object.

Result types must be stable across calls: a step whose result type changes
after preparation fails loudly instead of silently reseeding. Like every
[`PreparedADKernel`](@ref), the stored preparation is mutable and not
thread-safe; prepare one object per concurrent caller.

Forward-mode backends that push dual numbers through the differentiated
arguments (e.g. `AutoForwardDiff`) are not supported: the cache slots cannot
carry duals, and preparation fails loudly inside the backend. Reverse-mode
(`AutoEnzyme`) and primal-call-based (finite-difference) backends work.

The backend's package must be loaded in the calling session (`using Enzyme`
for `AutoEnzyme`); the backend value alone does not load
DifferentiationInterface's backend extension, and preparation without it
fails loudly naming the missing `using`.
"""
function prepare_ad(kernel::NonAllocatingKernel,
                    backend::DifferentiationInterface.AbstractADType,
                    args...; active, kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "a low-level NonAllocatingKernel has a positional HAVE boundary and " *
        "does not accept keywords; use a KernelSpec to preserve authored " *
        "keywords"))
    _prepare_ad(kernel, tuple, backend, args, NamedTuple(), active)
end

# A batched gradient callable owns the scalar AD preparation plus the same
# trailing-axis metadata as `ReplicatedKernel`. The distinction matters: its
# scalar target takes an already-reordered DI point and Constant contexts, not
# the ordinary kernel ABI.
struct _ReplicatedADKernel{B,BT,AT,K,IN}
    prepared::K
    inputs::IN
end

@inline function (k::_ReplicatedADKernel{B})(args...) where {B}
    length(args) == length(k.inputs) || throw(MethodError(k, args))
    _replica_ad_call(k, args, getfield(args, first(B)))
end

inputs(k::_ReplicatedADKernel) = k.inputs
outputs(k::_ReplicatedADKernel) = outputs(k.prepared.kernel)
code_expr(k::_ReplicatedADKernel) = code_expr(k.prepared.kernel)

function Base.show(io::IO, k::_ReplicatedADKernel{B}) where {B}
    names = Tuple(k.inputs[i].name for i in B)
    print(io, "ReplicatedADKernel(batched=", names, ", target=")
    show(io, k.prepared)
    print(io, ")")
end

function replica(prepared::PreparedADKernel{I}; batched) where {I}
    boundary = inputs(prepared.kernel)
    indices = _replica_batch_indices(boundary, batched)
    input_types = Tuple{(valtype(boundary[i]) for i in indices)...}
    foreach(_replica_rank, input_types.parameters)
    active_type = I isa Int ? valtype(boundary[I]) :
        Tuple{(valtype(boundary[index]) for index in I)...}
    _ReplicatedADKernel{indices,input_types,active_type,
                        typeof(prepared),typeof(boundary)}(
        prepared, boundary)
end

function _replica_ad_validation(k::_ReplicatedADKernel{B}, args, marker) where {B}
    replica_count = size(marker, ndims(marker))
    for index in B
        arg = getfield(args, index)
        input = k.inputs[index]
        expected_rank = _replica_rank(valtype(input)) + 1
        ndims(arg) == expected_rank || throw(DimensionMismatch(
            "replica port :$(input.name) has rank $(ndims(arg)); " *
            "expected $expected_rank (scalar rank plus one trailing replica axis)"))
        size(arg, ndims(arg)) == replica_count || throw(DimensionMismatch(
            "replica port :$(input.name) has $(size(arg, ndims(arg))) " *
            "replicas; expected $replica_count"))
    end
    replica_count
end

@inline _replica_ad_native_arg(arg, ::Type{T}, replica_index) where {T<:Number} =
    arg[replica_index]
@inline _replica_ad_native_arg(arg, ::Type{T}, replica_index) where {T<:AbstractArray} =
    copy(selectdim(arg, ndims(arg), replica_index))

function _replica_ad_scalar_args(
        ::_ReplicatedADKernel{B,BT}, args, replica_index) where {B,BT}
    ntuple(length(args)) do argument_index
        position = findfirst(==(argument_index), B)
        position === nothing ? getfield(args, argument_index) :
            _replica_ad_native_arg(
                getfield(args, argument_index),
                BT.parameters[position], replica_index)
    end
end

function _replica_ad_call(
        k::_ReplicatedADKernel{B,BT,AT}, args, marker) where {B,BT,AT}
    count = _replica_ad_validation(k, args, getfield(args, first(B)))
    results = map(1:count) do replica_index
        scalar_args = _replica_ad_scalar_args(k, args, replica_index)
        ad_value_and_gradient(k.prepared, scalar_args...)
    end
    _replica_ad_stack(first.(results), valtype(only(outputs(k)))),
        _replica_ad_stack(last.(results), AT)
end

function ad_value_and_gradient(
        k::_ReplicatedADKernel{B,BT,AT}, args...) where {B,BT,AT}
    _replica_ad_call(k, args, getfield(args, first(B)))
end

function ad_gradient(
        k::_ReplicatedADKernel, args...)
    last(_replica_ad_call(k, args, getfield(args, first(B))))
end

_replica_ad_stack(values, ::Type{T}) where {T<:Number} = collect(values)
_replica_ad_stack(values, ::Type{T}) where {T<:AbstractArray} = stack(values)
function _replica_ad_stack(values, ::Type{T}) where {T<:Tuple}
    ntuple(length(T.parameters)) do component
        _replica_ad_stack(
            map(value -> getfield(value, component), values),
            T.parameters[component])
    end
end
_replica_ad_stack(values, ::Type{T}) where {T} = throw(ArgumentError(
    "replica AD active ports must be Numbers or AbstractArrays; got $T"))

"""
    prepare_ad_pullback(spec, backend, seed, args...;
                        active, want, kwargs...) -> PreparedADPullback
    prepare_ad_pullback(kernel, backend, seed, args...;
                        active) -> PreparedADPullback

Prepare a reusable reverse pullback (vector-Jacobian product) for one explicit
`want` port. `seed` is an exemplar output cotangent. Unlike [`prepare_ad`](@ref),
the selected WANT may be non-scalar. One HAVE port or an ordered tuple of HAVE
ports is active, and all other current HAVE values are rebound as `Constant`
contexts on every call.

The backend's package must be loaded in the calling session (`using Enzyme`
for `AutoEnzyme`); the backend value alone does not load
DifferentiationInterface's backend extension, and preparation without it
fails loudly naming the missing `using`.
"""
function prepare_ad_pullback(
        spec::KernelSpec, backend::DifferentiationInterface.AbstractADType,
        seed, args...; active, want, kwargs...)
    kernel = _ad_spec_kernel(spec, want)
    _prepare_ad_pullback(
        kernel, _ad_resolver(spec), backend, seed, args,
        NamedTuple(kwargs), active)
end

function prepare_ad_pullback(
        kernel::PreparedKernel,
        backend::DifferentiationInterface.AbstractADType,
        seed, args...; active, kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "a low-level PreparedKernel has a positional HAVE boundary and does " *
        "not accept keywords; use a KernelSpec to preserve authored keywords"))
    _prepare_ad_pullback(
        kernel, tuple, backend, seed, args, NamedTuple(), active)
end

function prepare_ad_pullback(
        kernel::NonAllocatingKernel,
        backend::DifferentiationInterface.AbstractADType,
        seed, args...; active, kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "a low-level NonAllocatingKernel has a positional HAVE boundary and " *
        "does not accept keywords; use a KernelSpec to preserve authored " *
        "keywords"))
    _prepare_ad_pullback(
        kernel, tuple, backend, seed, args, NamedTuple(), active)
end

"""
    ad_gradient(spec, backend, args...; active, want, kwargs...)
    ad_gradient(kernel, backend, args...; active)
    ad_gradient(prepared, args...; kwargs...)

Compute a gradient with respect to one named active HAVE port or an ordered
tuple of ports. The `KernelSpec` form selects an explicit scalar `want` and
preserves authored defaults and keywords. The low-level `PreparedKernel` /
`NonAllocatingKernel` forms require an already selected, positional,
single-scalar boundary. The reusable form uses a [`PreparedADKernel`](@ref)
returned by [`prepare_ad`](@ref). Tuple selectors return tuple gradients in the
same order from one differentiation call.

The one-shot forms require the backend's package to be loaded in the calling
session (`using Enzyme` for `AutoEnzyme`); the backend value alone does not
load DifferentiationInterface's backend extension.
"""
function ad_gradient(spec::KernelSpec,
                     backend::DifferentiationInterface.AbstractADType,
                     args...; active, want, bound = NamedTuple(), kwargs...)
    _ad_require_backend_packages(backend)
    kernel = _ad_spec_kernel(spec, want, bound)
    if !isempty(bound)
        _ad_reject_bound_keywords(NamedTuple(kwargs))
        call, point, contexts, _, _ = _ad_call(kernel, args, active)
        _ad_trace_sanity(point, contexts)
        gradient = DifferentiationInterface.gradient(
            call, backend, point, contexts...)
        return _ad_restore_cotangent(point, gradient)
    end
    resolver = _ad_resolver(spec)
    resolved = _ad_resolve(resolver, args, NamedTuple(kwargs))
    call, point, contexts, _, _ = _ad_call(kernel, resolved, active)
    _ad_trace_sanity(point, contexts)
    gradient = DifferentiationInterface.gradient(
        call, backend, point, contexts...)
    _ad_restore_cotangent(point, gradient)
end

function ad_gradient(kernel::PreparedKernel,
                     backend::DifferentiationInterface.AbstractADType,
                     args...; active, kwargs...)
    _ad_require_backend_packages(backend)
    isempty(kwargs) || throw(ArgumentError(
        "a low-level PreparedKernel has a positional HAVE boundary and does " *
        "not accept keywords; use a KernelSpec to preserve authored keywords"))
    call, point, contexts, _, _ = _ad_call(kernel, args, active)
    _ad_trace_sanity(point, contexts)
    gradient = DifferentiationInterface.gradient(
        call, backend, point, contexts...)
    _ad_restore_cotangent(point, gradient)
end

function ad_gradient(kernel::NonAllocatingKernel,
                     backend::DifferentiationInterface.AbstractADType,
                     args...; active, kwargs...)
    _ad_require_backend_packages(backend)
    isempty(kwargs) || throw(ArgumentError(
        "a low-level NonAllocatingKernel has a positional HAVE boundary and " *
        "does not accept keywords; use a KernelSpec to preserve authored " *
        "keywords"))
    call, point, contexts, _, _ = _ad_call(kernel, args, active)
    _ad_trace_sanity(point, contexts)
    gradient = DifferentiationInterface.gradient(
        call, backend, point, contexts...)
    _ad_restore_cotangent(point, gradient)
end

function _ad_prepared_arguments(
        prepared::PreparedADKernel{I}, args, kwargs::NamedTuple) where {I}
    resolved = _ad_resolve(prepared.resolver, args, NamedTuple(kwargs))
    length(resolved) == length(inputs(prepared.kernel)) || throw(ArgumentError(
        "selected HAVE boundary expects $(length(inputs(prepared.kernel))) " *
        "values; got $(length(resolved))"))
    point, contexts =
        _ad_arguments(Val(I), (resolved..., prepared.external_values...))
    _ad_trace_sanity(point, contexts)
    point, contexts
end

function ad_gradient(prepared::PreparedADKernel, args...; kwargs...)
    point, contexts = _ad_prepared_arguments(
        prepared, args, NamedTuple(kwargs))
    gradient = DifferentiationInterface.gradient(
        prepared.call, prepared.preparation, prepared.backend,
        point, contexts...)
    _ad_restore_cotangent(point, gradient)
end

"""
    ad_value_and_gradient(prepared, args...; kwargs...)

Compute a scalar value and gradient without requiring caller-owned gradient
storage. This is the structured counterpart to [`ad_value_and_gradient!`](@ref):
it preserves DifferentiationInterface's returned cotangent structure, including
`NamedTuple` active inputs supported by the backend.

With Reactant loaded, a traced scalar or array active input stages this
operation inside the enclosing compiled program, using the same primal kernel
and DI backend. It does not reuse the native-input DI preparation in that trace.
"""
function ad_value_and_gradient(
        prepared::PreparedADKernel, args...; kwargs...)
    point, contexts = _ad_prepared_arguments(
        prepared, args, NamedTuple(kwargs))
    _ad_prepared_value_and_gradient(prepared, point, contexts)
end

function _ad_prepared_value_and_gradient(prepared, point, contexts)
    value, gradient = DifferentiationInterface.value_and_gradient(
        prepared.call, prepared.preparation, prepared.backend,
        point, contexts...)
    value, _ad_restore_cotangent(point, gradient)
end

"""
    ad_pullback(spec, backend, seed, args...; active, want, kwargs...)
    ad_pullback(kernel, backend, seed, args...; active)
    ad_pullback(prepared, seed, args...; kwargs...)

Evaluate a reverse pullback for one output cotangent `seed`, returning the
cotangent of the selected active HAVE port. For a vector-valued WANT this is the
vector-Jacobian product `J' * seed`; constructing a full Jacobian still requires
multiple seeds.

The one-shot forms require the backend's package to be loaded in the calling
session (`using Enzyme` for `AutoEnzyme`); the backend value alone does not
load DifferentiationInterface's backend extension.
"""
function ad_pullback(spec::KernelSpec,
                     backend::DifferentiationInterface.AbstractADType,
                     seed, args...; active, want, kwargs...)
    _ad_require_backend_packages(backend)
    kernel = _ad_spec_kernel(spec, want)
    resolver = _ad_resolver(spec)
    resolved = _ad_resolve(resolver, args, NamedTuple(kwargs))
    call, point, contexts, _, _ =
        _ad_call(kernel, resolved, active; scalar_output = false)
    _ad_trace_sanity(point, contexts)
    cotangent = only(DifferentiationInterface.pullback(
        call, backend, point, (seed,), contexts...))
    _ad_restore_cotangent(point, cotangent)
end

function ad_pullback(kernel::PreparedKernel,
                     backend::DifferentiationInterface.AbstractADType,
                     seed, args...; active, kwargs...)
    _ad_require_backend_packages(backend)
    isempty(kwargs) || throw(ArgumentError(
        "a low-level PreparedKernel has a positional HAVE boundary and does " *
        "not accept keywords; use a KernelSpec to preserve authored keywords"))
    call, point, contexts, _, _ =
        _ad_call(kernel, args, active; scalar_output = false)
    _ad_trace_sanity(point, contexts)
    cotangent = only(DifferentiationInterface.pullback(
        call, backend, point, (seed,), contexts...))
    _ad_restore_cotangent(point, cotangent)
end

function ad_pullback(kernel::NonAllocatingKernel,
                     backend::DifferentiationInterface.AbstractADType,
                     seed, args...; active, kwargs...)
    _ad_require_backend_packages(backend)
    isempty(kwargs) || throw(ArgumentError(
        "a low-level NonAllocatingKernel has a positional HAVE boundary and " *
        "does not accept keywords; use a KernelSpec to preserve authored " *
        "keywords"))
    call, point, contexts, _, _ =
        _ad_call(kernel, args, active; scalar_output = false)
    _ad_trace_sanity(point, contexts)
    cotangent = only(DifferentiationInterface.pullback(
        call, backend, point, (seed,), contexts...))
    _ad_restore_cotangent(point, cotangent)
end

function _ad_prepared_arguments(
        prepared::PreparedADPullback{I}, args, kwargs::NamedTuple) where {I}
    resolved = _ad_resolve(prepared.resolver, args, NamedTuple(kwargs))
    length(resolved) == length(inputs(prepared.kernel)) || throw(ArgumentError(
        "selected HAVE boundary expects $(length(inputs(prepared.kernel))) " *
        "values; got $(length(resolved))"))
    point, contexts =
        _ad_arguments(Val(I), (resolved..., prepared.external_values...))
    _ad_trace_sanity(point, contexts)
    point, contexts
end

function ad_pullback(prepared::PreparedADPullback, seed, args...; kwargs...)
    point, contexts = _ad_prepared_arguments(
        prepared, args, NamedTuple(kwargs))
    cotangent = only(DifferentiationInterface.pullback(
        prepared.call, prepared.preparation, prepared.backend,
        point, (seed,), contexts...))
    _ad_restore_cotangent(point, cotangent)
end

"""
    ad_value_and_pullback(prepared, seed, args...; kwargs...)

Evaluate the selected WANT and its reverse pullback together. Returns
`(value, active_cotangent)`, unwrapping DifferentiationInterface's
one-direction tuple. A tuple active selector produces a tuple cotangent in the
selector's order. The prepared object is reusable with new runtime arguments
and seeds compatible with its preparation exemplars.
"""
function ad_value_and_pullback(
        prepared::PreparedADPullback, seed, args...; kwargs...)
    point, contexts = _ad_prepared_arguments(
        prepared, args, NamedTuple(kwargs))
    value, pullbacks = DifferentiationInterface.value_and_pullback(
        prepared.call, prepared.preparation, prepared.backend,
        point, (seed,), contexts...)
    value, _ad_restore_cotangent(point, only(pullbacks))
end

"""
    ad_value_and_pullback!(prepared, cotangent, seed, args...; kwargs...)

Evaluate the selected WANT and reverse pullback while writing the active-input
cotangent into caller-owned `cotangent` storage. Returns `(value, cotangent)`.
For a scalar active selector, the destination must satisfy the differentiation
backend's in-place pullback contract. For a tuple selector, supply a matching
tuple of mutable arrays or `Ref` destinations; the structured cotangent is
copied into them after the one reverse pass.
"""
function ad_value_and_pullback!(
        prepared::PreparedADPullback, cotangent, seed, args...; kwargs...)
    point, contexts = _ad_prepared_arguments(
        prepared, args, NamedTuple(kwargs))
    _ad_prepared_value_and_pullback!(
        prepared, cotangent, seed, point, contexts)
end

function _ad_prepared_value_and_pullback!(
        prepared, cotangent, seed, point::Tuple, contexts)
    value, pullbacks = DifferentiationInterface.value_and_pullback(
        prepared.call, prepared.preparation, prepared.backend,
        point, (seed,), contexts...)
    restored = _ad_restore_cotangent(point, only(pullbacks))
    _ad_copy_cotangent!(cotangent, restored)
    value, cotangent
end

function _ad_prepared_value_and_pullback!(
        prepared, cotangent, seed, point, contexts)
    value, pullbacks = DifferentiationInterface.value_and_pullback!(
        prepared.call, (cotangent,), prepared.preparation, prepared.backend,
        point, (seed,), contexts...)
    value, only(pullbacks)
end

"""
    ad_value_and_gradient!(prepared, gradient, args...; kwargs...)

Compute the scalar value and gradient for a reusable [`PreparedADKernel`](@ref),
writing the gradient into `gradient`. Runtime arguments follow the original RK
HAVE boundary; the prepared active port or ordered tuple of ports is reordered
to DI position one, and every other current argument is rebound as a fresh
`DifferentiationInterface.Constant` context. For a tuple selector, `gradient`
is a matching tuple of mutable destinations. The backend's structured
out-of-place gradient is copied into those destinations after the same single
reverse pass.

Returns the `(value, gradient)` pair from
`DifferentiationInterface.value_and_gradient!`. The destination must be valid
for the active argument's gradient and is mutated in place. Like
[`ad_gradient`](@ref), this prepared object and its DI preparation are not
thread-safe; use one per concurrent caller.

A Reactant-traced active input stages the derivative in the enclosing compiled
program and writes it into the traced destination, without using the native DI
preparation there.
"""
@generated function ad_value_and_gradient!(
        prepared::PreparedADKernel{I,K,typeof(tuple),F,B,P,E}, gradient,
        args::Vararg{Any,N}) where {I,K,F,B,P,E,N}
    indices = _ad_selector_indices(I)
    all(index -> 1 <= index <= N, indices) || return :(throw(ArgumentError(
        "prepared active input selector $I is invalid for $N arguments")))
    contexts = [
        :(DifferentiationInterface.Constant(getfield(args, $index)))
        for index in 1:N if !(index in indices)
    ]
    append!(contexts, [
        E.parameters[index] <: DifferentiationInterface.Context ?
            :(getfield(prepared.external_values, $index)) :
            :(DifferentiationInterface.Constant(
                getfield(prepared.external_values, $index)))
        for index in 1:fieldcount(E)
    ])
    point = I isa Int ? :(getfield(args, $I)) :
        Expr(:tuple, (:(
            _ad_active_point_component(getfield(args, $index)))
            for index in indices)...)
    quote
        length(inputs(prepared.kernel)) == $N || throw(ArgumentError(
            "selected HAVE boundary expects " *
            string(length(inputs(prepared.kernel))) *
            " values; got $N"))
        _ad_prepared_value_and_gradient!(
            prepared, gradient, $point, ($(contexts...),))
    end
end

function ad_value_and_gradient!(
        prepared::PreparedADKernel, gradient, args...; kwargs...)
    point, contexts = _ad_prepared_arguments(
        prepared, args, NamedTuple(kwargs))
    _ad_prepared_value_and_gradient!(prepared, gradient, point, contexts)
end

function _ad_prepared_value_and_gradient!(prepared, gradient, point, contexts)
    DifferentiationInterface.value_and_gradient!(
        prepared.call, gradient, prepared.preparation, prepared.backend,
        point, contexts...)
end

function _ad_prepared_value_and_gradient!(
        prepared, gradient, point::Tuple, contexts)
    value, derivative = _ad_prepared_value_and_gradient(
        prepared, point, contexts)
    _ad_copy_cotangent!(gradient, derivative)
    value, gradient
end

function _ad_copy_cotangent!(destination::AbstractArray, source)
    copyto!(destination, source)
    destination
end


function _ad_copy_cotangent!(destination::Base.RefValue, source)
    destination[] = source
    destination
end

function _ad_copy_cotangent!(destination::Tuple, source::Tuple)
    length(destination) == length(source) || throw(DimensionMismatch(
        "gradient destination has $(length(destination)) components; " *
        "the active tuple has $(length(source))"))
    foreach(_ad_copy_cotangent!, destination, source)
    destination
end

function _ad_copy_cotangent!(destination::NamedTuple, source::NamedTuple)
    keys(destination) == keys(source) || throw(ArgumentError(
        "gradient destination fields $(keys(destination)) do not match " *
        "cotangent fields $(keys(source))"))
    foreach(_ad_copy_cotangent!, values(destination), values(source))
    destination
end


function _ad_copy_cotangent!(destination, source)
    throw(ArgumentError(
        "structured active gradients require matching mutable array, Ref, " *
        "or tuple destinations; got $(typeof(destination)) for a " *
        "$(typeof(source)) cotangent. Use ad_value_and_gradient for an " *
        "out-of-place structured gradient."))
end

inputs(prepared::PreparedADKernel) = inputs(prepared.kernel)
outputs(prepared::PreparedADKernel) = outputs(prepared.kernel)
code_expr(prepared::PreparedADKernel) = code_expr(prepared.kernel)
inputs(prepared::PreparedADPullback) = inputs(prepared.kernel)
outputs(prepared::PreparedADPullback) = outputs(prepared.kernel)
code_expr(prepared::PreparedADPullback) = code_expr(prepared.kernel)

function _show_ad_active(io::IO, boundary, index::Int)
    print(io, ":", boundary[index].name)
end

function _show_ad_active(io::IO, boundary, indices::Tuple)
    show(io, Tuple(boundary[index].name for index in indices))
end

function Base.show(io::IO, prepared::PreparedADKernel{I}) where {I}
    print(io, "PreparedADKernel(active=")
    _show_ad_active(io, inputs(prepared.kernel), I)
    print(io, ", want=:", only(outputs(prepared.kernel)).name, ", kernel=")
    show(io, prepared.kernel)
    print(io, ")")
end


function Base.show(io::IO, prepared::PreparedADPullback{I}) where {I}
    print(io, "PreparedADPullback(active=")
    _show_ad_active(io, inputs(prepared.kernel), I)
    print(io, ", want=:", only(outputs(prepared.kernel)).name, ", kernel=")
    show(io, prepared.kernel)
    print(io, ")")
end

# --- Reactant-compiled automatic differentiation -----------------------------
# The AD analog of the primal Reactant path (`@compile sync=true kernel(args...)`).
# These take a native `PreparedADKernel` — which already owns the scalar-WANT /
# active-port validation and the authored-HAVE-order reorder — and compile a
# DifferentiationInterface gradient (or value-and-gradient) through Reactant.
#
# The differentiation engine stays the caller's DifferentiationInterface backend
# (the one passed to `prepare_ad`), so ReactiveKernels imports no concrete AD
# engine here: a caller-configured reverse-mode backend stages differentiation
# inside the Reactant program. The real methods live in
# `ext/ReactiveKernelsReactantExt.jl` and are selected when the active argument is
# a Reactant-traced value; without the Reactant weak dependency loaded (or with a
# host-array active argument) these raise a clear, actionable error instead of a
# bare `MethodError`.

function _reactant_ad_marker(prepared::PreparedADKernel{I}, args::Tuple) where {I}
    length(args) == length(inputs(prepared.kernel)) || throw(ArgumentError(
        "Reactant AD compilation expects the $(length(inputs(prepared.kernel)))-value " *
        "HAVE boundary $(Tuple(input.name for input in inputs(prepared.kernel))); " *
        "got $(length(args)) argument(s)"))
    I isa Int ? getfield(args, I) :
        ntuple(position -> getfield(args, I[position]), length(I))
end

# Selected by the Reactant extension on `marker::Reactant.RArray` /
# `Reactant.RNumber`. This fallback fires when Reactant is not loaded or the
# active argument was not traced.
_reactant_compile_ad(::Val, ::PreparedADKernel, marker, args...; kwargs...) =
    throw(ArgumentError(
        "Reactant-compiled AD requires the Reactant weak dependency (`using Reactant`) " *
        "and a Reactant-traced active argument (e.g. `Reactant.to_rarray(active)`); " *
        "the active argument is a $(typeof(marker))"))

# Internal extension point used when a partially-evaluated kernel owns large
# bound arrays. The Reactant extension compiles the same DI operation after
# those arrays have been transferred explicitly as hidden compiler operands.
function _reactant_compile_ad_externalized end

"""
    compile_ad_gradient(prepared::PreparedADKernel, traced_args...; sync = true)

Reactant/XLA-compile the gradient of a [`PreparedADKernel`](@ref) with respect to
its active HAVE port or ordered tuple of active ports. `traced_args` are the
selected HAVE values in authored order, already traced with
`Reactant.to_rarray` (matching the primal Reactant path). Returns a compiled
callable that accepts the same traced boundary and returns the gradient with
respect to the active point; every inactive HAVE is held constant, exactly as
on the native [`ad_gradient`](@ref) path.

This is the AD analog of compiling the primal kernel with
`@compile sync = true kernel(traced_args...)`. It is only possible where the
primal kernel itself compiles through Reactant; where it does not, the underlying
`@compile` error propagates unchanged.

Requires the Reactant weak dependency to be loaded. The differentiation engine is
the DifferentiationInterface backend passed to [`prepare_ad`](@ref).

`optimize` selects the Reactant optimization pipeline: `nothing` (default)
runs Reactant's default pipeline; `:no_slice_slice` runs the default pipeline
minus the `slice_slice` transform, which miscompiles chained consumers of a
strided slice on Reactant 0.2.284 (see reactivekernels-use §7j); any other
value forwards verbatim to `Reactant.compile`'s `optimize` keyword.
"""
function compile_ad_gradient(prepared::PreparedADKernel, args...; sync::Bool = true,
                             optimize = nothing)
    prepared.kernel isa NonAllocatingKernel && throw(ArgumentError(
        "Reactant-compiled AD is not supported over a NonAllocatingKernel " *
        "(the mutating cache program does not stage); prepare the gradient " *
        "from the dataflow kernel instead"))
    _reactant_compile_ad(Val(:gradient), prepared,
                         _reactant_ad_marker(prepared, args), args...; sync,
                         optimize)
end

"""
    compile_ad_value_and_gradient(prepared::PreparedADKernel, traced_args...; sync = true)

Reactant/XLA-compile the scalar value and gradient of a [`PreparedADKernel`](@ref)
together, mirroring the native [`ad_value_and_gradient!`](@ref) boundary. Returns
a compiled callable that accepts the traced HAVE boundary in authored order and
returns the `(value, gradient)` pair (a traced scalar and a traced gradient with
respect to the active port or ordered active tuple). This is the sampler-facing
surface: one compiled call yields both the potential and its gradient.

Like [`compile_ad_gradient`](@ref), this requires the Reactant weak dependency,
reuses the DifferentiationInterface backend from [`prepare_ad`](@ref), and only
compiles where the primal kernel itself compiles through Reactant. The
`optimize` keyword is identical: `nothing` (default) runs Reactant's default
pipeline, `:no_slice_slice` removes the miscompiling `slice_slice` transform
(reactivekernels-use §7j), and any other value forwards to `Reactant.compile`.
"""
function compile_ad_value_and_gradient(prepared::PreparedADKernel, args...; sync::Bool = true,
                                        optimize = nothing)
    prepared.kernel isa NonAllocatingKernel && throw(ArgumentError(
        "Reactant-compiled AD is not supported over a NonAllocatingKernel " *
        "(the mutating cache program does not stage); prepare the gradient " *
        "from the dataflow kernel instead"))
    _reactant_compile_ad(Val(:value_and_gradient), prepared,
                         _reactant_ad_marker(prepared, args), args...; sync,
                         optimize)
end
