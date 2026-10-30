# Position batching of a rebound kernel (`vectorize(prepare(g; bound = …))`).
#
# `vectorize` lowers and compiles a position driver for its scalar kernel: the
# owning `GraphReplicatedKernel`, and with `reuse=true` and a
# `NativeScheduling` the borrowed driver, the shared prefix kernel and the
# per-worker residual driver as well. That is several lowerings and
# compilations of the whole graph per call, independent of the bound values
# (snag rebinding-a-comp-d03576ab: 400–440 ms and 134 MB of the 490–520 ms
# needed to bind and vectorize each design of a large composed simulation
# graph, 4.5–4.9 ms in all once kept). Every binding of one bound template (`_BoundTemplate`,
# partial_evaluation.jl) has the same plan up to its constant recipes'
# values, so its kernels share one `_BoundVectorizations`: per vectorize key,
# the products of the first binding lowered with that key, kept without that
# binding's constants (`_RebindTable`). A later binding's `vectorize` rebuilds
# the same products around the same compiled callables with its own
# constants in their slots, exactly as `_bound_rebind` rebuilds the scalar
# kernel.
#
# A product is kept only when each of its constants tables holds each
# constant's value exactly once, in plan order, and nothing else of that
# binding; every other product is lowered afresh per call, as before. The key
# includes the types of the kernel's two tables, so the constants' types are
# those the kept programs were lowered for. Nothing here is on a call path, and the products'
# types are as varied as the graphs, so these functions take their arguments
# unspecialized.

# The owning position driver without the binding's constants.
struct _RebindReplicated
    type::DataType   # the exact `GraphReplicatedKernel` type
    native::Any
    constants::_RebindTable
    callables::Tuple
    ast::Expr
end

# The borrowed driver around its owning driver.
struct _RebindBorrowed
    target::_RebindReplicated
    native::Any
    constants::_RebindTable
    callables::Tuple
    slots::Int
    ast::Expr
end

# A prepared kernel over a plan holding the constants (the shared prefix of a
# scheduled batch).
struct _RebindPrepared
    f::Any
    constants::_RebindTable
    callables::Vector{Any}
    inputs::Tuple
    outputs::Tuple
    have::Vector{Value}
    want::Vector{Value}
    recipes::_RebindTable
    ast::Expr
    lowered::_RebindTable
    signature::Type
    executable::_ExecutableTable # `callables` as this process executes them
end

# A scheduled batch: its serial driver, its prefix and one empty residual
# instance that every worker copies (it reads no constant).
struct _RebindScheduled
    type::DataType   # the exact `_ScheduledBatchedKernel` type
    serial::Union{_RebindReplicated,_RebindBorrowed}
    prefix::Union{Nothing,_RebindPrepared}
    residual::BorrowedBatchedKernel
end

# A kernel's constants: their values in its constants table and their
# readable recipes, at its template's constant slots, in plan order.
struct _RebindConstants
    values::Vector{Any}
    recipes::Vector{Any}
end
# `getfield`, not `getindex`: on an unspecialized kernel each `getindex` was a
# dynamic dispatch on the type of a table of hundreds of entries. Each table
# is read from the kernel once: a tuple field of an unspecialized kernel is
# boxed, a copy of the whole table, at every read, which inside the loop was
# 1.8 MB per binding of a simulator graph with 56 constants among 365 entries
# (snag per-binding-resi-990b18ab).
function _RebindConstants(@nospecialize(kernel::PreparedKernel), slots::Vector{Int})
    constants = getfield(kernel, :constants)
    recipes = getfield(kernel, :lowered_recipes)
    _RebindConstants(Any[getfield(constants, k) for k in slots],
                     Any[getfield(recipes, k) for k in slots])
end

_holds_constant(@nospecialize(table), constants::Vector{Any}) =
    any(x -> any(c -> x === c, constants), table)

_vectorize(::Nothing, kernel::PreparedKernel, @nospecialize(batched), reuse::Bool,
           @nospecialize(schedule)) =
    _position_schedule(replica_graph(kernel; batched), schedule, Val(reuse))

function _vectorize(vectorizations::_BoundVectorizations,
                    @nospecialize(kernel::PreparedKernel), @nospecialize(batched),
                    reuse::Bool, @nospecialize(schedule))
    key = (_replica_batch_indices(inputs(kernel), batched), reuse,
           schedule === nothing, fieldtype(typeof(kernel), :constants),
           fieldtype(typeof(kernel), :callables))
    kept = lock(() -> get(vectorizations.products, key, nothing), vectorizations.lock)
    constants = _RebindConstants(kernel, vectorizations.slots)
    kept === nothing || return _rebound(kept, kernel, constants, schedule)
    product = _vectorize(nothing, kernel, batched, reuse, schedule)
    kept = _rebind_template(product, kernel, constants)
    kept === nothing ||
        lock(() -> get!(vectorizations.products, key, kept), vectorizations.lock)
    product
end

# --- keeping a product without its binding's constants ----------------------

_rebind_template(@nospecialize(product), @nospecialize(kernel::PreparedKernel),
                 constants::_RebindConstants) = nothing

function _rebind_template(@nospecialize(product::GraphReplicatedKernel),
                          @nospecialize(kernel::PreparedKernel),
                          constants::_RebindConstants)
    product.target === kernel || return nothing
    table = _rebind_table(product.constants, constants.values)
    table === nothing ? nothing :
        _RebindReplicated(typeof(product), product.native, table, product.callables,
                          product.ast)
end

function _rebind_template(@nospecialize(product::BorrowedBatchedKernel),
                          @nospecialize(kernel::PreparedKernel),
                          constants::_RebindConstants)
    target = _rebind_template(product.target, kernel, constants)
    target === nothing && return nothing
    table = _rebind_table(product.constants, constants.values)
    table === nothing ? nothing :
        _RebindBorrowed(target, product.native, table, product.callables,
                        length(product.caches), product.ast)
end

function _rebind_prefix(@nospecialize(prefix::PreparedKernel), constants::_RebindConstants)
    table = _rebind_table(prefix.constants, constants.values)
    recipes = _rebind_table(prefix.plan.recipes, constants.recipes)
    lowered = _rebind_table(prefix.lowered_recipes, constants.recipes)
    (table === nothing || recipes === nothing || lowered === nothing) && return nothing
    callables = Any[prefix.callables...]
    _RebindPrepared(prefix.f, table, callables, prefix.inputs, prefix.outputs,
                    prefix.plan.have, prefix.plan.want, recipes, prefix.ast, lowered,
                    _exact_signature(prefix), _ExecutableTable(callables))
end

function _rebind_template(@nospecialize(product::_ScheduledBatchedKernel),
                          @nospecialize(kernel::PreparedKernel),
                          constants::_RebindConstants)
    serial = _rebind_template(product.serial, kernel, constants)
    serial === nothing && return nothing
    prefix = product.prefix === nothing ? nothing :
             _rebind_prefix(product.prefix, constants)
    product.prefix === nothing || prefix !== nothing || return nothing
    residual = first(product.workers)
    _holds_constant(residual.constants, constants.values) && return nothing
    _holds_constant(plan(residual).recipes, constants.recipes) && return nothing
    _RebindScheduled(typeof(product), serial, prefix, copy(residual))
end

# --- rebuilding it around another binding's constants -----------------------

function _rebound(kept::_RebindReplicated, @nospecialize(kernel::PreparedKernel),
                  constants::_RebindConstants, @nospecialize(schedule))
    table = Tuple(_rebound_table(kept.constants, constants.values))
    _kernel_new_instance(kept.type, (kernel, kept.native, table, kept.callables,
                                     kernel.plan, kept.ast, kernel.inputs, kernel.outputs))
end

function _rebound(kept::_RebindBorrowed, @nospecialize(kernel::PreparedKernel),
                  constants::_RebindConstants, @nospecialize(schedule))
    target = _rebound(kept.target, kernel, constants, schedule)
    table = Tuple(_rebound_table(kept.constants, constants.values))
    BorrowedBatchedKernel(target, kept.native, table, kept.callables,
                          _borrowed_batch_caches(kept.slots), kept.ast)
end

function _rebound(kept::_RebindPrepared, @nospecialize(kernel::PreparedKernel),
                  constants::_RebindConstants, @nospecialize(schedule))
    recipes = convert(Vector{Recipe}, _rebound_table(kept.recipes, constants.recipes))
    plan = _partial_subplan(kernel.plan, kept.have, kept.want, recipes)
    _executed_prepared_kernel(kept.f, Tuple(_rebound_table(kept.constants, constants.values)),
                              _executable_callables(kept.executable, kept.callables),
                              kept.inputs, kept.outputs, plan, kept.ast,
                              Tuple(_rebound_table(kept.lowered, constants.recipes)),
                              kept.signature, nothing)
end

function _rebound(kept::_RebindScheduled, @nospecialize(kernel::PreparedKernel),
                  constants::_RebindConstants, schedule::NativeScheduling)
    serial = _rebound(kept.serial, kernel, constants, schedule)
    prefix = kept.prefix === nothing ? nothing :
             _rebound(kept.prefix, kernel, constants, schedule)
    caches = _borrowed_batch_caches(length(outputs(kernel)))
    _kernel_new_instance(kept.type, (serial, prefix, _scheduled_workers(kept.residual, schedule),
                                     caches, schedule))
end

# Untyped, so compiled once for every graph; these directives put that in the
# package image, so a graph's first vectorized rebinding compiles nothing
# here either (compare `_prepare_bound!`, graphops.jl).
for (f, types) in (
        (_vectorize, (_BoundVectorizations, PreparedKernel, Any, Bool, Any)),
        (_rebind_template, (GraphReplicatedKernel, PreparedKernel, _RebindConstants)),
        (_rebind_template, (BorrowedBatchedKernel, PreparedKernel, _RebindConstants)),
        (_rebind_template, (_ScheduledBatchedKernel, PreparedKernel, _RebindConstants)),
        (_rebound, (_RebindReplicated, PreparedKernel, _RebindConstants, Any)),
        (_rebound, (_RebindBorrowed, PreparedKernel, _RebindConstants, Any)),
        (_rebound, (_RebindPrepared, PreparedKernel, _RebindConstants, Any)),
        (_rebound, (_RebindScheduled, PreparedKernel, _RebindConstants, NativeScheduling)))
    precompile(f, types) || error("no $f method for $types")
end
