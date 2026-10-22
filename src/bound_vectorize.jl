# Position batching of a rebound kernel (`vectorize(prepare(g; bound = …))`).
#
# `vectorize` lowers and compiles a position driver for its scalar kernel: the
# owning `GraphReplicatedKernel`, and with `reuse=true` and a
# `NativeScheduling` the borrowed driver, the shared prefix kernel and the
# per-worker residual driver as well. That is several lowerings and
# compilations of the whole graph per call, independent of the bound values
# (snag rebinding-a-comp-d03576ab: about 330 ms and 171 MB per design on a
# large composed simulation graph whose rebinding `prepare` costs under a
# millisecond). Every binding of one bound template (`_BoundTemplate`,
# partial_evaluation.jl) has the same plan up to the values of its leading
# constant recipes, so its kernels share one `_BoundVectorizations`: per
# vectorize key, the products of the first binding lowered with that key,
# kept without that binding's constants. A later binding's `vectorize`
# rebuilds the same products around the same compiled callables, with its own
# constants in the leading slots of each operation table, exactly as
# `_bound_rebind` rebuilds the scalar kernel.
#
# A product is kept only when each of its operation tables starts with exactly
# the binding's constants and holds none of them elsewhere; every other
# product is lowered afresh per call, as before. The key includes the type of
# the kernel's operation table, so the constants' types are those the kept
# programs were lowered for. Nothing here is on a call path, and the products'
# types are as varied as the graphs, so these functions take their arguments
# unspecialized.

# The owning position driver without the binding's constants.
struct _RebindReplicated
    type::DataType   # the exact `GraphReplicatedKernel` type
    native::Any
    tail::Tuple      # the operation table after the constant slots
    ast::Expr
end

# The borrowed driver around its owning driver.
struct _RebindBorrowed
    target::_RebindReplicated
    native::Any
    tail::Tuple
    slots::Int
    ast::Expr
end

# A prepared kernel over a plan whose leading recipes are the constants (the
# shared prefix of a scheduled batch).
struct _RebindPrepared
    f::Any
    tail::Tuple
    inputs::Tuple
    outputs::Tuple
    have::Vector{Value}
    want::Vector{Value}
    recipes::Vector{Recipe}   # the plan's recipes after the constants
    ast::Expr
    lowered::Tuple
    signature::Type
end

# A scheduled batch: its serial driver, its prefix and one empty residual
# instance that every worker copies (it reads no constant).
struct _RebindScheduled
    type::DataType   # the exact `_ScheduledBatchedKernel` type
    serial::Union{_RebindReplicated,_RebindBorrowed}
    prefix::Union{Nothing,_RebindPrepared}
    residual::BorrowedBatchedKernel
end

_vectorize(::Nothing, kernel::PreparedKernel, @nospecialize(batched), reuse::Bool,
           @nospecialize(schedule)) =
    _position_schedule(replica_graph(kernel; batched), schedule, Val(reuse))

function _vectorize(vectorizations::_BoundVectorizations,
                    @nospecialize(kernel::PreparedKernel), @nospecialize(batched),
                    reuse::Bool, @nospecialize(schedule))
    key = (_replica_batch_indices(inputs(kernel), batched), reuse,
           schedule === nothing, typeof(kernel.ops))
    kept = lock(() -> get(vectorizations.products, key, nothing), vectorizations.lock)
    n = vectorizations.constants
    kept === nothing || return _rebound(kept, kernel, n, schedule)
    product = _vectorize(nothing, kernel, batched, reuse, schedule)
    kept = _rebind_template(product, kernel, n)
    kept === nothing ||
        lock(() -> get!(vectorizations.products, key, kept), vectorizations.lock)
    product
end

# Whether `ops` starts with the kernel's `n` constants and holds none of them
# after that.
function _rebind_constants_lead(@nospecialize(ops::Tuple),
                                @nospecialize(kernel::PreparedKernel), n::Int)
    length(ops) >= n || return false
    for i in 1:n
        ops[i] === kernel.ops[i] || return false
    end
    !_rebind_holds_constant(ops[(n + 1):end], kernel, n)
end

function _rebind_holds_constant(@nospecialize(ops::Tuple),
                                @nospecialize(kernel::PreparedKernel), n::Int)
    for op in ops, i in 1:n
        op === kernel.ops[i] && return true
    end
    false
end

_rebind_constants(@nospecialize(kernel::PreparedKernel), n::Int) = kernel.ops[1:n]
_exact_signature(@nospecialize(kernel::PreparedKernel)) =
    (typeof(kernel)::DataType).parameters[6]

# --- keeping a product without its binding's constants ----------------------

_rebind_template(@nospecialize(product), @nospecialize(kernel::PreparedKernel),
                 n::Int) = nothing

function _rebind_template(@nospecialize(product::GraphReplicatedKernel),
                          @nospecialize(kernel::PreparedKernel), n::Int)
    product.target === kernel && _rebind_constants_lead(product.ops, kernel, n) ||
        return nothing
    _RebindReplicated(typeof(product), product.native, product.ops[(n + 1):end],
                      product.ast)
end

function _rebind_template(@nospecialize(product::BorrowedBatchedKernel),
                          @nospecialize(kernel::PreparedKernel), n::Int)
    target = _rebind_template(product.target, kernel, n)
    target === nothing && return nothing
    _rebind_constants_lead(product.ops, kernel, n) || return nothing
    _RebindBorrowed(target, product.native, product.ops[(n + 1):end],
                    length(product.caches), product.ast)
end

function _rebind_prefix(@nospecialize(prefix::PreparedKernel),
                        @nospecialize(kernel::PreparedKernel), n::Int)
    constants = kernel.plan.recipes
    recipes = prefix.plan.recipes
    length(recipes) >= n && length(prefix.lowered_recipes) >= n || return nothing
    for i in 1:n
        recipes[i] === constants[i] && prefix.lowered_recipes[i] === constants[i] ||
            return nothing
    end
    _rebind_constants_lead(prefix.ops, kernel, n) || return nothing
    _RebindPrepared(prefix.f, prefix.ops[(n + 1):end], prefix.inputs, prefix.outputs,
                    prefix.plan.have, prefix.plan.want, recipes[(n + 1):end], prefix.ast,
                    prefix.lowered_recipes[(n + 1):end], _exact_signature(prefix))
end

function _rebind_template(@nospecialize(product::_ScheduledBatchedKernel),
                          @nospecialize(kernel::PreparedKernel), n::Int)
    serial = _rebind_template(product.serial, kernel, n)
    serial === nothing && return nothing
    prefix = product.prefix === nothing ? nothing :
             _rebind_prefix(product.prefix, kernel, n)
    product.prefix === nothing || prefix !== nothing || return nothing
    residual = first(product.workers)
    for ops in (residual.ops, residual.target.ops, residual.target.target.ops)
        _rebind_holds_constant(ops, kernel, n) && return nothing
    end
    _RebindScheduled(typeof(product), serial, prefix, copy(residual))
end

# --- rebuilding it around another binding's constants -----------------------

function _rebound(kept::_RebindReplicated, @nospecialize(kernel::PreparedKernel), n::Int,
                  @nospecialize(schedule))
    ops = (_rebind_constants(kernel, n)..., kept.tail...)
    _kernel_new_instance(kept.type, (kernel, kept.native, ops, kernel.plan, kept.ast,
                                     kernel.inputs, kernel.outputs))
end

function _rebound(kept::_RebindBorrowed, @nospecialize(kernel::PreparedKernel), n::Int,
                  @nospecialize(schedule))
    target = _rebound(kept.target, kernel, n, schedule)
    ops = (_rebind_constants(kernel, n)..., kept.tail...)
    BorrowedBatchedKernel(target, kept.native, ops, _borrowed_batch_caches(kept.slots),
                          kept.ast)
end

function _rebound(kept::_RebindPrepared, @nospecialize(kernel::PreparedKernel), n::Int,
                  @nospecialize(schedule))
    constants = kernel.plan.recipes[1:n]
    plan = _partial_subplan(kernel.plan, kept.have, kept.want,
                            vcat(constants, kept.recipes))
    ops = (_rebind_constants(kernel, n)..., kept.tail...)
    _prepared_kernel(kept.f, ops, kept.inputs, kept.outputs, plan, kept.ast,
                     (constants..., kept.lowered...); exact_signature = kept.signature)
end

function _rebound(kept::_RebindScheduled, @nospecialize(kernel::PreparedKernel), n::Int,
                  schedule::NativeScheduling)
    serial = _rebound(kept.serial, kernel, n, schedule)
    prefix = kept.prefix === nothing ? nothing : _rebound(kept.prefix, kernel, n, schedule)
    workers = [copy(kept.residual) for _ in 1:min(schedule.workers, Threads.nthreads())]
    caches = _borrowed_batch_caches(length(outputs(kernel)))
    _kernel_new_instance(kept.type, (serial, prefix, workers, caches, schedule))
end

# Untyped, so compiled once for every graph; these directives put that in the
# package image, so a graph's first vectorized rebinding compiles nothing
# here either (compare `_prepare_bound!`, graphops.jl).
for (f, types) in ((_vectorize, (_BoundVectorizations, PreparedKernel, Any, Bool, Any)),
                   (_rebind_template, (GraphReplicatedKernel, PreparedKernel, Int)),
                   (_rebind_template, (BorrowedBatchedKernel, PreparedKernel, Int)),
                   (_rebind_template, (_ScheduledBatchedKernel, PreparedKernel, Int)),
                   (_rebound, (_RebindReplicated, PreparedKernel, Int, Any)),
                   (_rebound, (_RebindBorrowed, PreparedKernel, Int, Any)),
                   (_rebound, (_RebindPrepared, PreparedKernel, Int, Any)),
                   (_rebound, (_RebindScheduled, PreparedKernel, Int, NativeScheduling)))
    precompile(f, types) || error("no $f method for $types")
end
