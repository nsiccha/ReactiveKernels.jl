# Deep copies of the objects that hold code.
#
# A recipe's authored source and a prepared object's generated program are
# expressions, and an expression may hold a module as a literal value:
# ReactiveKernelsPPL rebuilds a resolved binding among a law's arguments as
# `Core.GlobalRef(<module>, :name)` so that the printed source replays, and a
# program that inlines that source calls it as written. `deepcopy` refuses a
# `Module`, so a per-chain deep copy of a prepared kernel failed (snag
# `prepared-query-h-b7bd890a`). A deep copy of these objects copies their
# expression trees and shares the modules in them, as the copy of a source
# operation shares its module (`Base.deepcopy_internal(::_KernelSourceOp, …)`).
# Everything else is copied as `deepcopy` copies it.
_deepcopy_code(@nospecialize(x), stackdict::IdDict) = Base.deepcopy_internal(x, stackdict)
_deepcopy_code(x::Module, ::IdDict) = x
_deepcopy_code(x::QuoteNode, stackdict::IdDict) =
    QuoteNode(_deepcopy_code(x.value, stackdict))
function _deepcopy_code(x::Expr, stackdict::IdDict)
    haskey(stackdict, x) && return stackdict[x]::Expr
    copy = Expr(x.head)
    stackdict[x] = copy
    for arg in x.args
        push!(copy.args, _deepcopy_code(arg, stackdict))
    end
    return copy
end

# An immutable object whose fields are each copied as `_deepcopy_code` copies
# them; every field of these types is defined.
Base.@nospecializeinfer _deepcopy_code_fields(@nospecialize(x), stackdict::IdDict) =
    _kernel_new_instance(typeof(x)::DataType,
        ntuple(i -> _deepcopy_code(getfield(x, i), stackdict), nfields(x)))

# The types with authored source (`Recipe.source`) or a generated program
# (an `Expr` field). `test/test_execution_sources.jl` checks that every type of
# this package with an `Expr` field is listed.
for T in (Recipe, PreparedKernel, GraphReplicatedKernel, BorrowedBatchedKernel,
          NonAllocatingKernel, _BoundTemplate, _RebindReplicated, _RebindBorrowed,
          _RebindPrepared, _CellEmitter, _ReactiveGetter, _ReactiveGetterInPlace)
    @eval Base.deepcopy_internal(x::$T, stackdict::IdDict) =
        _deepcopy_code_fields(x, stackdict)
end
