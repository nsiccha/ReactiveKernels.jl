# Benchmark imports only; tracing is owned by the optional Reactant extension.
const TracedSlotCompiler =
    Base.get_extension(ReactiveKernels,:ReactiveKernelsReactantExt).TracedSlotCompiler
using .TracedSlotCompiler:
    compile_traced_slots, SlotStatic, TracedSlotCall, traced_slot_batch
