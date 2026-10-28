# Typed-IR audit of prepared programs, shared by the ReactiveKernels and
# ReactiveKernelsPPL type-stability gates. The audit itself lives in
# `src/type_stability_audit.jl`, where preparation runs it too
# (`type_stability_level`); this module names it for the gates.
# `findings(f, argtypes)` infers and optimizes `f` for concrete argument types
# and reports, in the body and in every statically invoked callee that
# `recurse` accepts, used non-concrete values (`:root` where every argument is
# concrete, else `:nonconcrete`) and calls dispatched at run time (`:dynamic`).
module TypeStabilityAudit

export findings, ad_call_findings, rk_or_generated

import ReactiveKernels

const RK = ReactiveKernels

"""
    findings(f, argtypes; recurse = rk_or_generated) -> Vector

Audit `f` called with arguments of types `argtypes`, recursing into every
statically invoked callee whose method `recurse` accepts.
"""
findings(@nospecialize(f), @nospecialize(argtypes); recurse = rk_or_generated) =
    RK._code_findings(f, argtypes; recurse)

"""
    rk_or_generated(modules...) -> predicate

Recurse into ReactiveKernels (and its submodules), into the bodies RK
generates (`RuntimeGeneratedFunctions`), and into methods of `modules`, the
authoring modules of the audited kernels.
"""
rk_or_generated(modules::Module...) = RK._rk_or_generated(modules...)
rk_or_generated(m::Method) = rk_or_generated()(m)

"""
    ad_call_findings(prepared, args...; recurse) -> Vector

Audit the callable a `PreparedADKernel` hands its backend, at the point and
contexts `prepared` builds from `args`.
"""
function ad_call_findings(prepared, args...; recurse = rk_or_generated)
    point, contexts = RK._ad_prepared_arguments(prepared, args, (;))
    findings(prepared.call, RK._ad_call_argument_types(point, contexts); recurse)
end

end # module
