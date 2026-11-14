# Loading Enzyme invalidates part of the construction code RKPPL's package
# image caches: Enzyme and the packages it loads add methods (`ndims` and
# `eltype` for `Enzyme.TupleArray`, `convert` to `Array`, LibGit2's
# `Base.IteratorSize`) that match abstract calls on RK/RKPPL's construction
# path, so a fresh process that loaded Enzyme re-inferred lowering, binding and
# `build_kernel` on its first model (snag `rkppl-cold-port-72991c9f`). This
# extension's image is generated with Enzyme loaded after RK and RKPPL, so
# running the same construction workload here caches that code again for a
# process that loads them in that order. A process that loads Enzyme before RK
# rejects other cached code than the image replaced and still recompiles
# (WellsDist 53 s against 6 s on Julia 1.13.1; snag `downstream-workl-75f8cba4`).
#
# It prepares no sampler: DifferentiationInterface's Enzyme extension is not
# loaded while this image is generated, so `prepare_sampler` with `AutoEnzyme`
# raises a MethodError here, and a failing workload fails the extension.
module ReactiveKernelsPPLEnzymeExt

using ReactiveKernelsPPL: _construction_workload
using Enzyme: Enzyme
using PrecompileTools: @compile_workload

@compile_workload begin
    _construction_workload()
end

end
