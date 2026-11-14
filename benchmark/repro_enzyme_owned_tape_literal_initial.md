# Owned tape and a literal initial total through nested AD

`repro_enzyme_owned_tape_literal_initial.jl` imports only Enzyme,
DifferentiationInterface, and Test. It contains the exact emitted reverse cut
from the RK gathered sine plate, with its helper subset; the expression's
module references are relocated to that local helper module. It does not load
ReactiveKernels or RuntimeGeneratedFunctions.

Run in the pinned diagnostic environment:

```sh
julia +1.13.1 --startup-file=no --project=.scratch/env-tape113 benchmark/repro_enzyme_owned_tape_literal_initial.jl
```

The failing evaluation is forward over reverse HVP. With fresh tape the three
changed-input HVP checks pass. With the same rule and reused tape, all three
throw `EnzymeRuntimeActivityError` at the select of the constant initial total
and accumulated value. This reproduces on Julia 1.13.1, Enzyme
`d37eb44840cc3dd7f6fc013c57e04867110964de`, native Enzyme
`f25c881a0e56cfb39aabf818a4c31d5d27c3ca72`, DI 0.7.22.

The differentiating call passes literal `0.0` as its initial total. The earlier
control passes the same zero as a `DI.Constant` argument through that caller;
then both fresh and reused tapes pass. An outer empty-plate guard, index-run
scheduling, a noinline wrapper, and RuntimeGeneratedFunctions are unnecessary
to reproduce the failure. Simple carried-total loop reductions pass even with
the literal, so a literal alone is not a sufficient description of the defect.

Enzyme repair `constant_scalar_tangent`, landed in dogfood
`15ebe6cc77be51cd9fb6f611b9c4db680b7ecfb7`, fixes this failure. The unchanged
reproducer passes all six checks on that revision, and the RK consumer passes
2,309 checks including changed-point higher derivatives, tape ownership and
exception recovery. The native library is
`342bcb7a0f5a742ee8e52943225bb6870c64c39e`. No runtime-activity or mathematical
source workaround was added. Check the repair with
`Enzyme.has_dogfood_repair(:constant_scalar_tangent)`; the upstream report is
[Enzyme.jl #4060](https://github.com/EnzymeAD/Enzyme.jl/issues/4060).

Retained reduction run: fleet `d593601a85894f9a` on strato2, wrapper receipt
exit1 at 2026-11-14T17:21:51Z (251 seconds). Minimal loop and gathered-view
controls pass12 each. The emitted-cut, explicit-noinline-cut, and RGF-cut
variants each pass3 fresh and error3 reused. See the archived diagnostic
bundle linked from the Enzyme snag and RK todo `021y7re` for exact logs and
the effective diagnostic environment.
