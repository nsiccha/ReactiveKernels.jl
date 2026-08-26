# NUTS sampling

ReactiveKernels' No-U-Turn sampler is being reimplemented as a **production `@kernel`
NUTS** — a single, method-bearing `@kernel` authoring surface, modeled on the
ReactiveHMC.jl algorithm structure, with implicit field capture.

**No current implementation is documented here.** The earlier `@reactive` port is not
the product and is deliberately not shown; its history remains in Git. This page will
document the actual production `@kernel` NUTS — its build-executed kernels and how you
interact with them — once the executable, lowered implementation lands.

Correctness is established separately, in **isolated verification harnesses** (not shown
as docs content), by **mathematical, independent** gates — Hamiltonian and gradient
identities, leapfrog reversibility and controlled/expected energy-error behavior, NUTS
tree / multinomial / divergence / depth logic, adaptation recurrences, and analytic
target distributions under justified tolerances. ReactiveHMC.jl `ca9` is an
**algorithm-structure reference only** — not a bitwise or RNG target (improvements may
change arithmetic or ordering).
