# Distribution log-density benchmark environment

This environment pins the packages used by
`benchmark/distributions_comparison.jl`, including ProbabilityMeasures at the
exact reviewed commit. From the repository root:

```sh
julia --startup-file=no --project=benchmark/distributions benchmark/distributions/setup.jl
julia --startup-file=no --project=benchmark/distributions \
  benchmark/distributions_comparison.jl \
  --output=benchmark/receipts/distribution-logdensity-v1.toml
julia --startup-file=no benchmark/receipts/validate_distributions.jl \
  benchmark/receipts/distribution-logdensity-v1.toml
```

The benchmark refuses a dirty ReactiveKernels tree by default. Reactant
compilation and host-to-device transfers are outside the timed execution
region; every compiled call uses `sync=true`, and both model parameters remain
tracked runtime values. Per-shape compilation diagnostics are retained in the
receipt; the first RK compile includes Reactant service startup and is not used
as a cross-library compile-time comparison.
