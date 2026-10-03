# Rectangular-recurrence reverse reproducers

Standalone CPU reproducers for the Reactant/Enzyme-JAX reverse-mode defects that
the experimental rectangular PK fold exposed (RK issue #13, closed; both fixed in
Reactant 0.2.289+). The first two scripts need Reactant and Enzyme; the PK script also needs
ReactiveKernels, ReactiveKernelsPPL, and DifferentiationInterface:

- `repro_reactant_while_reverse.jl`: reverse of a traced `while` whose body holds a conditional.
- `repro_nested_if_reverse.jl`: nested-`if` result adjoint read without being zeroed.
- `repro_rectangular_reverse.jl`: two subjects through the ordinary PK subject plate and event scans.
  Native reverse runs; full compilation currently fails at fixed-size system-matrix batching.
