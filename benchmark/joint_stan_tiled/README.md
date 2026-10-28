# Rectangular-recurrence reverse reproducers

Standalone CPU reproducers for the Reactant/Enzyme-JAX reverse-mode defects that
the experimental rectangular PK fold exposed (RK issue #13, closed; both fixed in
Reactant 0.2.289+). These backend reproducers need Reactant and Enzyme:

- `repro_reactant_while_reverse.jl`: reverse of a traced `while` whose body holds a conditional.
- `repro_nested_if_reverse.jl`: nested-`if` result adjoint read without being zeroed.

PK helpers and their model acceptance belong to the downstream RKPPLBench consumer.
