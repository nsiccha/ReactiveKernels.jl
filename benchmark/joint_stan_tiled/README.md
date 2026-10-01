# Rectangular-recurrence reverse reproducers

Standalone CPU reproducers for the Reactant/Enzyme-JAX reverse-mode defects that
the experimental rectangular PK fold exposed (RK issue #13, closed; both fixed in
Reactant 0.2.289+). Each script needs only Reactant and Enzyme:

- `repro_reactant_while_reverse.jl`: reverse of a traced `while` whose body holds a conditional.
- `repro_nested_if_reverse.jl`: nested-`if` result adjoint read without being zeroed.
- `repro_rectangular_reverse.jl`: two subjects and six operations through the rectangular fold shape.
