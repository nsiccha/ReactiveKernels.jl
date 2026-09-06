# Experiment: does the ReactiveState layer recompute only the Markov blanket
# of a changed source, leaving unrelated terms cached?
using ReactiveKernels

# Global op-execution counters, one per named term.
const HITS = Dict{Symbol,Int}()
hit!(name) = (HITS[name] = get(HITS, name, 0) + 1)

g = Graph()
# Three "parameter blocks" as source Values.
a = value!(g, :a, Float64)
b = value!(g, :b, Float64)
c = value!(g, :c, Float64)

# Additive log-density-like terms with explicit, different Markov blankets.
t_a  = value!(g, :t_a, Float64)   # depends on a
t_ab = value!(g, :t_ab, Float64)  # depends on a,b
t_bc = value!(g, :t_bc, Float64)  # depends on b,c  (the "expensive/data" term)
total = value!(g, :total, Float64)

add!(g, a => t_a, x -> (hit!(:t_a); x^2))
add!(g, (a, b) => t_ab, (x, y) -> (hit!(:t_ab); x * y))
add!(g, (b, c) => t_bc, (y, z) -> (hit!(:t_bc); sin(y) + z^2); cost = 100.0)
add!(g, (t_a, t_ab, t_bc) => total, (p, q, r) -> (hit!(:total); p + q + r))

st = ReactiveState(g; materialize = (t_a, t_ab, t_bc, total))
set!(st, a, 1.0); set!(st, b, 2.0); set!(st, c, 3.0)

println("== initial get!(total) ==")
empty!(HITS)
v0 = get!(st, total)
println("total = ", v0, "   hits = ", HITS)

println("== set!(a) then get!(total): expect t_a, t_ab, total recompute; t_bc CACHED ==")
empty!(HITS)
set!(st, a, 1.5)
v1 = get!(st, total)
println("total = ", v1, "   hits = ", HITS)

println("== set!(c) then get!(total): expect t_bc, total recompute; t_a, t_ab CACHED ==")
empty!(HITS)
set!(st, c, 4.0)
v2 = get!(st, total)
println("total = ", v2, "   hits = ", HITS)

println("== set!(b) then get!(total): expect ALL recompute (b in every blanket) ==")
empty!(HITS)
set!(st, b, 2.5)
v3 = get!(st, total)
println("total = ", v3, "   hits = ", HITS)

# Correctness cross-check against direct evaluation.
direct(x, y, z) = x^2 + x * y + (sin(y) + z^2)
println("== correctness ==")
println("rk=", v3, "  direct=", direct(1.5, 2.5, 4.0), "  match=", v3 ≈ direct(1.5, 2.5, 4.0))
