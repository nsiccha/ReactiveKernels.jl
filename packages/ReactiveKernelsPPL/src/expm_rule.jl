# Owned matrix-exponential primitive with generated AD rules.
#
# `LinearAlgebra.exp` on a matrix cannot reverse under Enzyme
# (`EnzymeNoDerivativeError` on its LAPACK `ccall`), which is why the PK cells
# carry a hand-ported Pade approximant (`_pk_expm3` in `pkcells.jl`). This rule
# is the replacement path for native execution: one pure-math `@kernel` graph
# authors the primal plus its forward (JVP) and reverse (VJP) branches, and
# `derivative_rule` generates the owned callable plus every AD-protocol adapter
# from that graph's cuts. No hand-written rule, no rule on a foreign function
# (see `docs/src/constraints.md`).
#
# Under Reactant the cuts trace as plain graph mathematics only once the
# backend supports `exp` on traced arrays (no release carries that yet); until
# then the port stays for the compiled path.
using LinearAlgebra: exp

# Both derivative branches are one augmented-matrix exponential plus
# top-right-block extraction: the JVP is the Frechet derivative
# `L(A, A_dot)`, and the VJP identity `A_bar = L(A', Y_bar)` follows from its
# integral form (verified against finite differences in `test_expm_rule.jl`).
@kernel rk_expm_rule(A::Matrix{Float64}, A_dot::Matrix{Float64},
        Y_bar::Matrix{Float64}) = begin
    Y::Matrix{Float64} = exp(A)
    Zf::Matrix{Float64} = zero(A)
    Topf::Matrix{Float64} = hcat(A, A_dot)
    Botf::Matrix{Float64} = hcat(Zf, A)
    Mf::Matrix{Float64} = vcat(Topf, Botf)
    Ef::Matrix{Float64} = exp(Mf)
    nf::Int = size(A, 1)
    nf2::Int = 2 * nf
    nf1::Int = nf + 1
    rf1::UnitRange{Int} = UnitRange(1, nf)
    rf2::UnitRange{Int} = UnitRange(nf1, nf2)
    Y_dot::Matrix{Float64} = Ef[rf1, rf2]
    # Eager by design: a lazy `transpose` would stage a view aliasing the
    # input as a reverse residual, which the retain logic keeps unstaged.
    At::Matrix{Float64} = permutedims(A)
    Zr::Matrix{Float64} = zero(A)
    Topr::Matrix{Float64} = hcat(At, Y_bar)
    Botr::Matrix{Float64} = hcat(Zr, At)
    Mr::Matrix{Float64} = vcat(Topr, Botr)
    Er::Matrix{Float64} = exp(Mr)
    A_bar::Matrix{Float64} = Er[rf1, rf2]
    return Y, Y_dot, A_bar
end

"""
    rk_expm(A::Matrix{Float64}) -> Matrix{Float64}

Matrix exponential as an RK-owned primitive: `rk_expm(A) == exp(A)`, with
forward- and reverse-mode rules generated from the one pure-math graph
`rk_expm_rule` (JVP and VJP via augmented-matrix exponentials).
Differentiating through it with Enzyme — directly or via
`DifferentiationInterface` with `AutoEnzyme` — uses those generated cuts, so
the LAPACK `ccall` inside is never differentiated. No Reactant custom rule is
emitted (see `docs/src/manual-derivative-rules.md`); tracing the primal needs
backend `exp` support for traced arrays.
"""
const rk_expm = derivative_rule(rk_expm_rule; primal = :Y,
    directions = (A = :A_dot,), tangent = :Y_dot,
    covector = :Y_bar, cotangents = (A = :A_bar,), name = :rk_expm)
