# `rk_expm` is the owned matrix-exponential primitive
# (`src/expm_rule.jl`): its value and both AD directions come from the one
# pure-math graph `rk_expm_rule`, and the Enzyme adapter in ReactiveKernels'
# own extension reads them from that graph's cuts. Ordinary reverse Enzyme
# through `LinearAlgebra.exp` fails (`EnzymeNoDerivativeError` on the LAPACK
# `ccall`), so this rule is the native-execution replacement path for the
# hand-ported `_pk_expm3`; no Reactant custom rule is emitted (no release
# carries the upstream mechanism), so no XLA assertions here by design.
using DifferentiationInterface: AutoEnzyme, gradient
import Enzyme
using Enzyme: Const, Duplicated, Forward
using LinearAlgebra: dot, exp
using ReactiveKernels
using ReactiveKernelsPPL
using Test

_expm_fd_gradient(f, X; h = 1e-6) = begin
    G = similar(X, Float64)
    for i in eachindex(X)
        Xp = copy(X); Xp[i] += h
        Xm = copy(X); Xm[i] -= h
        G[i] = (f(Xp) - f(Xm)) / (2h)
    end
    G
end

# Deterministic inputs: the 3x3 PK shape plus a second size proving the rule
# is size-generic, not a small-case special.
const _EXPM_CASES = (
    [0.1 0.2 0.0; -0.1 0.1 0.3; 0.0 -0.2 0.2],
    [0.1 0.2 0.0 -0.1; -0.1 0.1 0.3 0.0; 0.0 -0.2 0.2 0.1; 0.05 0.0 -0.1 0.15],
)
_expm_direction(n) = [0.1 * sin(i + 2j) for i in 1:n, j in 1:n]
_expm_covector(n) = [0.2 * cos(2i + j) for i in 1:n, j in 1:n]

@testset "rk_expm is a generated rule on an owned callable" begin
    @test rk_expm isa DerivativeRule
    @test sprint(show, rk_expm) ==
        "DerivativeRule(:rk_expm; inputs = (:A,), branches = forward+reverse)"
    # Served by ReactiveKernels' generic adapter: this package defines no
    # AD extension of its own.
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt) !== nothing
    @test Base.get_extension(
        ReactiveKernelsPPL, :ReactiveKernelsPPLEnzymeExt) === nothing
    for A in _EXPM_CASES
        @test rk_expm(A) == exp(A)
    end
    # The cuts are ordinary prepared kernels over the one graph.
    A = first(_EXPM_CASES)
    dA = _expm_direction(3)
    yb = _expm_covector(3)
    primal = prepare(ReactiveKernelsPPL.rk_expm_rule; have = (:A,), want = :Y)
    fwd = prepare(ReactiveKernelsPPL.rk_expm_rule;
        have = (:A, :A_dot), want = (:Y, :Y_dot))
    rev = prepare(ReactiveKernelsPPL.rk_expm_rule;
        have = (:A, :Y_bar), want = :A_bar)
    @test primal(A) == rk_expm(A)
    @test fwd(A, dA) == forward_cut(rk_expm, A, dA)
    @test rev(A, yb) == only(reverse_cut(rk_expm, Val(1), A, yb))
end

@testset "rk_expm reverse through Enzyme matches finite differences" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for A in _EXPM_CASES
        g_rule = gradient(M -> sum(rk_expm(M)), backend, A)
        g_fd = _expm_fd_gradient(M -> sum(exp(M)), A)
        @test maximum(abs.(g_rule .- g_fd)) < 1e-8
        # The VJP branch at an arbitrary covector is the gradient of that
        # linear functional of the built-in.
        yb = _expm_covector(size(A, 1))
        (Ab,) = reverse_cut(rk_expm, Val(1), A, yb)
        @test maximum(abs.(Ab .- _expm_fd_gradient(M -> dot(yb, exp(M)), A))) < 1e-8
    end
end

@testset "rk_expm forward through Enzyme matches finite differences" begin
    for A in _EXPM_CASES
        n = size(A, 1)
        dA = _expm_direction(n)
        yb = _expm_covector(n)
        _, Yd = @inferred forward_cut(rk_expm, A, dA)
        fwd = Enzyme.autodiff(Forward, Const(rk_expm), Duplicated,
            Duplicated(A, dA))
        @test only(fwd) ≈ Yd
        h = 1e-7
        jvp_fd = (sum(exp(A + h * dA)) - sum(exp(A - h * dA))) / (2h)
        @test abs(sum(Yd) - jvp_fd) / abs(jvp_fd) < 1e-6
        # The two authored branches agree with each other.
        (Ab,) = reverse_cut(rk_expm, Val(1), A, yb)
        @test dot(yb, Yd) ≈ dot(Ab, dA)
    end
end
