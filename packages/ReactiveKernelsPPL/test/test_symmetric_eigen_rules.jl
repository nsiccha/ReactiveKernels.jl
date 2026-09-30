# `rk_symmetric_eigvals` / `rk_symmetric_eigvecs` are the owned
# symmetric-eigendecomposition primitives
# (`src/symmetric_eigen_rules.jl`): their values and both AD directions come
# from the one pure-math graph per output, and the Enzyme adapter in
# ReactiveKernels' own extension reads them from those graphs' cuts. Ordinary
# reverse Enzyme through `eigen(::Symmetric)` fails
# (`EnzymeNoDerivativeError` on the LAPACK `syevr!` `ccall`), so these rules
# are the native-execution replacement path for
# symmetric-eigendecomposition-based densities (the posteriordb `kronecker_gp`
# marginal likelihood); no Reactant custom rule is emitted (no release
# carries the upstream mechanism), so no XLA assertions here by design.
using DifferentiationInterface: AutoEnzyme, gradient
import Enzyme
using Enzyme: Const, Duplicated, Forward
using LinearAlgebra: Diagonal, Symmetric, dot, eigen, qr
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Deterministic symmetric inputs with distinct eigenvalues: a fixed
# orthogonal factor around a strictly increasing spectrum (gaps stay above
# 0.05, so the F matrix is well conditioned).
function _seig_case(n, s)
    B = [sin(i + s * j) + 0.5 * cos(2i - j) for i in 1:n, j in 1:n]
    Q0 = Matrix(qr(B).Q)
    lam = [0.5 + 0.07i + 0.01 * sin(s * i) for i in 1:n]
    Matrix(Symmetric(Q0 * Diagonal(lam) * Q0')), lam
end
const _SEIG_CASES = (_seig_case(4, 1.0), _seig_case(30, 2.5))
const _SEIG_S = (1.0, 2.5)

_seig_direction(n, s) = begin
    B = [0.1 * sin(i + 2j + s) for i in 1:n, j in 1:n]
    (B + B') / 2
end
_seig_rbar(n, s) = [0.2 * cos(2i + s) for i in 1:n]
_seig_qbar(n, s) = [0.2 * cos(2i + j + s) for i in 1:n, j in 1:n]

# Eigenvector columns are sign-ambiguous; align a perturbed factor to the
# base point so FD functionals of the vectors are smooth.
function _seig_align!(Qm, Q0)
    for j in axes(Qm, 2)
        dot(view(Qm, :, j), view(Q0, :, j)) < 0 && (view(Qm, :, j) .*= -1)
    end
    Qm
end

# Central FD over symmetric perturbations: the rules read only the upper
# triangle, so the perturbation stays in the documented symmetric domain.
function _seig_fd_gradient(f, X; h = 1e-6)
    G = similar(X, Float64)
    for i in eachindex(X)
        E = zeros(size(X)); E[i] += 1; E = (E + E') / 2
        G[i] = (f(X + h * E) - f(X - h * E)) / (2h)
    end
    (G + G') / 2
end

@testset "symmetric eigen rules are generated rules on owned callables" begin
    @test rk_symmetric_eigvals isa DerivativeRule
    @test rk_symmetric_eigvecs isa DerivativeRule
    @test sprint(show, rk_symmetric_eigvals) == "DerivativeRule(" *
        ":rk_symmetric_eigvals; inputs = (:A,), branches = forward+reverse)"
    @test sprint(show, rk_symmetric_eigvecs) == "DerivativeRule(" *
        ":rk_symmetric_eigvecs; inputs = (:A,), branches = forward+reverse)"
    # Served by ReactiveKernels' generic adapter: this package defines no
    # AD extension of its own.
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt) !== nothing
    @test Base.get_extension(
        ReactiveKernelsPPL, :ReactiveKernelsPPLEnzymeExt) === nothing
    for (A, _) in _SEIG_CASES
        F = eigen(Symmetric(A))
        @test rk_symmetric_eigvals(A) == F.values
        @test rk_symmetric_eigvecs(A) == F.vectors
    end
    # The cuts are ordinary prepared kernels over the one graph per output.
    A, _ = first(_SEIG_CASES)
    n = size(A, 1)
    dA = _seig_direction(n, 1.0)
    rb = _seig_rbar(n, 1.0)
    qb = _seig_qbar(n, 1.0)
    vprimal = prepare(ReactiveKernelsPPL.rk_symmetric_eigvals_rule;
        have = (:A,), want = :R)
    vfwd = prepare(ReactiveKernelsPPL.rk_symmetric_eigvals_rule;
        have = (:A, :A_dot), want = (:R, :R_dot))
    vrev = prepare(ReactiveKernelsPPL.rk_symmetric_eigvals_rule;
        have = (:A, :R_bar), want = :A_bar)
    @test vprimal(A) == rk_symmetric_eigvals(A)
    @test vfwd(A, dA) == forward_cut(rk_symmetric_eigvals, A, dA)
    @test vrev(A, rb) == only(reverse_cut(rk_symmetric_eigvals, Val(1), A, rb))
    qprimal = prepare(ReactiveKernelsPPL.rk_symmetric_eigvecs_rule;
        have = (:A,), want = :Q)
    qfwd = prepare(ReactiveKernelsPPL.rk_symmetric_eigvecs_rule;
        have = (:A, :A_dot), want = (:Q, :Q_dot))
    qrev = prepare(ReactiveKernelsPPL.rk_symmetric_eigvecs_rule;
        have = (:A, :Q_bar), want = :A_bar)
    @test qprimal(A) == rk_symmetric_eigvecs(A)
    @test qfwd(A, dA) == forward_cut(rk_symmetric_eigvecs, A, dA)
    @test qrev(A, qb) == only(reverse_cut(rk_symmetric_eigvecs, Val(1), A, qb))
end

@testset "symmetric eigen reverse through Enzyme matches finite differences" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for (case, (A, _)) in enumerate(_SEIG_CASES)
        s = _SEIG_S[case]
        n = size(A, 1)
        dA = _seig_direction(n, s)
        rb = _seig_rbar(n, s)
        qb = _seig_qbar(n, s)
        F = eigen(Symmetric(A))
        # Values: the generated reverse rule at a sum functional, plus the
        # VJP branch at an arbitrary covector. The sum-functional FD carries
        # LAPACK rounding noise (~eps*||A||/h per eigenvalue, summed over n),
        # so its bound is looser at n = 30; the rule itself is exact (the
        # staged checks below compare against the authored cut, not FD).
        g_rule = gradient(M -> sum(rk_symmetric_eigvals(M)), backend, A)
        g_fd = _seig_fd_gradient(M -> sum(eigen(Symmetric(M)).values), A)
        @test maximum(abs.(g_rule .- g_fd)) < 5e-8
        (Ab,) = reverse_cut(rk_symmetric_eigvals, Val(1), A, rb)
        @test maximum(abs.(Ab .-
            _seig_fd_gradient(M -> dot(rb, eigen(Symmetric(M)).values), A))) < 1e-8
        # Vectors: the VJP branch against the sign-aligned functional.
        (Abq,) = reverse_cut(rk_symmetric_eigvecs, Val(1), A, qb)
        aligned_dot(M) = dot(qb,
            _seig_align!(copy(eigen(Symmetric(M)).vectors), F.vectors))
        @test maximum(abs.(Abq .- _seig_fd_gradient(aligned_dot, A))) < 1e-7
        # Staged reverse equals the self-contained cut; residuals are the
        # covector-independent frontier (Q for values; Q and the F matrix
        # for vectors), computed once.
        yp, res = stage_primal(rk_symmetric_eigvals, Val(1), A)
        @test yp == F.values
        @test stage_residuals(rk_symmetric_eigvals, Val(1)) == (:Q,)
        @test only(stage_reverse(rk_symmetric_eigvals, Val(1), res, rb)) == Ab
        yq, resq = stage_primal(rk_symmetric_eigvecs, Val(1), A)
        @test yq == F.vectors
        @test stage_residuals(rk_symmetric_eigvecs, Val(1)) == (:Q, :Fm)
        @test only(stage_reverse(rk_symmetric_eigvecs, Val(1), resq, qb)) == Abq
        # The two authored branches agree with each other.
        _, Rd = forward_cut(rk_symmetric_eigvals, A, dA)
        @test dot(rb, Rd) ≈ dot(Ab, dA)
        _, Qd = forward_cut(rk_symmetric_eigvecs, A, dA)
        @test dot(qb, Qd) ≈ dot(Abq, dA)
    end
end

@testset "symmetric eigen forward through Enzyme matches finite differences" begin
    for (case, (A, _)) in enumerate(_SEIG_CASES)
        s = _SEIG_S[case]
        n = size(A, 1)
        dA = _seig_direction(n, s)
        F = eigen(Symmetric(A))
        _, Rd = @inferred forward_cut(rk_symmetric_eigvals, A, dA)
        fwd = Enzyme.autodiff(Forward, Const(rk_symmetric_eigvals), Duplicated,
            Duplicated(A, dA))
        @test only(fwd) ≈ Rd
        h = 1e-7
        jvp_fd = (sum(eigen(Symmetric(A + h * dA)).values) -
                  sum(eigen(Symmetric(A - h * dA)).values)) / (2h)
        @test abs(sum(Rd) - jvp_fd) / abs(jvp_fd) < 1e-6
        _, Qd = @inferred forward_cut(rk_symmetric_eigvecs, A, dA)
        fwdq = Enzyme.autodiff(Forward, Const(rk_symmetric_eigvecs), Duplicated,
            Duplicated(A, dA))
        @test only(fwdq) ≈ Qd
        Qp = _seig_align!(eigen(Symmetric(A + h * dA)).vectors, F.vectors)
        Qm = _seig_align!(eigen(Symmetric(A - h * dA)).vectors, F.vectors)
        @test maximum(abs.(Qd .- (Qp .- Qm) / (2h))) < 1e-6
    end
end

# A kernel using BOTH rules in a gauge-invariant loss (whitened quadratic
# plus log-eigenvalues, the kronecker_gp shape at n = 4): `prepare_ad` with
# `AutoEnzyme(Reverse)` differentiates it, which raw `eigen(Symmetric)`
# cannot do.
@kernel _seig_mini_loss(u::Vector{Float64}, y::Vector{Float64},
        M0::Matrix{Float64}) = begin
    A::Matrix{Float64} = M0 + u * transpose(u)
    R::Vector{Float64} = rk_symmetric_eigvals(A)
    Q::Matrix{Float64} = rk_symmetric_eigvecs(A)
    w::Vector{Float64} = transpose(Q) * y
    e::Vector{Float64} = R .+ 0.5
    quad::Float64 = sum(w .^ 2 ./ e)
    ld::Float64 = sum(log.(e))
    loss::Float64 = quad + ld
    return loss
end

@testset "prepare_ad reverse over a kernel using both eigen rules" begin
    M0, _ = _seig_case(4, 7.0)
    y = [0.5, -0.3, 0.8, 0.1]
    u0 = 0.1 .* sin.((1:4) .+ 0.5)
    k = prepare(_seig_mini_loss; have = (:u, :y, :M0), want = :loss)
    @test isfinite(k(u0, y, M0))
    prep = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), u0, y, M0;
        active = :u)
    value, g = ReactiveKernels.ad_value_and_gradient!(prep, similar(u0),
        u0, y, M0)
    f(uu) = k(uu, y, M0)
    h = 1e-6
    gfd = map(eachindex(u0)) do i
        up = copy(u0); up[i] += h
        um = copy(u0); um[i] -= h
        (f(up) - f(um)) / (2h)
    end
    @test value ≈ f(u0)
    @test g ≈ gfd rtol = 1e-5
end
