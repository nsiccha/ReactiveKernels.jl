# `rk_expm` is the owned matrix-exponential primitive
# (`src/expm_rule.jl`): its value and both AD directions come from the one
# pure-math graph `rk_expm_rule`, and the Enzyme adapter in ReactiveKernels'
# own extension reads them from that graph's cuts. Ordinary reverse Enzyme
# through `LinearAlgebra.exp` fails (`EnzymeNoDerivativeError` on the LAPACK
# `ccall`); no Reactant custom rule is emitted (no release
# carries the upstream mechanism), so no XLA assertions here by design.
using DifferentiationInterface: AutoEnzyme, gradient
import Enzyme
using Enzyme: Const, Duplicated, Forward
using LinearAlgebra: dot, exp, I
using ReactiveKernels
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

# Deterministic inputs at two sizes proving the rule
# is size-generic, not a small-case special.
const _EXPM_CASES = (
    [0.1 0.2 0.0; -0.1 0.1 0.3; 0.0 -0.2 0.2],
    [0.1 0.2 0.0 -0.1; -0.1 0.1 0.3 0.0; 0.0 -0.2 0.2 0.1; 0.05 0.0 -0.1 0.15],
)
_expm_direction(n) = [0.1 * sin(i + 2j) for i in 1:n, j in 1:n]
_expm_covector(n) = [0.2 * cos(2i + j) for i in 1:n, j in 1:n]

# Independent high-precision Taylor oracle for these small-norm fixtures.
# Eighty terms put the truncation error far below 256-bit rounding here.
function _expm_reference_sum(A)
    setprecision(256) do
        B = BigFloat.(A)
        term = Matrix{BigFloat}(I, size(A)...)
        result = copy(term)
        for k in 1:80
            term = term * B / k
            result += term
        end
        sum(result)
    end
end

@testset "rk_expm is a generated rule on an owned callable" begin
    @test rk_expm isa DerivativeRule
    @test sprint(show, rk_expm) ==
        "DerivativeRule(:rk_expm; inputs = (:A,), branches = forward+reverse)"
    # Served by ReactiveKernels' generic generated-rule adapter.
    @test Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt) !== nothing
    for A in _EXPM_CASES
        @test rk_expm(A) == exp(A)
    end
    # The cuts are ordinary prepared kernels over the one graph.
    A = first(_EXPM_CASES)
    dA = _expm_direction(3)
    yb = _expm_covector(3)
    primal = prepare(ReactiveKernels.rk_expm_rule; have = (:A,), want = :Y)
    fwd = prepare(ReactiveKernels.rk_expm_rule;
        have = (:A, :A_dot), want = (:Y, :Y_dot))
    rev = prepare(ReactiveKernels.rk_expm_rule;
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

module NativeExpmTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test
using LinearAlgebra: Diagonal, diag
const exp_alias = Base.exp
const _parent = parentmodule(@__MODULE__)

@kernel bare(A) = begin
    E = exp(A)
    total = sum(E)
    return total
end
@kernel qualified(A) = begin
    total = sum(Base.exp(A))
    return total
end
@kernel aliased(A) = begin
    total = sum(exp_alias(A))
    return total
end
@traceable transformed(A) = exp(A)
@kernel helper(A) = begin
    total = sum(transformed(A))
    return total
end
@kernel exponential(A) = begin
    E = exp(A)
    return E
end
@kernel pointwise(A) = begin
    E = exp.(A)
    return E
end
@kernel shadowed_port(exp, A) = begin
    E = exp(A)
    return E
end
@kernel shadowed_local(A) = begin
    E = let exp = x -> x .+ 1
        exp(A)
    end
    return E
end
@kernel shadowed_module(Base, A) = begin
    E = Base.exp(A)
    return E
end
module OtherExp
exp(A) = A .- 1
end
@kernel other_binding(A) = begin
    E = OtherExp.exp(A)
    return E
end
@kernel recurrence(a, steps, A) = begin
    trajectory = scan(steps, Ref(a), Ref(A); init = 0.0) do carry, dt, scale, system
        next = dt > 0 ? carry + sum(exp(system * (scale * dt))) : carry
        (next, next)
    end
    total = sum(trajectory)
    return total
end
@kernel guarded(A, enabled) = begin
    total = enabled ? sum(exp(A)) : sum(A)
    return total
end
@kernel batches(a, systems) = begin
    values = plate(systems, Ref(a)) do A, scale
        sum(exp(scale * A))
    end
    total = sum(values)
    return total
end

@testset "authored matrix exp uses the generated rule in ordinary Enzyme" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for A in _parent._EXPM_CASES
        expected = sum(exp(A))
        reference = _parent._expm_reference_sum(A)
        @test abs(BigFloat(expected) - reference) <= 4BigFloat(eps(expected))
        expected_gradient = _parent._expm_fd_gradient(M -> sum(exp(M)), A)
        saved = copy(A)
        for spec in (bare, qualified, aliased, helper)
            k = prepare(spec)
            ad = prepare_ad(k, backend, A; active = :A)
            value, gradient = ad_value_and_gradient(ad, A)
            # Ordinary AD can change a floating-point reduction's rounding.
            # Bound that scalar loss difference to two Float64 spacings;
            # exact matrix-primal checks and ordinary gradient checks remain.
            @test value ≈ expected rtol = 0 atol = 2eps(expected)
            @test abs(BigFloat(value) - reference) <= 4BigFloat(eps(expected))
            @test gradient ≈ expected_gradient rtol = 1e-7 atol = 1e-8
        end
        k = prepare(exponential)
        dA = _parent._expm_direction(size(A, 1))
        tangent = only(Enzyme.autodiff(Enzyme.Forward, Enzyme.Const(k),
            Enzyme.Duplicated, Enzyme.Duplicated(A, dA)))
        @test tangent ≈ last(forward_cut(rk_expm, A, dA))
        @test A == saved
    end
end

@testset "native exp preserves bindings and Base's other domains" begin
    A = first(_parent._EXPM_CASES)
    @test prepare(exponential)(A) == exp(A)
    @test prepare(exponential)(0.3) == exp(0.3)
    @test prepare(exponential)(Float32.(A)) == exp(Float32.(A))
    @test prepare(exponential)(Diagonal(diag(A))) == exp(Diagonal(diag(A)))
    @test prepare(pointwise)(A) == exp.(A)
    @test prepare(shadowed_port)(x -> 2x, A) == 2A
    @test prepare(shadowed_local)(A) == A .+ 1
    @test prepare(shadowed_module)((; exp = x -> 3x), A) == 3A
    @test prepare(other_binding)(A) == A .- 1

    # Resolve by function identity, including generated-source GlobalRefs.
    native(ex, names...) = ReactiveKernels._kernel_native_body(
        ex, @__MODULE__, Set{Symbol}(names))
    @test native(Expr(:call, GlobalRef(Base, :exp), :A), :A).args[1] ==
          GlobalRef(ReactiveKernels, :_native_exp)
    @test native(:(exp_alias(A)), :A).args[1] ==
          GlobalRef(ReactiveKernels, :_native_exp)
    @test native(:(exp(A)), :A, :exp).args[1] === :exp
    @test native(:(Base.exp(A)), :A, :Base).args[1] == :(Base.exp)
end

@testset "ordinary Reverse through matrix exp in a lazy scan branch" begin
    A = first(_parent._EXPM_CASES)
    steps = [0.0, 0.2, -1.0, 0.4]
    reference(a) = begin
        carry = 0.0
        total = 0.0
        for dt in steps
            dt > 0 && (carry += sum(exp(A * (a * dt))))
            total += carry
        end
        total
    end
    k = prepare(recurrence; bound = (; steps, A))
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for a in (0.3, -0.6)
        ad = prepare_ad(k, backend, a; active = :a)
        value, gradient = ad_value_and_gradient(ad, a)
        @test value ≈ reference(a) rtol = 1e-14
        @test gradient ≈ (reference(a + 1e-6) - reference(a - 1e-6)) / 2e-6 rtol = 1e-7
    end
    # An untaken matrix exponential must retain its guard, including when
    # evaluating it would throw for the input shape.
    nonsquare = ones(2, 3)
    inactive = prepare(guarded; bound = (; enabled = false))
    ad = prepare_ad(inactive, backend, nonsquare; active = :A)
    value, gradient = ad_value_and_gradient(ad, nonsquare)
    @test value == 6.0
    @test gradient == ones(2, 3)

    systems = collect(_parent._EXPM_CASES)
    cells = prepare(batches; bound = (; systems))
    cell_reference(a) = sum(sum(exp(a * M)) for M in systems)
    ad = prepare_ad(cells, backend, 0.3; active = :a)
    value, gradient = ad_value_and_gradient(ad, 0.3)
    @test value ≈ cell_reference(0.3)
    @test gradient ≈ (cell_reference(0.3 + 1e-6) - cell_reference(0.3 - 1e-6)) / 2e-6 rtol = 1e-7
end
end
