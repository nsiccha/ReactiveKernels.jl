using DifferentiationInterface: AutoEnzyme, gradient
import Enzyme
import LinearAlgebra
using LinearAlgebra: LowerTriangular, PosDefException, Symmetric, cholesky
using ReactiveKernels
using Test

function _chol_loss(A)
    L = rk_cholesky_lower(A)
    result = 0.0
    for i in axes(L, 1), j in 1:i
        result += (i + 2j) * L[i, j]
    end
    return result
end

function _chol_lapack_loss(A)
    L = cholesky(Symmetric(A, :L)).L
    return sum((i + 2j) * L[i, j] for i in axes(L, 1) for j in 1:i)
end

@testset "owned pure-Julia lower Cholesky" begin
    for n in (0, 1, 2, 7, 19)
        B = [sin(i + 2j) for i in 1:n, j in 1:n]
        A = B * B' + 2 * Matrix{Float64}(LinearAlgebra.I, n, n)
        expected = cholesky(Symmetric(A, :L)).L
        # Deliberately invalid upper entries must not affect the factor.
        for j in 1:n, i in 1:j-1
            A[i, j] = NaN
        end
        before = copy(A)
        L = rk_cholesky_lower(A)
        @test L isa LowerTriangular{Float64, Matrix{Float64}}
        @test L ≈ expected
        @test Matrix(L) * Matrix(L)' ≈ Matrix(Symmetric(A, :L))
        @test isequal(A, before)
        @test parent(L) !== A
        if n > 0
            L[1, 1] = -1
            @test isequal(A, before)
        end
    end
    A = [4 99; 2 5]
    @test rk_cholesky_lower(A) == LowerTriangular([2.0 0.0; 1.0 2.0])
    @test rk_cholesky_lower(Float32.(A)) == rk_cholesky_lower(A)
    @test rk_cholesky_lower(view(A, :, :)) == rk_cholesky_lower(A)
    @test_throws ArgumentError rk_cholesky_lower(zeros(2, 3))
    for (A, pivot) in (([0.0;;], 1), ([-1.0;;], 1), ([NaN;;], 1),
                      ([1.0 8.0; 1.0 1.0], 2))
        err = try
            rk_cholesky_lower(A)
        catch e
            e
        end
        @test err isa PosDefException
        @test err.info == pivot
    end
end

@testset "lower Cholesky ordinary Enzyme reverse" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for n in (2, 5)
        B = [cos(2i + j) for i in 1:n, j in 1:n]
        A = B * B' + 3 * Matrix{Float64}(LinearAlgebra.I, n, n)
        before = copy(A)
        g = gradient(_chol_loss, backend, A)
        @test A == before
        @test _chol_loss(A) ≈ _chol_lapack_loss(A)
        for j in 1:n, i in 1:n
            E = zeros(n, n)
            E[i, j] = 1
            h = 1e-6
            # Independent LAPACK oracle; :L also tests zero upper sensitivity.
            expected = (_chol_lapack_loss(A + h * E) -
                        _chol_lapack_loss(A - h * E)) / (2h)
            @test g[i, j] ≈ expected rtol = 2e-6 atol = 2e-8
        end
    end
end
