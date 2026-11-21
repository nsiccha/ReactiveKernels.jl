# Run with Enzyme in the active project. An RKPPL sampler reads a vector
# parameter as a view into its packed coordinates. Enzyme snag
# use-ordinary-enz-fd0accd2 reproduces with just that view and a dense solve.
using Enzyme, LinearAlgebra, Test

dense_rhs(A, x) = sum(A \ x)
packed_rhs(A, x, r) = sum(A \ view(x, r))

@testset "Dense solve with a packed right-hand side" begin
    A = [2.0 -0.4; 0.3 1.5]
    b = [0.35, -0.2]
    expected = transpose(A) \ ones(length(b))
    db = zero(b)
    autodiff(Reverse, dense_rhs, Active, Const(A), Duplicated(b, db))
    @test db ≈ expected

    packed = [4.0; b; 5.0]
    r = 2:3
    @test packed_rhs(A, packed, r) == dense_rhs(A, b)
    dp = zero(packed)
    autodiff(Reverse, packed_rhs, Active, Const(A), Duplicated(packed, dp), Const(r))
    @test dp[r] ≈ expected
    @test iszero(dp[1]) && iszero(dp[end])
end
