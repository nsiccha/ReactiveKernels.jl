# Run with Julia 1.13 and Enzyme dogfood in the active project.
# The GP Poisson sampler multiplies a lower factor by a contiguous view
# into its packed position. Either storage convention for the lower factor
# reaches LinearAlgebra._trimul!. Check the ordinary reverse for both the
# allocating product and the product that writes into a prepared vector.
using Enzyme, LinearAlgebra, Test

product_sum(L, x) = sum(L * x)
product_sum!(L, x, dest) = sum(mul!(dest, L, x))

function check_reverse(f, L, x, destinations...)
    dL = Enzyme.make_zero(L)
    dx = Enzyme.make_zero(x)
    outputs = map(dest -> Duplicated(dest, zero(dest)), destinations)
    report = Enzyme.derivative_report() do
        autodiff(Reverse, f, Active, Duplicated(L, dL), Duplicated(x, dx), outputs...)
    end

    weights = ones(length(x))
    expected = weights * transpose(x)
    @test Matrix(dL) ≈ (L isa LowerTriangular ? tril(expected) : expected)
    @test dx ≈ transpose(L) * weights

    @test any(d -> d.func === typeof(f), report.derivatives)
    fallbacks = String[]
    for d in report.derivatives, finding in something(d.findings)
        if !finding.throw_only && finding.kind === :enzyme_runtime_fallback
            push!(fallbacks, string(finding.callee, " at ", first(finding.frames)))
        end
    end
    @test isempty(fallbacks)
end

@testset "Reverse of triangular products with a packed vector view" begin
    packed = collect(1.0:13.0)
    x = view(packed, 3:13)
    A = reshape(sin.(1:121), 11, 11)
    # cholesky(Symmetric(K)).L wraps adjoint storage; rk_cholesky_lower(K)
    # wraps a dense matrix. Give both wrappers the same logical entries.
    lower = LowerTriangular(A)
    stock = LowerTriangular(copy(transpose(A))')
    for (label, L, rhs) in (("matrix-backed factor", lower, x),
                            ("adjoint-backed factor", stock, x),
                            ("dense matrix control", Matrix(lower), x),
                            ("dense vector control", lower, copy(x)))
        @testset "$label" begin
            check_reverse(product_sum, L, rhs)
            check_reverse(product_sum!, L, rhs, similar(rhs))
        end
    end
end
