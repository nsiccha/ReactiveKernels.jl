# Run with Julia 1.13.1 and an Enzyme build that provides derivative_report.
# The singleton-simplex branch constructs a fixed column for an empty
# coordinate axis. Both branches return a Matrix{Float64}. On dogfood
# 38e0256c the reverse contains a jl_apply_generic call to Enzyme.make_zero.
using Enzyme
using Test

branch_objective(x) = sum(abs2, size(x, 2) == 0 ? ones(size(x, 1), 1) : x)

@testset "reverse of a fresh fixed column or an active matrix" begin
    for x in (reshape(collect(1.0:6.0), 2, 3), zeros(2, 0))
        dx = zero(x)
        report = Enzyme.derivative_report() do
            Enzyme.autodiff(Reverse, branch_objective, Active, Duplicated(x, dx))
        end
        @test branch_objective(x) == (isempty(x) ? size(x, 1) : sum(abs2, x))
        @test dx == 2x
        @test !isempty(report.derivatives)
        dispatch = [finding for derivative in report.derivatives
                    for finding in something(derivative.findings, [])
                    if !finding.throw_only &&
                        finding.kind in (:dynamic_dispatch, :enzyme_runtime_fallback)]
        @test isempty(dispatch)
    end
end
