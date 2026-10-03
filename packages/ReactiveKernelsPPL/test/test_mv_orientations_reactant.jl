using ReactiveKernels, ReactiveKernelsPPL, Reactant, Enzyme, Test
using ReactiveKernelsPPL: _SliceRows, _SliceCols, _SliceWhole, _PerSlice,
    _mvnormal_cholesky_slices_logpdf, _mvnormal_slices_logpdf

@kernel _mv_orientation_cholesky(B, mu, F, o) =
    _mvnormal_cholesky_slices_logpdf(o, B, mu, F)
@kernel _mv_orientation_covariance(B, mu, Sigma, o) =
    _mvnormal_slices_logpdf(o, B, mu, Sigma)
@kernel _mv_permean_cholesky(B, M, F, o) =
    _mvnormal_cholesky_slices_logpdf(o, B, _PerSlice(o, M), F)
@kernel _mv_permean_covariance(B, M, Sigma, o) =
    _mvnormal_slices_logpdf(o, B, _PerSlice(o, M), Sigma)

@testset "compiled multivariate orientations and per-slice means" begin
    F = [.9 0.; .15 1.1]
    for covariance in (false, true), orientation in (:rows, :cols, :vector, :perrow, :percol)
        rows = [.2 -.3; .1 .4; -.5 .3]
        o = orientation in (:cols, :percol) ? _SliceCols() :
            orientation === :vector ? _SliceWhole() : _SliceRows()
        B = orientation in (:cols, :percol) ? permutedims(rows) :
            orientation === :vector ? rows[1, :] : rows
        permean = orientation in (:perrow, :percol)
        mu = permean ? (orientation === :percol ? permutedims(rows ./ 3) : rows ./ 3) : [.1, -.2]
        spec = permean ? (covariance ? _mv_permean_covariance : _mv_permean_cholesky) :
                         (covariance ? _mv_orientation_covariance : _mv_orientation_cholesky)
        factor = covariance ? F * F' : F
        kernel = prepare(spec; bound = (; o), on_error = :ignore)
        rb, rm, rf = Reactant.to_rarray(B), Reactant.to_rarray(mu), Reactant.to_rarray(factor)
        compiled = Reactant.@compile kernel(rb, rm, rf)
        @test Float64(compiled(rb, rm, rf)) ≈ kernel(B, mu, factor) rtol = 1e-10
        grad(B, mu, factor) = Enzyme.gradient(Enzyme.Reverse, kernel, B, mu, factor)
        native = grad(B, mu, factor)
        compiled_grad = Reactant.@compile grad(rb, rm, rf)
        result = compiled_grad(rb, rm, rf)
        for i in 1:3
            @test Array(result[i]) ≈ native[i] rtol = 1e-9 atol = 1e-11
        end
        @test Array(rb) == B && Array(rm) == mu && Array(rf) == factor
    end
end

@testset "empty multivariate slice wrappers" begin
    F = [.9 0.; .15 1.1]
    for covariance in (false, true), o in (_SliceRows(), _SliceCols()), permean in (false, true)
        B = o isa _SliceRows ? zeros(0, 2) : zeros(2, 0)
        mu = permean ? copy(B) : zeros(2)
        spec = permean ? (covariance ? _mv_permean_covariance : _mv_permean_cholesky) :
                         (covariance ? _mv_orientation_covariance : _mv_orientation_cholesky)
        factor = covariance ? F * F' : F
        kernel = prepare(spec; bound = (; o), on_error = :ignore)
        @test kernel(B, mu, factor) == 0.0
        native = Enzyme.gradient(Enzyme.Reverse, kernel, B, mu, factor)
        @test all(g -> all(iszero, g), native)
        rb, rm, rf = Reactant.to_rarray(B), Reactant.to_rarray(mu), Reactant.to_rarray(factor)
        compiled = Reactant.@compile kernel(rb, rm, rf)
        @test Float64(compiled(rb, rm, rf)) == 0.0
    end
end
