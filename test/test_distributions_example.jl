using Test
using Distributions

include(joinpath(@__DIR__, "..", "examples", "distributions.jl"))
using .DistributionExamples
using ReactiveKernels: code_expr

@testset "Native log-density examples" begin
    artifacts = map(evaluate_source, all_sources())

    @testset "sources build native recipes checked against a Distributions oracle" begin
        @test length(artifacts) == 8
        # Every source declares @kernel recipes.
        @test all(source -> occursin(r"@kernel \w+\(", source), all_sources())
        continuous, discrete, vectorized = all_sources()
        # No forced API demonstrations: these are plain native densities. Nothing
        # shoehorns a `compose` call; the vectorized source generates its batched
        # kernel with `plate`.
        @test all(source -> !occursin("compose(", source), all_sources())
        @test occursin("plate(", vectorized)
        # Regression guard against the exp-then-log round trip: with logσ in HAND
        # the density uses it directly (the -logσ term), never rebuilds σ via exp
        # only to take log(σ) again. Scan code only — drop `#` comments so prose
        # mentioning the anti-pattern doesn't trip the guard.
        code_only(src) = join(
            (first(split(line, "#")) for line in eachsplit(src, "\n")), "\n",
        )
        continuous_code = code_only(continuous)
        @test occursin("- logσ", continuous_code)
        @test !occursin("log(σ)", continuous_code)
        @test !occursin("log(exp", continuous_code)
        # The compute path is Distributions.jl-free.
        @test all(artifacts) do artifact
            !occursin("Distributions", string(code_expr(artifact.kernel)))
        end
        # Values match the independent Distributions.jl oracle.
        @test all(artifact -> isapprox(artifact.output, artifact.reference), artifacts)
        lognormal = artifacts[6]
        _, μ, logσ = Tuple(lognormal.inputs)
        @test lognormal.kernel(-1.0, μ, logσ) == -Inf

        mvnormal, ar1 = artifacts[7:8]
        mvn_code = string(code_expr(mvnormal.kernel))
        ar1_code = string(code_expr(ar1.kernel))
        @test all(name -> occursin(name, mvn_code),
                  ("centered", "whitened", "half_logdet_cov", "quadratic",
                   "logdensity"))
        @test all(name -> occursin(name, ar1_code),
                  ("centered", "innovations", "transition_ss", "valid", "logdensity"))
        @test occursin("LowerTriangular", MVNORMAL_SOURCE)
        @test occursin("cholesky(Symmetric(covariance))", MVNORMAL_SOURCE)
        @test occursin("cholesky(Symmetric(precision))", MVNORMAL_SOURCE)

        covariance = mvnormal.parametrization_inputs.covariance.covariance
        reference = logpdf(MvNormal(mvnormal.inputs.μ, covariance), mvnormal.inputs.x)
        @test all(output -> isapprox(output, reference),
                  values(mvnormal.parametrization_outputs))

        expected_haves = (;
            covariance = r"\bcovariance::Matrix",
            cholesky = r"\bchol::Matrix",
            precision = r"\bprecision::Matrix",
            precision_cholesky = r"\bprecision_chol::Matrix",
        )
        expected_recipe_counts = (;
            covariance = 6,
            cholesky = 5,
            precision = 6,
            precision_cholesky = 5,
        )
        for name in propertynames(mvnormal.kernels)
            kernel = getproperty(mvnormal.kernels, name)
            kernel_inputs = Tuple(getproperty(mvnormal.parametrization_inputs, name))
            kernel_argtypes = Tuple{map(typeof, kernel_inputs)...}
            generated = string(code_expr(kernel))
            @test occursin(getproperty(expected_haves, name), generated)
            for other in propertynames(mvnormal.kernels)
                other === name && continue
                @test !occursin(getproperty(expected_haves, other), generated)
            end
            @test length(kernel.plan.recipes) ==
                  getproperty(expected_recipe_counts, name)
            @test only(Base.return_types(kernel, kernel_argtypes)) === Float64
        end
        @test mvnormal.kernels.covariance.plan.cost >
              mvnormal.kernels.cholesky.plan.cost
        @test mvnormal.kernels.precision.plan.cost >
              mvnormal.kernels.precision_cholesky.plan.cost
        _, arμ, arϕ, arlogσ = Tuple(ar1.inputs)
        @test ar1.kernel(ar1.inputs.x, arμ, 1.0, arlogσ) == -Inf
        @test ar1.kernel(ar1.inputs.x, arμ, -1.0, arlogσ) == -Inf

        # `replica` adds one trailing independent-replica axis. It does not
        # scalar-plate coordinates/time: each vector or series remains one
        # complete call of the authored mathematical kernel.
        for artifact in (mvnormal, ar1)
            replica_inputs = Tuple(artifact.replica_inputs)
            matrix = first(replica_inputs)
            shared = Base.tail(replica_inputs)
            expected = [
                artifact.kernel(copy(matrix[:, i]), shared...)
                for i in axes(matrix, 2)
            ]
            @test artifact.replica_output ≈ expected
            @test artifact.replicated(replica_inputs...) ≈ expected
            @test code_expr(artifact.replicated) == code_expr(artifact.kernel)
        end
        # Every MVN HAVE boundary lifts the same whole-vector graph. The
        # parametrization changes only the selected recipes and call boundary.
        replica_x = mvnormal.replica_inputs.x
        for name in propertynames(mvnormal.kernels)
            kernel = getproperty(mvnormal.kernels, name)
            replicated = getproperty(mvnormal.replicated_kernels, name)
            scalar_inputs = Tuple(getproperty(mvnormal.parametrization_inputs, name))
            shared = Base.tail(scalar_inputs)
            expected = [kernel(copy(replica_x[:, i]), shared...)
                        for i in axes(replica_x, 2)]
            @test replicated(replica_x, shared...) ≈ expected
            @test code_expr(replicated) == code_expr(kernel)
        end
    end

    @testset "allocation is well-formed; the scalar densities are 0-alloc" begin
        @test all(artifact -> artifact.allocated_bytes isa Int, artifacts)
        @test all(artifact -> artifact.reference_allocated_bytes isa Int, artifacts)
        @test all(artifact -> artifact.allocated_bytes >= 0, artifacts)
        @test all(artifact -> artifact.reference_allocated_bytes >= 0, artifacts)
        # The scalar native kernels are fully non-allocating, and so is their oracle.
        @test all(artifact -> artifact.allocated_bytes == 0, artifacts[1:2])
        @test all(artifact -> artifact.reference_allocated_bytes == 0, artifacts[1:2])
    end

    @testset "concrete inference evidence matches exact result types" begin
        expected_returns = ntuple(_ -> Float64, 8)
        for (artifact, expected_return) in zip(artifacts, expected_returns)
            observed = artifact.kernel(Tuple(artifact.inputs)...)
            @test isconcretetype(artifact.inferred_return)
            @test artifact.inferred_return === typeof(observed)
            @test artifact.inferred_return === expected_return
            @test typeof(observed) === expected_return
        end
    end
end
