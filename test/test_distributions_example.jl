using Test
using DifferentiationInterface
using DifferentiationInterface: Constant
import Enzyme
using Distributions

include(joinpath(@__DIR__, "..", "examples", "distributions.jl"))
using .DistributionExamples
using ReactiveKernels: code_expr

const DISTRIBUTION_ENZYME_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

struct SecondDistributionInput{K}
    kernel::K
end
(call::SecondDistributionInput)(active, first_input) =
    call.kernel(first_input, active)

@testset "Native log-density examples" begin
    artifacts = map(evaluate_source, all_sources())

    @testset "sources build native recipes checked against a Distributions oracle" begin
        @test length(artifacts) == 11
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

        exponential, geometric, uniform = artifacts[7:9]
        @test exponential.kernel(-1.0, exponential.inputs.logθ) == -Inf
        @test geometric.kernel(-1, geometric.inputs.logitp) == -Inf
        @test uniform.kernel(-1.1, uniform.inputs.lower, uniform.inputs.upper) == -Inf
        @test uniform.kernel(
            uniform.inputs.x, uniform.inputs.upper, uniform.inputs.lower) == -Inf

        # Each newly authored scalar formula lifts through the generic `plate`
        # path without a family-specific vectorized implementation.
        for artifact in (exponential, geometric, uniform)
            plated_inputs = Tuple(artifact.plate_inputs)
            observations = first(plated_inputs)
            shared = Base.tail(plated_inputs)
            expected = sum(artifact.kernel(observation, shared...)
                           for observation in observations)
            @test artifact.plate_output ≈ expected
            @test artifact.plated(plated_inputs...) ≈ expected
            plated_code = string(code_expr(artifact.plated))
            @test occursin("logdensity", plated_code)
            @test !occursin("Distributions", plated_code)
        end

        mvnormal, ar1 = artifacts[10:11]
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
        # Scalar native kernels are fully non-allocating, including the added
        # Exponential/Geometric/Uniform formulas, and so are their oracles.
        scalar_artifacts = artifacts[[1, 2, 4, 5, 6, 7, 8, 9]]
        @test all(artifact -> artifact.allocated_bytes == 0, scalar_artifacts)
        @test all(artifact -> artifact.reference_allocated_bytes == 0,
                  scalar_artifacts)
    end

    @testset "concrete inference evidence matches exact result types" begin
        expected_returns = ntuple(_ -> Float64, 11)
        for (artifact, expected_return) in zip(artifacts, expected_returns)
            observed = artifact.kernel(Tuple(artifact.inputs)...)
            @test isconcretetype(artifact.inferred_return)
            @test artifact.inferred_return === typeof(observed)
            @test artifact.inferred_return === expected_return
            @test typeof(observed) === expected_return
        end
    end

    @testset "plain DI + Enzyme reverse mode covers every prepared density" begin
        gradients = Dict{Symbol,Any}()
        for artifact in artifacts
            values = Tuple(artifact.inputs)
            if artifact.name in (:discrete_bernoulli_logit, :geometric_logit)
                call = SecondDistributionInput(artifact.kernel)
                active = values[2]
                constants = (Constant(values[1]),)
            else
                call = artifact.kernel
                active = first(values)
                constants = map(Constant, Base.tail(values))
            end
            gradient = DifferentiationInterface.gradient(
                call, DISTRIBUTION_ENZYME_BACKEND, active, constants...)
            gradients[artifact.name] = gradient
            components = gradient isa Number ? (gradient,) : gradient
            @test all(isfinite, components)
        end

        normal = artifacts[1]
        x, μ, logσ = Tuple(normal.inputs)
        @test gradients[:continuous_normal] ≈ -(x - μ) / exp(2logσ)

        bernoulli = artifacts[2]
        observed, logit = Tuple(bernoulli.inputs)
        @test gradients[:discrete_bernoulli_logit] ≈
              Int(observed) - 1 / (1 + exp(-logit))

        vectorized = artifacts[3]
        xbatch, μbatch, logσbatch = Tuple(vectorized.inputs)
        @test gradients[:vectorized_normal] ≈
              @. -(xbatch - μbatch) / exp(2logσbatch)
    end
end
