using Test
using DifferentiationInterface
import Enzyme
using Distributions

using ReactiveKernelsDistributionKernels: DistributionKernelSources
using ReactiveKernelsKernelExamples.DistributionExamples
using ReactiveKernels: KernelObjectSpec, KernelSpec, @kernel, code_expr,
    extract, plan, plate, prepare

const NORMAL_LOGSCALE_KERNEL = prepare(NORMAL_LOGDENSITY;
    have = (:x, :location, :log_scale), want = :logpdf)
const CAUCHY_SCALE_KERNEL = prepare(CAUCHY_LOGDENSITY;
    have = (:x, :location, :scale), want = :logpdf)

@kernel embedded_distribution_logdensity(
        x::Float64, location::Float64, scale::Float64) = begin
    normal_term::Float64 = normal.logpdf(x)
    cauchy_term::Float64 = cauchy.logpdf(x)
    total::Float64 = normal_term + cauchy_term
    return total
end

const EMBEDDED_DISTRIBUTION_KERNEL = prepare(embedded_distribution_logdensity;
    have = (:x, :location, :scale))

const DISTRIBUTION_ENZYME_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

@testset "Native log-density examples" begin
    artifacts = map(evaluate_source, all_sources())

    @testset "reusable distribution objects compose without Distributions" begin
        @test all(object -> object isa KernelObjectSpec, (normal, cauchy, laplace))
        @test NORMAL_LOGDENSITY isa KernelSpec
        @test CAUCHY_LOGDENSITY isa KernelSpec
        @test !isdefined(DistributionKernelSources, :Distributions)

        x = 0.4
        location = -0.2
        scale = 1.3
        log_scale = log(scale)
        for (spec, reference) in (
                (NORMAL_LOGDENSITY, logpdf(Normal(location, scale), x)),
                (CAUCHY_LOGDENSITY, logpdf(Cauchy(location, scale), x)))
            scale_kernel = prepare(spec;
                have = (:x, :location, :scale), want = :logpdf)
            logscale_kernel = prepare(spec;
                have = (:x, :location, :log_scale), want = :logpdf)
            both_kernel = prepare(spec;
                have = (:x, :location, :scale, :log_scale),
                want = :logpdf)
            @test scale_kernel(x, location, scale) ≈ reference
            @test logscale_kernel(x, location, log_scale) ≈ reference
            @test both_kernel(x, location, scale, log_scale) ≈ reference
            @test only(Base.return_types(
                scale_kernel, Tuple{Float64,Float64,Float64})) === Float64
            @test only(Base.return_types(
                logscale_kernel, Tuple{Float64,Float64,Float64})) === Float64
            @test DistributionExamples._allocated(
                scale_kernel, x, location, scale) == 0
            @test DistributionExamples._allocated(
                logscale_kernel, x, location, log_scale) == 0
            @test !occursin("Distributions", string(code_expr(scale_kernel)))
            @test !occursin("Distributions", string(code_expr(logscale_kernel)))

            scale_plan = plan(spec;
                have = (:x, :location, :scale), want = :logpdf)
            logscale_plan = plan(spec;
                have = (:x, :location, :log_scale), want = :logpdf)
            both_plan = plan(spec;
                have = (:x, :location, :scale, :log_scale),
                want = :logpdf)
            recipe_outputs(selected_plan) =
                [only(recipe.outputs).name for recipe in selected_plan.recipes]
            scale_outputs = recipe_outputs(scale_plan)
            logscale_outputs = recipe_outputs(logscale_plan)
            both_outputs = recipe_outputs(both_plan)
            @test :log_scale in scale_outputs
            @test !(:scale in scale_outputs)
            @test :scale in logscale_outputs
            @test !(:log_scale in logscale_outputs)
            @test !(:scale in both_outputs)
            @test !(:log_scale in both_outputs)
            @test all(outputs -> :standardized in outputs,
                      (scale_outputs, logscale_outputs, both_outputs))
            @test all(outputs -> last(outputs) === :logpdf,
                      (scale_outputs, logscale_outputs, both_outputs))
        end

        joint = extract(normal;
            have = (:x, :location, :scale), want = (:logpdf, :cdf))
        joint_plan = plan(joint)
        @test count(recipe -> only(recipe.outputs).name === :standardized,
                    joint_plan.recipes) == 1
        @test all(isapprox.(prepare(joint)(x, location, scale),
            (logpdf(Normal(location, scale), x),
             cdf(Normal(location, scale), x))))
        @test prepare(normal.quantile)(location, scale, 0.73) ≈
              quantile(Normal(location, scale), 0.73)

        observations = [28.0, 8.0, -3.0, 7.0]
        effects = [1.0, 1.5, -0.5, 0.25]
        scales = [15.0, 10.0, 16.0, 11.0]
        likelihood = plate(NORMAL_LOGDENSITY;
            have = (:x, :location, :scale),
            want = :logpdf, batched = (:x, :location, :scale))
        @test likelihood(observations, effects, scales) ≈ sum(
            logpdf(Normal(effect, observation_scale), observation)
            for (observation, effect, observation_scale) in
                zip(observations, effects, scales))

        effects_prior = plate(NORMAL_LOGDENSITY;
            have = (:x, :location, :log_scale),
            want = :logpdf, batched = :x)
        @test effects_prior(effects, location, log_scale) ≈ sum(
            logpdf(Normal(location, scale), effect) for effect in effects)

        cauchy_logdensity = CAUCHY_SCALE_KERNEL(scale, 0.0, 5.0)
        @test log(2) + cauchy_logdensity ≈
              logpdf(truncated(Cauchy(0.0, 5.0), 0.0, Inf), scale)

        @test EMBEDDED_DISTRIBUTION_KERNEL(
            x, location, scale) ≈
            prepare(normal.logpdf)(location, scale, x) +
            CAUCHY_SCALE_KERNEL(x, location, scale)
        @test !occursin("Distributions",
                       string(code_expr(EMBEDDED_DISTRIBUTION_KERNEL)))
    end

    @testset "sources build native recipes checked against a Distributions oracle" begin
        @test length(artifacts) == 11
        continuous, discrete, vectorized = all_sources()
        # No forced API demonstrations: these are plain native densities. Nothing
        # shoehorns a `compose` call; the vectorized source generates its batched
        # kernel with `plate`.
        @test all(source -> !occursin("compose(", source), all_sources())
        @test occursin("plate(", vectorized)
        # The first example defines the shared family objects once and then
        # prepares the natural Normal logpdf endpoint.
        @test !occursin("@recipe", continuous)
        @test !occursin("cost =", continuous)
        @test !occursin("negative_log_scale", continuous)
        @test occursin("@kernel standard_normal()", continuous)
        @test occursin("@kernel location_scale(standard", continuous)
        @test occursin("@kernel normal = location_scale(standard_normal)", continuous)
        @test occursin("prepare(normal.logpdf)", continuous)
        @test occursin("plate(normal.logpdf", vectorized)
        @test !occursin("@kernel normal_logdensity", vectorized)
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
            active_index = artifact.name in (
                :discrete_bernoulli_logit, :geometric_logit,
            ) ? 2 : findfirst(==(:x), propertynames(artifact.inputs))
            active_name = inputs(artifact.kernel)[active_index].name
            prepared = prepare_ad(
                artifact.kernel, DISTRIBUTION_ENZYME_BACKEND, values...;
                active = active_name,
            )
            gradient = ad_gradient(prepared, values...)
            gradients[artifact.name] = gradient
            components = gradient isa Number ? (gradient,) : gradient
            @test all(isfinite, components)
        end

        normal = artifacts[1]
        μ, σ, x = Tuple(normal.inputs)
        @test gradients[:continuous_normal] ≈ -(x - μ) / σ^2

        bernoulli = artifacts[2]
        observed, logit = Tuple(bernoulli.inputs)
        @test gradients[:discrete_bernoulli_logit] ≈
              Int(observed) - 1 / (1 + exp(-logit))

        vectorized = artifacts[3]
        xbatch, μbatch, σbatch = Tuple(vectorized.inputs)
        @test gradients[:vectorized_normal] ≈
              @. -(xbatch - μbatch) / σbatch^2
    end
end
