using ReactiveKernelsPPLExamples.SumToZeroExample
using ReactiveKernelsPPLExamples.EightSchoolsExample:
    EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy

_sum_to_zero_reference_normal(x, location, scale) =
    -0.5 * log(2π) - log(scale) - 0.5 * ((x - location) / scale)^2

_sum_to_zero_reference_cauchy(x, location, scale) =
    -log(π) - log(scale) - log1p(((x - location) / scale)^2)

@testset "sum-to-zero PPL graph" begin
    artifact = evaluate_sum_to_zero_source()
    @test artifact.source == strip(SUM_TO_ZERO_SOURCE, '\n')
    @test artifact.normal_object === normal
    @test artifact.cauchy_object === cauchy
    @test length(findall("@kernel model(", SUM_TO_ZERO_SOURCE)) == 1
    @test !occursin("@kernel sum_to_zero", SUM_TO_ZERO_SOURCE)
    @test !occursin("prepare(", first(split(
        SUM_TO_ZERO_SOURCE, "\n\nunconstrained ="; limit = 2)))

    model = artifact.model
    K = length(EIGHT_SCHOOLS_Y)
    α_s2z = 0.5
    log_τ = log(2.0)
    effects_free = 0.25 .* collect(1:(K - 1))
    q = [α_s2z, log_τ, effects_free...]

    @testset "orthonormal constrain and distinct density adjustments" begin
        kernel = prepare(model;
            have = :unconstrained,
            want = (:parameters, :sum_to_zero_log_jacobian,
                    :log_jacobian))
        parameters, sum_to_zero_log_jacobian, log_jacobian = kernel(q)

        @test parameters.α_s2z == α_s2z
        @test parameters.τ ≈ exp(log_τ)
        @test length(parameters.effects_s2z) == K
        @test sum(parameters.effects_s2z) ≈ 0.0 atol = 8eps(Float64)
        @test sum(abs2, parameters.effects_s2z) ≈ sum(abs2, effects_free)
        @test sum_to_zero_log_jacobian == 0.0
        @test log_jacobian == log_τ

        adjustments = prepare(model;
            have = (:parameters,),
            want = (:sum_to_zero_log_jacobian,
                    :subspace_normalization))
        transform_adjustment, prior_adjustment = adjustments(parameters)
        @test transform_adjustment == 0.0
        @test prior_adjustment ≈ log(parameters.τ)
    end

    @testset "induced prior and posterior decomposition" begin
        kernel = prepare(model;
            have = (:unconstrained, :observations,
                    :observation_scales, :α_prior_sd),
            want = (:parameters, :prior, :likelihood,
                    :log_jacobian, :posterior))
        parameters, prior, likelihood, log_jacobian, posterior = kernel(
            q, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA, 5.0,
        )

        τ = parameters.τ
        effects = parameters.effects_s2z
        α_scale = sqrt(5.0^2 + τ^2 / K)
        reference_prior =
            _sum_to_zero_reference_normal(α_s2z, 0.0, α_scale) +
            log(2.0) + _sum_to_zero_reference_cauchy(τ, 0.0, 5.0) +
            sum(_sum_to_zero_reference_normal(effect, 0.0, τ)
                for effect in effects) +
            log(τ)
        reference_likelihood = sum(
            _sum_to_zero_reference_normal(
                EIGHT_SCHOOLS_Y[j],
                α_s2z + effects[j],
                EIGHT_SCHOOLS_SIGMA[j],
            ) for j in 1:K
        )

        @test prior ≈ reference_prior
        @test likelihood ≈ reference_likelihood
        @test posterior ≈ prior + likelihood + log_jacobian

        # Native code keeps the pivot transform linear and reduces each plate
        # directly into its scalar total. There are no pointwise buffers or
        # nested prepared-kernel calls left in the generated posterior.
        @test @inferred(kernel(
            q, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA, 5.0,
        )) == (parameters, prior, likelihood, log_jacobian, posterior)
        posterior_kernel = prepare(model;
            have = (:unconstrained, :observations,
                    :observation_scales, :α_prior_sd),
            want = :posterior)
        posterior_code = string(code_expr(posterior_kernel))
        @test count(line -> occursin("for ", line),
                    split(posterior_code, '\n')) == 2
        @test !occursin("similar", posterior_code)
        @test !any(op -> op isa ReactiveKernels.PreparedKernel,
                   posterior_kernel.ops)
    end

    @testset "stochastic common-shift recovery is an independent graph cut" begin
        parameters = prepare(model;
            have = :unconstrained,
            want = :parameters)(q)
        innovation = -0.25
        recovery_plan = plan(model;
            have = (:parameters, :α_prior_sd,
                    :reconstruction_innovation),
            want = :superpopulation)
        recovered = prepare(recovery_plan)(parameters, 5.0, innovation)

        mean_variance = parameters.τ^2 / K
        intercept_variance = 5.0^2
        weight = mean_variance / (intercept_variance + mean_variance)
        conditional_sd = sqrt(
            intercept_variance * mean_variance /
            (intercept_variance + mean_variance),
        )
        expected_mean = weight * parameters.α_s2z +
                        conditional_sd * innovation

        @test recovered.realized_effect_mean ≈ expected_mean
        @test recovered.α_bayes ≈ parameters.α_s2z - expected_mean
        @test recovered.effects_bayes ≈
              parameters.effects_s2z .+ expected_mean
        @test sum(recovered.effects_bayes) / K ≈ expected_mean
        @test recovered.α_bayes .+ recovered.effects_bayes ≈
              parameters.α_s2z .+ parameters.effects_s2z

        recovery_kernel = prepare(recovery_plan)
        @test @inferred(recovery_kernel(
            parameters, 5.0, innovation,
        )) == recovered
        recovery_code = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(
                code_expr(recovery_kernel), recovery_kernel,
            );
            context = :limit => false,
        )
        for pruned_source in ("observations", "normal(", "cauchy(",
                              "log_jacobian", "effects_free")
            @test !occursin(pruned_source, recovery_code)
        end
        @test !occursin("for ", recovery_code)
        @test count("effects_s2z .+ mean_effect_bayes", recovery_code) == 1

        produced = Set(
            canon_id(model.graph, output.id)
            for recipe in recovery_plan.recipes for output in recipe.outputs
        )
        for pruned in (model.effects_free, model.prior, model.likelihood,
                       model.log_jacobian, model.posterior)
            @test !(canon_id(model.graph, pruned.id) in produced)
        end
    end
end
