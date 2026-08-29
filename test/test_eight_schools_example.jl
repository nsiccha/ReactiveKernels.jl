using ReactiveKernels
using Test

include(joinpath(@__DIR__, "..", "examples", "eight_schools.jl"))
using .EightSchoolsExample

_eight_schools_likelihood_call(kernel, observations, effects, scales) =
    kernel(observations, effects, scales)

_eight_schools_likelihood_allocated(kernel, observations, effects, scales) =
    @allocated _eight_schools_likelihood_call(
        kernel, observations, effects, scales,
    )

_eight_schools_reference_normal(x, location, scale) =
    -0.5 * log(2π) - log(scale) - 0.5 * ((x - location) / scale)^2

_eight_schools_reference_cauchy(x, location, scale) =
    -log(π) - log(scale) - log1p(((x - location) / scale)^2)

function _build_and_constrain_eight_schools(q)
    model = build_eight_schools_graph()
    prepare(model; have = :unconstrained, want = :parameters)(q)
end

@testset "manual PPL graph — eight schools" begin
    artifact = evaluate_eight_schools_source()
    @test artifact.source == strip(EIGHT_SCHOOLS_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(
        artifact.kernel, Tuple(artifact.inputs)...,
    )
    model = artifact.model
    q = [1.5, log(2.0), (0.25 .* (1:8))...]

    @test occursin("NORMAL_LOGDENSITY", EIGHT_SCHOOLS_SOURCE)
    @test occursin("CAUCHY_LOGDENSITY", EIGHT_SCHOOLS_SOURCE)
    @test !occursin("@kernel school_", EIGHT_SCHOOLS_SOURCE)
    @test artifact.normal_spec === EightSchoolsExample.NORMAL_LOGDENSITY
    @test artifact.cauchy_spec === EightSchoolsExample.CAUCHY_LOGDENSITY
    @test artifact.normal_factor.plan.graph === artifact.normal_spec.graph
    @test artifact.cauchy_factor.plan.graph === artifact.cauchy_spec.graph
    @test artifact.effects_prior_kernel.plan.graph === artifact.normal_spec.graph
    @test artifact.pointwise_kernel.plan.graph === artifact.normal_spec.graph
    @test artifact.likelihood_kernel.plan.graph === artifact.normal_spec.graph

    @testset "one named model graph, with only the intentional transform alternative" begin
        # Every model QOI has exactly one producer; constrained parameters alone
        # intentionally have the two documented constrain-only vs
        # constrain-plus-Jacobian producers.
        producer_count(value) = count(model.graph.recipes) do recipe
            any(output -> canon_id(model.graph, output.id) ==
                          canon_id(model.graph, value.id), recipe.outputs)
        end
        @test producer_count(model.parameters) == 2
        for value in (model.log_jacobian, model.prior, model.likelihood,
                      model.pointwise, model.posterior, model.new_group)
            @test producer_count(value) == 1
        end
    end

    # The public constructor is callable from ordinary compiled code: it clones
    # a module-load template instead of defining fresh recipe methods via
    # Core.eval inside the caller's world.
    @test _build_and_constrain_eight_schools(q).θ == q[3:end]

    @testset "PPL accumulators are one static graph-output extraction" begin
        @test artifact.requested_nodes ==
              (:parameters, :prior, :likelihood)
        @test Tuple(value.name for value in outputs(artifact.kernel)) ==
              artifact.requested_nodes

        parameters, prior, likelihood = artifact.output
        @test parameters.μ == q[1]
        @test parameters.τ ≈ exp(q[2])
        @test parameters.θ == q[3:end]
        @test prior isa Float64
        @test likelihood isa Float64

        # The Wren-style Params/LogPrior/LogLikelihood selection uses the direct
        # reducing plate. Unrequested pointwise/Jacobian/posterior nodes are absent
        # from the compiled plan rather than skipped by runtime branches.
        produced = Set(
            canon_id(model.graph, output.id)
            for recipe in artifact.kernel.plan.recipes for output in recipe.outputs
        )
        @test canon_id(model.graph, model.likelihood.id) in produced
        @test !(canon_id(model.graph, model.pointwise.id) in produced)
        @test !(canon_id(model.graph, model.log_jacobian.id) in produced)
        @test !(canon_id(model.graph, model.posterior.id) in produced)
    end

    @testset "unconstrained -> constrained; Jacobian is optional" begin
        p = plan(model.graph;
                 have = (model.unconstrained,),
                 want = (model.parameters,))
        # split (μ, log_τ, θ) + τ + the parameters-only producer.
        @test length(p.recipes) == 5
        # The Jacobian-bearing producer and every posterior recipe are pruned.
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.log_jacobian.id) in produced)
        @test !(canon_id(model.graph, model.prior.id) in produced)

        parameters = prepare(p)(q)
        @test parameters isa NamedTuple
        @test parameters.μ == 1.5
        @test parameters.τ ≈ 2.0
        @test parameters.θ == [0.25 * i for i in 1:8]

        with_jacobian = prepare(model.graph;
            have = (model.unconstrained,),
            want = (model.parameters, model.log_jacobian))
        parameters2, log_jacobian = with_jacobian(q)
        @test parameters2 == parameters
        @test log_jacobian == q[2]
    end

    @testset "posterior decomposition; likelihood is the summed pointwise density" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.observations,
                         model.observation_scales),
                 want = (model.prior, model.log_jacobian,
                         model.likelihood, model.posterior))

        k = prepare(p)
        prior, log_jacobian, likelihood, posterior =
            k(q, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA)

        θ = q[3:end]
        μ, τ = q[1], exp(q[2])
        reference_prior =
            _eight_schools_reference_normal(μ, 0.0, 5.0) +
            log(2.0) + _eight_schools_reference_cauchy(τ, 0.0, 5.0) +
            sum(_eight_schools_reference_normal(value, μ, τ) for value in θ)
        reference_likelihood = sum(
            _eight_schools_reference_normal(
                EIGHT_SCHOOLS_Y[j], θ[j], EIGHT_SCHOOLS_SIGMA[j])
            for j in 1:8)
        @test prior ≈ reference_prior
        @test likelihood ≈ reference_likelihood
        @test posterior ≈ prior + log_jacobian + likelihood

        posterior_only = plan(model.graph;
            have = (model.unconstrained, model.observations,
                    model.observation_scales),
            want = (model.posterior,))
        @test length(posterior_only.recipes) > 1
        @test !any(r -> r.op === sum, posterior_only.recipes)
        posterior_kernel = prepare(posterior_only)
        @test count(line -> occursin("for ", line),
                    split(string(code_expr(posterior_kernel)), '\n')) == 2
        @test !occursin("similar", string(code_expr(posterior_kernel)))
        @test !any(op -> op isa ReactiveKernels.PreparedKernel,
                   posterior_kernel.ops)
    end

    @testset "corrected core contracts hold on PPL paths" begin
        @testset "authoritative constrained HAVE is a hard cut" begin
            p = plan(model.graph;
                     have = (model.unconstrained, model.μ),
                     want = (model.parameters,))
            @test inputs(p) == (model.unconstrained, model.μ)

            parameters = prepare(p)(q, 99.0)
            @test parameters.μ == 99.0
            @test parameters.τ ≈ 2.0
            @test parameters.θ == [0.25 * i for i in 1:8]

            state = ReactiveState(model.graph)
            set!(state, model.unconstrained, q)
            set!(state, model.μ, 99.0)
            @test get!(state, model.parameters) == parameters
        end

        @testset "overlapping transform producers admit a valid order" begin
            graph = Graph()
            raw = value!(graph, :ppl_unconstrained, Vector{Float64})
            location = value!(graph, :location, Float64)
            log_scale = value!(graph, :log_scale, Float64)
            bridge = value!(graph, :location_bridge, Float64)
            auxiliary = value!(graph, :auxiliary, Float64)

            overlapping = add!(graph, bridge => (location, auxiliary),
                               x -> (x - 10, 2x))
            source = add!(graph, raw => (location, log_scale),
                          x -> (x[1], x[2]))
            bridged = add!(graph, location => bridge, x -> x + 10)

            p = plan(graph; have = (raw,), want = (auxiliary, log_scale))
            @test [r.id for r in p.recipes] ==
                  [source.id, bridged.id, overlapping.id]
            auxiliary_value, log_scale_value =
                prepare(p)([3.0, log(2.0)])
            @test auxiliary_value == 26.0
            @test log_scale_value ≈ log(2.0)
        end

        @testset "nested density provenance remains reusable" begin
            nested = build_eight_schools_graph()
            checked_posterior = value!(nested.graph, :checked_log_posterior, Float64)
            posterior_calls = Ref(0)
            add!(nested.graph,
                 (nested.prior, nested.log_jacobian, nested.likelihood) =>
                     checked_posterior,
                 (prior, jacobian, likelihood) -> begin
                     posterior_calls[] += 1
                     prior + jacobian + likelihood
                 end)

            state = ReactiveState(
                nested.graph;
                materialize = (nested.parameters, nested.prior,
                               nested.likelihood, checked_posterior),
            )
            set!(state, nested.unconstrained, q)
            set!(state, nested.observations, EIGHT_SCHOOLS_Y)
            set!(state, nested.observation_scales, EIGHT_SCHOOLS_SIGMA)

            get!(state, nested.parameters)
            get!(state, nested.prior)
            get!(state, nested.likelihood)
            first_posterior = get!(state, checked_posterior)
            @test get!(state, checked_posterior) == first_posterior
            @test posterior_calls[] == 1
        end

        @testset "checkpoint rejects stale transformed parameters" begin
            checkpointed = build_eight_schools_graph()
            state = ReactiveState(checkpointed.graph;
                                  materialize = (checkpointed.parameters,))
            set!(state, checkpointed.unconstrained, q)
            get!(state, checkpointed.parameters)

            q2 = [7.0; q[2:end]]
            set!(state, checkpointed.unconstrained, q2)
            @test_throws ErrorException checkpoint(
                state, (checkpointed.parameters,))

            refreshed = get!(state, checkpointed.parameters)
            saved = checkpoint(state, (checkpointed.parameters,))
            @test refreshed.μ == 7.0
            @test saved[canon_id(checkpointed.graph,
                                 checkpointed.parameters.id)] == refreshed
        end

        @testset "generated bindings are hygienic for PPL names" begin
            graph = Graph()
            reserved = value!(graph, :__ops__, Vector{Float64})
            location = value!(graph, :location, Float64)
            log_scale = value!(graph, :log_scale, Float64)
            effects = value!(graph, :effects, Vector{Float64})
            add!(graph, reserved => (location, log_scale, effects),
                 x -> (x[1], x[2], x[3:end]))

            k = prepare(graph; have = (reserved,),
                        want = (location, log_scale, effects))
            @test k(q) == (q[1], q[2], q[3:end])
        end

        @testset "recipe costs preserve planner assumptions" begin
            diagnostic = value!(model.graph, :density_diagnostic, Float64)
            recipe_count = length(model.graph.recipes)
            for invalid_cost in (-1.0, Inf, NaN)
                @test_throws ArgumentError add!(
                    model.graph, model.posterior => diagnostic, identity;
                    cost = invalid_cost)
            end
            @test length(model.graph.recipes) == recipe_count
        end
    end

    @testset "generated quantities prune density work" begin
        parameters = (; μ = 1.0, τ = 4.0, θ = fill(2.0, 8))
        p = plan(model.graph;
                 have = (model.parameters, model.new_group_scale,
                         model.prediction_innovations),
                 want = (model.new_group,))

        # θ_new + y_new + the new_group NamedTuple; the split, transform, prior,
        # likelihood, and posterior recipes are all pruned.
        @test length(p.recipes) == 3
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.prior.id) in produced)
        @test !(canon_id(model.graph, model.posterior.id) in produced)

        prediction = prepare(p)(parameters, 12.0, [0.25, -1.0])
        @test prediction isa NamedTuple
        @test prediction.θ == 2.0
        @test prediction.y == -10.0
    end

    @testset "plate exposes pointwise values and a buffer-free total" begin
        θ = q[3:end]
        expected_pointwise = [
            _eight_schools_reference_normal(
                EIGHT_SCHOOLS_Y[j], θ[j], EIGHT_SCHOOLS_SIGMA[j])
            for j in 1:8
        ]

        @test artifact.pointwise ≈ expected_pointwise
        @test artifact.pointwise_kernel.plan.graph ===
              artifact.likelihood_kernel.plan.graph
        likelihood = _eight_schools_likelihood_call(
            artifact.likelihood_kernel,
            EIGHT_SCHOOLS_Y, θ, EIGHT_SCHOOLS_SIGMA,
        )
        @test likelihood ≈ sum(expected_pointwise)

        # Collect mode allocates exactly the requested pointwise result. The
        # reducing plate instead emits a scalar accumulator loop, with no
        # `similar` output buffer, and executes allocation-free after warmup.
        @test occursin("similar", string(code_expr(artifact.pointwise_kernel)))
        @test !occursin("similar", string(code_expr(artifact.likelihood_kernel)))
        _eight_schools_likelihood_call(
            artifact.likelihood_kernel,
            EIGHT_SCHOOLS_Y, θ, EIGHT_SCHOOLS_SIGMA,
        )
        @test _eight_schools_likelihood_allocated(
            artifact.likelihood_kernel,
            EIGHT_SCHOOLS_Y, θ, EIGHT_SCHOOLS_SIGMA,
        ) == 0
    end

end
