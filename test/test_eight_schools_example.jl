using ReactiveKernels
using Test

include(joinpath(@__DIR__, "..", "examples", "eight_schools.jl"))
using .EightSchoolsExample

@testset "manual PPL graph — eight schools" begin
    model = build_eight_schools_graph()
    q = [1.5, log(2.0), (0.25 .* (1:8))...]

    @testset "unconstrained -> constrained; Jacobian is optional" begin
        p = plan(model.graph;
                 have = (model.unconstrained,),
                 want = (model.parameters,))
        @test length(p.recipes) == 3
        @test !any(r -> r.op === EightSchoolsExample.log_abs_det_jacobian,
                   p.recipes)
        @test !any(r -> r.op === EightSchoolsExample.log_prior, p.recipes)

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

    @testset "density decomposition; likelihood is a plated (vectorized) kernel" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.observations,
                         model.observation_scales),
                 want = (model.prior, model.log_jacobian,
                         model.likelihood, model.density))

        k = prepare(p)
        prior, log_jacobian, likelihood, density =
            k(q, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA)

        θ = q[3:end]
        reference_likelihood = sum(
            EightSchoolsExample.normal_logpdf(
                EIGHT_SCHOOLS_Y[j], θ[j], EIGHT_SCHOOLS_SIGMA[j])
            for j in 1:8)
        @test likelihood ≈ reference_likelihood
        @test density ≈ prior + log_jacobian + likelihood
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
            checked_density = value!(nested.graph, :checked_log_density, Float64)
            density_calls = Ref(0)
            add!(nested.graph,
                 (nested.prior, nested.log_jacobian, nested.likelihood) =>
                     checked_density,
                 (prior, jacobian, likelihood) -> begin
                     density_calls[] += 1
                     prior + jacobian + likelihood
                 end)

            state = ReactiveState(
                nested.graph;
                materialize = (nested.parameters, nested.prior,
                               nested.likelihood, checked_density),
            )
            set!(state, nested.unconstrained, q)
            set!(state, nested.observations, EIGHT_SCHOOLS_Y)
            set!(state, nested.observation_scales, EIGHT_SCHOOLS_SIGMA)

            get!(state, nested.parameters)
            get!(state, nested.prior)
            get!(state, nested.likelihood)
            first_density = get!(state, checked_density)
            @test get!(state, checked_density) == first_density
            @test density_calls[] == 1
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
                 EightSchoolsExample.split_unconstrained)

            k = prepare(graph; have = (reserved,),
                        want = (location, log_scale, effects))
            @test k(q) == (q[1], q[2], q[3:end])
        end

        @testset "recipe costs preserve planner assumptions" begin
            diagnostic = value!(model.graph, :density_diagnostic, Float64)
            recipe_count = length(model.graph.recipes)
            for invalid_cost in (-1.0, Inf, NaN)
                @test_throws ArgumentError add!(
                    model.graph, model.density => diagnostic, identity;
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

        @test length(p.recipes) == 1
        @test !any(r -> r.op === EightSchoolsExample.split_unconstrained,
                   p.recipes)
        @test !any(r -> r.op === EightSchoolsExample.log_prior, p.recipes)

        prediction = prepare(p)(parameters, 12.0, [0.25, -1.0])
        @test prediction isa NamedTuple
        @test prediction.θ == 2.0
        @test prediction.y == -10.0
    end

    @testset "the plated likelihood equals the summed per-school density" begin
        # `plated_loglik` is the vectorized `plate` of the scalar per-school kernel.
        θ = q[3:end]
        expected = sum(
            EightSchoolsExample.normal_logpdf(
                EIGHT_SCHOOLS_Y[j], θ[j], EIGHT_SCHOOLS_SIGMA[j])
            for j in 1:8)
        @test EightSchoolsExample.plated_loglik(
            EIGHT_SCHOOLS_Y, θ, EIGHT_SCHOOLS_SIGMA) ≈ expected
    end

    @testset "invalid constrained inputs fail explicitly" begin
        @test_throws DomainError EightSchoolsExample.normal_logpdf(0.0, 0.0, 0.0)
        @test_throws DomainError EightSchoolsExample.predict_new_group(
            (; μ = 1.0, τ = 4.0, θ = fill(2.0, 8)), 0.0, [0.25, -1.0])
        @test_throws DomainError EightSchoolsExample.log_prior(
            (; μ = 1.0, τ = 0.0, θ = fill(2.0, 8)))
    end
end
