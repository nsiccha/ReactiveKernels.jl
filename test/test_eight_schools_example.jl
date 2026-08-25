using ReactiveKernels
using DifferentiationInterface
import Enzyme
using Test

include(joinpath(@__DIR__, "..", "examples", "eight_schools.jl"))
using .EightSchoolsExample

# Reverse-mode Enzyme through DifferentiationInterface. Runtime activity is
# enabled because the prepared density kernel closes over constant model data
# (the observations and their scales), which Enzyme's static activity analysis
# cannot prove non-differentiable; the closure (which captures the prepared
# kernel) is annotated `Const` because only the numeric input is differentiated,
# never the kernel itself.
const ENZYME_BACKEND = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const,
)

@testset "manual PPL graph — eight schools" begin
    model = build_eight_schools_graph()
    q = (1.5, log(2.0), ntuple(i -> 0.25 * i, 8)...)

    @testset "unconstrained -> constrained; Jacobian is optional" begin
        p = plan(model.graph;
                 have = (model.unconstrained,),
                 want = (model.parameters,))
        @test length(p.recipes) == 3
        @test !any(r -> r.op === EightSchoolsExample.log_abs_det_jacobian,
                   p.recipes)
        @test !any(r -> r.op === EightSchoolsExample.log_prior, p.recipes)

        parameters = prepare(p)(q)
        @test parameters isa EightSchoolsParameters
        @test parameters.μ == 1.5
        @test parameters.τ ≈ 2.0
        @test parameters.θ == ntuple(i -> 0.25 * i, 8)

        with_jacobian = prepare(model.graph;
            have = (model.unconstrained,),
            want = (model.parameters, model.log_jacobian))
        parameters2, log_jacobian = with_jacobian(q)
        @test parameters2 == parameters
        @test log_jacobian == q[2]
    end

    @testset "density decomposition and shared pointwise likelihood" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.observations,
                         model.observation_scales),
                 want = (model.prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.density))
        @test count(r -> r.op === EightSchoolsExample.pointwise_log_likelihood,
                    p.recipes) == 1

        k = prepare(p)
        prior, log_jacobian, pointwise, likelihood, density =
            k(q, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA)

        @test all(isfinite, pointwise)
        @test likelihood ≈ sum(pointwise)
        @test density ≈ prior + log_jacobian + likelihood
    end

    @testset "the log-density boundary accepts AD numbers" begin
        k = prepare(model.graph;
                    have = (model.unconstrained, model.observations,
                            model.observation_scales),
                    want = (model.density,))
        logdensity(qv) = k(Tuple(qv), EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA)

        qvec = collect(q)
        gradient = DifferentiationInterface.gradient(
            logdensity, ENZYME_BACKEND, qvec)
        @test length(gradient) == length(q)
        @test all(isfinite, gradient)

        # Enzyme derivative-aliasing caveat: DifferentiationInterface's
        # out-of-place gradient must return freshly allocated memory that does
        # not alias the mutable primal input.
        @test gradient !== qvec
        @test pointer(gradient) != pointer(qvec)

        # Correctness is validated with no second differentiation backend. The
        # μ-component has a closed form for this fixed q: the Normal(0, 5) prior
        # contributes -μ/25 = -0.06, and the eight Normal(μ, τ) group terms
        # contribute Σⱼ(θⱼ - μ)/τ² = (9 - 12)/4 = -0.75 (τ = exp(log 2) = 2), so
        # ∂density/∂μ = -0.81 exactly.
        @test gradient[1] ≈ -0.81
        # The full gradient is pinned to independently established constants;
        # these anchor a pure DI+Enzyme regression that needs no second
        # differentiation backend as a dependency.
        expected_gradient = [
            -0.81, -6.338362068965517, 0.4358333333333333, 0.325,
            0.1728515625, 0.17458677685950413, 0.034722222222222224,
            -0.004132231404958678, 0.1, -0.0941358024691358,
        ]
        @test gradient ≈ expected_gradient
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
            @test parameters.θ == ntuple(i -> 0.25 * i, 8)

            state = ReactiveState(model.graph)
            set!(state, model.unconstrained, q)
            set!(state, model.μ, 99.0)
            @test get!(state, model.parameters) == parameters
        end

        @testset "overlapping transform producers admit a valid order" begin
            graph = Graph()
            raw = value!(graph, :ppl_unconstrained, NTuple{2,Real})
            location = value!(graph, :location, Real)
            log_scale = value!(graph, :log_scale, Real)
            bridge = value!(graph, :location_bridge, Real)
            auxiliary = value!(graph, :auxiliary, Real)

            overlapping = add!(graph, bridge => (location, auxiliary),
                               x -> (x - 10, 2x))
            source = add!(graph, raw => (location, log_scale),
                          x -> (x[1], x[2]))
            bridged = add!(graph, location => bridge, x -> x + 10)

            p = plan(graph; have = (raw,), want = (auxiliary, log_scale))
            @test [r.id for r in p.recipes] ==
                  [source.id, bridged.id, overlapping.id]
            auxiliary_value, log_scale_value =
                prepare(p)((3.0, log(2.0)))
            @test auxiliary_value == 26.0
            @test log_scale_value ≈ log(2.0)
        end

        @testset "nested density provenance remains reusable" begin
            nested = build_eight_schools_graph()
            checked_density = value!(nested.graph, :checked_log_density, Real)
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
                               nested.pointwise, nested.likelihood,
                               checked_density),
            )
            set!(state, nested.unconstrained, q)
            set!(state, nested.observations, EIGHT_SCHOOLS_Y)
            set!(state, nested.observation_scales, EIGHT_SCHOOLS_SIGMA)

            get!(state, nested.parameters)
            get!(state, nested.prior)
            get!(state, nested.pointwise)
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

            q2 = (7.0, q[2:end]...)
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
            reserved = value!(graph, :__ops__,
                              EightSchoolsExample.UnconstrainedParameters)
            location = value!(graph, :location, Real)
            log_scale = value!(graph, :log_scale, Real)
            effects = value!(graph, :effects, EightSchoolsExample.SchoolVector)
            add!(graph, reserved => (location, log_scale, effects),
                 EightSchoolsExample.split_unconstrained)

            k = prepare(graph; have = (reserved,),
                        want = (location, log_scale, effects))
            @test k(q) == (q[1], q[2], q[3:end])
        end

        @testset "recipe costs preserve planner assumptions" begin
            diagnostic = value!(model.graph, :density_diagnostic, Real)
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
        parameters = EightSchoolsParameters(1.0, 4.0, ntuple(_ -> 2.0, 8))
        p = plan(model.graph;
                 have = (model.parameters, model.observations,
                         model.observation_scales, model.new_group_scale,
                         model.prediction_innovations),
                 want = (model.pointwise, model.new_group))

        @test length(p.recipes) == 2
        @test !any(r -> r.op === EightSchoolsExample.split_unconstrained,
                   p.recipes)
        @test !any(r -> r.op === EightSchoolsExample.log_prior, p.recipes)
        @test !any(r -> r.op === EightSchoolsExample.total_log_density,
                   p.recipes)

        pointwise, prediction = prepare(p)(
            parameters, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA,
            12.0, (0.25, -1.0))
        @test all(isfinite, pointwise)
        @test prediction isa NewGroupPrediction
        @test prediction.θ == 2.0
        @test prediction.y == -10.0
    end

    @testset "invalid constrained inputs fail explicitly" begin
        parameters = EightSchoolsParameters(1.0, 4.0, ntuple(_ -> 2.0, 8))
        bad_scales = (0.0, EIGHT_SCHOOLS_SIGMA[2:end]...)

        @test_throws DomainError EightSchoolsExample.pointwise_log_likelihood(
            parameters, EIGHT_SCHOOLS_Y, bad_scales)
        @test_throws DomainError EightSchoolsExample.predict_new_group(
            parameters, 0.0, (0.25, -1.0))
        @test_throws DomainError EightSchoolsExample.log_prior(
            EightSchoolsParameters(1.0, 0.0, ntuple(_ -> 2.0, 8)))
    end
end
