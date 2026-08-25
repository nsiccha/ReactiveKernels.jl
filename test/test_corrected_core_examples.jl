if !isdefined(@__MODULE__, :ReactiveHMCExamples)
    include(joinpath(@__DIR__, "..", "examples", "preexisting_reactivehmc.jl"))
end

using .ReactiveHMCExamples
using LinearAlgebra
using ReactiveKernels

@testset "corrected core through preexisting ecosystem patterns" begin
    @testset "partially supplied Riemannian geometry is authoritative" begin
        riemannian = riemannian_examples().gaussian
        v = riemannian.values
        pos = [0.25, -0.5]
        supplied_metric = Diagonal([3.0, 4.0])

        kernel = prepare(
            riemannian.graph;
            have = (v.pos, v.metric),
            want = (v.metric, v.dpot, v.metric_grad),
        )
        metric, dpot, metric_grad = kernel(pos, supplied_metric)
        @test metric === supplied_metric
        @test dpot == pos
        @test size(metric_grad) == (2, 2, 2)

        state = ReactiveState(riemannian.graph)
        set!(state, v.pos, pos)
        set!(state, v.metric, supplied_metric)
        metric, dpot, metric_grad = get!(state, (v.metric, v.dpot, v.metric_grad))
        @test metric === supplied_metric
        @test dpot == pos
        @test size(metric_grad) == (2, 2, 2)
    end

    @testset "overlapping HMC oracle producers retain a valid schedule" begin
        graph = Graph()
        pos = value!(graph, :pos, Float64)
        pot = value!(graph, :pot, Float64)
        grad = value!(graph, :grad, Float64)
        tape = value!(graph, :tape, Float64)
        hessian = value!(graph, :hessian, Float64)

        add!(graph, tape => (pot, hessian), t -> (t + 1, t + 2))
        add!(graph, pos => (pot, grad), p -> (p + 1, p + 2))
        add!(graph, pot => tape, p -> p + 10)

        kernel = prepare(graph; have = (pos,), want = (hessian, grad))
        @test kernel(1.0) == (14.0, 3.0)
        @test length(kernel.plan.recipes) == 3
    end

    @testset "effectful HMC diagnostics never alias pure oracle results" begin
        # ReactiveHMC-style oracle wrappers often share a stable cache key even
        # when one wrapper exists only to log a diagnostic.  The effect marker
        # is part of recipe identity: the pure result must remain plannable and
        # the diagnostic result must remain rejected, in either insertion order.
        for effectful_first in (false, true)
            graph = Graph()
            pos = value!(graph, :pos, Float64)
            potential = value!(graph, :potential, Float64)
            logged_potential = value!(graph, :logged_potential, Float64)
            pure = () -> add!(
                graph, pos => potential, abs2; cse_key = :potential_oracle,
            )
            diagnostic = () -> add!(
                graph, pos => logged_potential, abs2;
                cse_key = :potential_oracle, effectful = true,
            )

            if effectful_first
                diagnostic()
                pure()
            else
                pure()
                diagnostic()
            end

            @test length(graph.recipes) == 2
            @test prepare(
                graph; have = (pos,), want = (potential,),
            )(3.0) == 9.0
            @test_throws PlanningError plan(
                graph; have = (pos,), want = (logged_potential,),
            )
        end
    end

    @testset "nested reactive provenance remains valid" begin
        graph = Graph()
        x = value!(graph, :x, Float64)
        a = value!(graph, :a, Float64)
        b = value!(graph, :b, Float64)
        c = value!(graph, :c, Float64)
        calls = (a = Ref(0), b = Ref(0), c = Ref(0))

        add!(graph, x => a, x -> (calls.a[] += 1; x + 1))
        add!(graph, a => b, a -> (calls.b[] += 1; 2a))
        add!(graph, (a, b) => c, (a, b) -> (calls.c[] += 1; a + b))

        state = ReactiveState(graph; materialize = (a,))
        set!(state, x, 2.0)
        @test get!(state, a) == 3.0
        materialize!(state, b)
        @test get!(state, b) == 6.0
        materialize!(state, c)
        @test get!(state, c) == 9.0
        @test get!(state, c) == 9.0
        @test map(ref -> ref[], calls) == (a = 1, b = 1, c = 1)
    end

    @testset "checkpoint rejects stale warm-start state" begin
        graph = Graph()
        position = value!(graph, :position, Float64)
        adapted_scale = value!(graph, :adapted_scale, Float64)
        add!(graph, position => adapted_scale, x -> abs(x) + 1)

        state = ReactiveState(graph; materialize = (adapted_scale,))
        set!(state, position, 1.0)
        @test get!(state, adapted_scale) == 2.0
        set!(state, position, 10.0)
        error = try
            checkpoint(state, (adapted_scale,))
            nothing
        catch exception
            exception
        end
        @test error isa ErrorException
        @test occursin("stale", sprint(showerror, error))
        @test get!(state, adapted_scale) == 11.0
        @test only(values(checkpoint(state, (adapted_scale,)))) == 11.0
    end

    @testset "HMC query inputs and lowering names are validated" begin
        graph = Graph()
        user_ops = value!(graph, :__ops__, Float64)
        hamiltonian = value!(graph, :hamiltonian, Float64)
        add!(graph, user_ops => hamiltonian, x -> abs2(x))
        @test prepare(graph; have = (user_ops,), want = (hamiltonian,))(3.0) == 9.0

        riemannian = riemannian_examples().gaussian
        v = riemannian.values
        duplicate_have = prepare(
            riemannian.graph;
            have = (v.pos, v.pos, v.metric),
            want = (v.metric, v.dpot),
        )
        @test inputs(duplicate_have) == (v.pos, v.metric)
        supplied_metric = Diagonal([3.0, 4.0])
        @test duplicate_have([0.25, -0.5], supplied_metric)[1] === supplied_metric

        invalid_graph = Graph()
        pos = value!(invalid_graph, :pos, Float64)
        ham = value!(invalid_graph, :ham, Float64)
        for cost in (-1.0, -Inf, Inf, NaN)
            @test_throws ArgumentError add!(
                invalid_graph, pos => ham, abs2; cost,
            )
        end
        @test isempty(invalid_graph.recipes)
    end
end
