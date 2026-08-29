using DifferentiationInterface
import Enzyme

const TEST_AD_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

@testset "Prepared AD kernels" begin
    @testset "authored boundary, defaults, keywords, and fresh Constants" begin
        @kernel objective(q::Vector{Float64}, scale::Float64 = 1.25;
                          data::Vector{Float64}, offset::Float64 = 0.0) = begin
            density::Float64 =
                sum(q .* data) - scale * sum(abs2, q) + offset
        end

        data = [2.0, -1.0, 0.5]
        parameters = [0.3, -0.4, 0.2]
        expected = data .- 2(1.25) .* parameters

        @test ad_gradient(
            objective, TEST_AD_BACKEND, parameters;
            data, active = :q, want = :density,
        ) ≈ expected

        prepared = prepare_ad(
            objective, TEST_AD_BACKEND, parameters;
            data, active = objective.q, want = objective.density,
        )
        @test prepared isa PreparedADKernel
        @test inputs(prepared) == inputs(objective)
        @test outputs(prepared) == (objective.density,)
        @test code_expr(prepared) == code_expr(plan(objective; want = :density))
        @test occursin("active=:q", sprint(show, prepared))
        @test occursin("want=:density", sprint(show, prepared))

        # Preparation values are exemplars. Every call reconstructs Constants
        # from the current data/default/keyword values instead of retaining the
        # originals in the DI closure or preparation.
        for (newdata, newparameters, newscale, newoffset) in (
                ([1.0, 3.0, -2.0], [0.1, 0.2, -0.3], 0.75, 4.0),
                ([-0.5, 0.25, 4.0], [-0.2, 0.6, 0.4], 2.0, -3.0))
            @test ad_gradient(
                prepared, newparameters, newscale;
                data = newdata, offset = newoffset,
            ) ≈ newdata .- 2newscale .* newparameters
        end

        @test_throws UndefKeywordError prepare_ad(
            objective, TEST_AD_BACKEND, parameters;
            data, active = :q,
        )
        @test_throws ArgumentError prepare_ad(
            objective, TEST_AD_BACKEND, parameters;
            data, active = :missing, want = :density,
        )
    end

    @testset "low-level positional boundary" begin
        graph = Graph()
        observed = value!(graph, :observed, Bool)
        logit = value!(graph, :logit, Float64)
        density = value!(graph, :density, Float64)
        add!(graph, (observed, logit) => density,
             (y, z) -> (y ? z : zero(z)) - log1p(exp(z)))
        kernel = prepare(graph; have = (observed, logit), want = density)

        gradient = ad_gradient(
            kernel, TEST_AD_BACKEND, true, 0.4; active = :logit)
        @test gradient ≈ 1 - inv(1 + exp(-0.4))

        prepared = prepare_ad(
            kernel, TEST_AD_BACKEND, true, 0.4; active = :logit)
        @test ad_gradient(prepared, false, -0.7) ≈
              -inv(1 + exp(0.7))
        @test_throws ArgumentError ad_gradient(
            kernel, TEST_AD_BACKEND, true, 0.4;
            active = :logit, unsupported = 1,
        )
    end

    @testset "active dependency boundary" begin
        function boundary_kernel(kind)
            graph = Graph()
            q = value!(graph, :unconstrained, Vector{Float64})
            middle = value!(graph, :middle, Vector{Float64})
            buffer = value!(graph, :derived_buffer, Vector{Float64})
            density = value!(graph, :density, Float64)
            add!(graph, q => middle, copy)
            if kind === :direct
                add!(graph, q => buffer, copy)
            else
                add!(graph, middle => buffer, copy)
            end
            add!(graph, buffer => density, sum)
            prepare(graph; have = (q, buffer), want = density)
        end

        for kind in (:direct, :transitive)
            error = try
                prepare_ad(
                    boundary_kernel(kind), TEST_AD_BACKEND,
                    [1.0, 2.0], [1.0, 2.0]; active = :unconstrained,
                )
                nothing
            catch exception
                exception
            end
            @test error isa ArgumentError
            @test occursin(":derived_buffer", sprint(showerror, error))
            @test occursin("transitively downstream", sprint(showerror, error))
            @test occursin("Constant", sprint(showerror, error))
        end

        # A selected active port may itself be graph-derived from an inactive
        # root: the inactive root is upstream, so treating its current boundary
        # value as Constant severs no active dependency.
        graph = Graph()
        root = value!(graph, :root, Float64)
        active = value!(graph, :active_derived, Float64)
        density = value!(graph, :density, Float64)
        add!(graph, root => active, exp)
        add!(graph, (root, active) => density, (r, q) -> r + q^2)
        kernel = prepare(graph; have = (root, active), want = density)
        @test ad_gradient(
            kernel, TEST_AD_BACKEND, 3.0, 2.0;
            active = :active_derived,
        ) ≈ 4.0
    end

    @testset "invalid and aliased boundaries fail loudly" begin
        graph = Graph()
        x = value!(graph, :x, Float64)
        y = value!(graph, :y, Float64)
        z = value!(graph, :z, Float64)
        add!(graph, x => (y, z), t -> (t, t^2))
        multi = prepare(graph; have = x, want = (y, z))
        @test_throws ArgumentError prepare_ad(
            multi, TEST_AD_BACKEND, 1.0; active = :x)

        int_graph = Graph()
        count = value!(int_graph, :count, Int)
        integer_objective = value!(int_graph, :objective, Float64)
        add!(int_graph, count => integer_objective, Float64)
        @test_throws ArgumentError prepare_ad(
            prepare(int_graph; have = count, want = integer_objective),
            TEST_AD_BACKEND, 2; active = :count,
        )

        alias_graph = Graph()
        q = value!(alias_graph, :q, Float64)
        q_alias = value!(alias_graph, :q_alias, Float64)
        density = value!(alias_graph, :density, Float64)
        alias_graph.aliases[q_alias.id] = q.id
        add!(alias_graph, q => density, abs2)
        alias_spec = KernelSpec(
            alias_graph,
            Dict{Symbol,Value}(
                :q => q, :q_alias => q_alias, :density => density,
            ),
            [:q, :q_alias, :density], [:q, :q_alias], [:density],
        )
        alias_error = try
            prepare_ad(
                alias_spec, TEST_AD_BACKEND, 1.0, 1.0;
                active = :q, want = :density,
            )
            nothing
        catch exception
            exception
        end
        @test alias_error isa ArgumentError
        @test occursin(":q aliases :q_alias", sprint(showerror, alias_error))

        @kernel multiple_wants(q::Float64) = begin
            first::Float64 = q^2
            second::Float64 = q^3
        end
        @test ad_gradient(
            multiple_wants, TEST_AD_BACKEND, 2.0;
            active = :q, want = :second,
        ) ≈ 12.0
        @test_throws ArgumentError prepare_ad(
            multiple_wants, TEST_AD_BACKEND, 2.0;
            active = :q, want = (:first, :second),
        )
    end
end
