using DifferentiationInterface
import Enzyme

const TEST_AD_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

_test_ad_value_gradient_allocated(prepared, gradient, q, data) =
    @allocated ad_value_and_gradient!(prepared, gradient, q, data)

_test_ad_backend_value_gradient_allocated(prepared, gradient, q, data) =
    @allocated DifferentiationInterface.value_and_gradient!(
        prepared.call, gradient, prepared.preparation, prepared.backend,
        q, DifferentiationInterface.Constant(data))

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

        # The combined prepared boundary reuses the same active ordering and
        # freshly rebound Constant data, while writing into caller-owned
        # gradient storage.
        changed_data = [1.5, -2.0, 0.25]
        changed_parameters = [-0.1, 0.4, 0.8]
        changed_scale = 0.6
        changed_offset = -1.25
        destination = fill(NaN, length(changed_parameters))
        value, returned_gradient = ad_value_and_gradient!(
            prepared, destination, changed_parameters, changed_scale;
            data = changed_data, offset = changed_offset,
        )
        @test value ≈ sum(changed_parameters .* changed_data) -
              changed_scale * sum(abs2, changed_parameters) + changed_offset
        @test returned_gradient === destination
        @test destination ≈
              changed_data .- 2changed_scale .* changed_parameters

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

    @testset "structured active storage and prepared reverse pullbacks" begin
        parameter_type = NamedTuple{
            (:location, :effects),Tuple{Float64,Vector{Float64}}}
        graph = Graph()
        parameters = value!(graph, :parameters, parameter_type)
        scale = value!(graph, :scale, Float64)
        objective = value!(graph, :objective, Float64)
        add!(graph, (parameters, scale) => objective,
             (p, s) -> p.location^2 + s * sum(abs2, p.effects))
        structured = prepare(
            graph; have = (parameters, scale), want = objective)

        point = (; location = 1.5, effects = [0.25, -0.5, 0.75])
        gradient = ad_gradient(
            structured, TEST_AD_BACKEND, point, 2.0;
            active = :parameters)
        @test gradient isa NamedTuple
        @test keys(gradient) == keys(point)
        @test gradient.location ≈ 3.0
        @test gradient.effects ≈ 4 .* point.effects

        prepared_gradient = prepare_ad(
            structured, TEST_AD_BACKEND, point, 2.0;
            active = :parameters)
        changed = (; location = -0.75, effects = [1.0, -2.0, 0.5])
        changed_gradient = ad_gradient(prepared_gradient, changed, 0.25)
        @test changed_gradient.location ≈ -1.5
        @test changed_gradient.effects ≈ 0.5 .* changed.effects
        changed_value, combined_gradient = ad_value_and_gradient(
            prepared_gradient, changed, 0.25)
        @test changed_value ≈
            changed.location^2 + 0.25 * sum(abs2, changed.effects)
        @test combined_gradient == changed_gradient
        @test code_expr(prepared_gradient) === code_expr(structured)

        vector_graph = Graph()
        x = value!(vector_graph, :x, Vector{Float64})
        data = value!(vector_graph, :data, Vector{Float64})
        pointwise = value!(vector_graph, :pointwise, Vector{Float64})
        add!(vector_graph, (x, data) => pointwise,
             (q, d) -> q .^ 2 .+ q .* d)
        vector_kernel = prepare(
            vector_graph; have = (x, data), want = pointwise)
        q = [0.2, -0.4, 0.7]
        observed = [1.5, -0.5, 0.25]
        seed = [1.0, -2.0, 0.5]
        expected = seed .* (2 .* q .+ observed)

        @test_throws ArgumentError prepare_ad(
            vector_kernel, TEST_AD_BACKEND, q, observed; active = :x)
        @test ad_pullback(
            vector_kernel, TEST_AD_BACKEND, seed, q, observed;
            active = :x) ≈ expected

        prepared_pullback = prepare_ad_pullback(
            vector_kernel, TEST_AD_BACKEND, seed, q, observed;
            active = :x)
        @test prepared_pullback isa PreparedADPullback
        @test inputs(prepared_pullback) == inputs(vector_kernel)
        @test outputs(prepared_pullback) == outputs(vector_kernel)
        @test code_expr(prepared_pullback) === code_expr(vector_kernel)
        @test occursin("active=:x", sprint(show, prepared_pullback))
        @test occursin("want=:pointwise", sprint(show, prepared_pullback))
        @test ad_pullback(prepared_pullback, seed, q, observed) ≈ expected

        q2 = [-0.1, 0.8, 0.4]
        observed2 = [0.3, -1.0, 2.0]
        seed2 = [-0.5, 1.25, 2.0]
        value2, pullback2 = ad_value_and_pullback(
            prepared_pullback, seed2, q2, observed2)
        @test value2 ≈ q2 .^ 2 .+ q2 .* observed2
        @test pullback2 ≈ seed2 .* (2 .* q2 .+ observed2)

        destination = fill(NaN, length(q2))
        value3, returned = ad_value_and_pullback!(
            prepared_pullback, destination, seed2, q2, observed2)
        @test value3 ≈ value2
        @test returned === destination
        @test destination ≈ pullback2
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

        # Type annotations are optional authoring metadata. An abstractly typed
        # output is validated against the exemplar result at preparation.
        @kernel untyped_square(q) = q^2
        @test ad_gradient(
            untyped_square, TEST_AD_BACKEND, 3.0;
            active = :q, want = :untyped_square,
        ) ≈ 6.0
    end

    @testset "authored plate AD reuses the inspectable primal body" begin
        @kernel plate_objective(q::Vector{Float64},
                                data::Vector{Float64}) = begin
            pointwise = plate(q, data) do qi, di
                score::Float64 = qi * di - 0.5 * qi^2
                return score
            end
            objective::Float64 = sum(pointwise)
        end

        q = [0.2, -0.4, 0.7, 0.1]
        data = [1.5, -0.5, 0.25, 2.0]
        kernel = prepare(plate_objective;
            have = (:q, :data), want = :objective)
        primal_ast = code_expr(kernel)
        @test occursin("Base.Broadcast.preprocess", string(primal_ast))
        @test !occursin("_plate_dependency_changed", string(primal_ast))

        prepared = prepare_ad(
            kernel, TEST_AD_BACKEND, q, data; active = :q)
        ad_text = string(code_expr(prepared))
        @test prepared.call isa ReactiveKernels._ADNativeKernelCall
        @test prepared.call.native === kernel.f.native
        @test prepared.call.ops === kernel.ops
        @test code_expr(prepared) === primal_ast
        @test occursin("Base.Broadcast.preprocess", ad_text)
        @test !occursin("_plate_dependency_changed", ad_text)

        gradient = similar(q)
        value, returned = ad_value_and_gradient!(
            prepared, gradient, q, data)
        @test value ≈ sum(q .* data .- 0.5 .* q .^ 2)
        @test returned === gradient
        @test gradient ≈ data .- q
        _test_ad_value_gradient_allocated(prepared, gradient, q, data)
        _test_ad_backend_value_gradient_allocated(
            prepared, gradient, q, data)
        rk_allocated = _test_ad_value_gradient_allocated(
            prepared, gradient, q, data)
        backend_allocated = _test_ad_backend_value_gradient_allocated(
            prepared, gradient, q, data)
        # Julia 1.12's DI/Enzyme path currently performs bounded backend heap
        # work. The reusable RK wrapper must not add allocations beyond that
        # direct backend call; on runtimes where the backend is allocation-free,
        # this remains the original exact zero-allocation sentinel.
        @test rk_allocated <= backend_allocated
    end
end
