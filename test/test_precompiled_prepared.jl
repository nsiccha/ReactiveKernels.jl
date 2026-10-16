using ReactiveKernels, Test

@testset "prepared constants survive consumer package precompilation" begin
    root = dirname(dirname(pathof(ReactiveKernels)))
    fixture = joinpath(@__DIR__, "fixtures", "precompiled_prepared_consumer.jl")
    mktempdir() do env
        consumer = joinpath(env, "PrecompiledPreparedConsumer")
        derived = joinpath(env, "ImportedPreparedConsumer")
        for path in (consumer, derived)
            mkpath(joinpath(path, "src"))
        end
        write(joinpath(consumer, "Project.toml"), """
            name = "PrecompiledPreparedConsumer"
            uuid = "3cab329d-6ec4-4c05-9ae4-533c6802a85a"
            version = "0.1.0"
            [deps]
            ReactiveKernels = "78e9f072-d36b-4c73-b7a3-751b0eb26cf9"
            """)
        cp(fixture, joinpath(consumer, "src", "PrecompiledPreparedConsumer.jl"))
        write(joinpath(derived, "Project.toml"), """
            name = "ImportedPreparedConsumer"
            uuid = "38b88126-89f5-424d-a365-5156dd922d05"
            version = "0.1.0"
            [deps]
            ReactiveKernels = "78e9f072-d36b-4c73-b7a3-751b0eb26cf9"
            PrecompiledPreparedConsumer = "3cab329d-6ec4-4c05-9ae4-533c6802a85a"
            """)
        write(joinpath(derived, "src", "ImportedPreparedConsumer.jl"), """
            module ImportedPreparedConsumer
            using ReactiveKernels
            # A loading process can count up to the identities this image
            # gives its nested-plate law (below).
            foreach(_ -> value(:padding, Float64), 1:400)
            @kernel weighted_log_sum(scores::Vector{Float64},
                    weights::Vector{Float64}) = begin
                weighted = plate(scores, weights) do score, weight
                    log(weight) + score
                end
                return log(sum(exp.(weighted)))
            end
            @kernel nested_mixture(observed::Vector{Bool},
                    probabilities::Matrix{Float64}, weights::Vector{Float64}) = begin
                components = plate(1:size(probabilities, 2)) do k
                    cells = plate(1:length(observed)) do i
                        p = probabilities[i, k]
                        observed[i] ? log(p) : log1p(-p)
                    end
                    sum(cells)
                end
                return weighted_log_sum(components, weights)
            end
            import PrecompiledPreparedConsumer: trajectory, HAVE, borrowed_curve
            const WAS_PRECOMPILED = ccall(:jl_generating_output, Cint, ()) != 0
            # A value of the consumer image held by this image, and a table
            # this image keys by values of both images.
            const SHARED_INPUT = first(inputs(trajectory))
            # A graph template, y = 2x, that a loading process extends; its
            # numbers lie above any the law's composition below counts to.
            foreach(_ -> value(:padding, Float64), 1:2000)
            const TEMPLATE = let g = Graph()
                x = value!(g, :x, Float64)
                y = value!(g, :y, Float64)
                add!(g, (x,) => (y,), v -> 2v)
                (; graph = g, x, y)
            end
            const NAMES = Dict(SHARED_INPUT => :position, TEMPLATE.x => :x)
            const BATCH = prepare_batched(trajectory; have = HAVE,
                batched = :position, want = (:total, :conc))
            const BORROWED = prepare_batched(borrowed_curve;
                batched = :position, reuse = true)
            end
            """)

        # Keep the suite's active project untouched. This process builds the
        # consumer images; the next processes only load and execute them.
        setup = """
            using Pkg
            include($(repr(joinpath(root, "deps", "source_dependencies.jl"))))
            # A fresh environment needs explicit source dependencies on Julia
            # 1.10, which does not read the root package's [sources] table.
            Pkg.develop([PackageSpec[PackageSpec(path = path) for path in
                $(repr((root, consumer, derived)))]; rk_source_dependencies()])
            Pkg.precompile()
            """
        julia = Base.julia_cmd()
        setup_ok = success(pipeline(`$julia --startup-file=no --project=$env -e $setup`;
                                    stdout = stdout, stderr = stderr))
        @test setup_ok
        setup_ok || return
        probe = """
            using Test, ReactiveKernels, PrecompiledPreparedConsumer,
                  ImportedPreparedConsumer
            const C = PrecompiledPreparedConsumer
            @test C.WAS_PRECOMPILED
            @test ImportedPreparedConsumer.WAS_PRECOMPILED
            # Each image keeps the identities its kernels were authored with,
            # in a namespace of its own, and this process allocates in another.
            # A value one image refers to from the other keeps its identity.
            const RK = ReactiveKernels
            const D = ImportedPreparedConsumer
            here = RK._VALUE_NAMESPACE[]
            consumer_namespace = first(inputs(C.trajectory)).id.namespace
            derived_namespace = D.TEMPLATE.x.id.namespace
            @test consumer_namespace !== derived_namespace
            @test consumer_namespace !== here && derived_namespace !== here
            @test D.SHARED_INPUT == first(inputs(C.trajectory))
            @test D.SHARED_INPUT.id.namespace === consumer_namespace
            @test D.NAMES[first(inputs(C.trajectory))] === :position
            @test D.NAMES[D.TEMPLATE.x] === :x
            graph = C.pair_grid.graph
            right_id = RK.canon_id(graph, C.pair_grid.right.id)
            # Count to the right location's number, where a new bound plate
            # cache once reused the loaded identity.
            while here.count < right_id.number - 1
                value(:alignment, Float64)
            end
            original_values = copy(graph.values)
            original_recipes = copy(graph.recipes)
            for x in ([-0.7, 0.2, 0.2, 1.1], collect(range(-1.3, 0.8; length=9)))
                saved_x = copy(x)
                bound = prepare(C.pair_grid; bound=(; x))
                for scale in (0.6, 1.2)
                    expected_grid = [scale * exp(-abs(a - b)) for a in x, b in x]
                    @test C.PAIR_GRID(x, scale) ≈ expected_grid
                    @test bound(scale) ≈ expected_grid
                    @test bound(scale) ≈ transpose(bound(scale))
                    @test all(i -> bound(scale)[i, i] ≈ scale, eachindex(x))
                end
                @test x == saved_x
                caches = filter(r -> any(v -> startswith(string(v.name), "bound_plate_"),
                                          r.outputs), bound.plan.recipes)
                @test !isempty(caches)
                @test all(r -> all(v -> !haskey(original_values, v.id), r.outputs), caches)
            end
            @test graph.values == original_values
            @test graph.recipes == original_recipes

            # The builder also extends loaded graphs and composed fragments
            # without overwriting a value retained by either graph.
            copied = RK.Graph(copy(graph.values), copy(graph.recipes),
                              copy(graph.producers), copy(graph.aliases), graph.version)
            fresh_value = value!(copied, :fresh, Float64)
            @test !haskey(original_values, fresh_value.id)
            merged_graph = compose(graph)
            merged_value = value!(merged_graph, :fresh, Float64)
            @test !haskey(original_values, merged_value.id)
            positions = [1.0 2.0; 0.0 0.0]
            schedule = [0.5, 1.0]
            doses = [3.0, 4.0]
            args = (positions, 0.0, schedule, doses)
            expected = ([3.5 4.0; 5.0 6.0], [8.5, 10.0])

            # Instantiate saved pristine/used templates BEFORE any post-load
            # preparation. Copy must reuse the image-owned executable body.
            for template in (C.BORROWED, C.USED_BORROWED,
                             ImportedPreparedConsumer.BORROWED)
                instance = copy(template)
                @test instance.target.native === template.target.native
                @test instance.target.target === template.target.target
                @test code_expr(instance) === code_expr(template)
                @test all(slot[] === nothing for slot in instance.target.caches)
                first = instance([2.0, 3.0], [4.0, 5.0])
                @test first == (curve=[8.0 12.0; 10.0 15.0], total=[18.0, 27.0])
                published = deepcopy(first)
                @test copy(instance)([2.0, 3.0], [4.0, 5.0]; amount=2.0) ==
                    (curve=2 .* published.curve, total=2 .* published.total)
                @test first == published
            end
            @test C.BORROWED_WARMUP ==
                (curve=[6.0 12.0; 8.0 16.0], total=[14.0, 28.0])

            # Call every saved object before constructing anything after load.
            # Re-preparing first would silently repopulate the missing bodies.
            @test last(C.FIRST) ==
                (expected[1][:, 1:1], expected[2][1:1])
            for kernel in (first(C.FIRST), C.BATCH, C.VECTORIZED)
                @test kernel(args...) == expected
                @test Core.Compiler.return_type(kernel, typeof(args)) ===
                    typeof(expected)
                @test batched_ports(kernel) == (:position,)
                @test kernel(positions[:, 1:1], args[2:end]...) ==
                    (expected[1][:, 1:1], expected[2][1:1])
                @test_throws DimensionMismatch kernel(
                    positions[:, 1], args[2:end]...)
            end
            @test C.SCALAR(positions[:, 1], args[2:end]...) ==
                (expected[1][:, 1], expected[2][1])
            @test C.BOUND(positions[:, 1], 0.0, schedule) ==
                (expected[1][:, 1], expected[2][1])
            @test C.WARMUP == expected[1][:, 1:1]
            @test C.WARMED(view(positions, :, :), 0.0, schedule, doses) == expected[1]
            @test C.Nested.BATCH(args...) == expected[2]
            @test ImportedPreparedConsumer.BATCH(args...) == reverse(expected)

            # The image carries the graph's bound preparation from `BOUND`:
            # binding new data after load reuses its compiled residual.
            rebound = prepare(C.trajectory; have = C.HAVE, want = C.WANT,
                              bound = (dose_mgs = [5.0, 6.0],))
            @test rebound.f === C.BOUND.f
            @test rebound(positions[:, 1], 0.0, schedule) == ([5.5, 7.0], 12.5)
            @test C.BOUND(positions[:, 1], 0.0, schedule) ==
                (expected[1][:, 1], expected[2][1])

            # The cache belongs to the image being produced, even for an
            # imported graph or a preparation performed inside a submodule.
            owner(f) = parentmodule(typeof(
                ReactiveKernels._native_generated_function(f)).parameters[2])
            @test owner(C.BATCH.native) === C
            @test owner(C.Nested.BATCH.native) === C
            @test owner(ImportedPreparedConsumer.BATCH.native) ===
                ImportedPreparedConsumer
            @test parentmodule(typeof(C.BATCH.native).parameters[3]) === C

            fresh = prepare_batched(C.trajectory; have = C.HAVE,
                                    batched = :position, want = C.WANT)
            @test fresh(args...) == C.BATCH(args...)
            # Fresh lowering gives temporary variables new gensym suffixes.
            normalized(ast) = replace(string(ast), r"#[0-9]+" => "#")
            @test normalized(code_expr(fresh)) == normalized(code_expr(C.BATCH))
            @test inputs(fresh) == inputs(C.BATCH)
            @test outputs(fresh) == outputs(C.BATCH)
            @test owner(fresh.native) === ReactiveKernels
            # Inputs and earlier outputs retain their contents across calls.
            earlier = C.BATCH(args...)
            @test C.BATCH(2 .* positions, args[2:end]...) != earlier
            @test earlier == expected
            @test positions == [1.0 2.0; 0.0 0.0]
            @test schedule == [0.5, 1.0]
            @test doses == [3.0, 4.0]
            println("PRECOMPILED_PREPARED_PASS")
            """
        # Two independent loads rule out a process-local body-cache recovery.
        for _ in 1:2
            @test success(pipeline(
                `$julia --startup-file=no --check-bounds=yes --project=$env -e $probe`;
                stdout = stdout, stderr = stderr))
        end

        # A loaded nested-plate law composed into the cells of a plate authored
        # after load. The loading process's count does not include the image's
        # allocations, so it reaches the identities of the law's nested cells.
        # Each backend product composed the law's plates with fresh values
        # separately; once those took the law's own identities, preparation
        # refused the products as different (snag readable-recipe-75581e09).
        composition = """
            using Test, ReactiveKernels
            import ImportedPreparedConsumer: nested_mixture
            const RK = ReactiveKernels
            @kernel event_mixtures(events::Vector{Vector{Bool}},
                    probabilities::Matrix{Float64}, weights::Vector{Float64}) = begin
                scores = plate(events) do observed
                    nested_mixture(observed, probabilities, weights)
                end
                return sum(scores)
            end
            @test RK._needs_embedded_tensorization(plan(event_mixtures))
            function plate_value_ids!(ids, graph)
                for r in graph.recipes
                    r.op isa RK._AuthoredPlateOp || continue
                    inner = r.op.kernel.plan.graph
                    union!(ids, keys(inner.values))
                    plate_value_ids!(ids, inner)
                end
                ids
            end
            nested = setdiff(plate_value_ids!(Set{RK.ValueId}(), nested_mixture.graph),
                             keys(nested_mixture.graph.values))
            # Count to just below their numbers, as an ordered suite does.
            first_nested = minimum(id.number for id in nested)
            here = RK._VALUE_NAMESPACE[]
            @test here.count < first_nested - 1
            while here.count < first_nested - 1
                value(:ordered_suite, Float64)
            end
            mixtures = prepare(event_mixtures)
            events = [Bool[1, 0, 1], Bool[0, 0, 1], Bool[1, 1, 1]]
            probabilities = [0.3 0.8; 0.6 0.2; 0.5 0.5]
            weights = [0.4, 0.6]
            expected = sum(events) do observed
                scores = [sum(observed[i] ? log(probabilities[i, k]) :
                              log1p(-probabilities[i, k]) for i in eachindex(observed))
                          for k in axes(probabilities, 2)]
                log(sum(weights .* exp.(scores)))
            end
            @test mixtures(events, probabilities, weights) ≈ expected

            # Values made here, with the numbers of the loaded template's
            # values, wired into that template. Sharing identities made `a`
            # the template's `x` and `b` its `y`: planning `b` from `x` then
            # returned `y` (2x) instead of refusing the missing `a`.
            import ImportedPreparedConsumer: TEMPLATE
            while here.count < TEMPLATE.x.id.number - 1
                value(:ordered_suite, Float64)
            end
            a = value(:a, Float64)
            b = value(:b, Float64)
            @test a.id.number == TEMPLATE.x.id.number && a != TEMPLATE.x
            @test b.id.number == TEMPLATE.y.id.number && b != TEMPLATE.y
            g = compose(TEMPLATE.graph)
            add!(g, (a,) => (b,), v -> v + 100)
            @test g.values[TEMPLATE.x.id] === TEMPLATE.x
            @test prepare(g; have = (TEMPLATE.x,), want = (TEMPLATE.y,))(1.0) == 2.0
            @test prepare(g; have = (a,), want = (b,))(1.0) == 101.0
            @test_throws RK.PlanningError prepare(g; have = (TEMPLATE.x,), want = (b,))
            println("NESTED_LAW_COMPOSITION_PASS")
            """
        @test success(pipeline(
            `$julia --startup-file=no --check-bounds=yes --project=$env -e $composition`;
            stdout = stdout, stderr = stderr))
    end
end
