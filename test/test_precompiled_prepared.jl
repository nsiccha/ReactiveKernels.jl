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
            import PrecompiledPreparedConsumer: trajectory, HAVE
            const WAS_PRECOMPILED = ccall(:jl_generating_output, Cint, ()) != 0
            const BATCH = prepare_batched(trajectory; have = HAVE,
                batched = :position, want = (:total, :conc))
            end
            """)

        # Keep the suite's active project untouched. This process builds the
        # consumer images; the next processes only load and execute them.
        setup = """
            using Pkg
            Pkg.develop([PackageSpec(path = path) for path in
                $(repr((root, consumer, derived)))])
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
            positions = [1.0 2.0; 0.0 0.0]
            schedule = [0.5, 1.0]
            doses = [3.0, 4.0]
            args = (positions, 0.0, schedule, doses)
            expected = ([3.5 4.0; 5.0 6.0], [8.5, 10.0])

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
    end
end
