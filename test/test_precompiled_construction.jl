using ReactiveKernels, Test

# The child module binds nothing it is not given, and lowering reads some names
# the source spells only as syntax: `x'` calls `'`, `s -= y` calls `-`. A
# batched reader of a generated program built in a downstream image called
# such a recipe's closure and failed with "UndefVarError: `'` not defined"
# (snag `image-built-kern-ddaab84e`).
module ScopeAuthor
# Shadows Base's adjoint: the child must resolve `'` as this module does.
const var"'" = x -> (:shadowed, x)
end

@testset "a construction's child module resolves every name lowering reads" begin
    RK = ReactiveKernels
    samples = Any[
        :(let x = [1.0, 2.0]; x' * x end),
        :(let x = [1.0 2.0]; (y -> y')(x) end),
        :(let s = 1.0; s -= 0.25; s end),
        :(let s = [1.0, 2.0]; s .*= 3.0; s end),
        :(let s = 3; s ⊻= 5; s end),
        :(let s = 3 // 4; s //= 2; s end),
        :(let a = [1.0, 2.0]; a .+ 1.0 end),
        :(let a = [1.0, 2.0]; .-a end),
        :(let a = [1.0, 2.0]; a .^ 2 end),
        :(let a = [1.0, 4.0]; a .|> sqrt end),
        :(Int <: Real),
        :(Real >: Int),
        :([Int, String] .<: Real),
        :(let s = [1, 2]; s[2] += 3; s end),
        :(let f = x -> x'; f([1.0 2.0]) end),
    ]
    for ex in samples
        child = RK._kernel_scope_module(@__MODULE__, Main, ex)
        @test Core.eval(child, ex) == Core.eval(Main, ex)
    end
    ex = :(let x = 2.0; x' end)
    child = RK._kernel_scope_module(@__MODULE__, ScopeAuthor, ex)
    @test Core.eval(child, ex) == Core.eval(ScopeAuthor, ex) == (:shadowed, 2.0)
end

# Kernel construction evaluates recipe closures in the module a definition was
# authored for. While a downstream package's image is generated, Julia refuses
# evaluation into another package's module, so preparing an imported kernel
# whose constructed-endpoint branch arms are evaluated in the endpoint's
# package, or evaluating a generated definition authored for another package
# (as ReactiveKernelsPPL's `build_kernel` does), failed with "Evaluation into
# the closed module" (snag `rkppl-build-kern-4607ff90`). Construction now
# evaluates in a child module of the package being built whose names resolve
# as in the authoring module, and the image keeps the result.
@testset "constructions for another package's module in a package image" begin
    root = dirname(dirname(pathof(ReactiveKernels)))
    mktempdir() do env
        source = joinpath(env, "ArmSource")
        consumer = joinpath(env, "ArmConsumer")
        for path in (source, consumer)
            mkpath(joinpath(path, "src"))
        end
        write(joinpath(source, "Project.toml"), """
            name = "ArmSource"
            uuid = "8e4b6f1a-3c2d-4e5f-9a8b-7c6d5e4f3a2b"
            version = "0.1.0"
            [deps]
            ReactiveKernels = "78e9f072-d36b-4c73-b7a3-751b0eb26cf9"
            """)
        write(joinpath(source, "src", "ArmSource.jl"), """
            module ArmSource
            using ReactiveKernels
            # Visible only through this module: a construction evaluated
            # elsewhere must resolve it as this module does.
            scaled_log(scale) = log(scale)
            # Preparing the plate evaluates the endpoint's branch-arm graph in
            # this module (`_plate_branch_with_source_graphs`).
            @kernel endpoint_law(scale::Float64) = begin
                logpdf(x::Float64)::Float64 =
                    x > 0 ? (twice = 2x; 0.5twice * scaled_log(scale)) : -x
            end
            @kernel endpoint_law_plate(x::Vector{Float64}, scale::Float64) = begin
                cells = plate(x) do xi
                    endpoint_law(scale).logpdf(xi)
                end
                total::Float64 = sum(cells)
            end
            # `offset` is a top-level recipe closure, whose module the probe
            # reads (a bare call would use `scaled_log` itself as the recipe).
            const DEFINITION = :(generated(x::Vector{Float64}, scale::Float64) = begin
                offset = 2 * scaled_log(sqrt(scale))
                cells = plate(x) do xi
                    endpoint_law(scale).logpdf(xi) + offset
                end
                total::Float64 = sum(cells)
            end)
            # Recipes whose names lowering reads only from syntax (`'`, the
            # `-` of `-=`). Both summands' factors that read `x` but not
            # `scale` are split nodes, and a batched reader calls their
            # closures.
            const SYNTAX_DEFINITION = :(generated(x::Vector{Float64}, scale::Float64) = begin
                total::Float64 = scale * sum(x' * x) +
                                 abs(let s = sum(x); s -= 1.0; s end) * scale
            end)
            # A summary written once against a port `prediction`, and a model
            # whose prediction value has another name (snag
            # `kernel-input-bou-ce13eae3`).
            @kernel summaries(prediction, rows) = begin
                cells = plate(eachindex(rows)) do s
                    cell::Float64 = sum(prediction[rows[s]])
                end
                total::Float64 = sum(cells)
            end
            @kernel model(a, b, t) = begin
                mu = a .+ b .* t
            end
            end
            """)
        write(joinpath(consumer, "Project.toml"), """
            name = "ArmConsumer"
            uuid = "9f5c7a2b-4d3e-4f6a-8b9c-8d7e6f5a4b3c"
            version = "0.1.0"
            [deps]
            ArmSource = "8e4b6f1a-3c2d-4e5f-9a8b-7c6d5e4f3a2b"
            ReactiveKernels = "78e9f072-d36b-4c73-b7a3-751b0eb26cf9"
            """)
        write(joinpath(consumer, "src", "ArmConsumer.jl"), """
            module ArmConsumer
            using ReactiveKernels, ArmSource
            const WAS_PRECOMPILED = ccall(:jl_generating_output, Cint, ()) != 0
            const PREPARED = prepare(ArmSource.endpoint_law_plate)
            const VALUE = PREPARED([-2.0, 1.0, -3.0, 0.5, 4.0], 2.0)
            const GENERATED = ReactiveKernels._kernel_eval_definition(ArmSource,
                deepcopy(ArmSource.DEFINITION))
            const SYNTAX = ReactiveKernels._kernel_eval_definition(ArmSource,
                deepcopy(ArmSource.SYNTAX_DEFINITION))
            const SYNTAX_READER = vectorize(SYNTAX; batched = :scale, reuse = true)
            const SYNTAX_VALUE = copy(SYNTAX_READER([-2.0, 1.0, -3.0, 0.5, 4.0], [2.0, 3.0]))
            # The prediction value's name is known only at run time. Renaming
            # the closed dependency's port evaluates nothing.
            prediction_name() = Symbol(:m, :u)
            const RENAMED = compose(ArmSource.model,
                ReactiveKernels.rename(ArmSource.summaries, :prediction => prediction_name()))
            const RENAMED_READER = prepare(RENAMED; have = (:a, :b, :t, :rows), want = :total)
            const RENAMED_VALUE = RENAMED_READER(1.0, 2.0, [0.0, 0.5, 1.0], [[1, 2], [3]])
            end
            """)
        setup = """
            using Pkg
            include($(repr(joinpath(root, "deps", "source_dependencies.jl"))))
            # Julia 1.10 does not read the root package's [sources] table.
            Pkg.develop([PackageSpec[PackageSpec(path = path) for path in
                $(repr((root, source, consumer)))]; rk_source_dependencies()])
            Pkg.precompile()
            """
        julia = Base.julia_cmd()
        setup_ok = success(pipeline(`$julia --startup-file=no --project=$env -e $setup`;
                                    stdout = stdout, stderr = stderr))
        @test setup_ok
        setup_ok || return
        probe = """
            using Test, ReactiveKernels, ArmSource, ArmConsumer
            const RK = ReactiveKernels
            # The module whose image owns each source companion (the cache of
            # its generated function), and each source's authoring module.
            companion_roots(spec) = Set(Base.moduleroot(parentmodule(
                    typeof(RK._native_generated_function(f.companion)).parameters[2]))
                for r in spec.graph.recipes if r.op isa RK._KernelSourceOp
                for f in (RK._source_record(r.op.f),) if f !== nothing && f.companion !== nothing)
            source_modules(spec) = Set(RK._kernel_endpoint_source_module(r.op)
                for r in spec.graph.recipes if r.op isa RK._KernelSourceOp)
            @testset "loaded image" begin
            @test ArmConsumer.WAS_PRECOMPILED
            x = [-2.0, 1.0, -3.0, 0.5, 4.0]
            saved = copy(x)
            law(scale) = 5.0 + 5.5log(scale)
            @test ArmConsumer.VALUE ≈ law(2.0)
            for scale in (0.5, 2.0, 3.0)
                @test ArmConsumer.PREPARED(x, scale) ≈ law(scale)
                @test prepare(ArmSource.endpoint_law_plate)(x, scale) ≈ law(scale)
                @test prepare(ArmConsumer.GENERATED)(x, scale) ≈ law(scale) + 5log(scale)
            end
            @test x == saved
            # The image owns the generated companions of a construction built
            # while it was generated, and a construction of the same sources
            # at run time reuses them (`compile` returns a loaded image's
            # generated function). Both resolve their sources' names in the
            # authoring module.
            @test ArmConsumer in companion_roots(ArmConsumer.GENERATED)
            @test source_modules(ArmConsumer.GENERATED) == Set([ArmSource])
            runtime = RK._kernel_eval_definition(ArmSource, deepcopy(ArmSource.DEFINITION))
            @test companion_roots(runtime) == companion_roots(ArmConsumer.GENERATED)
            @test source_modules(runtime) == Set([ArmSource])
            @test prepare(runtime)(x, 2.0) ≈ law(2.0) + 5log(2.0)
            # Names lowering reads from syntax, through a batched reader.
            syntax(scale) = scale * (sum(abs2, x) + abs(sum(x) - 1.0))
            @test ArmConsumer in companion_roots(ArmConsumer.SYNTAX)
            @test ArmConsumer.SYNTAX_VALUE ≈ syntax.([2.0, 3.0])
            @test ArmConsumer.SYNTAX_READER(x, [0.5, 4.0]) ≈ syntax.([0.5, 4.0])
            @test vectorize(ArmConsumer.SYNTAX; batched = :scale)(x, [1.5]) ≈ [syntax(1.5)]
            @test x == saved
            # A kernel composed through a renamed port of a closed dependency.
            t = [0.0, 0.5, 1.0]; rows = [[1, 2], [3]]
            total(a, b) = sum(a .+ b .* t)
            @test ArmConsumer.RENAMED_VALUE == total(1.0, 2.0)
            @test ArmConsumer.RENAMED_READER(-1.0, 4.0, t, rows) == total(-1.0, 4.0)
            @test haskey(ArmConsumer.RENAMED, :mu) && !haskey(ArmConsumer.RENAMED, :prediction)
            end
            println("PRECOMPILED_CONSTRUCTION_PASS")
            """
        @test success(pipeline(`$julia --startup-file=no --project=$env -e $probe`;
                               stdout = stdout, stderr = stderr))
    end
end
