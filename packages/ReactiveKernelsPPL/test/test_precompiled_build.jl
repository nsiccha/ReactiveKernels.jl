using ReactiveKernels, ReactiveKernelsPPL, ReactiveKernelsDistributionKernels, Test

# A downstream package builds an RKPPL model while its own image is generated:
# in a PrecompileTools workload, and as a constant the image keeps. Julia
# refuses evaluation into `PPLGeneratedModels` from another package's
# precompilation, so `build_kernel` failed there with "Evaluation into the
# closed module" (snag `rkppl-build-kern-4607ff90`). A second package prepares
# the first image's build, and builds another binding, while its image is
# generated.
@testset "build_kernel while a downstream package image is generated" begin
    root = dirname(dirname(pathof(ReactiveKernels)))
    paths = (root, dirname(dirname(pathof(ReactiveKernelsDistributionKernels))),
             dirname(dirname(pathof(ReactiveKernelsPPL))))
    mktempdir() do env
        built = joinpath(env, "PrecompiledPPLBuild")
        prepared = joinpath(env, "PrecompiledPPLPrepare")
        for path in (built, prepared)
            mkpath(joinpath(path, "src"))
        end
        write(joinpath(built, "Project.toml"), """
            name = "PrecompiledPPLBuild"
            uuid = "5a2d8c3e-6f4b-4a7c-9d1e-2b3c4d5e6f7a"
            version = "0.1.0"
            [deps]
            Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
            PrecompileTools = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
            ReactiveKernelsPPL = "e266ea2a-6817-46cd-8bec-6238f0035f44"
            """)
        write(joinpath(built, "src", "PrecompiledPPLBuild.jl"), """
            module PrecompiledPPLBuild
            using ReactiveKernelsPPL, Distributions
            using PrecompileTools: @setup_workload, @compile_workload
            const WAS_PRECOMPILED = ccall(:jl_generating_output, Cint, ()) != 0
            const MODEL = @rkppl begin
                a ~ Normal(0, 5)
                b ~ Normal(0, 2)
                sigma ~ Exponential(1.0)
                mu = a .+ b .* x
                y .~ Normal.(mu, sigma)
            end
            const PLAN = MODEL(; x = [0.0, 1.0, 2.0]) | (; y = [0.1, 0.9, 2.2])
            const BUILT = build_kernel(PLAN)
            @setup_workload begin
                @compile_workload begin
                    plan = MODEL(; x = [0.5, 1.5]) | (; y = [0.4, 1.7])
                    prepare_query(build_kernel(plan), plan, :sampler)([0.3, -0.2, 0.1])
                end
            end
            # Prepared after that workload, so its generated functions are the
            # image's own rather than world-age wrappers. A process that
            # prepares this build after loading the image reuses them.
            const QUERY = prepare_query(BUILT, PLAN, :sampler)
            const VALUE = QUERY([0.3, -0.2, 0.1])
            end
            """)
        write(joinpath(prepared, "Project.toml"), """
            name = "PrecompiledPPLPrepare"
            uuid = "6b3e9d4f-7a5c-4b8d-8e2f-3c4d5e6f7a8b"
            version = "0.1.0"
            [deps]
            ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
            Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
            PrecompileTools = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
            PrecompiledPPLBuild = "5a2d8c3e-6f4b-4a7c-9d1e-2b3c4d5e6f7a"
            ReactiveKernelsPPL = "e266ea2a-6817-46cd-8bec-6238f0035f44"
            """)
        write(joinpath(prepared, "src", "PrecompiledPPLPrepare.jl"), """
            module PrecompiledPPLPrepare
            using ReactiveKernelsPPL, PrecompiledPPLBuild, Enzyme
            using ADTypes: AutoEnzyme
            using PrecompileTools: @setup_workload, @compile_workload
            const B = PrecompiledPPLBuild
            const WAS_PRECOMPILED = ccall(:jl_generating_output, Cint, ()) != 0
            const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
            # Prepares B's build for reverse-mode gradients without taking one
            # (rkppl-use §1). The workload comes first: the image does not keep
            # a lowering that the constants below compiled first.
            @setup_workload begin
                @compile_workload begin
                    plan = B.MODEL(; x = [0.2, 1.2]) | (; y = [0.5, 1.1])
                    prepare_sampler(B.BUILT, plan, [0.3, -0.2, 0.1]; backend = BACKEND)(
                        [0.3, -0.2, 0.1])
                end
            end
            const QUERY = prepare_query(B.BUILT, B.PLAN, :sampler)
            const VALUE = QUERY([0.3, -0.2, 0.1])
            const PLAN = B.MODEL(; x = [0.5, 1.5, 2.5, 3.5]) | (; y = [0.4, 1.7, 2.1, 3.9])
            const BUILT = build_kernel(PLAN)
            end
            """)
        setup = """
            using Pkg
            include($(repr(joinpath(root, "deps", "source_dependencies.jl"))))
            # Julia 1.10 does not read the root package's [sources] table.
            Pkg.develop([PackageSpec[PackageSpec(path = path) for path in
                $(repr((paths..., built, prepared)))]; rk_source_dependencies()])
            # The probe takes reverse gradients after loading the images.
            Pkg.add(["DifferentiationInterface", "Enzyme"])
            Pkg.precompile()
            """
        julia = Base.julia_cmd()
        setup_ok = success(pipeline(`$julia --startup-file=no --project=$env -e $setup`;
                                    stdout = stdout, stderr = stderr))
        @test setup_ok
        setup_ok || return
        probe = """
            using Test, DifferentiationInterface, Enzyme, ReactiveKernels,
                  ReactiveKernelsPPL, PrecompiledPPLBuild, PrecompiledPPLPrepare
            const RK = ReactiveKernels
            const B = PrecompiledPPLBuild
            const P = PrecompiledPPLPrepare
            normal_logpdf(z, m, s) = -0.5log(2pi) - log(s) - 0.5((z - m) / s)^2
            # Independent posterior density on the unconstrained scale:
            # sigma = exp(u[3]) contributes its Exponential(1) prior and Jacobian.
            density(u, x, y) = let a = u[1], b = u[2], s = exp(u[3])
                normal_logpdf(a, 0, 5) + normal_logpdf(b, 0, 2) - s + u[3] +
                    sum(normal_logpdf.(y, a .+ b .* x, s))
            end
            function density_gradient(u, x, y)
                a, b, s = u[1], u[2], exp(u[3])
                r = y .- a .- b .* x
                [-a / 25 + sum(r) / s^2, -b / 4 + sum(x .* r) / s^2,
                 1 - s + sum(r .^ 2 ./ s^2 .- 1)]
            end
            # The module whose image owns each source companion (the cache of
            # its generated function), and each source's authoring module.
            companion_roots(spec) = Set(Base.moduleroot(parentmodule(
                    typeof(RK._native_generated_function(f.companion)).parameters[2]))
                for r in spec.graph.recipes if r.op isa RK._KernelSourceOp
                for f in (RK._source_record(r.op.f),) if f !== nothing && f.companion !== nothing)
            source_modules(spec) = Set(RK._kernel_endpoint_source_module(r.op)
                for r in spec.graph.recipes if r.op isa RK._KernelSourceOp)
            @testset "loaded images" begin
            @test B.WAS_PRECOMPILED && P.WAS_PRECOMPILED
            x, y = [0.0, 1.0, 2.0], [0.1, 0.9, 2.2]
            x2, y2 = [0.5, 1.5, 2.5, 3.5], [0.4, 1.7, 2.1, 3.9]
            @test P.VALUE ≈ density([0.3, -0.2, 0.1], x, y)
            for u in ([0.3, -0.2, 0.1], [-1.1, 0.7, -0.4], [2.0, 0.1, 0.9])
                @test prepare_query(B.BUILT, B.PLAN, :sampler)(u) ≈ density(u, x, y)
                @test P.QUERY(u) ≈ density(u, x, y)
                @test prepare_query(P.BUILT, P.PLAN, :sampler)(u) ≈ density(u, x2, y2)
                # One build serves another binding of the same program.
                @test prepare_query(B.BUILT, P.PLAN, :sampler)(u) ≈ density(u, x2, y2)
            end
            # Each image owns the source companions of its builds, and a build
            # of the same program at run time reuses them (`compile` returns a
            # loaded image's generated function). Every build resolves its
            # sources' names in `PPLGeneratedModels`.
            @test B in companion_roots(B.BUILT.spec)
            @test P in companion_roots(P.BUILT.spec)
            runtime = build_kernel(B.PLAN)
            @test !isempty(companion_roots(runtime.spec))
            @test companion_roots(runtime.spec) ⊆
                  union(companion_roots(B.BUILT.spec), companion_roots(P.BUILT.spec))
            @test all(m -> Base.moduleroot(m) === ReactiveKernelsPPL, source_modules(runtime.spec))
            @test prepare_query(runtime, B.PLAN, :sampler)([0.3, -0.2, 0.1]) ≈
                  density([0.3, -0.2, 0.1], x, y)
            end
            # B's image compiled the query of its build, and P's image prepared
            # the same program. Preparing that build after loading, on either
            # binding, reuses the image's generated functions: the query has
            # the image query's type, so the code the image compiled for it
            # applies, rather than compiling the program again in this process.
            @testset "a run-time preparation reuses the image's program" begin
            x, y = [0.0, 1.0, 2.0], [0.1, 0.9, 2.2]
            x2, y2 = [0.5, 1.5, 2.5, 3.5], [0.4, 1.7, 2.1, 3.9]
            @test B.VALUE ≈ density([0.3, -0.2, 0.1], x, y)
            @test typeof(P.QUERY) === typeof(B.QUERY)
            for plan in (B.PLAN, P.PLAN)
                @test typeof(prepare_query(B.BUILT, plan, :sampler)) === typeof(B.QUERY)
            end
            # A rebuild of the program at run time is evaluated in
            # `PPLGeneratedModels`, the image's build in a scope module of B.
            # Neither module enters an executable type, so the rebuild prepares
            # to the image query's type as well (snag `compiled-reader-be3cadbb`).
            rebuilt = build_kernel(B.PLAN)
            @test typeof(prepare_query(rebuilt, B.PLAN, :sampler)) === typeof(B.QUERY)
            backend = AutoEnzyme(; mode = Enzyme.Reverse)
            for (plan, x, y) in ((B.PLAN, x, y), (P.PLAN, x2, y2)),
                    u in ([0.3, -0.2, 0.1], [-1.1, 0.7, -0.4])
                q = prepare_sampler(B.BUILT, plan, u; backend)
                g = similar(u)
                value, _ = sampler_value_and_gradient!(q, g, u)
                @test value ≈ density(u, x, y)
                @test g ≈ density_gradient(u, x, y)
            end
            end
            println("PRECOMPILED_PPL_BUILD_PASS")
            """
        @test success(pipeline(`$julia --startup-file=no --project=$env -e $probe`;
                               stdout = stdout, stderr = stderr))

        # P's image prepared B's build as a reverse-mode sampler, so preparing
        # that build on another binding after loading compiles no lowering.
        # Julia 1.10 kept RK's recursive lowering in such an image for its
        # return type only, without code, and every process compiled it again.
        trace = joinpath(env, "trace_compile.jl")
        lowering = """
            using ReactiveKernelsPPL, PrecompiledPPLBuild, PrecompiledPPLPrepare
            const P = PrecompiledPPLPrepare
            prepare_sampler(PrecompiledPPLBuild.BUILT, P.PLAN, [0.3, -0.2, 0.1];
                            backend = P.BACKEND)
            """
        @test success(pipeline(
            `$julia --startup-file=no --project=$env --trace-compile=$trace -e $lowering`;
            stdout = stdout, stderr = stderr))
        @test isempty(filter(contains("_lower_with_ops_collecting"), readlines(trace)))
    end
end
