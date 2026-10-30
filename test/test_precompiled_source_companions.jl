using ReactiveKernels, Test

# A graph that a package image defines inside one compiled call compiles every
# source companion in the world that defines the image's generated method, so
# `compile` wraps each in `_PrecompileWarmFunction`. A kernel prepared from
# that graph, after loading the image or while building it, executes the
# companions as a graph built at run time does: no run-time dispatched call in
# its body. Since execution copies started to hold the companions, the
# wrapper's image check and latest-world branch ran in every cell, 4.6-11x the
# same plate built at run time (snag `generator-sum-in-862b7041`).
@testset "image-built source companions run without a dynamic call" begin
    root = dirname(dirname(pathof(ReactiveKernels)))
    audit = joinpath(@__DIR__, "type_stability_audit.jl")
    mktempdir() do env
        consumer = joinpath(env, "WarmSourceConsumer")
        mkpath(joinpath(consumer, "src"))
        write(joinpath(consumer, "Project.toml"), """
            name = "WarmSourceConsumer"
            uuid = "5f0c6a0e-2b7d-4c43-9a51-8d3e1f6b2c47"
            version = "0.1.0"
            [deps]
            ReactiveKernels = "78e9f072-d36b-4c73-b7a3-751b0eb26cf9"
            """)
        write(joinpath(consumer, "src", "WarmSourceConsumer.jl"), """
            module WarmSourceConsumer
            using ReactiveKernels
            const WAS_PRECOMPILED = ccall(:jl_generating_output, Cint, ()) != 0
            # Built inside the image's first compiling call, so no return to
            # top level makes its generated method visible before the last
            # companion. The second graph is also prepared there, before the
            # method is visible.
            function make_graphs()
                folds = @kernel weighted_folds(w::Vector{Float64},
                        x::Vector{Float64}) = begin
                    y = plate(x) do xi
                        sum(w[j] * xi for j in eachindex(w); init = 0.0)
                    end
                    return y
                end
                scaled = @kernel scaled_folds(w::Vector{Float64},
                        x::Vector{Float64}) = begin
                    y = plate(x) do xi
                        sum(w[j] * (xi + j) for j in eachindex(w); init = 0.0)
                    end
                    return y
                end
                # A bound preparation kept in the image, whose residual calls
                # a companion at its top level.
                dot = @kernel weighted_dot(w::Vector{Float64},
                        x::Vector{Float64}) = begin
                    s = sum(w[j] * x[j] for j in eachindex(w); init = 0.0)
                    return s
                end
                bound = prepare(dot; bound = (; w = collect(range(0.1, 1.4; length = 14))))
                folds, scaled, prepare(scaled), dot, bound
            end
            const FOLDS, SCALED, SCALED_PREPARED, DOT, BOUND_DOT = make_graphs()
            # Whether the bound template kept an execution copy of its table
            # while the image was generated.
            const KEPT_EXECUTABLE = any(values(only(values(
                    ReactiveKernels._graph_preparations(DOT.graph).bound)).templates)) do t
                t isa ReactiveKernels._BoundTemplate && t.executable.entries[] !== nothing
            end
            end
            """)
        setup = """
            using Pkg
            include($(repr(joinpath(root, "deps", "source_dependencies.jl"))))
            # Julia 1.10 does not read the root package's [sources] table.
            Pkg.develop([PackageSpec[PackageSpec(path = path) for path in
                $(repr((root, consumer)))]; rk_source_dependencies()])
            # The typed-IR audit imports it; ReactiveKernels already depends on it.
            Pkg.add("DifferentiationInterface")
            Pkg.precompile()
            """
        julia = Base.julia_cmd()
        setup_ok = success(pipeline(`$julia --startup-file=no --project=$env -e $setup`;
                                    stdout = stdout, stderr = stderr))
        @test setup_ok
        setup_ok || return
        probe = """
            using Test, ReactiveKernels, WarmSourceConsumer
            include($(repr(audit)))
            using .TypeStabilityAudit: findings, rk_or_generated
            const RK = ReactiveKernels
            const C = WarmSourceConsumer
            # Every source closure of the graph, with its companion.
            function sources!(found, x, seen = IdDict{Any,Nothing}())
                (isprimitivetype(typeof(x)) || x isa Union{Symbol,String,Module,Type,Expr}) &&
                    return found
                haskey(seen, x) && return found
                seen[x] = nothing
                x isa RK._KernelSourceFunction && push!(found, x)
                if x isa Union{Tuple,NamedTuple}
                    foreach(v -> sources!(found, v, seen), x)
                elseif x isa AbstractArray
                    foreach(i -> isassigned(x, i) && sources!(found, x[i], seen), eachindex(x))
                elseif typeof(x) isa DataType
                    foreach(i -> isdefined(x, i) && sources!(found, getfield(x, i), seen),
                            1:fieldcount(typeof(x)))
                end
                found
            end
            warm(x) = any(f -> f.runtime_f isa RK._PrecompileWarmFunction,
                          sources!(Any[], x))
            warm(spec::RK.KernelSpec) = warm(spec.graph)
            @testset "loaded image" begin
            @test C.WAS_PRECOMPILED
            # The case under test: the image holds warm-wrapped companions.
            @test warm(C.FOLDS)
            @test warm(C.SCALED)
            w = collect(range(0.1, 1.4; length = 14))
            x = collect(range(-1.0, 1.0; length = 257))
            saved = (copy(w), copy(x))
            argtypes = (Vector{Float64}, Vector{Float64})
            recurse = rk_or_generated(C)
            report(found) = (foreach(f -> println(stderr, "  ", f), found); isempty(found))
            for (kernel, term) in ((prepare(C.FOLDS), (j, xi) -> w[j] * xi),
                                   (prepare(C.SCALED), (j, xi) -> w[j] * (xi + j)),
                                   (C.SCALED_PREPARED, (j, xi) -> w[j] * (xi + j)))
                @test kernel(w, x) == [sum(term(j, xi) for j in eachindex(w); init = 0.0)
                                       for xi in x]
                # The generated native body, whose cells call the companions.
                native = kernel.f isa RK._ArrayFunction ? kernel.f.native : kernel.f
                body = RK._native_generated_function(native)
                @test report(findings(body, (typeof(kernel.ops), argtypes...); recurse))
            end
            @test report(findings(prepare(C.FOLDS), argtypes; recurse))
            # A kernel prepared while the image was built holds its body behind
            # the warm wrapper; once the image is loaded the wrapper's method
            # query folds, so no latest-world branch stays at the call.
            @test report(findings(C.SCALED_PREPARED, argtypes; recurse))
            # A rebinding assembles its kernel from the bound template's
            # execution copy of its table, kept once per process (snag
            # per-binding-resi-990b18ab). A template the image kept has none:
            # its companions were warm-wrapped while it was generated, so
            # the first rebinding after load executes the table, as every
            # rebinding did before.
            @test !C.KEPT_EXECUTABLE
            w2 = 2 .* w
            xs = x[eachindex(w)]
            rebound = prepare(C.DOT; bound = (; w = w2))
            @test rebound.f === C.BOUND_DOT.f
            @test rebound.ops === RK._execution_source(rebound.ops)
            template = only(t for t in values(only(values(
                RK._graph_preparations(C.DOT.graph).bound)).templates)
                if t isa RK._BoundTemplate)
            @test template.executable.entries[] isa Vector{Any}
            @test rebound(xs) == sum(w2[j] * xs[j] for j in eachindex(w2); init = 0.0)
            again = prepare(C.DOT; bound = (; w))
            @test again.f === C.BOUND_DOT.f && typeof(again) === typeof(rebound)
            @test again(xs) == prepare(C.DOT)(w, xs)
            @test (w, x) == saved
            end
            println("PRECOMPILED_SOURCE_COMPANIONS_PASS")
            """
        @test success(pipeline(
            `$julia --startup-file=no --check-bounds=yes --project=$env -e $probe`;
            stdout = stdout, stderr = stderr))
    end
end
