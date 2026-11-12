using ReactiveKernels, Test

# A partition plan whose row bookkeeping sits behind a label-only guard caches
# it in a comprehension (`_partition_guarded_plan`). In a compiled body that
# was a closure, which `RuntimeGeneratedFunctions` makes an opaque closure, and
# Julia refuses to expand a generated function holding one while a package
# image is generated: a downstream workload could neither type nor compute
# such a plan ("Impossible to correctly handle OpaqueClosure inside @generated
# returned during precompile process.", snag `partition-plan-o-1d9db0fc`).
# `_compile` lifts the comprehension into a program of its own. A consumer
# plans a guarded body at top level and inside one compiling call (where the
# compiled planner is warm-wrapped) while its image is generated; a fresh
# process compares those plans with plans computed at run time.
_cc_has_generator(x) = x isa Expr && x.head !== :quote &&
    (x.head === :generator || any(_cc_has_generator, x.args))
_cc_lifted(f) = !_cc_has_generator(ReactiveKernels._program_expression(
    ReactiveKernels._native_generated_function(f)))

@testset "compiled comprehensions run as programs of their own" begin
    compile = ReactiveKernels.compile
    # Lifted: the element reads the body's locals and parameters.
    f = compile(:(function (a, n)
        b = 2a
        xs = [b * i + a for i in 1:n]
        s = sum(b * i for i in 1:n; init = 0.0)
        return xs, s
    end))
    @test _cc_lifted(f)
    @test f(0.5, 3) == ([1.5, 2.5, 3.5], 6.0)
    # No element: the element type is still the inferred one.
    @test f(0.5, 0) == (Float64[], 0.0) && f(0.5, 0)[1] isa Vector{Float64}
    # A `let` binding hides a parameter of the same name in its body only.
    g = compile(:(function (a, n)
        [let a = a + i
            a * 2
        end for i in 1:n]
    end))
    @test _cc_lifted(g)
    @test g(1, 2) == [4, 6]
    # Wide: the element reads more locals than a program call takes operands,
    # as a wide model's guarded plan does (109 and 175 in one real plan,
    # snag `partition-plan-o-f7bf3fd9`). They travel as one tuple.
    width = ReactiveKernels._PLATE_CALL_MAX_ARGUMENTS + 6
    ks = [Symbol(:k, j) for j in 1:width]
    w = compile(Expr(:function, Expr(:tuple, :a, :n), Expr(:block,
        (:($k = a + $j) for (j, k) in enumerate(ks))...,
        :(xs = [i * $(Expr(:call, :+, ks...)) for i in 1:n]),
        :(s = sum(i - $(ks[end]) for i in 1:n; init = 0.0)),
        :(return xs, s))))
    @test _cc_lifted(w)
    packed(x) = x === GlobalRef(ReactiveKernels, :_PackedComprehensionElement) ||
        x isa Expr && any(packed, x.args)
    @test packed(ReactiveKernels._program_expression(
        ReactiveKernels._native_generated_function(w)))
    total = sum(1:width)
    @test w(0, 3) == ([total, 2total, 3total], sum(i - width for i in 1:3; init = 0.0))
    @test w(0.5, 2) == ([total + width / 2, 2total + width], 3 - 2(width + 0.5))
    @test w(0, 0) == (Int[], 0.0) && w(0, 0)[1] isa Vector{Int}
    # Left as written: an element assigning a local of the body, a nested
    # `for` (which `RuntimeGeneratedFunctions` itself cannot compile), a
    # typed comprehension and a local function's body.
    for (ast, expected) in (
            (:(function (n)
                last = 0
                xs = [(last = i; 2i) for i in 1:n]
                return xs, last
            end), ([2, 4, 6], 3)),
            (:(function (n) [i + j for i in 1:n for j in 1:i] end), nothing),
            (:(function (n) Float64[i for i in 1:n] end), [1.0, 2.0, 3.0]),
            (:(function (n)
                h(k) = [k * i for i in 1:n]
                return h(2)
            end), [2, 4, 6]))
        @test ReactiveKernels._lift_comprehensions(ast) == ast
        expected === nothing || @test compile(ast)(3) == expected
    end
end

@testset "a guarded partition plan is computed while an image is generated" begin
    root = dirname(dirname(pathof(ReactiveKernels)))
    mktempdir() do env
        consumer = joinpath(env, "GuardedPlanConsumer")
        mkpath(joinpath(consumer, "src"))
        write(joinpath(consumer, "Project.toml"), """
            name = "GuardedPlanConsumer"
            uuid = "8d2f4b6a-1c3e-4f5a-9b7d-2e4f6a8c0b1d"
            version = "0.1.0"
            [deps]
            ReactiveKernels = "78e9f072-d36b-4c73-b7a3-751b0eb26cf9"
            """)
        write(joinpath(consumer, "src", "GuardedPlanConsumer.jl"), """
            module GuardedPlanConsumer
            using ReactiveKernels
            const RK = ReactiveKernels
            const WAS_PRECOMPILED = ccall(:jl_generating_output, Cint, ()) != 0
            # Bookkeeping that only a label with rows computes, read lexically.
            const BODY = quote
                rows = collect(1:label)
                if !isempty(rows)
                    selected = let r = rows
                        r[[1]]
                    end
                    y = u * sum(selected)
                else
                    y = zero(u)
                end
                return y
            end
            # The same bookkeeping reading more locals than a program call
            # takes operands, as a wide model's guarded plan does.
            const KS = [Symbol(:k, j) for j in 1:RK._PLATE_CALL_MAX_ARGUMENTS + 6]
            const WIDE_BODY = Expr(:block, :(rows = collect(1:label)),
                (:(\$k = label + \$j) for (j, k) in enumerate(KS))...,
                :(if !isempty(rows)
                      s = \$(Expr(:call, :+, KS...)) + rows[end]
                      y = u * s
                  else
                      y = zero(u)
                  end),
                :(return y))
            planner_ast(split) = Expr(:function, Expr(:tuple, :u, :label),
                Expr(:block, split[1]..., Expr(:return, Expr(:tuple, split[3]...))))
            planned_ast(split) = Expr(:function, Expr(:tuple, :u, :label, :p),
                Expr(:block, (Expr(:(=), name, :(p[\$i])) for (i, name) in
                              enumerate(split[3]))..., split[2]...))
            const SPLIT = RK._partition_plan_split(BODY, [:u], Any[])
            const PLANNER_AST = planner_ast(SPLIT)
            const WIDE_SPLIT = RK._partition_plan_split(WIDE_BODY, [:u], Any[])
            const WIDE_PLANNER_AST = planner_ast(WIDE_SPLIT)
            # The image's first compiles, called in the same call, before their
            # generated methods are visible: the planners are warm-wrapped.
            function warm_plans()
                planner = RK.compile(PLANNER_AST)
                wide = RK.compile(WIDE_PLANNER_AST)
                planner isa RK._PrecompileWarmFunction && wide isa RK._PrecompileWarmFunction,
                    (planner(-2.0, 0), planner(-2.0, 4)), (wide(-2.0, 0), wide(-2.0, 4))
            end
            const WARM, WARM_PLANS, WARM_WIDE_PLANS = warm_plans()
            const PLANNER = RK.compile(PLANNER_AST)
            const PLANS = (PLANNER(-2.0, 0), PLANNER(-2.0, 4))
            const WIDE_PLANNER = RK.compile(WIDE_PLANNER_AST)
            const WIDE_PLANS = (WIDE_PLANNER(-2.0, 0), WIDE_PLANNER(-2.0, 4))
            # The bodies reading those plans, and the unplanned bodies.
            const PLANNED = RK.compile(planned_ast(SPLIT))
            const PLAIN = RK.compile(Expr(:function, Expr(:tuple, :u, :label), BODY))
            const WIDE_PLANNED = RK.compile(planned_ast(WIDE_SPLIT))
            const WIDE_PLAIN = RK.compile(Expr(:function, Expr(:tuple, :u, :label), WIDE_BODY))
            end
            """)
        setup = """
            using Pkg
            include($(repr(joinpath(root, "deps", "source_dependencies.jl"))))
            # Julia 1.10 does not read the root package's [sources] table.
            Pkg.develop([PackageSpec[PackageSpec(path = path) for path in
                $(repr((root, consumer)))]; rk_source_dependencies()])
            # The fresh process differentiates the image's planned body.
            Pkg.add("Enzyme")
            Pkg.precompile()
            """
        julia = Base.julia_cmd()
        setup_ok = success(pipeline(`$julia --startup-file=no --project=$env -e $setup`;
                                    stdout = stdout, stderr = stderr))
        @test setup_ok
        setup_ok || return
        probe = """
            using Test, ReactiveKernels, Enzyme, GuardedPlanConsumer
            const RK = ReactiveKernels
            const C = GuardedPlanConsumer
            @testset "loaded image" begin
                @test C.WAS_PRECOMPILED
                @test C.WARM
                # The split caches the guarded bookkeeping in a comprehension;
                # the compiled planner holds none.
                has_generator(x) = x isa Expr && x.head !== :quote &&
                    (x.head === :generator || any(has_generator, x.args))
                @test has_generator(Expr(:block, C.SPLIT[1]...))
                @test !has_generator(RK._program_expression(RK._native_generated_function(C.PLANNER)))
                runtime = RK.compile(C.PLANNER_AST)
                T = typeof(C.PLANS[2])
                @test isconcretetype(T)
                for (k, label) in enumerate((0, 4))
                    for plans in (C.PLANS, C.WARM_PLANS)
                        @test isequal(plans[k], runtime(-2.0, label))
                        @test typeof(plans[k]) === T
                    end
                    @test isequal(C.PLANNER(-2.0, label), C.PLANS[k])
                end
                # The cache is empty for a label without rows and holds one
                # entry for a label with rows, of one element type.
                cache(plan) = only(x for x in plan if x isa Vector{<:Tuple})
                @test isempty(cache(C.PLANS[1]))
                @test cache(C.PLANS[2]) == [([1],)]
                # The image's planned body over the image's plans: values and
                # Reverse derivatives of the unplanned body, in a fresh process.
                gradient(f, args...) = first(first(Enzyme.autodiff(Enzyme.Reverse, f,
                    Enzyme.Active, Enzyme.Active(3.0), map(Enzyme.Const, args)...)))
                for (label, plan) in zip((0, 4), C.PLANS)
                    @test C.PLANNED(3.0, label, plan) == C.PLAIN(3.0, label)
                    @test gradient(C.PLANNED, label, plan) == gradient(C.PLAIN, label)
                end
                @test gradient(C.PLANNED, 4, C.PLANS[2]) == 1.0
                # The wide bookkeeping: its element reads more locals than a
                # program call takes operands and travels them as one tuple.
                @test has_generator(Expr(:block, C.WIDE_SPLIT[1]...))
                wide = RK._program_expression(RK._native_generated_function(C.WIDE_PLANNER))
                @test !has_generator(wide)
                packed(x) = x === GlobalRef(RK, :_PackedComprehensionElement) ||
                    x isa Expr && any(packed, x.args)
                @test packed(wide)
                wide_runtime = RK.compile(C.WIDE_PLANNER_AST)
                W = typeof(C.WIDE_PLANS[2])
                @test isconcretetype(W)
                s = sum(1:length(C.KS)) + length(C.KS) * 4 + 4
                for (k, label) in enumerate((0, 4))
                    for plans in (C.WIDE_PLANS, C.WARM_WIDE_PLANS)
                        @test isequal(plans[k], wide_runtime(-2.0, label))
                        @test typeof(plans[k]) === W
                    end
                end
                @test isempty(cache(C.WIDE_PLANS[1]))
                @test cache(C.WIDE_PLANS[2]) == [(s,)]
                for (label, plan) in zip((0, 4), C.WIDE_PLANS)
                    @test C.WIDE_PLANNED(3.0, label, plan) == C.WIDE_PLAIN(3.0, label)
                    @test gradient(C.WIDE_PLANNED, label, plan) == gradient(C.WIDE_PLAIN, label)
                end
                @test gradient(C.WIDE_PLANNED, 4, C.WIDE_PLANS[2]) == s
            end
            println("PRECOMPILED_GUARDED_PLAN_PASS")
            """
        @test success(pipeline(`$julia --startup-file=no --project=$env -e $probe`;
                               stdout = stdout, stderr = stderr))
    end
end
