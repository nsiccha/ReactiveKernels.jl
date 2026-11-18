using DifferentiationInterface
import Enzyme
using ReactiveKernels
using Test

# `ad_cost_profile`: statement scopes in compiled programs, and a profile of
# a prepared program's value and native-Reverse gradient split by statement.

@testset "statement scopes in compiled programs" begin
    RK = ReactiveKernels
    # A scope around each statement that assigns; binding lists, iteration
    # specifications and signatures stay as they are, and a block's last
    # statement, whose value the block returns, stays unwrapped.
    ast = :(function (a, b)
        x = a + 1
        y = let z = x
            w = z * 2
            w + b
        end
        for i in 1:3, j in 1:2
            out[i] = i + j
            print(i)
        end
        var"##plate_term#12" = x
        (p, q) = (x, y)
        r = p + q
    end)
    marked = RK._source_markers(ast)
    names(ex) = (found = String[]; walk(x) = (RK._is_source_scope(x) &&
        push!(found, String(x.args[2])[4:end]); x isa Expr && foreach(walk, x.args));
        walk(ex); found)
    @test names(marked) == ["(other statements)", "x", "y", "w", "out", "plate_term", "p, q"]
    @test marked.args[1] == ast.args[1]
    # Marking is idempotent, and stripping restores the expression.
    @test RK._source_markers(marked) == marked
    @test RK._strip_source_markers(marked) == ast
    # A scoped program computes what the program computes.
    out = zeros(3)
    scoped = Core.eval(@__MODULE__, RK._source_markers(:((a, b, out) -> begin
        x = a + 1
        y = let z = x
            w = z * 2
            w + b
        end
        for i in 1:3, j in 1:2
            out[i] = i + j
        end
        r = y + 1
    end)))
    @test Base.invokelatest(scoped, 1, 2, out) == 7
    @test out == [3.0, 4.0, 5.0]

    # A prepared kernel's compiled program carries scopes; its displayed
    # code and the source the package reads back do not.
    spec = @kernel marked_kernel(theta, x) = begin
        mu = theta[1] .+ theta[2] .* x
        total = sum(abs2, mu)
    end
    k = prepare(spec; have = (:theta, :x), want = :total)
    native = RK._native_generated_function(k.f isa RK._ArrayFunction ? k.f.native : k.f)
    @test RK._has_source_markers(RK.RuntimeGeneratedFunctions.get_expression(native))
    @test !RK._has_source_markers(RK._program_expression(native))
    @test !RK._has_source_markers(code_expr(k))
    @test k([0.5, 2.0], [1.0, -1.0, 3.0]) == sum(abs2, 0.5 .+ 2.0 .* [1.0, -1.0, 3.0])
end

@testset "sample labels" begin
    RK = ReactiveKernels
    frame(func, file) = Base.StackTraces.StackFrame(Symbol(func), Symbol(file), 1)
    @test RK._sample_label([frame("exp", "special/exp.jl"), frame("macro expansion", "./rk:plate_heavy"),
                            frame("macro expansion", "./rk:total")]) == "plate_heavy"
    @test RK._sample_label([frame("jl_gc_collect", "gc.c"), frame("f", "x.jl")]) ==
        "(garbage collection)"
    @test RK._sample_label([frame("f", "x.jl")]) == "(outside generated code)"
end

@testset "samples spent managing memory, collecting and in called functions" begin
    RK = ReactiveKernels
    frame(func, file; c = false) =
        Base.StackTraces.StackFrame(Symbol(func), Symbol(file), 1, nothing, c, false, UInt64(0))
    marker(name) = frame("macro expansion", "./rk:" * name)
    # A collection counts as garbage collection even inside a statement: which
    # statement's allocation starts one is not the statement's cost.
    @test RK._sample_info([frame("gc_sweep_page", "gc.c"; c = true),
                           frame("_jl_gc_collect", "gc.c"; c = true),
                           frame("jl_gc_pool_alloc_inner", "gc.c"; c = true), marker("mu")]) ==
        ("(garbage collection)", false, nothing)
    # Allocation alone is not a collection; it is the statement's memory time.
    @test RK._sample_info([frame("jl_gc_pool_alloc_inner", "gc.c"; c = true),
                           frame("ijl_alloc_array_1d", "array.c"; c = true),
                           frame("Array", "boot.jl"), marker("mu")]) == ("mu", true, nothing)
    # The C library's allocator, named or (without its symbols) not.
    @test RK._sample_info([frame("_int_malloc", "/lib/libc.so.6"; c = true),
                           marker("(other statements)")]) == ("(other statements)", true, nothing)
    @test RK._sample_info([frame("", "/lib/x86_64-linux-gnu/libc.so.6"; c = true),
                           frame("", "/lib/x86_64-linux-gnu/libc.so.6"; c = true),
                           marker("inbounds_value")]) == ("inbounds_value", true, nothing)
    @test RK._sample_info([frame("__libc_free", "libc.so.6"; c = true), frame("f", "/x/f.jl")]) ==
        ("(outside generated code)", true, nothing)
    # A named C-library function that does not manage memory, and arithmetic,
    # are not memory time.
    @test RK._sample_info([frame("clock_gettime", "/lib/libc.so.6"; c = true), marker("mu")]) ==
        ("mu", false, nothing)
    @test RK._sample_info([frame("exp", "./special/exp.jl"), marker("mu")]) == ("mu", false, nothing)
    # The statement's outermost call into a function neither Julia's nor RK's.
    @test RK._sample_info([frame("fill!", "./array.jl"),
                           frame("step_propagator", "/pkg/PK/src/pk.jl"),
                           frame("jcall", "/pkg/Blocks/src/functions.jl"),
                           frame("pk_propagator", "/pkg/PK/src/pk.jl"),
                           marker("propagators"), marker("(other statements)")]) ==
        ("propagators", false, "pk_propagator (pk.jl)")
    # RK's own source and extensions are not callees; what they call is.
    @test RK._sample_info([frame("model_helper", "/pkg/Model/src/model.jl"),
                           frame("cache_fill!", joinpath(dirname(RK._RK_SOURCE), "ext",
                                                         "ReactiveKernelsMutatingFunctionsExt.jl")),
                           marker("step")]) == ("step", false, "model_helper (model.jl)")
    # The display names a callee only when it holds at least half of the part.
    callee(name, g) = (; name, primal_seconds = 0.0, gradient_seconds = g)
    part(callees) = RK.ADCostPart("p", 1.0, 1.0, 1, 1, 0.0, 0.0, 0.0, 0.0, callees)
    @test RK._cost_callee(part([callee("f (f.jl)", 0.91)])) == "f (f.jl) 91%"
    @test RK._cost_callee(part([callee("f (f.jl)", 0.02)])) == ""
    @test RK._cost_callee(part(eltype(part([]).callees)[])) == ""
    @test RK._sample_info([frame("_native_mul", joinpath(RK._RK_SOURCE, "core.jl")),
                           frame("gemm!", "/opt/julia/share/julia/stdlib/v1.10/LinearAlgebra/src/blas.jl"),
                           marker("mu")]) == ("mu", false, nothing)
end

# A model helper that allocates, outside RK, so its part names it.
cost_profile_helper(m) = (v = fill(m, 32); sum(abs2, v) / 32)

@testset "ad_cost_profile names memory time and the helper a statement calls" begin
    spec = @kernel cost_profiled_helper(theta, x) = begin
        mu = theta[1] .+ theta[2] .* x
        @plate for i in eachindex(mu)
            term[i] = cost_profile_helper(mu[i])
        end
        total = sum(term)
    end
    x = [sin(0.37i) for i in 1:2_000]
    theta = [0.1, 0.5]
    k = prepare(spec; have = (:theta, :x), want = :total)
    ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), theta, x; active = :theta)
    value, gradient = ad_value_and_gradient(ad, theta, x)
    p = ad_cost_profile(ad, theta, x; seconds = 0.5)
    @test p.profiled
    @test sum(part.gradient_seconds for part in p.parts) ≈ p.gradient_seconds
    @test all(part -> 0 <= part.gradient_memory_seconds <= part.gradient_seconds * (1 + 1e-12) &&
                      0 <= part.primal_memory_seconds <= part.primal_seconds * (1 + 1e-12), p.parts)
    # The statement that calls the helper names it as where its time goes.
    calling = filter(part -> !isempty(part.callees) &&
                             startswith(first(part.callees).name, "cost_profile_helper ("), p.parts)
    @test !isempty(calling)
    # Each cell allocates a vector, in the value and in the gradient.
    @test sum(part.primal_memory_seconds for part in calling) > 0
    @test sum(part.primal_memory_seconds for part in p.parts) > 0
    shown = sprint(show, MIME"text/plain"(), p)
    @test occursin("memory management: gradient", shown)
    @test occursin("cost_profile_helper", shown)
    collection = filter(part -> part.name == "(garbage collection)", p.parts)
    @test all(part -> part.primal_memory_seconds == 0 && part.gradient_memory_seconds == 0, collection)
    @test ad_value_and_gradient(ad, theta, x) == (value, gradient)
end

@testset "ad_cost_profile splits the gradient by statement" begin
    spec = @kernel cost_profiled(theta, x, y) = begin
        a = theta[1]
        b = theta[2]
        s = exp(theta[3])
        mu = a .+ b .* x
        @plate for i in eachindex(y, mu)
            heavy = log1p(exp(mu[i])) * sin(mu[i]) * cos(mu[i]) * atan(mu[i]) *
                    log1p(exp(-mu[i]))
            term[i] = -0.5 * ((y[i] - mu[i] - 0.01 * heavy) / s)^2 - log(s)
        end
        total = sum(term)
    end
    n = 20_000
    x = [sin(0.37i) for i in 1:n]
    y = [0.3 + 0.7x[i] + 0.2cos(1.3i) for i in 1:n]
    theta = [0.1, 0.5, -0.3]
    k = prepare(spec; have = (:theta, :x, :y), want = :total)
    ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), theta, x, y; active = :theta)
    value, gradient = ad_value_and_gradient(ad, theta, x, y)
    p = ad_cost_profile(ad, theta, x, y; seconds = 0.5)
    @test p isa ADCostProfile
    @test p.profiled
    @test p.ratio ≈ p.gradient_seconds / p.primal_seconds
    @test p.gradient_samples > 100 && p.primal_samples > 100
    @test issorted(p.parts; by = part -> part.gradient_seconds, rev = true)
    # The parts partition the whole.
    @test sum(part.gradient_seconds for part in p.parts) ≈ p.gradient_seconds
    @test sum(part.primal_seconds for part in p.parts) ≈ p.primal_seconds
    @test sum(part.gradient_samples for part in p.parts) == p.gradient_samples
    # Five transcendental calls per cell dominate both passes.
    @test first(p.parts).name == "plate_heavy"
    heavy = only(filter(part -> part.name == "plate_heavy", p.parts))
    @test heavy.primal_seconds > 0.3 * p.primal_seconds
    @test all(part -> part.gradient_bytes >= 0 && part.primal_bytes >= 0, p.parts)
    @test occursin("plate_heavy", sprint(show, MIME"text/plain"(), p))
    # Profiling does not change the program.
    @test ad_value_and_gradient(ad, theta, x, y) == (value, gradient)

    # Below the threshold nothing is profiled.
    gated = ad_cost_profile(ad, theta, x, y; seconds = 0.2, threshold = 1.0e6)
    @test !gated.profiled
    @test isempty(gated.parts)
    @test occursin("not profiled", sprint(show, MIME"text/plain"(), gated))
    @test_throws ArgumentError ad_cost_profile(ad, theta, x, y; seconds = 0)
end
