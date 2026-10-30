module ExecutionSourceTests
using ReactiveKernels, Enzyme, Test
const RK = ReactiveKernels

@testset "equivalent graphs disambiguate locals independently of value IDs" begin
    kernels = map((0, 7)) do padding
        graph = Graph()
        for i in 1:padding
            value!(graph, Symbol(:unused, i), Float64)
        end
        ports = map(name -> value!(graph, name, Float64), (:a, :a, :a_2, :__ops__))
        output = value!(graph, :a, Float64)
        add!(graph, ports => output, (x, y, z, w) -> 1000x + 100y + 10z + w)
        prepare(graph; have=ports, want=output)
    end
    @test code_expr(kernels[1]) == code_expr(kernels[2])
    @test typeof(kernels[1].f) === typeof(kernels[2].f)
    # The generated-function cache retains the body; the callable is its type.
    @test Base.issingletontype(typeof(kernels[1].f))
    for kernel in kernels
        @test kernel(1.0, 2.0, 3.0, 4.0) == 1234.0
        @test kernel(4.0, 3.0, 2.0, 1.0) == 4321.0
    end
end

@testset "execution shares code while preserving definition provenance" begin
    mod = Module(gensym(:ExecutionSource))
    Core.eval(mod, :(using ReactiveKernels; shift(x) = x + 1.0))
    definition = :(model(x::Float64) = begin
        value::Float64 = shift(x) + 2.0
        return value
    end)
    specs = [RK._kernel_eval_definition(mod, deepcopy(definition)) for _ in 1:2]
    kernels = [prepare(spec) for spec in specs]
    @test typeof(kernels[1].f) === typeof(kernels[2].f)
    @test typeof(kernels[1].ops) === typeof(kernels[2].ops)
    @test Base.invokelatest(kernels[1], 2.0) == Base.invokelatest(kernels[2], 2.0) == 5.0
    source = [only(filter(r -> r.op isa RK._KernelSourceOp, spec.graph.recipes)) for spec in specs]
    @test RK._kernel_source_token(source[1].op) !== RK._kernel_source_token(source[2].op)
    # The differentiated callable holds only executable state: no token,
    # parameter names or module value that Enzyme must prove read-only.
    @test isbitstype(typeof(kernels[1].ops))
    for (kernel, original) in zip(kernels, source),
            (op, recipe) in zip(kernel.ops, kernel.lowered_recipes)
        op isa RK._KernelSourceOp || continue
        @test RK.kernel_sourceop_token(op) === nothing
        @test RK.kernel_sourceop_token(recipe.op) === RK._kernel_source_token(original.op)
        @test RK._kernel_endpoint_source_module(recipe.op) === mod
        # refused: an execution copy's type names no authoring module, or each
        # lowering module would make one program a new executable (snag
        # `compiled-reader-be3cadbb`); lowering reads the original operation.
        @test_throws ArgumentError RK._kernel_endpoint_source_module(op)
        @test RK._source_parameter_names(recipe.op, recipe.source, length(recipe.inputs)) !==
            nothing
        # refused: keying a token-free copy would merge separately authored
        # definitions in graph CSE; recipes must hold the original operation.
        @test_throws ArgumentError RK._kernel_source_token(op)
    end
    # Generated-source reuse must still observe ordinary Julia method changes.
    Core.eval(mod, :(shift(x) = 3.0x))
    newer = prepare(RK._kernel_eval_definition(mod, deepcopy(definition)))
    @test Base.invokelatest(newer, 2.0) == 8.0
    @test Base.invokelatest(kernels[1], 2.0) == 8.0
end

@testset "vectorized readers of fresh identical builds share executable types" begin
    # A rebuilt graph's position readers hold the token-free execution copies
    # `prepare` holds, so a rebuild of one program reuses the reader compiled
    # for the first build (snag `compiled-reader-be3cadbb`).
    mod = Module(gensym(:ExecutionSource))
    Core.eval(mod, :(using ReactiveKernels; shift(x) = x + 1.0))
    definition = :(model(position, shared::Float64) = begin
        invariant::Float64 = 2.0 * shared
        result = shift(position) * invariant
        return result
    end)
    specs = [RK._kernel_eval_definition(mod, deepcopy(definition)) for _ in 1:2]
    kernels = [prepare(spec) for spec in specs]
    positions = [1.0, 2.0, 3.0]
    expected = (positions .+ 1.0) .* 3.0
    schedule = NativeScheduling(; chunk_size = 2)
    for options in ((; reuse = false), (; reuse = true), (; reuse = true, schedule))
        readers = [Base.invokelatest(vectorize, kernel; batched = :position, options...)
                   for kernel in kernels]
        @test typeof(readers[1]) === typeof(readers[2])
        for reader in readers
            @test Base.invokelatest(reader, positions, 1.5) == expected
        end
    end
    # Each reader's lowered recipes still name its own definition.
    source = [first(filter(r -> r.op isa RK._KernelSourceOp, kernel.lowered_recipes))
              for kernel in kernels]
    @test RK.kernel_sourceop_token(source[1].op) !== RK.kernel_sourceop_token(source[2].op)
end

@testset "reader construction compiles once, not once per program" begin
    # The type-stability check and the position-reader constructors read only
    # a kernel's plan and fields; specialized on a large program's kernel type
    # they took a third of its first read (snag `compiled-reader-be3cadbb`).
    function reader(scale)
        graph = Graph()
        x = value!(graph, :x, Float64)
        y = value!(graph, :y, Float64)
        add!(graph, (x,) => y, v -> scale * v + Float64(length(string(scale))))
        vectorize(prepare(graph; have = (x,), want = y); batched = :x, reuse = true)
    end
    count_specializations(f) = sum(m -> count(!isnothing, Base.specializations(m)), methods(f))
    copy_method = which(copy, (RK.BorrowedBatchedKernel,))
    count_specializations(::typeof(copy)) = count(!isnothing, Base.specializations(copy_method))
    readers = [reader(2.0), reader(3)]
    @test typeof(readers[1]) !== typeof(readers[2])
    @test readers[1]([1.0, 2.0]) == [5.0, 7.0]
    @test readers[2]([1.0, 2.0]) == [4.0, 7.0]
    copies = map(copy, readers)
    @test copies[1]([1.0, 2.0]) == [5.0, 7.0]
    before = map(count_specializations, (RK._check_uninferred_values, RK._replica_graph,
                                         RK._borrowed_batch, copy))
    readers = [reader(4.0f0), reader(5//1)]
    copies = map(copy, readers)
    @test copies[2]([1.0, 2.0]) == [9.0, 14.0]
    @test map(count_specializations, (RK._check_uninferred_values, RK._replica_graph,
                                      RK._borrowed_batch, copy)) == before
end

@testset "generated code orders values by declaration, not by identity" begin
    # A value's identity orders by the key of the namespace that allocated it,
    # which differs in every process: values a package image made and values
    # made at run time sorted one way while an image was generated and another
    # at run time, so the image's compiled reader did not apply (snag
    # `compiled-reader-be3cadbb`). Generated code takes the plan's order.
    function in_namespace(f, key)
        previous = RK._VALUE_NAMESPACE[]
        RK._VALUE_NAMESPACE[] = RK.ValueNamespace(UInt(key))
        try
            f()
        finally
            RK._VALUE_NAMESPACE[] = previous
        end
    end
    graph = Graph()
    a = in_namespace(() -> value!(graph, :a, Float64), 2)
    b = in_namespace(() -> value!(graph, :b, Float64), 1)
    c = value!(graph, :c, Float64)
    add!(graph, (a, b) => c, +)
    p = plan(graph; have = (a, b), want = c)
    ids = [RK.canon_id(graph, v.id) for v in (c, b, a)]
    @test sort(ids) != ids[[3, 2, 1]]
    @test RK._declaration_order(p, ids) == ids[[3, 2, 1]]
    @test_throws ArgumentError RK._declaration_order(p, [RK.canon_id(graph, value!(graph, :d, Float64).id)])
end

@testset "generated bodies name globals by their owners" begin
    # A lowering module (a program generator's, or an image's scope module
    # standing in for it) qualifies the globals a body reads; naming them by the
    # bindings they resolve to gives one body in every lowering and process
    # (snag `compiled-reader-be3cadbb`).
    named = Module(:LoweringModule)
    Core.eval(named, :(const data = [1.0]; helper(x) = x))
    scope = RK._kernel_scope_module(Module(gensym(:Image)), named,
        :(reshape(view(data, :), 1) .* Float64(helper(1))))
    for (name, owner) in ((:reshape, Base), (:view, Base), (:(:), Base), (:Float64, Core))
        @test RK._canonical_global(GlobalRef(named, name)) == GlobalRef(owner, name)
        @test RK._canonical_global(GlobalRef(scope, name)) == GlobalRef(owner, name)
    end
    for name in (:data, :helper)
        @test RK._canonical_global(GlobalRef(named, name)) == GlobalRef(named, name)
        @test RK._canonical_global(GlobalRef(scope, name)) == GlobalRef(named, name)
    end
end

@testset "a name imported under another name is named by its owner's binding" begin
    # `import X: f as g` binds `g` to X's `f`; X's own `g`, if any, is another
    # binding. RKPPL's generated module imports `cexpexp as _ppl_cexpexp`, and
    # its bodies read `LogExpFunctions._ppl_cexpexp`, which is undefined.
    parent = Module(gensym(:Aliases))
    Core.eval(parent, Expr(:toplevel, (quote
        module A
            f(x) = x
            g(x) = 2x
            const c = 1
            v = [1.0]
        end
        module M
            import ..A: f as g, f as h, c as d, v as w
            using ..A: f as u
        end
    end).args...))
    A, M = (Base.invokelatest(getglobal, parent, n) for n in (:A, :M))
    f = Base.invokelatest(getglobal, A, :f)
    for name in (:g, :h, :u)
        @test RK._canonical_global(GlobalRef(M, name)) == GlobalRef(A, :f)
        @test RK._global_value(M, name) === Some(f)
        @test RK._global_isconst(M, name)
    end
    for name in (:d, :w)
        ref = RK._canonical_global(GlobalRef(M, name))
        @test Base.invokelatest(getglobal, ref.mod, ref.name) ===
            Base.invokelatest(getglobal, M, name)
    end
end

@testset "lowering reads a module's names as Julia resolves them, binding none" begin
    # Before Julia 1.12 the first read of a name a module has only from a
    # `using` binds it to that import for good, and the module can then not
    # define the name (snag `docs-distributio-30373f1f`). Each reader answers
    # what `Base.binding_module` answers in a twin module that binds the name.
    function readers(usings)
        parent = Module(gensym(:Bindings))
        Core.eval(parent, Expr(:toplevel, (quote
            module A
                export f, c, shared, unset
                f(x) = x
                const c = 1
                const shared = 2
            end
            module B
                export c, shared, g
                const c = 3
                const shared = 2
                g = 4
            end
            module Reexport
                using ..A
                export f
            end
            module X
                export h
            end
            module Y
                using ..X
                export h
            end
        end).args...))
        latest(name) = Base.invokelatest(getglobal, parent, name)
        Core.eval(latest(:X), :(using ..Y))
        twins = map((:Reader, :Twin)) do name
            Core.eval(parent, Meta.parse(
                "module $name; using " * join(("..$u" for u in usings), ", ") * "; end"))
        end
        latest(:Reexport), twins...
    end
    # What `name` denotes in `m`, binding it.
    resolve(m, name) = (Base.binding_module(m, name), isdefined(m, name) ?
        Some(getglobal(m, name)) : nothing, isdefined(m, name) && isconst(m, name))
    cases = [
        ((:A,), (:f, :c, :shared, :unset, :g, :absent)),
        ((:A, :B), (:c, :f, :g)),              # `c` is ambiguous
        ((:Reexport,), (:f,)),                 # an export Reexport has only from a `using`
        ((:Reexport, :A), (:f,)),
        ((:X,), (:h,)),                        # imports that cycle without a definition
    ]
    # One constant two imports export; Julia 1.12 names no owner for it.
    VERSION < v"1.12.0-" && push!(cases, ((:A, :B), (:shared,)), ((:B, :A), (:shared,)))
    for (usings, names) in cases, name in names
        reexport, reader, twin = readers(usings)
        owner = RK._binding_owner(reader, name)
        value = RK._global_value(reader, name)
        isconstant = RK._global_isconst(reader, name)
        @static if VERSION < v"1.12.0-"
            @test !Base.isbindingresolved(reader, name)
            @test !Base.isbindingresolved(reexport, :f)
        end
        resolved = redirect_stderr(() -> Base.invokelatest(resolve, twin, name), devnull)
        # A name that denotes no binding answers the module itself.
        @test (owner === reader ? twin : owner) === resolved[1]
        @test value === resolved[2]
        @test isconstant === resolved[3]
        Core.eval(reader, :($name = :own))
        @test RK._global_value(reader, name) === Some(:own)
    end
end

@testset "a constant two imports export as one value reads it" begin
    # Two `using`s export distinct constant bindings of one value (one
    # object: Julia compares the stored values by identity, and a small `Int`
    # is one cached box). Julia 1.10 and 1.11 name the oldest import its
    # owner; from 1.12 `Base.binding_module` throws "Constant binding was
    # imported from multiple modules". A body reading the name reads its
    # value either way.
    parent = Module(gensym(:SharedConstants))
    Core.eval(parent, Expr(:toplevel, (quote
        module A
            export shared
            const shared = 2
        end
        module B
            export shared
            const shared = 2
        end
        module R
            using ..A, ..B
        end
    end).args...))
    R = Base.invokelatest(getglobal, parent, :R)
    Core.eval(R, :(using ReactiveKernels))
    @test RK._global_value(R, :shared) === Some(2)
    @test RK._global_isconst(R, :shared)
    ref = RK._canonical_global(GlobalRef(R, :shared))
    @test Base.invokelatest(getglobal, ref.mod, ref.name) === 2
    Core.eval(R, :(@kernel scaled(x) = begin
        value = shared .* x
        return value
    end))
    kernel = Core.eval(R, :(prepare(scaled; have = (:x,), want = :value)))
    @test Base.invokelatest(kernel, [1.5, -0.5]) == [3.0, -1.0]
end

@testset "a kernel another module lowered leaves this module's names free" begin
    # A docs page prepared the uniform example after another module had lowered
    # the same kernel; comparing the two modules' bindings for its branch
    # closure, which has a parameter `lower`, bound the page's `lower` to the
    # exported `ReactiveKernels.lower`, and `lower = -1.0` then failed (snag
    # `docs-distributio-30373f1f`). The closure is still shared.
    source = :(@kernel bounded(lower::Float64, upper::Float64) = begin
        ccdf(x::Float64)::Float64 =
            x < lower ? 1.0 : (x >= upper ? 0.0 : (upper - x) / (upper - lower))
    end)
    modules = [Module(gensym(:BoundedPage)) for _ in 1:2]
    kernels = map(modules) do mod
        Core.eval(mod, :(using ReactiveKernels))
        Core.eval(mod, deepcopy(source))
        Core.eval(mod, :(prepare(bounded.ccdf; have = (:x, :lower, :upper), want = :ccdf)))
    end
    @test typeof(kernels[1].ops) === typeof(kernels[2].ops)
    for mod in modules
        Core.eval(mod, :(lower = -1.0))
        @test Base.invokelatest(getglobal, mod, :lower) === -1.0
    end
    for kernel in kernels
        @test Base.invokelatest(kernel, 0.5, -1.0, 2.0) == 0.5
        @test Base.invokelatest(kernel, -2.0, -1.0, 2.0) == 1.0
        @test Base.invokelatest(kernel, 3.0, -1.0, 2.0) == 0.0
    end
end

@testset "execution branch metadata preserves host branch behavior" begin
    call(x, scale) = scale > 0 ? x * log(scale) : -x
    branch = RK._KernelBranch(Val((2,)), Val((1, 2)), Val((1,)), call,
        scale -> scale > 0, (x, scale) -> x * log(scale), x -> -x)
    source = RK._KernelSourceOp(Val(gensym(:branch)), Val(:fused), branch)
    op = RK._execution_source(source)
    @test RK.kernel_sourceop_token(op) === nothing
    @test op.f isa RK._KernelBranch
    @test [op(x, 2.0) for x in (2.0, 3.0)] ≈ [2.0, 3.0] .* log(2.0)
    @test [op(x, -2.0) for x in (2.0, 3.0)] == [-2.0, -3.0]
end

@testset "runtime source conversion preserves lexical captures" begin
    function make(offset)
        @kernel captured(x::Float64) = begin
            y::Float64 = x + offset
            return y
        end
    end
    a, b = prepare(make(1.0)), prepare(make(4.0))
    @test typeof(a.ops) === typeof(b.ops)
    @test a(2.0) == 3.0
    @test b(2.0) == 6.0
    function make_ref(offset)
        @kernel captured_ref(x::Float64) = begin
            y::Float64 = x + offset[]
            return y
        end
    end
    offset = Ref(1.0)
    live = prepare(make_ref(offset))
    @test live(2.0) == 3.0
    offset[] = 4.0
    @test live(2.0) == 6.0
    @test only(only(Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(live),
        Enzyme.Active, Enzyme.Active(2.0)))) == 1.0
end
end
