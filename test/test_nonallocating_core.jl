using ReactiveKernels
using Test

@testset "optional non-allocating preparation interface" begin
    @test Base.get_extension(ReactiveKernels,
                             :ReactiveKernelsMutatingFunctionsExt) === nothing

    g = Graph()
    x = value!(g, :x, Vector{Float64})
    y = value!(g, :y, Vector{Float64})
    add!(g, x => y, copy)
    p = plan(g; have = (x,), want = (y,))

    err = try
        prepare_nonallocating(p)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("optional MutatingFunctions extension", sprint(showerror, err))
    @test occursin("using MutatingFunctions", sprint(showerror, err))

    rerr = try
        prepare_reactive_nonallocating(g; have = (x,), want = (y,))
    catch e
        e
    end
    @test rerr isa ArgumentError
    @test occursin("optional MutatingFunctions extension", sprint(showerror, rerr))
    @test occursin("using MutatingFunctions", sprint(showerror, rerr))
end

# The in-place reactive-getter hook is MutatingFunctions-agnostic: the core only
# needs a `cache_apply(cache, op, args...) -> newcache` callable. This whole
# testset supplies a hand-written one (no MutatingFunctions loaded), which both
# proves the agnosticism and exercises all the generic non-HMC semantics.
@testset "in-place reactive getters via a hand-written cache_apply" begin
    double(v) = 2 .* v
    # conforming hand-written mutating forms: reuse the buffer in place.
    hca(cache::AbstractVector, ::typeof(copy), A) =
        (length(cache) == length(A) || resize!(cache, length(A)); copyto!(cache, A); cache)
    hca(cache::AbstractVector, ::typeof(double), v) =
        (length(cache) == length(v) || resize!(cache, length(v)); cache .= 2 .* v; cache)
    # contract fallback: immutable/isbits/unregistered cache => passthrough recompute.
    hca(cache, op, args...) = op(args...)

    g = Graph()
    x = value!(g, :x, Vector{Float64})
    a = value!(g, :a, Vector{Float64})
    c = value!(g, :c, Vector{Float64})
    add!(g, x => a, copy)
    add!(g, a => c, double)

    pure = prepare_reactive(g; have = (x,), want = (c,))
    inplace = ReactiveKernels._prepare_reactive(g; have = (x,), want = (c,),
                                                cache_apply = hca)

    @testset "default program AST is unchanged; in-place routes through the hook" begin
        pure_ast = string(code_expr(pure, c))
        @test occursin("__slots__", pure_ast)
        @test !occursin("__cache_apply__", pure_ast)
        # Match the in-place CALL form, not the bare name: every getter in an
        # in-place program names `__cache_apply__` in its 4-arg signature.
        ip_ast = string(code_expr(inplace, c))
        @test occursin("__cache_apply__((__slots__", ip_ast)
        @test occursin("isassigned", ip_ast)
    end

    input = [1.0, 2.0, 3.0]

    @testset "parity in-place vs pure over an invalidate->recompute sequence" begin
        sp = pure(copy(input)); si = inplace(copy(input))
        cp = statevalue(pure, c); ch = statevalue(si, c)
        xp = statevalue(sp, x); xi = statevalue(si, x)
        for v in ([1.0, 2.0, 3.0], [5.0, -2.0, 7.0], [0.0, 0.0, 0.0], [10.0, 11.0, 12.0])
            set!(sp, xp, v); set!(si, xi, v)
            @test get!(sp, cp) == get!(si, ch) == 2 .* v
        end
        @test @inferred(get!(si, ch))::Vector{Float64} == [20.0, 22.0, 24.0]
    end

    @testset "steady-state 0 bytes across invalidate->recompute (function barrier)" begin
        si = inplace(copy(input))
        ch = statevalue(si, c); xi = statevalue(si, x)
        function cycle!(state, xh, c,  v, n)
            for _ in 1:n
                set!(state, xh, v)
                get!(state, c)
            end
        end
        buf = get!(si, ch)                 # first touch establishes buffers (may allocate)
        cycle!(si, xi, ch, input, 1)       # warm the steady path
        @test @allocated(cycle!(si, xi, ch, input, 1000)) == 0
        @test get!(si, ch) === buf         # same buffer reused in place, by identity
    end

    @testset "two independent instances do not alias caches" begin
        s1 = inplace([1.0, 2.0, 3.0]); s2 = inplace([4.0, 5.0, 6.0])
        ch = statevalue(s1, c); ah = statevalue(s1, a)
        @test get!(s1, ch) == [2.0, 4.0, 6.0]
        @test get!(s2, ch) == [8.0, 10.0, 12.0]
        @test get!(s1, ch) !== get!(s2, ch)
        @test get!(s1, ah) !== get!(s2, ah)
        # mutating s1's source must not disturb s2
        set!(s1, statevalue(s1, x), [0.0, 0.0, 0.0])
        @test get!(s1, ch) == [0.0, 0.0, 0.0]
        @test get!(s2, ch) == [8.0, 10.0, 12.0]
    end

    @testset "copy / copyto! keep per-instance buffers" begin
        s1 = inplace([1.0, 2.0, 3.0])
        ch = statevalue(s1, c)
        get!(s1, ch)
        s2 = copy(s1)
        @test get!(s2, ch) == get!(s1, ch)
        @test get!(s2, ch) !== get!(s1, ch)       # deep-copied, not aliased
        set!(s2, statevalue(s2, x), [7.0, 8.0, 9.0])
        @test get!(s2, ch) == [14.0, 16.0, 18.0]
        @test get!(s1, ch) == [2.0, 4.0, 6.0]     # source instance untouched

        s3 = inplace([0.0, 0.0, 0.0]); get!(s3, statevalue(s3, c))
        copyto!(s3, s1)
        @test get!(s3, ch) == get!(s1, ch)
        @test get!(s3, ch) !== get!(s1, ch)
    end

    @testset "exception during recompute leaves the slot recomputable" begin
        boom(v) = (any(iszero, v) && error("boom"); 2 .* v)
        hca2(cache, op, args...) = op(args...)
        hca2(cache::AbstractVector, ::typeof(copy), A) =
            (length(cache) == length(A) || resize!(cache, length(A)); copyto!(cache, A); cache)
        gg = Graph()
        xx = value!(gg, :xx, Vector{Float64})
        aa = value!(gg, :aa, Vector{Float64})
        cc = value!(gg, :cc, Vector{Float64})
        add!(gg, xx => aa, copy)
        add!(gg, aa => cc, boom)
        prog = ReactiveKernels._prepare_reactive(gg; have = (xx,), want = (cc,),
                                                 cache_apply = hca2)
        state = prog([0.0, 1.0])
        cch = statevalue(state, cc); xxh = statevalue(state, xx)
        @test_throws ErrorException get!(state, cch)   # throws inside the op
        set!(state, xxh, [3.0, 4.0])
        @test get!(state, cch) == [6.0, 8.0]           # recomputes cleanly afterwards
    end

    @testset "multi-output recipes stay on the pure allocating branch" begin
        gm = Graph()
        xm = value!(gm, :xm, Vector{Float64})
        um = value!(gm, :um, Vector{Float64})
        vm = value!(gm, :vm, Vector{Float64})
        add!(gm; inputs = xm, outputs = (um, vm), op = z -> (copy(z), 2 .* z))
        st = ReactiveKernels._prepare_reactive(gm; have = (xm,), want = (um, vm),
                                                cache_apply = hca)
        @test !occursin("__cache_apply__((__slots__", string(code_expr(st, um)))
        @test !occursin("__cache_apply__((__slots__", string(code_expr(st, vm)))
        inst = st([1.0, 2.0])
        @test get!(inst, statevalue(inst, um)) == [1.0, 2.0]
        @test get!(inst, statevalue(inst, vm)) == [2.0, 4.0]
    end

    @testset "default selector: array outputs in-place, scalars pure" begin
        gs = Graph()
        xs = value!(gs, :xs, Vector{Float64})
        as = value!(gs, :as, Vector{Float64})
        ss = value!(gs, :ss, Float64)
        add!(gs, xs => as, copy)
        add!(gs, as => ss, sum)
        ps = plan(gs; have = (xs,), want = (ss,))
        selected = Dict(only(r.outputs).name => ReactiveKernels._default_is_mutating(r)
                        for r in ps.recipes)
        @test selected[:as] == true        # array output -> in-place candidate
        @test selected[:ss] == false       # scalar output -> pure branch
        st = ReactiveKernels._prepare_reactive(gs; have = (xs,), want = (ss,),
                                               cache_apply = hca)
        @test occursin("__cache_apply__((__slots__", string(code_expr(st, as)))
        @test !occursin("__cache_apply__((__slots__[3])", string(code_expr(st, ss)))
        inst = st([1.0, 2.0, 3.0])
        @test get!(inst, statevalue(inst, ss)) == 6.0
    end
end
