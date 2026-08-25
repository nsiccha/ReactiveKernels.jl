using ReactiveKernels
using MutatingFunctions
using LinearAlgebra
using Test

# End-to-end check of the public `prepare_reactive_nonallocating` entry with the
# real `MutatingFunctions.apply!!` adapter. The MF-agnostic core semantics are
# covered allocation-free in test_nonallocating_core.jl; here we confirm that
# registered `apply!!` forms (copy/reverse and the mul!-backed `*`) reuse the
# owned slot buffers at zero steady-state allocation through the public API.
@testset "MutatingFunctions-backed in-place reactive getters" begin
    @test Base.get_extension(ReactiveKernels,
                             :ReactiveKernelsMutatingFunctionsExt) !== nothing

    g = Graph()
    x = value!(g, :x, Vector{Float64})
    a = value!(g, :a, Vector{Float64})       # a = copy(x)     (registered apply!!)
    b = value!(g, :b, Vector{Float64})       # b = reverse(a)  (registered apply!!)
    add!(g, x => a, copy)
    add!(g, a => b, reverse)

    pure = prepare_reactive(g; have = (x,), want = (b,))
    ip = prepare_reactive_nonallocating(g; have = (x,), want = (b,))

    @test occursin("__cache_apply__((__slots__", string(code_expr(ip, b)))
    @test !occursin("__cache_apply__", string(code_expr(pure, b)))

    input = [1.0, 2.0, 3.0]
    sp = pure(copy(input)); si = ip(copy(input))
    bp = statevalue(sp, b); bi = statevalue(si, b)
    xp = statevalue(sp, x); xi = statevalue(si, x)

    @testset "parity vs pure over invalidate->recompute" begin
        for v in ([1.0, 2.0, 3.0], [4.0, -1.0, 8.0], [0.0, 5.0, 9.0])
            set!(sp, xp, v); set!(si, xi, v)
            @test get!(sp, bp) == get!(si, bi) == reverse(v)
        end
        @test @inferred(get!(si, bi))::Vector{Float64} == [9.0, 5.0, 0.0]
    end

    @testset "registered apply!! ops reuse the slot buffer at 0 bytes" begin
        s = ip(copy(input))
        bh = statevalue(s, b); ah = statevalue(s, a); xh = statevalue(s, x)
        function cycle!(state, xhandle, bhandle, v, n)
            for _ in 1:n
                set!(state, xhandle, v)
                get!(state, bhandle)
            end
        end
        abuf = get!(s, ah); bbuf = get!(s, bh)   # establish buffers
        cycle!(s, xh, bh, input, 1)
        alloc = @allocated cycle!(s, xh, bh, input, 1000)
        println("REACTIVE_NONALLOCATING_ALLOC_BYTES\tcopy_reverse\t", alloc)
        @test alloc == 0
        @test get!(s, ah) === abuf && get!(s, bh) === bbuf
    end

    @testset "mul!-backed `*` recipe reuses the buffer at 0 bytes" begin
        gm = Graph()
        A = value!(gm, :A, Matrix{Float64})
        q = value!(gm, :q, Vector{Float64})
        Aq = value!(gm, :Aq, Vector{Float64})     # Aq = A * q  -> mul!(cache, A, q)
        add!(gm; inputs = (A, q), outputs = Aq, op = *)
        prog = prepare_reactive_nonallocating(gm; have = (A, q), want = (Aq,))
        @test occursin("__cache_apply__((__slots__", string(code_expr(prog, Aq)))

        M = [2.0 0.0; 0.0 3.0]; v0 = [1.0, 1.0]
        st = prog(M, v0)
        Aqh = statevalue(st, Aq); qh = statevalue(st, q)
        @test get!(st, Aqh) == M * v0
        buf = get!(st, Aqh)
        function cyc!(state, qhandle, ahandle, w, n)
            for _ in 1:n
                set!(state, qhandle, w)
                get!(state, ahandle)
            end
        end
        cyc!(st, qh, Aqh, v0, 1)
        alloc = @allocated cyc!(st, qh, Aqh, v0, 1000)
        println("REACTIVE_NONALLOCATING_ALLOC_BYTES\tmul\t", alloc)
        @test alloc == 0
        @test get!(st, Aqh) === buf
    end

    @testset "per-instance cache ownership: copy does not alias" begin
        s1 = ip([1.0, 2.0, 3.0]); get!(s1, statevalue(s1, b))
        s2 = copy(s1)
        @test get!(s2, statevalue(s2, b)) == get!(s1, statevalue(s1, b))
        @test get!(s2, statevalue(s2, b)) !== get!(s1, statevalue(s1, b))
        set!(s2, statevalue(s2, x), [7.0, 8.0, 9.0])
        @test get!(s1, statevalue(s1, b)) == [3.0, 2.0, 1.0]     # reverse([1,2,3])
        @test get!(s2, statevalue(s2, b)) == [9.0, 8.0, 7.0]     # reverse([7,8,9])
    end
end
