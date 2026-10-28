using ReactiveKernels, Reactant, Test
using StaticArrays: SMatrix
import Enzyme

# Independent synthetic fixed-storage matrix: no domain package or model.
# The loop must keep its wrapper and lift each host seed entry into a distinct
# tracer, as it does for immutable arrays with fixed tuple storage.
@traceable function _loop_matrix_recurrence(B::SMatrix{2,2}, n)
    R = SMatrix{2,2}(1.0, 0.0, 0.0, 1.0)
    i = zero(n)
    while i < n
        R = R * B
        B = B * B
        i = i + one(i)
    end
    R
end
@traceable _loop_matrix_sum(A::SMatrix{2,2}) =
    A[1, 1] + A[2, 1] + A[1, 2] + A[2, 2]

@kernel fixed_tuple_matrix_loop(q, n) = begin
    B = SMatrix{2,2}(q[1], 0.1, 0.2, q[2])
    R = _loop_matrix_recurrence(B, n)
    # The typed helper proves wrapper preservation, and B is read after the
    # loop to detect accidental mutation of the caller's tracer handles.
    total = _loop_matrix_sum(R) + _loop_matrix_sum(B)
    return total
end

@testset "fixed-tuple matrix carries retain their wrapper and caller values" begin
    k = prepare(fixed_tuple_matrix_loop)
    q = [0.6, 0.7]
    rq = Reactant.to_rarray(q)
    function inventory(hlo)
        counts = Dict{String,Int}()
        for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|tensor)\.\w+", hlo)
            counts[m.match] = get(counts, m.match, 0) + 1
        end
        counts
    end
    for T in (Int32, Int64)
        rn = Reactant.to_rarray(T(3); track_numbers=true)
        compiled = Reactant.@compile k(rq, rn)
        inventories = Dict{String,Int}[]
        for n in (0, 1, 2, 5, 17)
            input = Reactant.to_rarray(T(n); track_numbers=true)
            for point in (q, [0.45, 0.75])
                B = [point[1] 0.2; 0.1 point[2]]
                reference = sum(B^(2^n - 1)) + sum(B)
                @test k(point, T(n)) ≈ reference
                @test Float64(compiled(Reactant.to_rarray(point), input)) ≈ reference
            end
            hlo = repr(Reactant.@code_hlo optimize=true k(rq, input))
            push!(inventories, inventory(hlo))
        end
        @test all(==(first(inventories)), inventories)
        @test first(inventories)["stablehlo.while"] == 1
        println("fixed-tuple matrix default inventory $T: ", first(inventories))
    end
    gradient(v) = only(Enzyme.gradient(Enzyme.Reverse, w -> k(w, 3), v))
    h = 1e-6
    finite_gradient = [(e = zeros(2); e[j] = h;
        (k(q + e, 3) - k(q - e, 3)) / (2h)) for j in 1:2]
    @test gradient(q) ≈ finite_gradient rtol=1e-8
end

# An authored `for` inside a recipe keeps its iteration under Reactant: the
# loop body is emitted once inside one `stablehlo.while` region, whatever the
# (data-derived) trip count (docs/src/constraints.md).

@kernel running_sum_loop(x::Vector{Float64}, scale::Float64) = begin
    prefix::Vector{Float64} = let
        n = length(x)
        out = zero(x)
        acc = zero(scale)
        for i in 1:n
            acc = acc + scale * x[i]
            out[i] = acc
        end
        out
    end
    total::Float64 = sum(prefix)
    return total
end

_traced(v) = v isa AbstractArray ? Reactant.to_rarray(v) :
             Reactant.to_rarray(v; track_numbers = true)
_host(v) = v isa Reactant.AbstractConcreteArray ? Array(v) : Reactant.to_number(v)

@testset "an authored recipe loop lowers to one retained while region" begin
    k = prepare(running_sum_loop; want = :total)
    sizes = Int[]
    for n in (8, 32)
        x = collect(range(-1.0, 1.0; length = n))
        hlo = repr(Reactant.@code_hlo optimize = false k(_traced(x), _traced(0.5)))
        @test count("stablehlo.while", hlo) == 1
        push!(sizes, count("\n", hlo))
        compiled = Reactant.@compile k(_traced(x), _traced(0.5))
        @test _host(compiled(_traced(x), _traced(0.5))) ≈ k(x, 0.5)
    end
    # Only tensor shapes differ: the body is not replicated per iteration.
    @test sizes[1] == sizes[2]
end

@testset "the retained loop differentiates like the native loop" begin
    k = prepare(running_sum_loop; want = :total)
    x = [0.5, -1.0, 2.0, 0.25]
    native_gradient(v) = Enzyme.gradient(Enzyme.Reverse, w -> k(w, 0.5), v)
    gradient(v) = Enzyme.gradient(Enzyme.Reverse, w -> k(w, 0.5), v)
    compiled = Reactant.@compile gradient(_traced(x))
    @test _host(only(compiled(_traced(x)))) ≈ only(native_gradient(x))
end

@testset "the pointwise loop output is available" begin
    k = prepare(running_sum_loop; want = :prefix)
    x = [0.5, -1.0, 2.0, 0.25]
    compiled = Reactant.@compile k(_traced(x), _traced(0.5))
    @test _host(compiled(_traced(x), _traced(0.5))) ≈ k(x, 0.5)
end

@kernel literal_array_reads(x::Vector{Float64}, M::Matrix{Float64}) = begin
    total::Float64 = x[2] + M[2, 1]
    return total
end

@testset "authored scalar reads at literal indices are backend slices" begin
    k = prepare(literal_array_reads)
    x, M = [0.5, 1.2], [0.7 0.8; 1.4 0.9]
    compiled = Reactant.@compile k(_traced(x), _traced(M))
    @test _host(compiled(_traced(x), _traced(M))) == k(x, M)
    gradient(v) = Enzyme.gradient(Enzyme.Reverse, w -> k(w, M), v)
    cg = Reactant.@compile gradient(_traced(x))
    @test _host(only(cg(_traced(x)))) == [0.0, 1.0]
    @test x == [0.5, 1.2]
    @test M == [0.7 0.8; 1.4 0.9]
end

# `ReactantCore.@trace` carries a loop value by updating the traced object that
# exists before the loop; it never assigns results back to the enclosing
# variables. A carry seeded on the host (`acc = 0.0`, `zeros(n)`) therefore had
# no traced object, and the compiled loop silently returned its seed. The loop
# companion now re-binds each carried local to a fresh traced copy, and passes
# read-only host structs through untraced (snag one-natural-supe-39da86a4).
@kernel host_seed_loop(x::Vector{Float64}) = begin
    total::Float64 = begin
        acc = 0.0
        for i in eachindex(x)
            acc = acc + x[i] * x[i]
        end
        acc
    end
    return total
end

@kernel host_buffer_loop(x::Vector{Float64}) = begin
    prefix::Vector{Float64} = begin
        out = zeros(length(x))
        acc = 0.0
        for i in eachindex(x)
            acc = acc + x[i]
            out[i] = acc
        end
        out
    end
    return prefix
end

struct _HostLoopPlan
    picks::Vector{Int}
end

@kernel host_struct_loop(plan, x::Vector{Float64}) = begin
    total::Float64 = begin
        acc = 0.0
        for j in eachindex(plan.picks)
            acc = acc + x[plan.picks[j]]
        end
        acc
    end
    return total
end

# `@trace for` resolves `step` in the scope it expands into (reactivekernels-use
# §7m); with a port named `step` the loop is traced per iteration instead.
@kernel shadowed_step_loop(x::Vector{Float64}, step::Float64) = begin
    total::Float64 = begin
        acc = 0.0
        for i in eachindex(x)
            acc = acc + step * x[i]
        end
        acc
    end
    return total
end

# `@trace` writes each loop result back into every traced object the body
# reads; the loop reads the caller's inputs through fresh tracers, so a
# zero-sized one is not returned as an aliased `tensor.empty` output.
@kernel loop_reads_empty(x::Vector{Float64}, z::Vector{Float64}) = begin
    total::Float64 = begin
        acc = 0.0
        for i in eachindex(x)
            acc = acc + x[i] + sum(z)
        end
        acc
    end
    return total
end

@testset "a loop carry seeded on the host is carried, not dropped" begin
    sizes = Int[]
    for n in (8, 32)
        x = collect(range(-1.0, 1.0; length = n))
        k = prepare(host_seed_loop)
        hlo = repr(Reactant.@code_hlo optimize = false k(_traced(x)))
        @test count("stablehlo.while", hlo) == 1
        push!(sizes, count("\n", hlo))
        compiled = Reactant.@compile k(_traced(x))
        @test _host(compiled(_traced(x))) ≈ k(x)
        kb = prepare(host_buffer_loop)
        compiled_buffer = Reactant.@compile kb(_traced(x))
        @test _host(compiled_buffer(_traced(x))) ≈ kb(x)
    end
    @test sizes[1] == sizes[2]

    x = [0.5, -1.0, 2.0, 0.25, 3.0]
    plan = _HostLoopPlan([5, 1, 3])
    ks = prepare(host_struct_loop)
    compiled_struct = Reactant.@compile ks(plan, _traced(x))
    @test _host(compiled_struct(plan, _traced(x))) ≈ ks(plan, x)

    # `@trace for` traces its body once even for zero iterations, and indexing
    # a zero-length array there emits an invalid slice. A loop over an empty
    # host range is skipped instead, as the plain loop runs zero times (the
    # doses of an empty schedule).
    k = prepare(host_seed_loop)
    empty = Float64[]
    hlo_empty = repr(Reactant.@code_hlo optimize = false k(_traced(empty)))
    @test count("stablehlo.while", hlo_empty) == 0
    compiled_empty = Reactant.@compile k(_traced(empty))
    @test Float64(compiled_empty(_traced(empty))) === k(empty) === 0.0
    unplanned = _HostLoopPlan(Int[])
    compiled_unplanned = Reactant.@compile ks(unplanned, _traced(x))
    @test Float64(compiled_unplanned(unplanned, _traced(x))) === ks(unplanned, x) === 0.0

    kz = prepare(loop_reads_empty)
    for z in ([0.5], Float64[])
        compiled_z = Reactant.@compile sync = true kz(_traced(x), _traced(z))
        @test Float64(compiled_z(_traced(x), _traced(z))) ≈ kz(x, z)
    end

    kstep = prepare(shadowed_step_loop)
    hlo_step = repr(Reactant.@code_hlo optimize = false kstep(_traced(x), _traced(0.5)))
    @test count("stablehlo.while", hlo_step) == 0
    compiled_step = Reactant.@compile kstep(_traced(x), _traced(0.5))
    @test _host(compiled_step(_traced(x), _traced(0.5))) ≈ kstep(x, 0.5)
end

# A data-length generator sum's index is traced, so an element read with one
# index per dimension is a gather at traced indices: `W[i, j]` over a bound
# host table and over a traced matrix, and `W[i, 1]` with one concrete index
# (snag 2-d-traced-index-b107cbfc; before, only a vector read lowered and
# these failed with `Scalar indexing is disallowed`).
@kernel table_quadratic(W, x::Vector{Float64}) = begin
    total::Float64 = sum(sum(x[i] * W[i, j] * x[j] for i in axes(W, 1); init = 0.0)
                         for j in axes(W, 2); init = 0.0)
    column::Float64 = sum(W[i, 1] * x[i] for i in axes(W, 1); init = 0.0)
    return total
end

# A loop reading a partly traced named tuple keeps its host leaves host: the
# matrix crosses the loop as itself, not as a matrix of traced scalars.
@kernel model_quadratic(W, scale::Float64, x::Vector{Float64}) = begin
    model = (; W, scale)
    total::Float64 = sum(sum(model.scale * x[i] * model.W[i, j] * x[j]
                             for i in axes(model.W, 1); init = 0.0)
                         for j in axes(model.W, 2); init = 0.0)
    return total
end

@testset "a 2-D read at traced loop indices is one gather" begin
    for want in (:total, :column)
        sizes = Int[]
        for m in (3, 6)
            W = [sin(a + 2b) / (a + b) for a in 1:m, b in 1:m]
            x = collect(range(-1.0, 1.0; length = m))
            k = prepare(table_quadratic; want)
            native = k(W, x)
            bound = prepare(table_quadratic; want, bound = (; W))
            @test _host((Reactant.@compile bound(_traced(x)))(_traced(x))) ≈ native
            @test _host((Reactant.@compile k(_traced(W), _traced(x)))(
                _traced(W), _traced(x))) ≈ native
            hlo = repr(Reactant.@code_hlo optimize = false bound(_traced(x)))
            @test count("stablehlo.while", hlo) == (want === :total ? 2 : 1)
            push!(sizes, count("\n", hlo))
        end
        # Twice the matrix side, the same program.
        @test allequal(sizes)
    end

    k = prepare(model_quadratic)
    W = [sin(a + 2b) / (a + b) for a in 1:4, b in 1:4]
    x = [0.5, -1.0, 2.0, 0.25]
    compiled = Reactant.@compile k(W, _traced(1.5), _traced(x))
    @test _host(compiled(W, _traced(1.5), _traced(x))) ≈ k(W, 1.5, x)
    @test _host(compiled(W, _traced(-0.5), _traced(x))) ≈ k(W, -0.5, x)

    # Reverse through the gathers, against central differences.
    h = 1e-6
    fd_x = [(e = zeros(4); e[c] = h; (k(W, 1.5, x .+ e) - k(W, 1.5, x .- e)) / 2h)
            for c in 1:4]
    gradient_x(v) = Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(w -> k(W, 1.5, w)), v)
    @test isapprox(_host(only((Reactant.@compile gradient_x(_traced(x)))(_traced(x)))),
                   fd_x; rtol = 1e-6)
    t = prepare(table_quadratic)
    fd_W = [(E = zeros(4, 4); E[c] = h; (t(W .+ E, x) - t(W .- E, x)) / 2h)
            for c in CartesianIndices(W)]
    gradient_W(w) = Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(v -> t(v, x)), w)
    @test isapprox(_host(only((Reactant.@compile gradient_W(_traced(W)))(_traced(W)))),
                   fd_W; rtol = 1e-6)
end
