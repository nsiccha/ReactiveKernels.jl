using BenchmarkTools
using InteractiveUtils

_handwritten_chain(x::Float64) = 2x + 1
_handwritten_shared(x::Float64) = (2(x + 1), 3(x + 1))

function _benchmark_comparison(name, prepared, handwritten, input)
    prepared(input)
    handwritten(input)

    prepared_result = @inferred prepared(input)
    handwritten_result = @inferred handwritten(input)
    @test prepared_result == handwritten_result

    prepared_allocations = @allocated prepared(input)
    handwritten_allocations = @allocated handwritten(input)
    @test prepared_allocations == 0
    @test handwritten_allocations == 0

    prepared_types = Base.return_types(prepared, (typeof(input),))
    handwritten_types = Base.return_types(handwritten, (typeof(input),))
    @test prepared_types == [typeof(prepared_result)]
    @test handwritten_types == [typeof(handwritten_result)]

    typed = only(code_typed(prepared, Tuple{typeof(input)}; optimize = true))
    @test typed.second === typeof(prepared_result)
    warntype = sprint(io -> code_warntype(
        io, prepared, Tuple{typeof(input)}; debuginfo = :none,
    ))
    @test !occursin("::Any", warntype)

    prepared_trial = @benchmark $prepared($input) samples = 10_000 seconds = 0.25
    handwritten_trial = @benchmark $handwritten($input) samples = 10_000 seconds = 0.25
    prepared_median = median(prepared_trial)
    handwritten_median = median(handwritten_trial)
    measurement = (;
        name,
        prepared_ns = prepared_median.time,
        handwritten_ns = handwritten_median.time,
        ratio = prepared_median.time / handwritten_median.time,
        prepared_memory = prepared_median.memory,
        handwritten_memory = handwritten_median.memory,
        prepared_allocations = prepared_median.allocs,
        handwritten_allocations = handwritten_median.allocs,
        inferred_return = only(prepared_types),
    )
    println("BenchmarkTools comparison: ", measurement)
    measurement
end

@testset "BenchmarkTools: prepared versus handwritten kernels" begin
    chain_graph = Graph()
    chain_x = value!(chain_graph, :x, Float64)
    chain_z = value!(chain_graph, :z, Float64)
    chain_w = value!(chain_graph, :w, Float64)
    add!(chain_graph, chain_x => chain_z, x -> 2x)
    add!(chain_graph, chain_z => chain_w, z -> z + 1)
    chain_kernel = prepare(chain_graph; have = (chain_x,), want = (chain_w,))

    shared_graph = Graph()
    shared_x = value!(shared_graph, :x, Float64)
    shared_a = value!(shared_graph, :a, Float64)
    shared_b = value!(shared_graph, :b, Float64)
    shared_c = value!(shared_graph, :c, Float64)
    add!(shared_graph, shared_x => shared_a, x -> x + 1)
    add!(shared_graph, shared_a => shared_b, a -> 2a)
    add!(shared_graph, shared_a => shared_c, a -> 3a)
    shared_kernel = prepare(
        shared_graph; have = (shared_x,), want = (shared_b, shared_c),
    )

    chain = _benchmark_comparison(
        :reactiveobjects_chain, chain_kernel, _handwritten_chain, 3.0,
    )
    shared = _benchmark_comparison(
        :reactiveobjects_shared, shared_kernel, _handwritten_shared, 5.0,
    )

    @test chain.inferred_return === Float64
    @test shared.inferred_return === Tuple{Float64, Float64}
end
