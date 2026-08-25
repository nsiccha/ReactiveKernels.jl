using BenchmarkTools
using InteractiveUtils

_handwritten_chain(x::Float64) = 2x + 1
_handwritten_shared(x::Float64) = (2(x + 1), 3(x + 1))

_output_sum(x::Number) = x
_output_sum(x::Tuple) = sum(x)

function _hotloop(f, x::Float64)
    total = 0.0
    for i in 1:256
        total += _output_sum(f(x + 0.0001i))
    end
    total
end

_middle(values) = sort!(collect(values))[cld(length(values), 2)]

function _llvm_text(f, input_type)
    sprint(io -> code_llvm(
        io, f, Tuple{input_type}; raw = false, dump_module = false,
        optimize = true, debuginfo = :none,
    ))
end

function _llvm_arithmetic(text)
    [m.match for m in eachmatch(r"\b(?:fadd|fsub|fmul|fdiv)\b", text)]
end

function _llvm_has_data_load(text)
    any(eachline(IOBuffer(text))) do line
        occursin(r"\bload\b", line) &&
            !occursin(r"\bload volatile\b.*\binttoptr\b", line)
    end
end

function _typed_arithmetic(code)
    [m.match for m in eachmatch(
        r"Base\.(?:add_float|sub_float|mul_float|div_float)",
        sprint(show, code),
    )]
end

function _no_more_arithmetic(prepared, handwritten)
    all(op -> count(==(op), prepared) <= count(==(op), handwritten),
        union(prepared, handwritten))
end

function _benchmark_comparison(name, prepared, handwritten, input)
    prepared(input)
    handwritten(input)

    prepared_result = @inferred prepared(input)
    handwritten_result = @inferred handwritten(input)
    @test prepared_result == handwritten_result

    prepared_memory = @allocated prepared(input)
    handwritten_memory = @allocated handwritten(input)
    @test prepared_memory == 0
    @test handwritten_memory == 0

    prepared_types = Base.return_types(prepared, (typeof(input),))
    handwritten_types = Base.return_types(handwritten, (typeof(input),))
    @test prepared_types == [typeof(prepared_result)]
    @test handwritten_types == [typeof(handwritten_result)]

    typed = only(code_typed(prepared, Tuple{typeof(input)}; optimize = true))
    handwritten_typed = only(code_typed(
        handwritten, Tuple{typeof(input)}; optimize = true,
    ))
    @test typed.second === typeof(prepared_result)
    @test handwritten_typed.second === typeof(handwritten_result)
    prepared_typed_arithmetic = _typed_arithmetic(typed.first)
    handwritten_typed_arithmetic = _typed_arithmetic(handwritten_typed.first)
    @test _no_more_arithmetic(
        prepared_typed_arithmetic, handwritten_typed_arithmetic,
    )
    warntype = sprint(io -> code_warntype(
        io, prepared, Tuple{typeof(input)}; debuginfo = :none,
    ))
    @test !occursin("::Any", warntype)

    prepared_llvm = _llvm_text(prepared, typeof(input))
    handwritten_llvm = _llvm_text(handwritten, typeof(input))
    prepared_arithmetic = _llvm_arithmetic(prepared_llvm)
    handwritten_arithmetic = _llvm_arithmetic(handwritten_llvm)
    @test prepared_arithmetic == handwritten_arithmetic
    @test !occursin(r"\b(?:call|invoke)\b", prepared_llvm)
    # `julia-actions/julia-runtest` enables source coverage by default. Its
    # counters are volatile loads from fixed `inttoptr` addresses; those are
    # instrumentation, not loads from the PreparedKernel wrapper. Keep the
    # zero-wrapper-load assertion while ignoring only that recognizable form.
    prepared_has_data_load = _llvm_has_data_load(prepared_llvm)
    @test !prepared_has_data_load

    # Nanosecond kernels are sensitive to timer quantization and CPU state.
    # Alternate measurement order and report the median of nine independent
    # BenchmarkTools trials instead of interpreting one favorable sample.
    prepared_times = Float64[]
    handwritten_times = Float64[]
    prepared_benchmark_allocations = 0
    handwritten_benchmark_allocations = 0
    for trial in 1:9
        if isodd(trial)
            prepared_trial = @benchmark $prepared($input) samples = 1_000 evals = 1_000 seconds = 0.15
            handwritten_trial = @benchmark $handwritten($input) samples = 1_000 evals = 1_000 seconds = 0.15
        else
            handwritten_trial = @benchmark $handwritten($input) samples = 1_000 evals = 1_000 seconds = 0.15
            prepared_trial = @benchmark $prepared($input) samples = 1_000 evals = 1_000 seconds = 0.15
        end
        prepared_median = median(prepared_trial)
        handwritten_median = median(handwritten_trial)
        push!(prepared_times, prepared_median.time)
        push!(handwritten_times, handwritten_median.time)
        prepared_benchmark_allocations = prepared_median.allocs
        handwritten_benchmark_allocations = handwritten_median.allocs
    end
    prepared_ns = _middle(prepared_times)
    handwritten_ns = _middle(handwritten_times)

    prepared_hotloop_times = Float64[]
    handwritten_hotloop_times = Float64[]
    for trial in 1:9
        if isodd(trial)
            prepared_hotloop = @benchmark _hotloop($prepared, $input) samples = 1_000 evals = 10 seconds = 0.15
            handwritten_hotloop = @benchmark _hotloop($handwritten, $input) samples = 1_000 evals = 10 seconds = 0.15
        else
            handwritten_hotloop = @benchmark _hotloop($handwritten, $input) samples = 1_000 evals = 10 seconds = 0.15
            prepared_hotloop = @benchmark _hotloop($prepared, $input) samples = 1_000 evals = 10 seconds = 0.15
        end
        push!(prepared_hotloop_times, median(prepared_hotloop).time / 256)
        push!(handwritten_hotloop_times, median(handwritten_hotloop).time / 256)
    end
    prepared_hotloop_ns = _middle(prepared_hotloop_times)
    handwritten_hotloop_ns = _middle(handwritten_hotloop_times)
    measurement = (;
        name,
        prepared_ns,
        handwritten_ns,
        delta_ns = prepared_ns - handwritten_ns,
        ratio = prepared_ns / handwritten_ns,
        prepared_trials_ns = prepared_times,
        handwritten_trials_ns = handwritten_times,
        prepared_hotloop_ns,
        handwritten_hotloop_ns,
        hotloop_delta_ns = prepared_hotloop_ns - handwritten_hotloop_ns,
        hotloop_ratio = prepared_hotloop_ns / handwritten_hotloop_ns,
        prepared_hotloop_trials_ns = prepared_hotloop_times,
        handwritten_hotloop_trials_ns = handwritten_hotloop_times,
        prepared_memory,
        handwritten_memory,
        prepared_allocations = prepared_benchmark_allocations,
        handwritten_allocations = handwritten_benchmark_allocations,
        inferred_return = only(prepared_types),
        typed_arithmetic = prepared_typed_arithmetic,
        handwritten_typed_arithmetic,
        llvm_arithmetic = prepared_arithmetic,
        llvm_has_calls = occursin(r"\b(?:call|invoke)\b", prepared_llvm),
        llvm_loads_kernel = prepared_has_data_load,
    )
    println("BenchmarkTools comparison: ", measurement)
    measurement
end

@testset "BenchmarkTools: prepared versus handwritten kernels" begin
    @test !_llvm_has_data_load(
        "%lcnt = load volatile i64, i64* inttoptr (i64 123 to i64*), align 8",
    )
    @test _llvm_has_data_load("%value = load double, double* %kernel, align 8")
    @test _llvm_has_data_load(
        "%value = load volatile i64, i64* %kernel, align 8",
    )

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
