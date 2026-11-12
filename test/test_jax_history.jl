module JAXHistoryStorageTests
using ReactiveKernels, Test, DifferentiationInterface
import Enzyme
include("jax_python.jl")

@traceable function checked_scale(a)
    a > 0 || throw(ArgumentError("history_value _history_read"))
    a
end

@kernel lagged(a::Float64, x::Vector{Float64}) = begin
    ys = scan(eachindex(x); init = 0.0, history = 2.0) do carry, t, history
        previous = t > 1 ? history[t - 1] : 0.0
        (carry, checked_scale(a) * previous + x[t] + history[t])
    end
    total = sum(ys)
end

@kernel seeded(a::Float64, x::Vector{Float64}, seeds::Vector{Float64}) = begin
    m = length(seeds)
    ys = scan(eachindex(x); init = 0.0, history = seeds) do carry, t, history
        # Reading current/future cells gives zero; leading entries and
        # outputs retain their absolute, seed-prefixed positions.
        (carry, a * history[t] + x[t] + history[t + m])
    end
    total = sum(ys)
end

@kernel mixed_lags(a::Float64, x::Vector{Float64}) = begin
    ys = scan(eachindex(x); init = 0.0, history = 0.0) do carry, t, history
        previous = t <= 2 ? 0.0 : t % 2 == 0 ? history[t - 1] : history[t - 2]
        (carry, a * previous + x[t])
    end
    total = sum(ys)
end

@kernel blocks(a::Float64, x::Vector{Float64}, weights::Vector{Float64}) = begin
    K = length(weights)
    N = length(x)
    ys = scan(1:(N * K); init = 0.0, history = 0.0) do carry, cell, history
        t = (cell - 1) ÷ K + 1
        j = (cell - 1) % K + 1
        previous = t == 1 ? weights[j] :
            sum(history[(t - 2) * K + i] * weights[i] for i in 1:K; init = 0.0)
        (carry, a * previous + x[t])
    end
    total = sum(ys)
end

@kernel arbitrary(a::Float64, indices::Vector{Int}) = begin
    ys = scan(eachindex(indices); init = 0.0, history = 1.0) do carry, t, history
        (carry, a * history[indices[t]])
    end
    total = sum(ys)
end

@kernel seeded_blocks(a::Float64, x::Vector{Float64}, weights::Vector{Float64}) = begin
    K = length(weights)
    N = length(x)
    ys = scan((K + 1):(N * K); init = 0.0, history = weights) do carry, cell, history
        t = (cell - 1) ÷ K + 1
        previous = sum(history[(t - 2) * K + i] * weights[i] for i in 1:K; init = 0.0)
        (carry, a * previous + x[t])
    end
    total = sum(ys)
end

@kernel whole_history(a::Float64, x::Vector{Float64}) = begin
    ys = scan(x; init = 0.0, history = 1.0) do carry, value, history
        (carry, a * sum(history) + value)
    end
    total = sum(ys)
end

@traceable function repeat_from_index(a, n)
    total = 0.0
    for j in n:(n + 2)
        total += a * j
    end
    total
end

@kernel range_history(a::Float64, counts::UnitRange{Int}) = begin
    ys = scan(counts; init = 0.0, history = 0.0) do carry, n, history
        previous = n > 1 ? history[n - 1] : 0.0
        next = previous + repeat_from_index(a, n)
        (next, next)
    end
    total = sum(ys)
end

@testset "JAX history indices retain prepared loop bounds" begin
    have = (:a, :counts)
    src = jax_source(range_history; have, want = :total, data = (; counts = 1:4))
    @test !occursin("compact_history = False", src.text)
    native = prepare(range_history; have, want = :total)
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for counts in (1:0, 1:4, 1:19)
        @test src.text == jax_source(range_history; have, want = :total, data = (; counts)).text
        if rk_jax_available()
            gradient = ad_gradient(range_history, backend, 0.4, counts; active = :a, want = :total)
            for mode in (true, :arguments)
                result = rk_jax_evaluate(src, :kernel, (0.4, counts); grad = true, prepare = mode)
                @test result.value ≈ native(0.4, counts) rtol = 1e-12 atol = 1e-12
                @test result.grad ≈ gradient rtol = 1e-12 atol = 1e-12
            end
        else
            @test_skip "RK_JAX_PYTHON not set"
        end
    end
end

@testset "JAX history scans retain only proven bounded lag in the carry" begin
    cases = ((lagged, (:a, :x), (Float64[],)),
             (lagged, (:a, :x), ([0.2],)),
             (lagged, (:a, :x), (sin.(1:19),)),
             (mixed_lags, (:a, :x), (sin.(1:19),)),
             (seeded, (:a, :x, :seeds), (Float64[], [1.0, 2.0])),
             (seeded, (:a, :x, :seeds), (sin.(1:19), Float64[])),
             (seeded, (:a, :x, :seeds), (sin.(1:19), [1.0, 2.0])),
             (seeded, (:a, :x, :seeds), (sin.(1:19), collect(1.0:5.0))),
             (blocks, (:a, :x, :weights), (Float64[], [0.3, 0.7])),
             (blocks, (:a, :x, :weights), (sin.(1:19), [1.0])),
             (blocks, (:a, :x, :weights), (sin.(1:19), [0.3, 0.7])),
             (blocks, (:a, :x, :weights), (sin.(1:19), fill(0.2, 5))),
             (seeded_blocks, (:a, :x, :weights), (Float64[], [0.3, 0.7])),
             (seeded_blocks, (:a, :x, :weights), (sin.(1:19), [0.3, 0.7])),
             (arbitrary, (:a, :indices), ([1, 1, 2, 7, 1, 5, 3],)),
             (whole_history, (:a, :x), (sin.(1:9),)))
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    a = 0.4
    for (kernel, have, data) in cases
        src = jax_source(kernel; have, want = :total)
        kernel === lagged && @test occursin("\"history_value _history_read\"", src.text)
        native = prepare(kernel; have, want = :total)
        value = native(a, data...)
        gradient = ad_gradient(kernel, backend, a, data...; active = :a, want = :total)
        if kernel === arbitrary || kernel === whole_history
            @test occursin("compact_history = False", src.text)
        else
            @test !occursin("compact_history = False", src.text)
        end
        if rk_jax_available()
            result = rk_jax_evaluate(src, :kernel, (a, data...); grad = true)
            @test result.value ≈ value rtol = 1e-12 atol = 1e-12
            @test result.grad ≈ gradient rtol = 1e-12 atol = 1e-12
        else
            @test_skip "RK_JAX_PYTHON not set"
        end
    end
    if rk_jax_available()
        x, initial = sin.(1:19), [1.0, 2.0, 3.0]
        src = jax_source(seeded; have = (:seeds, :a, :x), want = :total)
        result = rk_jax_evaluate(src, :kernel, (initial, a, x); grad = true)
        @test result.grad ≈ ad_gradient(seeded, backend, a, x, initial;
                                       active = :seeds, want = :total) rtol = 1e-12
    end
    # Changing data lengths or the row width never specializes the source.
    src = jax_source(blocks; want = :total, data = (; x = sin.(1:8), weights = [0.3, 0.7]))
    for (n, k) in ((0, 1), (8, 2), (128, 5))
        @test src.text == jax_source(blocks; want = :total,
                                    data = (; x = sin.(1:n), weights = fill(1.0 / k, k))).text
    end
    if rk_jax_available()
        mktempdir() do dir
            path = joinpath(dir, "blocks.py")
            write(path, src)
            program = """
                import importlib.util
                import jax, numpy as np
                spec = importlib.util.spec_from_file_location('blocks', $(_rk_py(path)))
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                sizes = []
                for n in (8, 128):
                    prepared = m.prepare_data(x=np.sin(np.arange(1, n+1)), weights=np.array([0.3, 0.7]))
                    f = lambda a: m.kernel(a, **prepared)
                    jp = jax.make_jaxpr(f)($a).jaxpr
                    scans = [e for e in jp.eqns if e.primitive.name == 'scan']
                    assert scans
                    def carry(scan):
                        if 'ft_in' in scan.params:
                            return scan.params['ft_in'].update(scan.invars).unpack()[1]
                        start = scan.params['num_consts']
                        return scan.invars[start:start+scan.params['num_carry']]
                    carried = [v for scan in scans for v in carry(scan)]
                    sizes.append(max(np.prod(v.aval.shape) for v in carried))
                # The dependency spans at most two rows less one cell.
                assert max(sizes) <= 3 and sizes[0] == sizes[1], sizes
                """
            @test success(pipeline(`$RK_JAX_PYTHON -c $program`; stdout, stderr))
        end
    end
end
end
