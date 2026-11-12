using ReactiveKernels, ReactiveKernelsPPL, Test

isdefined(@__MODULE__, :rk_jax_evaluate) ||
    include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

# The native window tests define the same authored program and independent
# Julia oracle. Both scan shapes must retain their meaning in standalone JAX.
@testset "scan window sums: JAX values, gradients and runtime lengths" begin
    if !rk_jax_available()
        @info "RK_JAX_PYTHON is not set: scan window JAX parity skipped"
        @test_skip false
    else
        for coupled in (false, true), history in (false, true)
            lines = Int[]
            for T in (2, 8, 24), enabled in (false, true)
                data = (; T, enabled, y = 0.1 .* sin.(1:T))
                plan = bind_data(lower_rkppl(_window_program(coupled; history), data;
                    conditioned = (:y,)), data)
                built = build_kernel(plan)
                src = jax_source(built, plan)
                @test occursin("UNTRANSLATED = {}", string(src))
                names = first.(only(f for f in src.functions if f.name == "prepare_data").inputs)
                raw = Tuple(getproperty(data, Symbol(n)) for n in names)
                u = 0.13 .* sin.(1:built.layout.total)
                oracle(x) = _window_oracle(x, T, coupled, enabled, history)
                result = rk_jax_evaluate(src, "log_density", (u, raw...);
                    grad = true, prepare = true)
                @test result.value ≈ oracle(u) rtol = 1e-10
                @test result.grad ≈ rk_fd_gradient(oracle, u) rtol = 1e-5 atol = 1e-7
                @test rk_jax_evaluate(src, "log_density", (u, raw...);
                    prepare = :arguments).value ≈ oracle(u) rtol = 1e-10
                if enabled && T > 2
                    push!(lines, count(==('\n'), string(src)))
                end
            end
            @test lines[1] == lines[2]
        end
    end
end
