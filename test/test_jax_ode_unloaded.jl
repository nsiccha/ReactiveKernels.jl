# The native ODE bridges without their solver packages (the core test
# environment has none): a call names what to load, and `jax_source`, which
# needs no Julia solver, still translates code calling them. With the solvers,
# `test_jax_ode.jl` (PPLExamples) checks the emitted solve against native.
using ReactiveKernels, Test

_jax_unloaded_decay(t, y, k) = -k .* y
# BayesianRegressionModels' `ode_rk45` binding has this shape.
_jax_unloaded_rk45(f::F, y0, t0, ts, args::Vararg{Any,N}) where {F,N} =
    rk_ode_rk45_tol(f, y0, t0, ts, 1e-6, 1e-6, 10^6, args...)

@kernel jax_ode_unloaded(u::Vector{Float64}, ts::Vector{Float64}) = begin
    pred = _jax_unloaded_rk45(_jax_unloaded_decay, [1.0], 0.0, ts, exp(u[1]))[:, 1]
    total = sum(pred)
    return total
end

@testset "ODE bridges without their solver packages" begin
    if Base.get_extension(ReactiveKernels, :ReactiveKernelsLowOrderRKExt) === nothing &&
       Base.get_extension(ReactiveKernels, :ReactiveKernelsBDFExt) === nothing
        args = (_jax_unloaded_decay, [1.0], 0.0, [0.5, 1.0], 1e-6, 1e-6, 100, 0.7)
        @test Core.Compiler.return_type(rk_ode_rk45_tol, Tuple{map(typeof, args)...}) ===
              Matrix{Float64}
        @test_throws "rk_ode_rk45_tol needs OrdinaryDiffEqLowOrderRK, SciMLBase and SciMLSensitivity" rk_ode_rk45_tol(args...)
        @test_throws "rk_ode_bdf_tol needs OrdinaryDiffEqBDF, SciMLBase and SciMLSensitivity" rk_ode_bdf_tol(args...)
        source = string(jax_source(jax_ode_unloaded; have = (:u, :ts), want = :total))
        @test occursin("diffrax.Dopri5()", source)
        @test !occursin("<function", source)
    else
        @test_skip false
    end
end
