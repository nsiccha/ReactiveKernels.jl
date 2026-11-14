using ReactiveKernels, OrdinaryDiffEqBDF, OrdinaryDiffEqLowOrderRK,
    SciMLBase, SciMLSensitivity, Enzyme, Test
using DifferentiationInterface: AutoEnzyme, gradient
isdefined(@__MODULE__, :rk_jax_evaluate) ||
    include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

# The emitted module solves ODEs with diffrax; the parity checks need it in
# the `RK_JAX_PYTHON` environment (`pip install diffrax`).
const _JAX_ODE_PYTHON = rk_jax_available() && success(`$RK_JAX_PYTHON -c "import diffrax"`)

_jax_ode_elimination(t, y, k) = -k .* y
_jax_ode_absorption(t, y, ka, cl, v) = [-ka * y[1], ka * y[1] - cl / v * y[2]]
# Robertson's stiff chemical kinetics.
_jax_ode_robertson(t, y, k1, k2, k3) =
    [-k1 * y[1] + k3 * y[2] * y[3], k1 * y[1] - k2 * y[2]^2 - k3 * y[2] * y[3], k2 * y[2]^2]

# One subject per plate cell, each with its own number of output times.
@kernel jax_ode_cells(u::Vector{Float64}, doses::Vector{Float64},
                      times::Vector{Vector{Float64}}) = begin
    ka = exp(u[1])
    cl = exp(u[2])
    v = exp(u[3])
    values = plate(doses, times) do d, ts
        sol = rk_ode_rk45_tol(_jax_ode_absorption, [d, 0.0], 0.0, ts, 1e-9, 1e-9, 10^6, ka, cl, v)
        sum(log.(sol[:, 2]))
    end
    total = sum(values)
    return total
end

@kernel jax_ode_none(u::Vector{Float64}) = begin
    total = sum(exp.(u))
    return total
end

@kernel jax_ode_budget(u::Vector{Float64}, ts::Vector{Float64}, limit::Int) = begin
    sol = rk_ode_rk45_tol(_jax_ode_elimination, [1.0], 0.0, ts, 1e-10, 1e-10, limit, exp(u[1]))
    total = sum(sol)
    return total
end

@kernel jax_ode_stiff(u::Vector{Float64}, ts::Vector{Float64}) = begin
    sol = rk_ode_bdf_tol(_jax_ode_robertson, [1.0, 0.0, 0.0], 0.0, ts, 1e-8, 1e-10, 10^6,
                         exp(u[1]), 3e7, 1e4)
    total = sum(sol[:, 1]) + 1e4 * sum(sol[:, 2])
    return total
end

# A helper that splats its trailing arguments into the bridge, as
# BayesianRegressionModels' `ode_rk45` binding does: the function stays a
# function through the splat.
_jax_ode_rk45(f::F, y0, t0, ts, args::Vararg{Any,N}) where {F,N} =
    rk_ode_rk45_tol(f, y0, t0, ts, 1e-6, 1e-6, 10^6, args...)

@kernel jax_ode_wrapped(u::Vector{Float64}, doses::Vector{Float64},
                        times::Vector{Vector{Float64}}) = begin
    k = exp(u[1])
    values = plate(doses, times) do d, ts
        pred = _jax_ode_rk45(_jax_ode_elimination, fill(d, 1), 0.0, ts, k)[:, 1]
        sum(pred)
    end
    total = sum(values)
    return total
end

function jax_ode_case(n)
    doses = [80.0 + 10i for i in 1:n]
    times = [collect(0.5:0.75:(0.5 + 0.75 * (1 + i % 4))) for i in 1:n]
    return (; doses, times)
end

@testset "jax_source solves rk_ode_rk45_tol and rk_ode_bdf_tol with diffrax" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    source = jax_source(jax_ode_cells; have = (:u, :doses, :times), want = :total)
    text = string(source)
    @test occursin("import diffrax", text)
    @test occursin("diffrax.Dopri5()", text)
    # One solve, vmapped over the cells: no body per subject or output time.
    @test count("_rk_ode_solve(", text) == 2
    # A module without an ODE does not import diffrax.
    @test !occursin("diffrax", string(jax_source(jax_ode_none; have = (:u,), want = :total)))
    native = prepare(jax_ode_cells; have = (:u, :doses, :times), want = :total)
    doses = [100.0, 80.0, 120.0]
    times = [[0.5, 1.0, 2.0], [0.5, 3.0], [1.0, 2.0, 4.0, 8.0]]
    if _JAX_ODE_PYTHON
        for u in ([0.3, -0.5, 1.2], [-0.2, 0.1, 0.8])
            query = x -> native(x, doses, times)
            got = rk_jax_evaluate(source, "kernel", (u, doses, times); grad = true)
            @test got.value ≈ query(u) rtol = 1e-8
            @test got.grad ≈ gradient(query, backend, u) rtol = 1e-6
        end
    else
        @test_skip false
    end
    stiff = jax_source(jax_ode_stiff; have = (:u, :ts), want = :total)
    @test occursin("diffrax.Kvaerno5()", string(stiff))
    stiff_native = prepare(jax_ode_stiff; have = (:u, :ts), want = :total)
    ts = [0.4, 4.0, 40.0, 400.0]
    if _JAX_ODE_PYTHON
        u = [log(0.04)]
        query = x -> stiff_native(x, ts)
        got = rk_jax_evaluate(stiff, "kernel", (u, ts); grad = true)
        @test got.value ≈ query(u) rtol = 1e-6
        @test got.grad ≈ gradient(query, backend, u) rtol = 1e-4
    else
        @test_skip false
    end
end

@testset "an ODE solve through a splatting helper" begin
    source = jax_source(jax_ode_wrapped; have = (:u, :doses, :times), want = :total)
    @test !occursin("<function", string(source))
    native = prepare(jax_ode_wrapped; have = (:u, :doses, :times), want = :total)
    doses, times = [100.0, 80.0, 120.0], [[0.5, 1.0, 2.0], [0.5, 3.0], [1.0, 2.0, 4.0, 8.0]]
    if _JAX_ODE_PYTHON
        u = [log(0.3)]
        query = x -> native(x, doses, times)
        got = rk_jax_evaluate(source, "kernel", (u, doses, times); grad = true)
        @test got.value ≈ query(u) rtol = 1e-5
        @test got.grad ≈ gradient(query, AutoEnzyme(; mode = Enzyme.Reverse), u) rtol = 1e-4
    else
        @test_skip false
    end
end

@testset "jax_source ODE failures raise as the native bridge throws" begin
    source = jax_source(jax_ode_budget; have = (:u, :ts, :limit), want = :total)
    native = prepare(jax_ode_budget; have = (:u, :ts, :limit), want = :total)
    u, ts = [log(2.0)], [1.0, 2.0, 30.0]
    @test native(u, ts, 10_000) ≈ sum(exp.(-2.0 .* ts)) rtol = 1e-8
    @test_throws ErrorException native(u, ts, 2)
    if _JAX_ODE_PYTHON
        @test rk_jax_evaluate(source, "kernel", (u, ts, 10_000)).value ≈ sum(exp.(-2.0 .* ts)) rtol = 1e-8
        @test_throws r"exceeded maxsteps between requested outputs" rk_jax_evaluate(source, "kernel", (u, ts, 2))
        @test_throws r"invalid initial state, output times" rk_jax_evaluate(source, "kernel", (u, [1.0, 0.5, 2.0], 10_000))
    else
        @test_skip false
    end
end

@testset "a data-bound ODE plate is one JAX module across data sizes" begin
    cases = jax_ode_case.((1, 3, 12))
    sources = [jax_source(jax_ode_cells; have = (:u, :doses, :times), want = :total, data)
               for data in cases]
    @test all(==(string(first(sources))), string.(sources))
    @test count("diffrax.diffeqsolve(", string(first(sources))) == 1
    native = prepare(jax_ode_cells; have = (:u, :doses, :times), want = :total)
    if _JAX_ODE_PYTHON
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        source = first(sources)
        inputs = first.(only(f for f in source.functions if f.name == "prepare_data").inputs)
        u = [0.1, -0.3, 1.0]
        for data in cases
            query = x -> native(x, data.doses, data.times)
            raw = Tuple(getproperty(data, Symbol(name)) for name in inputs)
            got = rk_jax_evaluate(source, "kernel", (u, raw...); grad = true, prepare = :arguments)
            @test got.value ≈ query(u) rtol = 1e-8
            @test got.grad ≈ gradient(query, backend, u) rtol = 1e-6
        end
    else
        @test_skip false
    end
end
