using ReactiveKernels, OrdinaryDiffEqBDF, OrdinaryDiffEqLowOrderRK,
    SciMLBase, SciMLSensitivity, Enzyme, Test
using DifferentiationInterface: AutoEnzyme, gradient
isdefined(@__MODULE__, :rk_jax_evaluate) ||
    include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

# The emitted module solves ODEs with diffrax; the parity checks need it in
# the `RK_JAX_PYTHON` environment (`pip install diffrax`).
const _JAX_ODE_PYTHON = rk_jax_available() && success(`$RK_JAX_PYTHON -c "import diffrax"`)

# The value a function answers alone and under `jax.value_and_grad`. On a CUDA
# device these are two programs, the per-lane kernel without and with the
# tangents, and they agree to rounding: in the NaN case below by one unit in the
# last place (-2.6271005178169045 against -2.627100517816905). Elsewhere they
# agree exactly.
const _JAX_ODE_CUDA = _JAX_ODE_PYTHON &&
    readchomp(`$RK_JAX_PYTHON -c "import jax; print(jax.default_backend())"`) == "gpu"
jax_ode_same_value(a, b) = _JAX_ODE_CUDA ? isapprox(a, b; rtol = 1e-12) : a == b

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

# The solution after t = 1 is log1p(k * (t - 1)^2 / 2). A large trial step
# across the onset overflows exp(-y) even though the solution stays finite.
_jax_ode_ramp(t, y, k) = [k * max(t - 1, 0.0) * exp(-y[1])]
@kernel jax_ode_retry(u::Vector{Float64}, ts::Vector{Float64}) = begin
    sol = rk_ode_rk45_tol(_jax_ode_ramp, [0.0], 0.0, ts, 1e-8, 1e-8, 10^4, u[1])
    total = sol[end, 1]
    return total
end

@kernel jax_ode_retry_cells(rates::Vector{Float64}, ts::Vector{Float64}) = begin
    values = plate(rates) do k
        sol = rk_ode_rk45_tol(_jax_ode_ramp, [0.0], 0.0, ts, 1e-8, 1e-8, 10^4, k)
        sol[end, 1]
    end
    total = sum(values)
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

@kernel jax_ode_precision(k::Float64, rt::Float64, ts::Vector{Float64}) = begin
    total = k > 0 ? sum(rk_ode_rk45_tol(_jax_ode_elimination, [1.0], 0.0, ts, rt, 1e-7, 1000, k)) : 0.0
    return total
end

_jax_ode_relaxation(t, y, p) = [p - y[1]]
@kernel jax_ode_equilibrium(q::Vector{Float64}, ts::Vector{Float64}) = begin
    sol = rk_ode_rk45_tol(_jax_ode_relaxation, [q[1]], q[3], ts, 1e-7, 1e-8, 10000, q[2])
    total = sum(sol)
    return total
end

@testset "RK45 controls the response to perturbations at equilibrium" begin
    q, ts = zeros(3), [100.0]
    query = prepare(jax_ode_equilibrium; have = (:q, :ts), want = :total)
    @test query(q, ts) == 0
    @test gradient(x -> query(x, ts), AutoEnzyme(; mode = Enzyme.Reverse), q) ≈ [0, 1, 0] atol = 1e-6
    if _JAX_ODE_PYTHON
        mktempdir() do dir
            for T in (Float32, Float64)
                write(joinpath(dir, "model.py"), jax_source(jax_ode_equilibrium;
                    have = (:q, :ts), want = :total, float_type = T))
                driver = joinpath(@__DIR__, "jax_ode_sensitivity.py")
                cmd = addenv(`$RK_JAX_PYTHON -B $driver $dir`,
                             "JAX_ENABLE_X64" => (T === Float64 ? "1" : "0"))
                @test run(cmd).exitcode == 0
            end
        end
    else
        @test_skip false
    end
end

@testset "runtime ODE relative tolerance in the solve precision" begin
    if _JAX_ODE_PYTHON
        mktempdir() do dir
            for T in (Float32, Float64)
                source = jax_source(jax_ode_precision; have = (:k, :rt, :ts), want = :total, float_type = T)
                write(joinpath(dir, "model.py"), source)
                # Separate processes exercise each module with its intended JAX dtype.
                driver = joinpath(@__DIR__, "jax_ode_precision.py")
                @test success(addenv(`$RK_JAX_PYTHON -B $driver $dir`,
                                     "JAX_ENABLE_X64" => (T === Float64 ? "1" : "0")))
            end
        end
    else
        @test_skip false
    end
end

@testset "jax_source solves rk_ode_rk45_tol and rk_ode_bdf_tol with diffrax" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    source = jax_source(jax_ode_cells; have = (:u, :doses, :times), want = :total)
    text = string(source)
    @test occursin("import diffrax", text)
    @test occursin("_RKDopri5()", text)
    # One call to the ODE bridge, vmapped over the cells.
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
    @test occursin("_RKKvaerno5()", string(stiff))
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

@testset "a rejected non-finite ODE trial leaves a finite solution and gradient" begin
    u, ts = [100.0], [1.0, 2.0]
    source = jax_source(jax_ode_retry; have = (:u, :ts), want = :total)
    native = prepare(jax_ode_retry; have = (:u, :ts), want = :total)
    query = x -> native(x, ts)
    value = query(u)
    grad = gradient(query, AutoEnzyme(; mode = Enzyme.Reverse), u)
    @test value ≈ log1p(u[1] / 2) rtol = 1e-6
    @test only(grad) ≈ 1 / (2 + u[1]) rtol = 1e-6
    if _JAX_ODE_PYTHON
        got = rk_jax_evaluate(source, "kernel", (u, ts); grad = true)
        @test got.value ≈ value rtol = 1e-6
        @test got.grad ≈ grad rtol = 1e-6
        batched = jax_source(jax_ode_retry_cells; have = (:rates, :ts), want = :total)
        for rates in ([10.0, 12.0], [10.0, 100.0])
            got = rk_jax_evaluate(batched, "kernel", (rates, ts); grad = :joint)
            @test got.joint_value ≈ sum(log1p.(rates ./ 2)) rtol = 1e-6
            @test got.grad ≈ 1 ./ (2 .+ rates) rtol = 1e-6
        end
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
    @test_throws ReactiveKernels.ODESolveFailure native(u, ts, 2)
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
    # One ODE solve (its definition and one call).
    @test count("_rk_ode_solve(", string(first(sources))) == 2
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

# A failed solve (an exceeded per-output budget, a finite-time blow-up, a
# right-hand side that turns NaN) is a numerical failure: natively
# `ReactiveKernels.ODESolveFailure`, and in JAX a `checkify` check that leaves
# every `vmap`, `scan` and `cond` as a value. A function `rejecting` names
# answers -inf there; another raises; invalid inputs and other runtime checks
# still raise (snag rk-ode-solve-fai-3c664466).
_jax_ode_blowup(t, y, k) = k .* y .^ 2
# A branch, so its authored body is what JAX translates.
@traceable function _jax_ode_nan_below(t, y, k)
    return [y[1] < 0.5 ? NaN : -k * y[1]]
end
@traceable function _jax_ode_checked_rate(k)
    k < 100 || throw(ArgumentError("the elimination rate must be below 100"))
    k
end

# One solve per plate cell, with the cell's own step budget.
@kernel jax_ode_reject_cells(u::Vector{Float64}, limits::Vector{Int}, ts::Vector{Float64}) = begin
    k = _jax_ode_checked_rate(exp(u[1]))
    values = plate(limits) do limit
        sol = rk_ode_rk45_tol(_jax_ode_elimination, [1.0], 0.0, ts, 1e-10, 1e-10, limit, k)
        sum(sol)
    end
    total = sum(values) - u[1]^2
    return total
end
# One solve per scan step.
@kernel jax_ode_reject_scan(u::Vector{Float64}, limits::Vector{Int}, ts::Vector{Float64}) = begin
    k = exp(u[1])
    path = scan(limits; init = 0.0) do acc, limit
        next = acc + sum(rk_ode_rk45_tol(_jax_ode_elimination, [1.0], 0.0, ts, 1e-10, 1e-10, limit, k))
        (next, next)
    end
    total = path[end] - u[1]^2
    return total
end
# A solve in the arm of a branch on the parameter (`jax.lax.cond`), and in one
# of a branch on data inside a plate cell, whose vmap evaluates both arms.
@kernel jax_ode_reject_branch(u::Vector{Float64}, limits::Vector{Int}, signs::Vector{Float64},
                              ts::Vector{Float64}) = begin
    k = exp(u[1])
    values = plate(limits, signs) do limit, s
        s > 0 ? sum(rk_ode_rk45_tol(_jax_ode_elimination, [1.0], 0.0, ts, 1e-10, 1e-10, limit, k)) :
            s * u[1]
    end
    head = u[2] > 0 ? sum(rk_ode_rk45_tol(_jax_ode_elimination, [1.0], 0.0, ts, 1e-10, 1e-10,
                                          limits[1], k)) : -u[2]^2
    total = sum(values) + head
    return total
end
# A solve that blows up in finite time (t = 1 / (k y0)) or turns NaN: both
# fail promptly, as natively, instead of rejecting steps until diffrax's step
# limit.
@kernel jax_ode_reject_blowup(u::Vector{Float64}, y0s::Vector{Float64}, ts::Vector{Float64}) = begin
    k = exp(u[1])
    values = plate(y0s) do y0
        sum(rk_ode_rk45_tol(_jax_ode_blowup, [y0], 0.0, ts, 1e-8, 1e-8, 10^4, k))
    end
    total = sum(values) - u[1]^2
    return total
end
@kernel jax_ode_reject_nan(u::Vector{Float64}, ts::Vector{Float64}) = begin
    sol = rk_ode_rk45_tol(_jax_ode_nan_below, [1.0], 0.0, ts, 1e-8, 1e-8, 10^4, exp(u[1]))
    total = sum(sol) - u[1]^2
    return total
end
# Two solves in sequence: the second starts from the first's last state, after
# the first's check (where unfixed JAX checkify differentiates neither).
@kernel jax_ode_reject_chain(u::Vector{Float64}, limits::Vector{Int}, ts::Vector{Float64}) = begin
    k = exp(u[1])
    first_sol = rk_ode_rk45_tol(_jax_ode_elimination, [1.0], 0.0, ts, 1e-10, 1e-10, limits[1], k)
    second_sol = rk_ode_rk45_tol(_jax_ode_elimination, [first_sol[end, 1]], 0.0, ts, 1e-10, 1e-10,
                                 limits[2], 2 * k)
    total = sum(first_sol) + sum(second_sol) - u[1]^2
    return total
end
# One solve per iteration of a helper's counted loop (a `jax.lax.fori_loop`).
@traceable function _jax_ode_loop_sum(limits, ts, k)
    acc = zero(k)
    for i in eachindex(limits)
        acc = acc + sum(rk_ode_rk45_tol(_jax_ode_elimination, [one(k)], 0.0, ts, 1e-10, 1e-10,
                                        limits[i], k))
    end
    return acc
end
@kernel jax_ode_reject_loop(u::Vector{Float64}, limits::Vector{Int}, ts::Vector{Float64}) = begin
    k = exp(u[1])
    total = _jax_ode_loop_sum(limits, ts, k) - u[1]^2
    return total
end

# (kernel, inputs, parameters, data that solves, data where one solve fails)
function jax_ode_reject_cases()
    ts = [1.0, 2.0, 30.0]
    good, bad = [10_000, 10_000, 10_000], [10_000, 2, 10_000]
    return [
        (jax_ode_reject_cells, (:u, :limits, :ts), [0.7], (good, ts), (bad, ts)),
        (jax_ode_reject_scan, (:u, :limits, :ts), [0.7], (good, ts), (bad, ts)),
        # The failing cell's arm is taken (its sign is positive), then not.
        (jax_ode_reject_branch, (:u, :limits, :signs, :ts), [0.7, -0.4],
         (bad, [1.0, -1.0, 1.0], ts), (bad, [1.0, 1.0, 1.0], ts)),
        # The parameter branch takes the solve with the first cell's budget.
        (jax_ode_reject_branch, (:u, :limits, :signs, :ts), [0.7, 0.4],
         ([10_000, 2, 10_000], [-1.0, -1.0, -1.0], ts), ([2, 10_000, 10_000], [-1.0, -1.0, -1.0], ts)),
        # k = exp(-2): the solve from 0.3 blows up at t = 24.6.
        (jax_ode_reject_blowup, (:u, :y0s, :ts), [-2.0], ([0.2, 0.3, 0.25], [0.5, 1.0, 2.0]),
         ([0.2, 0.3, 0.25], [0.5, 1.0, 30.0])),
        (jax_ode_reject_nan, (:u, :ts), [log(0.1)], ([0.5, 1.0, 2.0],), ([0.5, 1.0, 20.0],)),
        # The second solve fails, then the first.
        (jax_ode_reject_chain, (:u, :limits, :ts), [0.7], (good, ts), (bad, ts)),
        (jax_ode_reject_chain, (:u, :limits, :ts), [0.7], (good, ts), ([2, 10_000, 10_000], ts)),
        (jax_ode_reject_loop, (:u, :limits, :ts), [0.7], (good, ts), (bad, ts)),
    ]
end

@testset "a failed solve answers -inf in a rejecting function and raises in the others" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    ts = [1.0, 2.0, 30.0]
    good, bad = [10_000, 10_000, 10_000], [10_000, 2, 10_000]
    for (kernel, have, u, solves, fails) in jax_ode_reject_cases()
        source = jax_source(kernel; have, functions = (density = :total, value = :total),
                            rejecting = (:density,))
        text = string(source)
        @test occursin("checkify.check(", text) && occursin("_rk_rejecting(", text) &&
              occursin("_rk_raising(", text) && occursin("error.has_error()", text)
        # Without checkify's fix and `has_error` it refuses to import.
        @test occursin("if not hasattr(checkify.Error, \"has_error\"):", text)
        native = prepare(kernel; have, want = :total)
        @test isfinite(native(u, solves...))
        @test_throws ReactiveKernels.ODESolveFailure native(u, fails...)
        if _JAX_ODE_PYTHON
            query = x -> native(x, solves...)
            got = rk_jax_evaluate(source, "density", (u, solves...); grad = :joint)
            @test got.value ≈ query(u) rtol = 1e-6
            @test jax_ode_same_value(got.joint_value, got.value)
            @test got.grad ≈ gradient(query, backend, u) rtol = 1e-4
            @test rk_jax_evaluate(source, "value", (u, solves...)).value ≈ got.value
            # -inf also from jax.value_and_grad, as a sampler evaluates the
            # point: an error lost under differentiation would read finite.
            failed = rk_jax_evaluate(source, "density", (u, fails...); grad = :joint)
            @test failed.value == -Inf && failed.joint_value == -Inf
            @test_throws r"exceeded maxsteps between requested outputs|\) failed" rk_jax_evaluate(
                source, "value", (u, fails...))
        else
            @test_skip false
        end
    end
    # The loop case's solve does sit in a JAX loop body.
    @test occursin("jax.lax.fori_loop(", string(jax_source(jax_ode_reject_loop;
        have = (:u, :limits, :ts), functions = (density = :total,), rejecting = (:density,))))
    source = jax_source(jax_ode_reject_cells; have = (:u, :limits, :ts),
                        functions = (density = :total,), rejecting = (:density,))
    native = prepare(jax_ode_reject_cells; have = (:u, :limits, :ts), want = :total)
    # Invalid inputs and a runtime check that is no numerical failure raise,
    # as natively.
    @test_throws DomainError native([0.7], good, [1.0, 0.5, 2.0])
    @test_throws ArgumentError native([log(200.0)], good, ts)
    if _JAX_ODE_PYTHON
        @test_throws r"invalid initial state, output times" rk_jax_evaluate(
            source, "density", ([0.7], good, [1.0, 0.5, 2.0]))
        @test_throws r"the elimination rate must be below 100" rk_jax_evaluate(
            source, "density", ([log(200.0)], good, ts))
        # A batch of points under `jax.vmap`, as a sampler's chains: each
        # point answers its own value, -inf where its solve blows up before
        # t = 2 (k > 1 / (0.3 * 2), u[1] > 0.51).
        blowup = jax_source(jax_ode_reject_blowup; have = (:u, :y0s, :ts),
                            functions = (density = :total,), rejecting = (:density,))
        mktempdir() do dir
            path = joinpath(dir, "reject.py")
            write(path, blowup)
            program = """
                import importlib.util
                import jax, jax.numpy as jnp, numpy as np
                spec = importlib.util.spec_from_file_location('reject', $(_rk_py(path)))
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                y0s, ts = np.array([0.2, 0.3, 0.25]), np.array([0.5, 1.0, 2.0])
                us = jnp.array([[-2.0], [0.0], [1.0], [2.0]])
                f = lambda u: m.density(u, y0s, ts)
                batch = np.asarray(jax.jit(jax.vmap(f))(us))
                one = np.asarray([jax.jit(f)(u) for u in us])
                print(" ".join(repr(float(v)) for v in batch), "|", " ".join(repr(float(v)) for v in one))
                """
            batch, one = split(readchomp(`$RK_JAX_PYTHON -c $program`), " | ")
            batch, one = parse.(Float64, split(batch)), parse.(Float64, split(one))
            native = prepare(jax_ode_reject_blowup; have = (:u, :y0s, :ts), want = :total)
            @test batch[1:2] ≈ [native([u], [0.2, 0.3, 0.25], [0.5, 1.0, 2.0]) for u in (-2.0, 0.0)] rtol = 1e-6
            @test batch[3:4] == [-Inf, -Inf]
            # Batched and one-point programs may round differently.
            @test all(isapprox.(batch, one; rtol = 1e-12))
        end
        # Per-chain gradients under `jax.vmap`, as a sampler takes them, of a
        # model whose solve sits in a scan step: those of one point at a time.
        scanned = jax_source(jax_ode_reject_scan; have = (:u, :limits, :ts),
                             functions = (density = :total,), rejecting = (:density,))
        mktempdir() do dir
            path = joinpath(dir, "reject_scan.py")
            write(path, scanned)
            program = """
                import importlib.util
                import jax, jax.numpy as jnp, numpy as np
                spec = importlib.util.spec_from_file_location('reject_scan', $(_rk_py(path)))
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                limits, ts = np.array([10000, 10000, 10000]), np.array([1.0, 2.0, 30.0])
                us = jnp.array([[0.7], [0.2], [-0.3]])
                g = jax.grad(lambda u: m.density(u, limits, ts))
                batch = np.asarray(jax.jit(jax.vmap(g))(us)).ravel()
                one = np.asarray([jax.jit(g)(u) for u in us]).ravel()
                print(" ".join(repr(float(v)) for v in batch), "|", " ".join(repr(float(v)) for v in one))
                """
            batch, one = split(readchomp(`$RK_JAX_PYTHON -c $program`), " | ")
            batch, one = parse.(Float64, split(batch)), parse.(Float64, split(one))
            native = prepare(jax_ode_reject_scan; have = (:u, :limits, :ts), want = :total)
            @test batch ≈ [only(gradient(x -> native(x, good, ts), backend, [u]))
                           for u in (0.7, 0.2, -0.3)] rtol = 1e-4
            @test batch ≈ one rtol = 1e-12
        end
    else
        @test_skip false
    end
    # A function the module does not define, or with several values.
    @test_throws ArgumentError jax_source(jax_ode_reject_cells; have = (:u, :limits, :ts),
                                          functions = (density = :total,), rejecting = (:other,))
    @test_throws ArgumentError jax_source(jax_ode_reject_cells; have = (:u, :limits, :ts),
                                          functions = (density = (:total, :k),), rejecting = (:density,))
    # A module without a numerical failure is unchanged.
    plain = string(jax_source(jax_ode_none; have = (:u,), functions = (density = :total,),
                              rejecting = (:density,)))
    @test plain == string(jax_source(jax_ode_none; have = (:u,), functions = (density = :total,)))
    @test !occursin("checkify", plain)
end

# On CUDA a module integrates every solve per lane in one Pallas kernel, with
# forward-mode sensitivities (snag jax-ode-solve-gp-d427eb5a); `rk_ode_bdf_tol`
# keeps diffrax there, also with forward-mode sensitivities.
# `RK_JAX_ODE_KERNEL=interpret` runs that path on any backend, its kernel in
# Pallas' interpreter, so these checks need no GPU. Run on a CUDA device, the
# testsets above check the compiled kernel the same way.
const _JAX_ODE_PALLAS = _JAX_ODE_PYTHON &&
    success(`$RK_JAX_PYTHON -c "from jax.experimental import pallas"`)

# Pallas' interpreter also runs a kernel that Triton refuses to compile: Triton
# takes `select_n` only with two cases and a bool predicate, and a branch in a
# right-hand side once picked with an int32 one. So this reads the jaxpr of
# `jax.value_and_grad(name)` at `args`, as the lanes trace it: the number of
# Pallas kernels and of the selects in them that Triton refuses. It also checks
# the selection tree against `select_n` of the clamped index, at every index from
# below to above range, for one to five cases.
const _JAX_ODE_TRITON_SELECTS = raw"""
import importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location("rk_module", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import jax, jax.numpy as jnp
from jax.extend.core import ClosedJaxpr, Jaxpr
exec(open(sys.argv[2]).read())
f = getattr(module, sys.argv[3])
first, rest = ARGS[0], ARGS[1:]
counts = {"kernels": 0, "refused": 0}

def walk(jaxpr, inside):
    for e in jaxpr.eqns:
        kernel = e.primitive.name == "pallas_call"
        counts["kernels"] += kernel
        if inside and e.primitive.name == "select_n" and (
                len(e.invars) != 3 or e.invars[0].aval.dtype != np.bool_):
            counts["refused"] += 1
        for v in e.params.values():
            for x in (v if isinstance(v, (tuple, list)) else (v,)):
                if isinstance(x, (ClosedJaxpr, Jaxpr)):
                    walk(getattr(x, "jaxpr", x), inside or kernel)

walk(jax.make_jaxpr(jax.value_and_grad(lambda u: f(u, *rest)))(first).jaxpr, False)
which = jnp.arange(-2, 7, dtype=jnp.int32)
tree = True
for n in range(1, 6):
    cases = [jnp.full(which.shape, float(k)) for k in range(n)]
    got = module._rk_soa_select(which.shape, module._RKLane(which), *map(module._RKLane, cases))
    tree &= bool(jnp.all(got.v == jax.lax.select_n(jnp.clip(which, 0, n - 1), *cases)))
print(counts["kernels"], counts["refused"], tree)
"""

function jax_ode_triton_selects(src, name, args)
    dir = mktempdir()
    check, path, inputs = joinpath.(dir, ("check.py", "module.py", "args.py"))
    write(check, _JAX_ODE_TRITON_SELECTS)
    write(path, src)
    write(inputs, _rk_jax_args(rk_jax_inputs(args), nothing))
    kernels, refused, tree = split(readchomp(`$RK_JAX_PYTHON $check $path $inputs $name`))
    return (; kernels = parse(Int, kernels), refused = parse(Int, refused), tree = tree == "True")
end

@testset "the CUDA path's per-lane solves match the native bridge (Pallas' interpreter)" begin
    @test occursin("_rk_lane_kernel(", string(jax_source(jax_ode_cells; have = (:u, :doses, :times),
                                                           want = :total)))
    if _JAX_ODE_PALLAS
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        lanes(f) = withenv(f, "RK_JAX_ODE_KERNEL" => "interpret")
        diffrax_only(f) = withenv(f, "RK_JAX_ODE_KERNEL" => "off")
        # Plates of solves with ragged output times: the native values and
        # gradients, and diffrax's own to rounding, as the lanes take its steps.
        source = jax_source(jax_ode_cells; have = (:u, :doses, :times), want = :total)
        native = prepare(jax_ode_cells; have = (:u, :doses, :times), want = :total)
        doses = [100.0, 80.0, 120.0]
        times = [[0.5, 1.0, 2.0], [0.5, 3.0], [1.0, 2.0, 4.0, 8.0]]
        for u in ([0.3, -0.5, 1.2], [-0.2, 0.1, 0.8])
            query = x -> native(x, doses, times)
            got = lanes(() -> rk_jax_evaluate(source, "kernel", (u, doses, times); grad = :joint))
            @test got.value ≈ query(u) rtol = 1e-8
            @test jax_ode_same_value(got.joint_value, got.value)
            @test got.grad ≈ gradient(query, backend, u) rtol = 1e-6
            ref = diffrax_only(() -> rk_jax_evaluate(source, "kernel", (u, doses, times); grad = true))
            @test got.value ≈ ref.value rtol = 1e-11
            @test got.grad ≈ ref.grad rtol = 1e-8
        end
        # The stiff solver: diffrax with forward-mode sensitivities.
        stiff = jax_source(jax_ode_stiff; have = (:u, :ts), want = :total)
        stiff_native = prepare(jax_ode_stiff; have = (:u, :ts), want = :total)
        ts = [0.4, 4.0, 40.0, 400.0]
        u = [log(0.04)]
        got = lanes(() -> rk_jax_evaluate(stiff, "kernel", (u, ts); grad = true))
        @test got.value ≈ stiff_native(u, ts) rtol = 1e-6
        @test got.grad ≈ gradient(x -> stiff_native(x, ts), backend, u) rtol = 1e-4
        # A trial that overflows is retried at a smaller step, alone and beside
        # a lane that needs no retry.
        u, ts = [100.0], [1.0, 2.0]
        retry = jax_source(jax_ode_retry; have = (:u, :ts), want = :total)
        got = lanes(() -> rk_jax_evaluate(retry, "kernel", (u, ts); grad = true))
        @test got.value ≈ log1p(u[1] / 2) rtol = 1e-6
        @test only(got.grad) ≈ 1 / (2 + u[1]) rtol = 1e-6
        batched = jax_source(jax_ode_retry_cells; have = (:rates, :ts), want = :total)
        for rates in ([10.0, 12.0], [10.0, 100.0])
            got = lanes(() -> rk_jax_evaluate(batched, "kernel", (rates, ts); grad = :joint))
            @test got.joint_value ≈ sum(log1p.(rates ./ 2)) rtol = 1e-6
            @test got.grad ≈ 1 ./ (2 .+ rates) rtol = 1e-6
        end
        # A right-hand side reached through a splatting helper.
        wrapped = jax_source(jax_ode_wrapped; have = (:u, :doses, :times), want = :total)
        wrapped_native = prepare(jax_ode_wrapped; have = (:u, :doses, :times), want = :total)
        u = [log(0.3)]
        query = x -> wrapped_native(x, doses, times)
        got = lanes(() -> rk_jax_evaluate(wrapped, "kernel", (u, doses, times); grad = true))
        @test got.value ≈ query(u) rtol = 1e-5
        @test got.grad ≈ gradient(query, backend, u) rtol = 1e-4
        # Budgets and invalid inputs fail as natively.
        budget = jax_source(jax_ode_budget; have = (:u, :ts, :limit), want = :total)
        u, ts = [log(2.0)], [1.0, 2.0, 30.0]
        @test lanes(() -> rk_jax_evaluate(budget, "kernel", (u, ts, 10_000))).value ≈
              sum(exp.(-2.0 .* ts)) rtol = 1e-8
        @test_throws r"exceeded maxsteps between requested outputs" lanes(
            () -> rk_jax_evaluate(budget, "kernel", (u, ts, 2)))
        @test_throws r"invalid initial state, output times" lanes(
            () -> rk_jax_evaluate(budget, "kernel", (u, [1.0, 0.5, 2.0], 10_000)))
        # A failed solve answers -inf in a rejecting function, in plates, scans,
        # branches, chained solves and loops, after a blow-up or a NaN.
        for (kernel, have, u, solves, fails) in jax_ode_reject_cases()
            source = jax_source(kernel; have, functions = (density = :total, value = :total),
                                rejecting = (:density,))
            native = prepare(kernel; have, want = :total)
            query = x -> native(x, solves...)
            got = lanes(() -> rk_jax_evaluate(source, "density", (u, solves...); grad = :joint))
            @test got.value ≈ query(u) rtol = 1e-6
            @test jax_ode_same_value(got.joint_value, got.value)
            @test got.grad ≈ gradient(query, backend, u) rtol = 1e-4
            failed = lanes(() -> rk_jax_evaluate(source, "density", (u, fails...); grad = :joint))
            @test failed.value == -Inf && failed.joint_value == -Inf
            @test_throws r"exceeded maxsteps between requested outputs|\) failed" lanes(
                () -> rk_jax_evaluate(source, "value", (u, fails...)))
            # Triton compiles every select in the kernels, among them the NaN
            # case's branch in its right-hand side.
            selects = lanes(() -> jax_ode_triton_selects(source, "density", (u, solves...)))
            @test selects.kernels >= 1 && selects.refused == 0 && selects.tree
        end
    else
        @test_skip false
    end
end
