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

@testset "a failed solve answers -inf in a rejecting function and raises in the others" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    ts = [1.0, 2.0, 30.0]
    good, bad = [10_000, 10_000, 10_000], [10_000, 2, 10_000]
    # (kernel, inputs, parameters, data that solves, data where one solve fails)
    cases = [
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
    for (kernel, have, u, solves, fails) in cases
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
            got = rk_jax_evaluate(source, "density", (u, solves...); grad = true)
            @test got.value ≈ query(u) rtol = 1e-6
            @test got.grad ≈ gradient(query, backend, u) rtol = 1e-4
            @test rk_jax_evaluate(source, "value", (u, solves...)).value ≈ got.value
            @test rk_jax_evaluate(source, "density", (u, fails...)).value == -Inf
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
            @test batch == one
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
