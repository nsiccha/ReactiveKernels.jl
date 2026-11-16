using ReactiveKernels, ReactiveKernelsDistributionKernels, Test
using ReactiveKernels: js_source, js_run
using ReactiveKernelsDistributionKernels.DistributionKernelSources: beta_binomial
import SpecialFunctions
isdefined(@__MODULE__, :rk_jax_evaluate) ||
    include(joinpath(@__DIR__, "../../../test/jax_python.jl"))

@kernel _beta_binomial_jax_cells(shapes::Vector{Float64}, ns::Vector{Int}, ys::Vector{Int}) = begin
    terms = plate(ns, ys) do n, y
        beta_binomial(n, shapes[1], shapes[2]).logpdf(y)
    end
    total = sum(terms)
    return total
end

@testset "beta-binomial JavaScript values" begin
    source = js_source(_beta_binomial_jax_cells;
        have = (:shapes, :ns, :ys), want = :total)
    if ReactiveKernels._js_default_node() !== nothing
        cases = [([2.0, 5.0], Int[], Int[]),
            ([2.0, 5.0], [0, 32, 64, 65, 1500], [0, 17, 40, 30, 900])]
        native = prepare(_beta_binomial_jax_cells;
            have = (:shapes, :ns, :ys), want = :total)
        for (args, result) in zip(cases, js_run(source, :kernel, cases))
            @test result.error === nothing
            @test result.value ≈ native(args...) rtol = 1e-10 atol = 1e-10
        end
    else
        @test_skip false
    end
end

@testset "beta-binomial JAX retains product loops" begin
    have = (:shapes, :ns, :ys)
    native = prepare(_beta_binomial_jax_cells; have, want = :total)
    source = jax_source(_beta_binomial_jax_cells; have, want = :total)
    @test occursin("jax.lax.while_loop", string(source))
    # Array sizes remain data; generating with different known data must not
    # replicate the numerical loop bodies. Compare definitions, not constants.
    emitted = [string(jax_source(_beta_binomial_jax_cells; have, want = :total,
        data = (; ns = fill(32, size), ys = fill(17, size)))) for size in (1, 17, 129)]
    @test allequal(count("jax.lax.while_loop", src) for src in emitted)
    if rk_jax_available()
        # Inspect reachable primal and reverse traces, including nested bodies.
        # These counts diagnose this fixed program's retained iteration as the
        # data length grows; they are not a general optimizer policy.
        mktempdir() do dir
            write(joinpath(dir, "bb.py"), source)
            write(joinpath(dir, "trace.py"), raw"""
import jax
import numpy as np
import bb
def primitives(jaxpr):
    jaxpr = getattr(jaxpr, 'jaxpr', jaxpr)
    names = []
    for eqn in jaxpr.eqns:
        names.append(eqn.primitive.name)
        for child in eqn.params.values():
            for part in child if isinstance(child, (tuple, list)) else (child,):
                if hasattr(part, 'jaxpr') or hasattr(part, 'eqns'):
                    names += primitives(part)
    return names
shapes = np.array([2.0, 5.0])
inventories = []
for size in (1, 17, 129):
    ns = np.full(size, 32, dtype=np.int64)
    ys = np.full(size, 17, dtype=np.int64)
    inventory = []
    for f in (bb.kernel, jax.grad(bb.kernel)):
        names = primitives(jax.make_jaxpr(f)(shapes, ns, ys))
        assert 'while' in names, names
        inventory.append((len(names), names.count('while'), names.count('scan')))
    inventories.append(inventory)
assert inventories[0] == inventories[1] == inventories[2], inventories
print('BETA_BINOMIAL_TRACE_OK', inventories)
""")
            output = read(Cmd(`$RK_JAX_PYTHON $(joinpath(dir, "trace.py"))`; dir), String)
            @test occursin("BETA_BINOMIAL_TRACE_OK", output)
            println(output)
        end
        for (ns, ys) in ((Int[], Int[]), ([0, 1, 31, 32, 33, 64], [0, 1, 15, 17, 33, 40]),
                         ([64, 65, 1500], [40, 30, 900]))
            shapes = [2.0, 5.0]
            result = rk_jax_evaluate(source, :kernel, (shapes, ns, ys); grad = true)
            @test result.value ≈ native(shapes, ns, ys) rtol = 1e-10 atol = 1e-10
            da = sum((SpecialFunctions.digamma(y + 2.0) - SpecialFunctions.digamma(2.0) +
                SpecialFunctions.digamma(7.0) - SpecialFunctions.digamma(n + 7.0)
                for (n, y) in zip(ns, ys)); init = 0.0)
            db = sum((SpecialFunctions.digamma(n - y + 5.0) - SpecialFunctions.digamma(5.0) +
                SpecialFunctions.digamma(7.0) - SpecialFunctions.digamma(n + 7.0)
                for (n, y) in zip(ns, ys)); init = 0.0)
            @test result.grad ≈ [da, db] rtol = 1e-10 atol = 1e-10
        end
        inactive = rk_jax_evaluate(source, :kernel, ([2.0, 5.0], [10, 10], [-1, 11]); grad = true)
        @test inactive.value == -Inf
        @test inactive.grad == [0.0, 0.0]
    else
        @test_skip false # RK_JAX_PYTHON enables execution and gradient parity.
    end
end
