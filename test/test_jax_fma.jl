using ReactiveKernels, Test, Enzyme

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

# `fma(x, y, z)` rounds `x * y + z` once (snag jax-scalar-fma-h-ad6e34c6).
@kernel jax_fma_scalar(x::Float64, y::Float64, z::Float64) = begin
    result = fma(x, y, z)
    return result
end
@kernel jax_fma_scalar32(x::Float32, y::Float32, z::Float32) = begin
    result = fma(x, y, z)
    return result
end
@kernel jax_fma_mixed(x::Float32, y::Float64, n::Int) = begin
    result = fma(x, y, n)
    return result
end
@kernel jax_fma_integer(i::Int, j::Int, k::Int) = begin
    result = fma(i, j, k)
    return result
end
# Julia's `muladd` may or may not fuse: it is `x * y + z`, which XLA may fuse.
@kernel jax_muladd_scalar(x::Float64, y::Float64, z::Float64) = begin
    result = muladd(x, y, z)
    return result
end
jax_muladd_helper(x, y, z) = muladd.(x, y, z)
@kernel jax_muladd_broadcast(x, y, z) = begin
    value = jax_muladd_helper(x, y, z)
    return value
end
# A plain helper, as a consumer writes one.
jax_fma_helper(x, y, z) = fma.(x, y, z)
@kernel jax_fma_broadcast(x, y, z) = begin
    value = jax_fma_helper(x, y, z)
    total = sum(value)
    return total
end
# A Ref-scoped elementwise wrapper, as a consumer's macro generates one.
struct JaxFmaScope end
jax_fma_scoped(::Val{f}, ::JaxFmaScope, args...) where {f} = f(args...)
jax_fma_compat(x, y, z) = jax_fma_scoped.(Ref(Val(fma)), Ref(JaxFmaScope()), x, y, z)
@kernel jax_fma_compat_kernel(x, y, z) = begin
    value = jax_fma_compat(x, y, z)
    return value
end
@kernel jax_fma_literal(scale::Float64) = begin
    value = scale * fma(0.1, 10.0, -1.0)
    return value
end
@kernel jax_fma_prepared(scale::Float64, x::Vector{Float64}, y::Vector{Float64}) = begin
    errors = jax_fma_helper(x, y, -1.0)
    total = scale * sum(errors)
    return total
end

# Operands whose unfused `x * y + z` differs from `fma`: the product's rounding
# error is the whole result, or the product is a tie that `z` decides.
const JAX_FMA_CASES = (
    (1.0 + 2.0^-30, 1.0 - 2.0^-30, -1.0),
    (0.1, 10.0, -1.0),
    (1.0 + 2.0^-52, 1.0 + 2.0^-52, -(1.0 + 2.0^-51)),
    (1.0 + 2.0^-52, 1.5, -2.0^-200),
    (3.0e200, 7.0e100, -2.1e301),
    (-0.7, 1.3, 0.91),
)
jax_fma_bits_equal(a, b) = isequal(Float64.(a), Float64.(b))

# A module whose inputs are all data: `prepare_data` computes the result.
function jax_fma_all_data(src, args)
    dir = mktempdir()
    module_path = joinpath(dir, "rk_module.py")
    write(module_path, src)
    args_path = joinpath(dir, "args.py")
    write(args_path, "ARGS = [" * join(map(_rk_py_input, Any[args...]), ", ") * "]\n")
    driver = joinpath(dir, "all_data.py")
    write(driver, raw"""
import importlib.util, sys
import numpy as np
s = importlib.util.spec_from_file_location('rk_module', sys.argv[1])
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
exec(open(sys.argv[2]).read())
v = np.atleast_1d(np.asarray(m.kernel(**m.prepare_data(*ARGS)), dtype=np.float64))
print(" ".join(repr(float(t)) for t in v))
""")
    parse.(Float64, split(read(`$RK_JAX_PYTHON $driver $module_path $args_path`, String)))
end

@testset "jax_source: fma rounds x * y + z once" begin
    src = jax_source(jax_fma_scalar; have = (:x, :y, :z), want = :result)
    text = string(src)
    @test occursin("def _rk_fma(", text)
    @test occursin("jax.lax.optimization_barrier(", text)
    @test !occursin("custom_jvp", text)
    @test !occursin("have_fma", text)
    @test occursin("def _rk_fma32(", string(jax_source(jax_fma_scalar32;
        have = (:x, :y, :z), want = :result)))
    integer = string(jax_source(jax_fma_integer; have = (:i, :j, :k), want = :result))
    @test !occursin("_rk_fma", integer)
    @test occursin("i * j + k", integer)
    @test occursin("_rk_fma_np(", string(jax_source(jax_fma_literal; have = (:scale,),
                                                     want = :value)))
    @test count(((x, y, z),) -> fma(x, y, z) != x * y + z, JAX_FMA_CASES) >= 4

    # Native ordinary Reverse is the derivative reference, without custom rules.
    x, y, z = 0.3, -1.7, 2.5
    @test Enzyme.autodiff(Enzyme.Reverse, fma, Enzyme.Active, Enzyme.Active(x),
                          Enzyme.Active(y), Enzyme.Active(z))[1] == (y, x, 1.0)

    if !rk_jax_available()
        @test_skip false
    else
        for (x, y, z) in JAX_FMA_CASES
            r = rk_jax_evaluate(src, :kernel, (x, y, z); grad = true)
            @test jax_fma_bits_equal(r.value, fma(x, y, z))
            @test r.grad == y
        end
        r = rk_jax_evaluate(src, :kernel, (2.0, 3.0, 4.0))
        @test r.value === 10.0
        r = rk_jax_evaluate(src, :kernel, (Inf, 0.0, 1.0))
        @test isnan(r.value)
        @test rk_jax_evaluate(src, :kernel, (2.0, 3.0, -Inf)).value == -Inf
        @test rk_jax_evaluate(src, :kernel, (1.0e300, 1.0e10, -Inf)).value == -Inf

        src32 = jax_source(jax_fma_scalar32; have = (:x, :y, :z), want = :result)
        for (x, y, z) in ((1f0 + 2f0^-13, 1f0 - 2f0^-13, -1f0), (0.1f0, 10f0, -1f0),
                          (3f20, 2f10, -6f30), (-0.7f0, 1.3f0, 0.91f0))
            r = rk_jax_evaluate(src32, :kernel, (x, y, z); grad = true)
            @test Float32(r.value) === fma(x, y, z)
            @test Float32(r.grad) === y
        end
        @test fma(1f0 + 2f0^-13, 1f0 - 2f0^-13, -1f0) != (1f0 + 2f0^-13) * (1f0 - 2f0^-13) - 1f0

        xs = [c[1] for c in JAX_FMA_CASES]
        ys = [c[2] for c in JAX_FMA_CASES]
        zs = [c[3] for c in JAX_FMA_CASES]
        muladds = jax_source(jax_muladd_scalar; have = (:x, :y, :z), want = :result)
        @test !occursin("_rk_fma", string(muladds))
        @test occursin("x * y + z", string(muladds))
        for (x, y, z) in JAX_FMA_CASES
            r = rk_jax_evaluate(muladds, :kernel, (x, y, z); grad = true)
            @test r.value in (fma(x, y, z), x * y + z)
            @test r.grad == y
        end
        for (x, y, z) in ((xs, 0.1, zs), (0.1, ys, -1.0), (xs, ys, zs))
            types = (x = typeof(x), y = typeof(y), z = typeof(z))
            r = rk_jax_evaluate(jax_source(jax_muladd_broadcast; have = (:x, :y, :z),
                                           want = :value, types), :kernel, (x, y, z))
            operands = broadcast(tuple, x, y, z)
            @test length(r.value) == length(operands)
            @test all(v in (fma(t...), t[1] * t[2] + t[3]) for (v, t) in zip(r.value, operands))
        end

        mixed = jax_source(jax_fma_mixed; have = (:x, :y, :n), want = :result)
        @test occursin("_rk_fma(", string(mixed))
        r = rk_jax_evaluate(mixed, :kernel, (0.1f0, 10.0, -1))
        @test r.value === fma(0.1f0, 10.0, -1)
        @test rk_jax_evaluate(jax_source(jax_fma_integer; have = (:i, :j, :k),
            want = :result), :kernel, (7, -3, 25)).value == fma(7, -3, 25)
        @test rk_jax_evaluate(jax_source(jax_fma_literal; have = (:scale,), want = :value),
                              :kernel, (3.0,)).value === 3.0 * fma(0.1, 10.0, -1.0)

        # Scalar and container permutations, including empty and long vectors.
        for (x, y, z) in ((xs, ys, zs), (xs, 0.1, zs), (0.1, ys, -1.0), (xs, ys, -1.0),
                          (Float64[], 2.0, Float64[]), ([0.1], [10.0], [-1.0]),
                          (fill(0.1, 73), 10.0, -1.0))
            types = (x = typeof(x), y = typeof(y), z = typeof(z))
            values = jax_source(jax_fma_broadcast; have = (:x, :y, :z), want = :value, types)
            got = rk_jax_evaluate(values, :kernel, (x, y, z)).value
            want = fma.(x, y, z)
            @test jax_fma_bits_equal(got, want)
            total = jax_source(jax_fma_broadcast; have = (:x, :y, :z), want = :total, types)
            r = rk_jax_evaluate(total, :kernel, (x, y, z); grad = true)
            @test r.value ≈ sum(want)
            @test r.grad ≈ (x isa Number ? sum(y .+ zero.(z)) : y .+ zero.(x))
        end

        # The Ref-scoped wrapper, in the functions and wholly in prepare_data.
        for (x, y, z) in ((xs, 0.1, zs), (0.1, ys, -1.0), (xs, ys, zs))
            types = (x = typeof(x), y = typeof(y), z = typeof(z))
            want = fma.(x, y, z)
            device = jax_source(jax_fma_compat_kernel; have = (:x, :y, :z), want = :value,
                                types)
            @test jax_fma_bits_equal(rk_jax_evaluate(device, :kernel, (x, y, z)).value, want)
            host = jax_source(jax_fma_compat_kernel; have = (:x, :y, :z), want = :value,
                              types, data = (:x, :y, :z))
            @test occursin("_rk_fma_np(", string(host))
            @test jax_fma_bits_equal(jax_fma_all_data(host, (x, y, z)), want)
        end

        # Data-only fma runs in host prepare_data, with the same rounding.
        prep = jax_source(jax_fma_prepared; have = (:scale, :x, :y), want = :total,
                          data = (x = xs, y = ys))
        @test occursin("_rk_fma_np(", string(prep))
        for n in (0, 1, length(xs))
            x, y = xs[1:n], ys[1:n]
            r = rk_jax_evaluate(prep, :kernel, (0.3, x, y); grad = true, prepare = true)
            @test r.value ≈ 0.3 * sum(fma.(x, y, -1.0))
            @test r.grad ≈ sum(fma.(x, y, -1.0))
        end

        # Data lengths change batched array extents, never the emitted trace.
        mktempdir() do dir
            module_path = joinpath(dir, "fma.py")
            write(module_path, jax_source(jax_fma_broadcast; have = (:x, :y, :z),
                want = :total, types = (x = Vector{Float64}, y = Vector{Float64},
                                         z = Vector{Float64})))
            driver = joinpath(dir, "structure.py")
            write(driver, raw"""
import importlib.util, sys
import jax, jax.numpy as jnp
s = importlib.util.spec_from_file_location('fma', sys.argv[1])
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
counts = [len(jax.make_jaxpr(m.kernel)(jnp.ones(n), jnp.ones(n), jnp.ones(n)).jaxpr.eqns)
          for n in (1, 19, 73)]
assert counts[0] > 0 and len(set(counts)) == 1, counts
""")
            @test success(`$RK_JAX_PYTHON $driver $module_path`)
        end
    end
end
