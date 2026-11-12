using ReactiveKernels, Test, Enzyme

isdefined(@__MODULE__, :rk_jax_available) || include(joinpath(@__DIR__, "jax_python.jl"))

struct JaxElementwiseScope end
jax_elementwise_scalar(::Val{f}, ::JaxElementwiseScope, args...) where {f} = f(args...)
jax_matching_container(tag, ref, x) = ref
jax_matching_container(tag, ::Nothing, x::AbstractArray) = x
jax_matching_container(tag, ref::AbstractArray, x::AbstractArray) =
    (jax_matching_sizes(ref, x); ref)
jax_matching_sizes(x, y) = size(x) == size(y) ||
    throw(DimensionMismatch("elementwise arguments must have equal sizes"))

# One ordinary Julia implementation: a captured callable checks the fixed
# argument list, then Ref keeps the tag and scope scalar during broadcasting.
function jax_elementwise(tag, scope, args...)
    Base.afoldl((ref, x) -> jax_matching_container(tag, ref, x), nothing, args...)
    jax_elementwise_scalar.(Ref(tag), Ref(scope), args...)
end
jax_elementwise_exp(x) = jax_elementwise(Val(exp), JaxElementwiseScope(), x)
jax_elementwise_hypot(x, y) = jax_elementwise(Val(hypot), JaxElementwiseScope(), x, y)
jax_elementwise_affine(x, y, z) = x * y + z
jax_elementwise_three(x, y, z) =
    jax_elementwise(Val(jax_elementwise_affine), JaxElementwiseScope(), x, y, z)
jax_elementwise_exp_sum(x) = sum(jax_elementwise_exp(x))

@kernel jax_elementwise_unary(x) = begin
    value = jax_elementwise_exp(x)
    total = sum(value)
    return total
end
@kernel jax_elementwise_binary(x, y) = begin
    value = jax_elementwise_hypot(x, y)
    total = sum(value)
    return total
end
@kernel jax_elementwise_ternary(x, y, z) = begin
    value = jax_elementwise_three(x, y, z)
    total = sum(value)
    return total
end
@kernel jax_elementwise_prepared(scale::Float64, x, y) = begin
    values = jax_elementwise_hypot(x, y)
    total = scale * sum(values)
    return total
end

# Numeric captures stay runtime values. Ref of an array passes that whole
# array to each scalar invocation, including its ragged extent metadata.
jax_captured_scale(a, x) = ((v) -> a * v).(x)
jax_ref_weighted(w, x) = ((weights, v) -> sum(weights) * v).(Ref(w), x)
jax_captured_call(a, x) = ((v) -> a * v)(x)
jax_captured_array(w, x) = ((v) -> sum(w) * v).(x)
jax_ref_scalar(x) = Float64.(Ref(x))
jax_ref_array(x) = identity.(Ref(x))
@kernel jax_elementwise_captures(a::Float64, w::Vector{Float64}, x::Vector{Float64}) = begin
    values = jax_captured_scale(a, x)
    weighted = jax_captured_array(w, values)
    total = jax_captured_call(a, sum(weighted))
    return total
end
@kernel jax_elementwise_refs(x::Vector{Float64}, scale::Float64) = begin
    array = jax_ref_array(x)
    scalar = jax_ref_scalar(scale)
    value = array .* scalar
    return value
end
@kernel jax_elementwise_escaping_ref(x::Vector{Float64}) = begin
    value = Ref(x)
    return value
end
@kernel jax_elementwise_ragged(xs::Vector{Vector{Float64}}) = begin
    values = plate(xs) do x
        sum(jax_ref_array(jax_ref_weighted(x, jax_elementwise_exp(x))))
    end
    total = sum(values)
    return total
end
jax_elementwise_guarded(x) = x > 0 ? log(x) : 0.0
@kernel jax_elementwise_inactive(x::Vector{Float64}) = begin
    values = jax_elementwise(Val(jax_elementwise_guarded), JaxElementwiseScope(), x)
    total = sum(values)
    return total
end

@testset "jax_source: closures and Ref elementwise calls" begin
    unary = jax_source(jax_elementwise_unary; have = (:x,), want = :total,
                       types = (x = Vector{Float64},))
    binary = jax_source(jax_elementwise_binary; have = (:x, :y), want = :total,
                        types = (x = Vector{Float64}, y = Vector{Float64}))
    @test occursin("jnp.vectorize(", string(unary))
    @test occursin("raise ValueError(", string(binary))
    @test !occursin("<closure>", string(unary))
    @test !occursin("<ref>", string(unary))
    @test_throws JAXEmissionError jax_source(jax_elementwise_escaping_ref;
                                            have = (:x,), want = :value)
    @test jax_elementwise_exp([0.1 0.2; 0.0 0.3]) == exp.([0.1 0.2; 0.0 0.3])
    @test jax_elementwise_exp([0.1 0.2; 0.0 0.3]) != exp([0.1 0.2; 0.0 0.3])
    @test_throws DimensionMismatch jax_elementwise_hypot([1.0], [2.0, 3.0])

    # Native ordinary Reverse is the derivative reference, without custom rules.
    x = [0.1, -0.3, 0.7]
    dx = zeros(length(x))
    Enzyme.autodiff(Enzyme.Reverse, jax_elementwise_exp_sum, Enzyme.Active,
                    Enzyme.Duplicated(x, dx))
    @test dx ≈ exp.(x)

    if !rk_jax_available()
        @test_skip false
    else
        for x in (0.2, Float64[], [0.1], collect(range(-0.4, 0.8; length = 19)),
                  [0.1 0.2; 0.0 0.3])
            source = jax_source(jax_elementwise_unary; have = (:x,), want = :total,
                                types = (x = typeof(x),))
            r = rk_jax_evaluate(source, :kernel, (x,); grad = true)
            @test r.value ≈ sum(exp.(x))
            expected = x isa Matrix ? collect(eachrow(exp.(x))) : exp.(x)
            @test r.grad ≈ expected
            shaped = jax_source(jax_elementwise_unary; have = (:x,), want = :value,
                                types = (x = typeof(x),))
            @test rk_jax_evaluate(shaped, :kernel, (x,)).value ≈ expected
        end
        for (x, y) in ((0.4, 0.7), ([0.2, -0.3], 0.7), (0.4, [0.2, -0.3]),
                       ([0.2, -0.3], [0.7, 0.4]), ([0.2 0.5; -0.3 0.8], 0.7))
            source = jax_source(jax_elementwise_binary; have = (:x, :y), want = :total,
                                types = (x = typeof(x), y = typeof(y)))
            r = rk_jax_evaluate(source, :kernel, (x, y); grad = true)
            @test r.value ≈ sum(hypot.(x, y))
            g = x ./ hypot.(x, y)
            expected = x isa Number ? sum(g) : x isa Matrix ? collect(eachrow(g)) : g
            @test r.grad ≈ expected
        end
        for (x, y, z) in ((0.3, 0.5, 0.7), ([0.2, -0.3], 0.5, [0.7, 0.4]),
                          (0.3, [0.5, 0.6], [0.7, 0.4]), ([0.2, -0.3], [0.5, 0.6], 0.7))
            source = jax_source(jax_elementwise_ternary; have = (:x, :y, :z), want = :total,
                                types = (x = typeof(x), y = typeof(y), z = typeof(z)))
            r = rk_jax_evaluate(source, :kernel, (x, y, z); grad = true)
            @test r.value ≈ sum(x .* y .+ z)
            expected = x isa Number ? sum(y) : y isa Number ? fill(y, size(x)) : y
            @test r.grad ≈ expected
        end

        prep = jax_source(jax_elementwise_prepared; have = (:scale, :x, :y), want = :total,
                          types = (x = Vector{Float64}, y = Vector{Float64}),
                          data = (x = [1.0], y = [2.0]))
        @test occursin("np.vectorize(", string(prep))
        for n in (0, 1, 19)
            x, y = fill(0.4, n), fill(0.7, n)
            r = rk_jax_evaluate(prep, :kernel, (0.3, x, y); grad = true, prepare = :arguments)
            @test r.value ≈ 0.3 * sum(hypot.(x, y))
            @test r.grad ≈ sum(hypot.(x, y))
            @test string(prep) == string(jax_source(jax_elementwise_prepared;
                have = (:scale, :x, :y), want = :total,
                types = (x = Vector{Float64}, y = Vector{Float64}), data = (; x, y)))
        end
        for (x, y) in (([1.0], [2.0, 3.0]), ([1.0, 2.0], [3.0]))
            for (source, args, preparation) in ((binary, (x, y), false),
                                                (prep, (0.3, x, y), :arguments))
                err = try
                    rk_jax_evaluate(source, :kernel, args; prepare = preparation)
                    nothing
                catch e
                    e
                end
                @test err isa ErrorException
                @test occursin("elementwise arguments must have equal sizes", sprint(showerror, err))
            end
        end

        captures = jax_source(jax_elementwise_captures; have = (:a, :w, :x), want = :total)
        for a in (0.3, 1.7)
            w, x = [0.1, 0.2], [0.4, -0.8, 0.7]
            r = rk_jax_evaluate(captures, :kernel, (a, w, x); grad = true)
            @test r.value ≈ a^2 * sum(w) * sum(x)
            @test r.grad ≈ 2a * sum(w) * sum(x)
        end
        refs = jax_source(jax_elementwise_refs; have = (:x, :scale), want = :value)
        @test rk_jax_evaluate(refs, :kernel, ([0.2, 0.7], 0.4)).value ≈ [0.08, 0.28]
        ragged = jax_source(jax_elementwise_ragged; have = (:xs,), want = :total)
        rows = [[0.2, -0.1, 0.8], Float64[], [0.4]]
        @test rk_jax_evaluate(ragged, :kernel, (rows,)).value ≈
            sum(sum(x) * sum(exp.(x)) for x in rows)
        inactive = jax_source(jax_elementwise_inactive; have = (:x,), want = :total)
        r = rk_jax_evaluate(inactive, :kernel, ([-2.0, 0.0, 2.0],); grad = true)
        @test r.value ≈ log(2.0)
        @test r.grad ≈ [0.0, 0.0, 0.5]

        # Data lengths change batched array extents, never emitted bodies or
        # JAX regions. This checks the trace as well as the source spelling.
        mktempdir() do dir
            module_path = joinpath(dir, "elementwise.py")
            write(module_path, binary)
            driver = joinpath(dir, "structure.py")
            write(driver, raw"""
import importlib.util, sys
import jax, jax.numpy as jnp
s = importlib.util.spec_from_file_location('elementwise', sys.argv[1])
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
def regions(jaxpr):
    jaxpr = getattr(jaxpr, 'jaxpr', jaxpr)
    count = len(jaxpr.eqns)
    for eq in jaxpr.eqns:
        for v in eq.params.values():
            for child in v if isinstance(v, (tuple, list)) else (v,):
                if hasattr(child, 'jaxpr') or hasattr(child, 'eqns'):
                    count += regions(child)
    return count
counts = [regions(jax.make_jaxpr(m.kernel)(jnp.ones(n), jnp.ones(n))) for n in (1, 19, 73)]
assert counts[0] > 0 and max(counts) <= 2 * min(counts), counts
print('batched trace equation counts:', counts)
""")
            @test success(`$RK_JAX_PYTHON $driver $module_path`)
        end
    end
end
