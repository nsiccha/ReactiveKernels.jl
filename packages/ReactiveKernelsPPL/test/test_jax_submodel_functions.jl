using ReactiveKernels, ReactiveKernelsPPL, Test
using DifferentiationInterface, Enzyme
include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

@rkppl jax_named_effect(loc, scale, spread_rate, n) = begin
    "Mean of the effect."
    mean ~ Normal(loc, scale)
    "Spread of the members around the mean."
    spread ~ Exponential(spread_rate)
    "Standardised member deviations."
    z[1:n] .~ Normal.(0, 1)
    return mean .+ spread .* z
end

const jax_named_model = @rkppl begin
    a ~ jax_named_effect(a_loc, a_scale, a_rate, n)
    b ~ jax_named_effect(b_loc, b_scale, b_rate, n)
    y .~ Normal.(a .+ b, noise)
end

function named_effect_case(model, n)
    raw = (; n, a_loc = -0.2, a_scale = 1.3, a_rate = 0.8,
           b_loc = 0.4, b_scale = 0.7, b_rate = 1.2, noise = 0.6,
           y = collect(range(-0.1, 0.3; length = n)))
    plan = model(; Base.structdiff(raw, (; y = raw.y))...) | (; raw.y)
    built = build_kernel(plan)
    src = jax_source(built, plan)
    names = first.(only(f for f in src.functions if f.name == "prepare_data").inputs)
    args = Tuple(getproperty(raw, Symbol(name)) for name in names)
    plan, built, src, args
end

@testset "one callable per repeated authored submodel" begin
    texts = String[]
    programs = Expr[]
    reference_source = nothing
    for n in (0, 3, 19)
        plan, built, src, args = named_effect_case(jax_named_model, n)
        text = src.text
        push!(texts, text)
        push!(programs, built.program)
        reference_source === nothing && (reference_source = src)
        @test count(r"(?m)^def jax_named_effect\w*\(", text) == 1
        @test occursin("Mean of the effect.", text)
        @test occursin("def model(", text)
        @test count(" = prepared[\"a_loc\"]", text) == 1
        @test count(r"mean\w* \+ spread\w* \* z\w*", text) == 1
        u = collect(range(-0.2, 0.3; length = built.layout.total))
        if rk_jax_available()
            for (entry, workflow) in ((:log_density, :sampler), (:log_prior, :prior),
                                      (:log_likelihood, :likelihood), (:log_jacobian, :log_jacobian))
                native = prepare_query(built, plan, workflow)
                got = rk_jax_evaluate(reference_source, entry, (u, args...); grad = true, prepare = :arguments)
                @test got.value ≈ native(u) atol = 1e-11
                finite_difference = [(native(u + 1e-5 .* (1:length(u) .== i)) -
                                      native(u - 1e-5 .* (1:length(u) .== i))) / 2e-5
                                     for i in eachindex(u)]
                @test got.grad ≈ finite_difference rtol = 2e-6 atol = 1e-7
                if workflow === :sampler && n == 3
                    sampler = prepare_sampler(built, plan, u;
                        backend = AutoEnzyme(; mode = Enzyme.Reverse))
                    gradient = similar(u)
                    value, _ = sampler_value_and_gradient!(sampler, gradient, u)
                    @test got.value ≈ value atol = 1e-11
                    @test got.grad ≈ gradient rtol = 1e-10 atol = 1e-10
                end
            end
            constrained = rk_jax_evaluate(src, :constrain, (u, args...); prepare = :arguments).value
            @test Set(keys(constrained)) == Set(["a", "b"])
            @test length(constrained["a"]["z"]) == n
        else
            @test_skip false
        end
    end
    # Increasing the bound data length keeps the same executable bodies;
    # alias choices may affect local names without replicating any work.
    structure(text) = [count(pattern, text) for pattern in
        (r"(?m)^\s*def ", "jax.vmap(", "jax.lax.cond(", "jax.lax.fori_loop(")]
    @test structure(texts[1]) == structure(texts[2]) == structure(texts[3])
    @test all(p -> ReactiveKernelsPPL._same_program(first(programs), p), programs)
end

@testset "scoped rewrites retain distinct executable bodies" begin
    variant = merge(jax_named_model, :(a.mean ~ Laplace(a_loc, a_scale)))
    plan, built, src, args = named_effect_case(variant, 3)
    @test count(r"(?m)^def jax_named_effect\w*\(", src.text) == 2
    if rk_jax_available()
        u = zeros(built.layout.total)
        @test rk_jax_evaluate(src, :log_density, (u, args...); prepare = true).value ≈
              prepare_query(built, plan, :sampler)(u)
    else
        @test_skip false
    end
end

# The diagonal of an LKJ factor is a fixed-shape allocation, so its value
# keeps that shape as translation metadata. The lower factor is assembled
# outside the submodel that computes the diagonal, so the caller reads a
# submodel output that carries such metadata.
@rkppl jax_correlated_effect(eta, n) = begin
    L ~ LKJCholesky(3, eta)
    z[1:n, 1:3] .~ Normal.(0, 1)
    return z * transpose(L)
end

const jax_correlated_model = @rkppl begin
    b ~ jax_correlated_effect(eta, n)
    y .~ Normal.(b[:, 1] .+ b[:, 2] .- b[:, 3], noise)
end

@testset "a submodel's fixed-shape output is read from its result" begin
    texts = String[]
    for n in (0, 4, 9)
        raw = (; n, eta = 2.0, noise = 0.7, y = collect(range(-0.4, 0.5; length = n)))
        plan = jax_correlated_model(; Base.structdiff(raw, (; y = raw.y))...) | (; raw.y)
        built = build_kernel(plan)
        src = jax_source(built, plan)
        push!(texts, src.text)
        # The factor assembly sits between the submodel's density part and
        # its returned value, which stay one function each.
        @test count(r"(?m)^def jax_correlated_effect\w*\(", src.text) == 2
        # Every array the caller's factor assembly reads is an entry of the
        # submodel's result, never a name local to the submodel's body.
        calls = [m.match for m in eachmatch(r"(?m)^\s+\w+ = _ppl_lkj_lower\(.*$", src.text)]
        @test length(calls) == 1
        @test occursin(r"_ppl_lkj_lower\((\w+)\[\"\w+\"\], \1\[\"\w+\"\], 3, 3\)", only(calls))
        if rk_jax_available()
            names = first.(only(f for f in src.functions if f.name == "prepare_data").inputs)
            args = Tuple(getproperty(raw, Symbol(name)) for name in names)
            u = collect(range(-0.3, 0.4; length = built.layout.total))
            for (entry, workflow) in ((:log_density, :sampler), (:log_prior, :prior),
                                      (:log_likelihood, :likelihood), (:log_jacobian, :log_jacobian))
                native = prepare_query(built, plan, workflow)
                got = rk_jax_evaluate(src, entry, (u, args...); grad = true, prepare = :arguments)
                @test got.value ≈ native(u) atol = 1e-11
                finite_difference = [(native(u + 1e-5 .* (1:length(u) .== i)) -
                                      native(u - 1e-5 .* (1:length(u) .== i))) / 2e-5
                                     for i in eachindex(u)]
                @test got.grad ≈ finite_difference rtol = 2e-6 atol = 1e-7
            end
        else
            @test_skip false
        end
    end
    # A longer grouping axis keeps the same executable bodies.
    structure(text) = [count(pattern, text) for pattern in
        (r"(?m)^\s*def ", "jax.vmap(", "jax.lax.cond(", "jax.lax.fori_loop(")]
    @test structure(texts[1]) == structure(texts[2]) == structure(texts[3])
end

@kernel function jax_lkj_weighted_partials(u::Vector{Float64}, margins::Vector{Float64})
    weights = ReactiveKernelsPPL._ppl_lkj_weights(length(margins), length(u))
    total = sum(u .* weights)
    return total
end

@testset "shape-only LKJ weights in a traced gradient" begin
    src = jax_source(jax_lkj_weighted_partials; have=(:u, :margins), want=:total)
    if rk_jax_available()
        mktempdir() do dir
            path = joinpath(dir, "weights.py")
            write(path, src)
            check = raw"""
import jax, jax.numpy as jnp, numpy as np, runpy, sys
jax.config.update("jax_enable_x64", True)
gradient = jax.value_and_grad(runpy.run_path(sys.argv[1])["kernel"])
for k in (1, 3, 8):
    weights = np.array([(j-i+1)/2 for j in range(2, k+1) for i in range(1, j)])
    u, margins = jnp.arange(weights.size, dtype=jnp.float64), jnp.zeros(k)
    value, grad = jax.jit(gradient)(u, margins)
    np.testing.assert_array_equal(grad, weights)
    np.testing.assert_equal(float(value), np.dot(np.asarray(u), weights))
    # The packed triangular weights need no device loop on each gradient call.
    trace = str(jax.make_jaxpr(gradient)(u, margins))
    assert "scan[" not in trace and "while[" not in trace, trace
"""
            @test success(`$RK_JAX_PYTHON -c $check $path`)
        end
    else
        @test_skip false
    end
end

@testset "LKJ row products in a bound lower or upper sampler factor" begin
    for (K, uplo) in ((1, 'L'), (3, 'U'), (8, 'L'))
        model = lower_rkppl(quote
            L ~ LKJCholesky(K, eta, $uplo)
            y ~ Normal(sum(L .* weights), 0.7)
        end, (:K, :eta, :weights, :y); conditioned = (:y,))
        raw = (; K, eta = 1.5, weights = reshape(sin.(1:K^2), K, K), y = 0.3)
        bound = bind_data(model, Dict(pairs(raw)))
        built = build_kernel(bound)
        u = 0.2 .* cos.(1:built.layout.total)
        native = prepare_query(built, bound, :sampler)
        direction = sin.(eachindex(u))
        h = 1e-5
        deriv = (Base.invokelatest(native, u + h * direction) -
                 Base.invokelatest(native, u - h * direction)) / (2h)
        for layout in (:padded, :groups)
            src = jax_source(built, bound; plate_layout = layout)
            @test occursin("jax.lax.scan(advance", src.text)
            if rk_jax_available()
                names = first.(only(f for f in src.functions if f.name == "prepare_data").inputs)
                data = Tuple(getproperty(raw, Symbol(n)) for n in names)
                got = rk_jax_evaluate(src, :log_density, (u, data...); grad = true, prepare = :arguments)
                @test got.value ≈ Base.invokelatest(native, u) atol = 2e-12
                @test sum(got.grad .* direction; init = 0.0) ≈ deriv atol = 2e-8
            end
        end
    end
end
