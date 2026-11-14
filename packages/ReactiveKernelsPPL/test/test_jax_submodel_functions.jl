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
