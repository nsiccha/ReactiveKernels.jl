module IntegerMarginalJAXTests
using ReactiveKernels, ReactiveKernelsPPL, Test
include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

@testset "finite integer marginalization in emitted JAX" begin
    mixture = @rkppl begin
        p ~ Beta(2,3)
        k ~ Binomial(n,p)
        y .~ Normal.(k,1)
    end
    joint = @rkppl begin
        p ~ Beta(2,3)
        k ~ Bernoulli(p)
        z ~ Bernoulli(k)
        y ~ Normal(k+z,1)
    end
    rag = @rkppl begin
        p ~ Beta(a,b)
        c ~ Binomial(n,p)
        r = h-c
        r ~ Binomial(n-c,0.5)
    end
    mktempdir() do dir
        plans = (
            ("mixture",marginalize(mixture,:k)(;n=5) | (;y=[1.2,2.1])),
            ("joint",marginalize(joint,:k,:z)() | (;y=0.3)),
            ("rag",bind_data(marginalize(rag,:c)(;n=10,h=8,a=2.0,b=3.0))))
        for (name,plan) in plans
            @testset "$name marginal outputs" begin
                source = jax_source(build_kernel(plan),plan;
                    values=(:pointwise,:_ppl_latents))
                @test occursin("def log_density",source.text)
                write(joinpath(dir,name*".py"),source)
                if rk_jax_available()
                    driver = joinpath(@__DIR__,"integer_marginal_jax.py")
                    @test success(pipeline(`$RK_JAX_PYTHON $driver $dir $name`;stdout,stderr))
                else
                    @test_skip "RK_JAX_PYTHON not set"
                end
            end
        end
    end
end


@testset "elementwise integer-array marginals in JAX" begin
    guide = @rkppl begin
        w ~ Dirichlet(ones(K))
        mu[1:K] .~ Normal.(0, 5)
        z[axes(y, 1)] .~ Categorical(w)
        y .~ Normal.(mu[z], 1.0)
    end
    counts = @rkppl begin
        p ~ Beta(2, 3)
        z[axes(trials, 1)] .~ Binomial.(trials, p)
        y .~ Normal.(z, 0.7)
    end
    matrix = @rkppl begin
        p ~ Beta(2, 3)
        z[axes(y, 1), axes(y, 2)] .~ Bernoulli(p)
        y .~ Normal.(z, 1.0)
    end
    joint = @rkppl begin
        p ~ Beta(2, 3)
        a ~ Bernoulli(p)
        z[axes(y, 1)] .~ Bernoulli(0.3)
        y .~ Normal.(z .+ a, 1.0)
    end
    mktempdir() do dir
        for (name, model, eliminated) in (("array_guide", guide, (:z,)),
                ("array_counts", counts, (:z,)), ("array_matrix", matrix, (:z,)),
                ("array_joint", joint, (:a, :z)))
            @testset "$name" begin
                program = nothing
                for n in (name == "array_joint" ? (2, 13) : (2, 13, 0))
                    y = name == "array_matrix" ?
                        reshape(collect(range(-0.2, 0.8; length = 2n)), n, 2) :
                        collect(range(-0.2, 0.8; length = n))
                    inputs = name == "array_guide" ? (; K = n == 2 ? 3 : 5) :
                        name == "array_counts" ? (; trials = mod.(1:n, n)) : (;)
                    plan = marginalize(model(; inputs...), eliminated...) | (; y)
                    built = build_kernel(plan)
                    if program === nothing
                        program = built.program
                        source = jax_source(built, plan; values = (:pointwise, :_ppl_latents))
                        @test occursin("def log_density", source.text)
                        write(joinpath(dir, name * ".py"), source)
                    else
                        @test built.program == program
                    end
                    u = [0.15 + 0.2 * (j - 1) for j in 1:built.layout.total]
                    density = prepare_query(built, plan, :sampler)
                    gradient = [(density(u .+ step) - density(u .- step)) / 2e-5
                        for step in (1e-5 .* ((1:length(u)) .== i) for i in eachindex(u))]
                    posterior = prepare_discrete_latents(built, plan)(u).parameters
                    element = posterior.elements.z
                    pw = prepare_query(built, plan, :pointwise)(u).y
                    open(joinpath(dir, name * string(n) * "_reference.py"), "w") do io
                        println(io, "raw = {", join([repr(string(k)) * ": " * _rk_py_input(v)
                            for (k, v) in pairs((; inputs..., y))], ", "), "}")
                        for (key, v) in pairs((; u, density = density(u), gradient, pw,
                                values = element.values, probabilities = element.probabilities))
                            println(io, key, " = ", _rk_py_input(v))
                        end
                    end
                end
                if rk_jax_available()
                    driver = joinpath(@__DIR__, "integer_marginal_jax.py")
                    @test success(pipeline(`$RK_JAX_PYTHON $driver $dir $name`; stdout, stderr))
                else
                    @test_skip "RK_JAX_PYTHON not set"
                end
            end
        end
    end
end
end
