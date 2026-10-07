using Test, Distributions, ReactiveKernels, ReactiveKernelsPPL

function _defaults_plate(n)
    ast = quote
        a ~ Normal()
        @plate for i in eachindex(y)
            mu[i] = a * x[i]
            y[i] ~ Normal(mu[i])
        end
    end
    data = Dict(:x => collect(range(0.2, 1.2; length=2n)),
        :y => collect(range(0.3, 0.8; length=2n)))
    before = deepcopy(data)
    unbound = lower_rkppl(ast, data; conditioned=data)
    bound = bind_data(unbound, data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (a=0.3,))
    kernel = prepare_query(built, bound, :sampler)
    expected = logpdf(Normal(), 0.3) +
        sum(logpdf.(Normal.(0.3 .* data[:x]), data[:y]))
    @test data == before
    return built, bound, kernel, u, expected
end

@testset "Normal default scale in plate observations" begin
    for n in (3, 7)
        built, bound, kernel, u, expected = _defaults_plate(n)
        @test Base.invokelatest(kernel, u) ≈ expected atol=1e-10 rtol=1e-10
        Base.invokelatest(_defaults_native_reverse, built, bound, kernel, u, expected)
    end
end
