using Test, Distributions, ReactiveKernels, ReactiveKernelsPPL

function _defaults_plate(n)
    ast = quote
        a ~ Normal()
        pred ~ plate(x, y; subjects=nsubjects) do xs, ys
            mu = a .* xs
            ys .~ Normal.(mu)
            mu
        end
    end
    data = Dict(:x => collect(range(0.2, 1.2; length=2n)),
        :y => collect(range(0.3, 0.8; length=2n)))
    before = deepcopy(data)
    unbound = lower_rkppl(ast, data; conditioned=data)
    bound = bind_data(unbound, data; dims=Dict(:nsubjects=>n, :kernel_T_pred=>2))
    built = build_kernel(bound)
    u = unconstrain(built.layout, (a=0.3,))
    kernel = prepare_query(built, bound, :sampler)
    expected = logpdf(Normal(), 0.3) +
        sum(logpdf.(Normal.(0.3 .* data[:x]), data[:y]))
    @test data == before
    return built, bound, kernel, u, expected
end

@testset "Normal default scale in plate observations" begin
    structures = Dict{String,Int}[]
    for n in (3, 7)
        println("PLATE_DEFAULTS_BACKEND_BEGIN n=", n)
        flush(stdout)
        built, bound, kernel, u, expected = _defaults_plate(n)
        @test Base.invokelatest(kernel, u) ≈ expected atol=1e-10 rtol=1e-10
        push!(structures, Base.invokelatest(_defaults_backends,
            built, bound, kernel, u, expected))
    end
    @test structures[1] == structures[2]
end
