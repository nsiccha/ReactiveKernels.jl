using Test, ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Reactant

@testset "Boolean response values: native and compiled reverse, retained structure" begin
    for name in (:negative_binomial, :binomial, :binomial_mixture, :beta_binomial, :hurdle_poisson,
            :inverse_gaussian, :weibull, :von_mises, :exponential, :lognormal)
        @testset "$name" begin
            structures = Dict{String,Int}[]
            for n in (3, 7)
                println("BOOL_BACKEND_BEGIN ", name, " n=", n)
                flush(stdout)
                ast, data, q, expected = _bool_response_case(name, n)
                built, bound, kernel, u = _defaults_build(ast, data, q)
                push!(structures, Base.invokelatest(_defaults_backends,
                    built, bound, kernel, u, expected))
            end
            @test structures[1] == structures[2]
        end
    end
end
