using Test, ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Reactant
import SpecialFunctions

# The native defaults file supplies cases and independent Distributions oracles.
function _defaults_backends(built, bound, kernel, u, expected)
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ expected atol=1e-10 rtol=1e-10
    finite = similar(u)
    for i in eachindex(u)
        up, down = copy(u), copy(u)
        up[i] += 1e-5
        down[i] -= 1e-5
        finite[i] = (kernel(up) - kernel(down)) / 2e-5
    end
    @test grad ≈ finite atol=1e-7 rtol=1e-6
    ru = Reactant.to_rarray(u)
    hlo = repr(Reactant.@code_hlo optimize=false kernel(ru))
    compiled = Reactant.@compile kernel(ru)
    @test Float64(compiled(ru)) ≈ expected atol=1e-10 rtol=1e-10
    cad = compile_ad_value_and_gradient(sampler.ad, ru)
    rvalue, rgrad = cad(ru)
    @test Float64(rvalue) ≈ value atol=1e-10 rtol=1e-10
    @test Array(rgrad) ≈ grad atol=1e-9 rtol=1e-9
    ops = Dict{String,Int}()
    for op in eachmatch(r"stablehlo\.[a-z_]+", hlo)
        ops[op.match] = get(ops, op.match, 0) + 1
    end
    return ops
end

@testset "constructor defaults: native and compiled reverse, retained structure" begin
    for name in (:normal, :negative_binomial, :inverse_gaussian, :weibull,
            :von_mises, :von_mises_literal, :lognormal, :swapped_beta, :uniform_mixture)
        @testset "$name" begin
            structures = Dict{String,Int}[]
            for n in (3, 7)
                println("DEFAULTS_BACKEND_BEGIN ", name, " n=", n)
                flush(stdout)
                ast, explicit, data, q, expected = _defaults_response_case(name, n)
                built, bound, kernel, u = _defaults_build(ast, data, q)
                try
                    push!(structures, Base.invokelatest(_defaults_backends,
                        built, bound, kernel, u, expected))
                    # The existing live-concentration Reactant limitation in
                    # test_vonmises.jl must fire when upstream gains support.
                    name === :von_mises && @test_broken true
                catch error
                    name === :von_mises && error isa MethodError &&
                        error.f === SpecialFunctions.besseli && length(error.args) == 2 &&
                        error.args[1] == 0 && error.args[2] isa Reactant.TracedRNumber || rethrow()
                    @test_broken false
                end
            end
            isempty(structures) || @test structures[1] == structures[2]
        end
    end
end

@testset "TDist prior: native and compiled reverse" begin
    structures = Dict{String,Int}[]
    for n in (3, 7)
        ast = quote
            x ~ TDist(3.0)
            y .~ Normal.(x)
        end
        data = Dict(:y => collect(range(0.2, 0.7; length=n)))
        q = (x=0.4,)
        expected = logpdf(TDist(3.0), q.x) + sum(logpdf.(Normal(q.x), data[:y]))
        built, bound, kernel, u = _defaults_build(ast, data, q)
        push!(structures, Base.invokelatest(_defaults_backends,
            built, bound, kernel, u, expected))
    end
    @test structures[1] == structures[2]
end
