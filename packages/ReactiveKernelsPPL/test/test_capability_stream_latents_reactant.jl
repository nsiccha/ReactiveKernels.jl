using Reactant

function _cap_stream_operations(hlo)
    out = Dict{String, Int}()
    for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
        out[m.match] = get(out, m.match, 0)+1
    end
    out
end

@testset "Reactant: empty conditioned priors retain the pointwise identity" begin
    fx = _cap_empty_conditioned_prior()
    ru = Reactant.to_rarray(fx.u)
    cad = compile_ad_value_and_gradient(fx.sampler.ad, ru)
    value, gradient = cad(ru)
    @test Float64(value) ≈ logpdf(Normal(), only(fx.u))
    @test Array(gradient) ≈ -fx.u
    pointwise = prepare_query(fx.built, fx.bound, :pointwise)
    cp = Reactant.@compile pointwise(ru)
    @test isempty(Array(cp(ru).z))
    @test Array(ru) == fx.u
end

@testset "Reactant: generative streams and per-cell predictor pins" begin
    for kind in (:vector, :scalar, :cell, :pin)
        primal, reverse = Dict{String, Int}[], Dict{String, Int}[]
        for n in (0, 4, 10)
            fx = _cap_stream_fixture(kind, n)
            ru = Reactant.to_rarray(fx.u)
            kernel = fx.sampler.kernel
            n == 0 || push!(primal, _cap_stream_operations(repr(Reactant.@code_hlo optimize=false kernel(ru))))
            cp = Reactant.@compile kernel(ru)
            @test Float64(cp(ru)) ≈ fx.oracle(fx.u)
            cad = compile_ad_value_and_gradient(fx.sampler.ad, ru)
            rv, rg = cad(ru)
            @test Float64(rv) ≈ fx.oracle(fx.u)
            @test Array(rg) ≈ _cap_stream_fd(fx.oracle, fx.u) rtol=6e-6
            grad = v -> only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(kernel), v))
            n == 0 || push!(reverse, _cap_stream_operations(repr(Reactant.@code_hlo optimize=:only_enzyme grad(ru))))
        end
        @test primal[1] == primal[2]
        @test reverse[1] == reverse[2]
    end
end
