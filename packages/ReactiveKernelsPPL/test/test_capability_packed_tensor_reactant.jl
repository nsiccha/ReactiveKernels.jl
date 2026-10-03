using Reactant

@testset "Reactant: packed tensor families retain row iteration" begin
    for kind in (:multinomial, :joint, :glm, :bernoulli_glm, :poisson_glm)
        primal, reverse = Dict{String,Int}[], Dict{String,Int}[]
        for n in (6, 18)
            fx = _cap_packed_tensor(kind, n)
            ru = Reactant.to_rarray(fx.u)
            kernel = fx.sampler.kernel
            push!(primal, _cap_range_operations(repr(Reactant.@code_hlo optimize=false kernel(ru))))
            compiled = Reactant.@compile kernel(ru)
            @test Float64(compiled(ru)) ≈ fx.oracle(fx.u)
            value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
            cad = compile_ad_value_and_gradient(fx.sampler.ad, ru)
            cv, cg = cad(ru)
            @test Float64(cv) ≈ value
            @test Array(cg) ≈ gradient rtol=1e-8
            changed = fx.u .+ 0.05
            cr = Reactant.to_rarray(changed)
            cv2, cg2 = cad(cr)
            @test Float64(cv2) ≈ fx.oracle(changed)
            @test Array(cg2) ≈ _cap_range_fd(fx.oracle, changed) rtol=1e-5 atol=1e-7
            @test fx.data == fx.saved
            @test Array(ru) == fx.u
            grad = v -> only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(kernel), v))
            push!(reverse, _cap_range_operations(repr(Reactant.@code_hlo optimize=:only_enzyme grad(ru))))
        end
        @test primal[1] == primal[2]
        @test reverse[1] == reverse[2]
    end
end

@testset "Reactant: zero Multinomial counts preserve lazy likelihood cuts" begin
    primal, reverse = Dict{String,Int}[], Dict{String,Int}[]
    for n in (6, 18)
        fx = _cap_packed_zero_counts(n)
        rp = Reactant.to_rarray(fx.p)
        compiled = compile_ad_value_and_gradient(fx.ad, rp)
        value, gradient = compiled(rp)
        @test Float64(value) == 0.0
        @test Array(gradient) ≈ fx.expected
        @test Array(rp) == fx.p
        @test fx.data == fx.saved
        kernel = fx.kernel
        push!(primal, _cap_range_operations(repr(Reactant.@code_hlo optimize=false kernel(rp))))
        grad = p -> only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(kernel), p))
        push!(reverse, _cap_range_operations(repr(Reactant.@code_hlo optimize=:only_enzyme grad(rp))))
    end
    @test primal[1] == primal[2]
    @test reverse[1] == reverse[2]
end
