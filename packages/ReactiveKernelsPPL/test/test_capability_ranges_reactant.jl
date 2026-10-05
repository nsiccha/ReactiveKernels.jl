using Reactant

function _cap_range_operations(hlo)
    out = Dict{String,Int}()
    for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
        out[m.match] = get(out, m.match, 0) + 1
    end
    out
end

@testset "Reactant: response slices and retained selected cells" begin
    for kind in (:cross_index, :cross_axis, :colon, :top_singleton, :cross_cell, :singleton, :inactive, :matrix, :singleton_latent, :free_matrix, :longer_inactive, :axis1_matrix_inactive,
            :literal_tail, :literal_whole, :literal_single, :literal_matrix)
        primal, reverse = Dict{String,Int}[], Dict{String,Int}[]
        # Sizes 4 and 9: no selected cell count equals the two parameters,
        # whose shared tensor shape would otherwise let XLA share constants.
        for n in (4, 9)
            fx = _cap_range_fixture(kind, n)
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
            pointwise = prepare_query(fx.built, fx.bound, :pointwise)
            cpw = Reactant.@compile pointwise(ru)
            output = cpw(ru)
            if kind === :free_matrix
                @test output == (;)
            else
                @test Array(output.y) ≈ fx.pointwise(fx.u)
            end
            grad = v -> only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(kernel), v))
            push!(reverse, _cap_range_operations(repr(Reactant.@code_hlo optimize=:only_enzyme grad(ru))))
        end
        @test primal[1] == primal[2]
        @test reverse[1] == reverse[2]
    end
end

@testset "Reactant: empty indexed observations retain prior gradients" begin
    for kind in (:cross_index, :cross_axis, :colon, :cross_cell, :matrix, :free_matrix,
            :literal_tail, :literal_whole, :literal_matrix)
        fx = _cap_range_fixture(kind, 0)
        ru = Reactant.to_rarray(fx.u)
        cad = compile_ad_value_and_gradient(fx.sampler.ad, ru)
        value, gradient = cad(ru)
        @test Float64(value) ≈ fx.oracle(fx.u)
        @test Array(gradient) ≈ _cap_range_fd(fx.oracle, fx.u) rtol=5e-6
    end
end
