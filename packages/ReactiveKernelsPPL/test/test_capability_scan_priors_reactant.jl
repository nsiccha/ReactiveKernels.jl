using Reactant

function _csi_operations(hlo)
    ops = Dict{String,Int}()
    for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
        ops[m.match] = get(ops, m.match, 0) + 1
    end
    return ops
end

@testset "Reactant: prior-only and shared simplex density" begin
    fx = _csi_build(_CSI_PRIOR_ONLY, NamedTuple())
    u = [0.2, -0.3, 0.1, 0.4]
    q = prepare_sampler(fx.built, fx.bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    ru = Reactant.to_rarray(u)
    kern = q.kernel
    compiled = Reactant.@compile kern(ru)
    @test Float64(compiled(ru)) ≈ _csi_prior_oracle(fx.built.layout, u)
    value, gradient = sampler_value_and_gradient!(q, similar(u), u)
    cad = compile_ad_value_and_gradient(q.ad, ru)
    rvalue, rgrad = cad(ru)
    @test Float64(rvalue) ≈ value
    @test Array(rgrad) ≈ gradient rtol=1e-8
    structures = Dict{String,Int}[]
    for n in (6, 15)
        data = _csi_mo_data(n)
        fx = _csi_build(_CSI_SHARED_MONOTONIC, data)
        q = prepare_sampler(fx.built, fx.bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
        kern = q.kernel
        push!(structures, _csi_operations(repr(Reactant.@code_hlo optimize=false kern(ru))))
        compiled = Reactant.@compile kern(ru)
        @test Float64(compiled(ru)) ≈ _csi_mo_oracle(fx.built.layout, u, data)
        value, gradient = sampler_value_and_gradient!(q, similar(u), u)
        cad = compile_ad_value_and_gradient(q.ad, ru)
        rvalue, rgrad = cad(ru)
        @test Float64(rvalue) ≈ value
        @test Array(rgrad) ≈ gradient rtol=1e-8
    end
    @test structures[1] == structures[2]
end

@testset "Reactant: index-valued scans retain their loop" begin
    for centered in (true, false)
        structures = Dict{String,Int}[]
        for n in (4, 9)
            data = (; y=[0.2sin(i) for i in 1:n])
            fx = _csi_build(_csi_index_model(centered), data)
            u = 0.05cos.(1:fx.built.layout.total)
            q = prepare_sampler(fx.built, fx.bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
            ru = Reactant.to_rarray(u)
            kern = q.kernel
            ops = _csi_operations(repr(Reactant.@code_hlo optimize=false kern(ru)))
            centered || @test get(ops, "stablehlo.while", 0) > 0
            push!(structures, ops)
            compiled = Reactant.@compile kern(ru)
            @test Float64(compiled(ru)) ≈ _csi_index_oracle(fx.built.layout, u, data, centered)
            value, gradient = sampler_value_and_gradient!(q, similar(u), u)
            cad = compile_ad_value_and_gradient(q.ad, ru)
            rvalue, rgrad = cad(ru)
            @test Float64(rvalue) ≈ value
            @test Array(rgrad) ≈ gradient rtol=1e-8
        end
        @test structures[1] == structures[2]
    end
end
