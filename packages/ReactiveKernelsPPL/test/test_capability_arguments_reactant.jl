using Reactant

function _arg_operations(hlo)
    ops = Dict{String,Int}()
    for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
        ops[m.match] = get(ops, m.match, 0)+1
    end
    return ops
end

@testset "Reactant: Gamma log-rate owner retains its numerical range" begin
    kernel = prepare(_ARG_GAMMA_LOG_ROUTE; have=(:u,), want=:density)
    ad = prepare_ad(kernel, AutoEnzyme(; mode=Enzyme.Reverse), [0.4]; active=:u)
    ru = Reactant.to_rarray([0.4])
    compiled = Reactant.@compile kernel(ru)
    cad = compile_ad_value_and_gradient(ad, ru)
    for log_rate in (-1000.0, 0.4)
        ru = Reactant.to_rarray([log_rate])
        @test Float64(compiled(ru)) ≈ 2log_rate-exp(log_rate)
        value, grad = cad(ru)
        @test Float64(value) ≈ 2log_rate-exp(log_rate)
        @test Array(grad) ≈ [2-exp(log_rate)]
    end
end

@testset "Reactant: ordinary distribution arguments" begin
    for (kind, ast, u) in _ARG_MODELS
        @testset "$kind" begin
            structures = Dict{String,Int}[]
            for n in (3, 8)
                data = _arg_data(n)
                fx = _arg_build(ast, data)
                q = prepare_sampler(fx.built, fx.bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
                ru = Reactant.to_rarray(u)
                kern = q.kernel
                push!(structures, _arg_operations(repr(Reactant.@code_hlo optimize=false kern(ru))))
                compiled = Reactant.@compile kern(ru)
                @test Float64(compiled(ru)) ≈ _arg_oracle(kind, fx.built.layout, u, data)
                if kind in (:weibull, :gamma_mixed, :gamma_raw_scale)
                    invalid = kind in (:weibull, :gamma_raw_scale) ? [-0.5] :
                        unconstrain(fx.built.layout, (; a=0.2, b=0.1, c=-1.4, d=0.1))
                    @test Float64(compiled(Reactant.to_rarray(invalid))) == -Inf
                end
                if !isempty(u)
                    value, gradient = sampler_value_and_gradient!(q, similar(u), u)
                    cad = compile_ad_value_and_gradient(q.ad, ru)
                    rvalue, rgrad = cad(ru)
                    @test Float64(rvalue) ≈ value
                    @test Array(rgrad) ≈ gradient rtol=1e-8
                    if kind === :gamma_raw_scale
                        ivalue, igrad = cad(Reactant.to_rarray([-0.5]))
                        @test Float64(ivalue) == -Inf
                        @test Array(igrad) ≈ [0.5]
                    end
                end
            end
            @test structures[1] == structures[2]
        end
    end
end
