using Reactant

function _vc_backend_operations(sampler, ru)
    post, ad = sampler.kernel, sampler.ad
    both(v) = ad_value_and_gradient(ad, v)
    primal = string(Reactant.@code_hlo post(ru))
    derivative = repr(Reactant.@code_hlo optimize=false both(ru))
    return [prefix * m.match for (prefix, hlo) in
        (("primal.", primal), ("ad.", derivative)) for m in
        eachmatch(r"\b(?:stablehlo|chlo|enzyme|func|arith)\.\w+", hlo)]
end

@testset "ordinary composed values: compiled density, gradient and structure" begin
    operations = Dict{String,Vector{String}}()
    for n in (3, 7, 0), case in (n == 0 ? _vc_empty_cases() : _vc_cases(n))
        # Keep the contrast's observation counts above its three levels:
        # n=3 is an identity gather that optimization removes. The other
        # cases use 3/7, away from the four-parameter fixture's shape case.
        if case.label == "library contrast composition"
            case = only(c for c in _vc_cases(n + 1) if c.label == case.label)
        end
        @testset "$(case.label) / n=$(length(case.data.y))" begin
            plan, built = _vc_model(case.ast, case.data)
            u = [0.2 * sin(i) for i in 1:built.layout.total]
            oracle = w -> _vc_oracle(case, built.layout, w)
            sampler = prepare_sampler(built, plan, u; backend = _GEN_BACKEND)
            grad = similar(u)
            native, _ = sampler_value_and_gradient!(sampler, grad, u)
            ru = Reactant.to_rarray(u)
            compiled = Reactant.@compile sampler.kernel(ru)
            @test Float64(compiled(ru)) ≈ oracle(u) rtol=1e-12
            cad = compile_ad_value_and_gradient(sampler.ad, ru)
            value, gradient = cad(ru)
            @test Float64(value) ≈ native
            @test Array(gradient) ≈ grad
            @test Array(gradient) ≈ _findiff_grad(oracle, u) rtol=1e-5 atol=1e-7
            ops = Base.invokelatest(_vc_backend_operations, sampler, ru)
            @test !isempty(ops)
            if n == 3
                operations[case.label] = ops
            elseif n == 7
                @test ops == operations[case.label]
            end
        end
    end
end
