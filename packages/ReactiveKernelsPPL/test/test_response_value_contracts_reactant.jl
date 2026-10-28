using Reactant

@testset "response value contracts: default compiled primal and reverse" begin
    for (provider, reference) in ((_rvc_beta, _rvc_beta_reference),
            (_rvc_scan, _rvc_scan_reference), (_rvc_effects, _rvc_effects_reference))
        for n in (0, 1, 4, 8, 16)
            fx = provider(n)
            before = deepcopy(fx.data)
            sampler = prepare_sampler(fx.built, fx.bound, fx.u;
                backend = AutoEnzyme(; mode = Enzyme.Reverse))
            ru = Reactant.to_rarray(fx.u)
            primal = Reactant.@compile sampler.kernel(ru)
            reverse = compile_ad_value_and_gradient(sampler.ad, ru)
            for u in (fx.u, fx.u .+ 0.03)
                input = Reactant.to_rarray(u)
                value, grad = reverse(input)
                @test Float64(primal(input)) ≈ reference(fx, u) rtol = 1e-10
                @test Float64(value) ≈ reference(fx, u) rtol = 1e-10
                step = cbrt(eps(Float64))
                fd = map(eachindex(u)) do i
                    up, dn = copy(u), copy(u)
                    up[i] += step
                    dn[i] -= step
                    (reference(fx, up) - reference(fx, dn)) / (2step)
                end
                @test Array(grad) ≈ fd rtol = 1e-5 atol = 1e-7
                @test Array(input) == u
            end
            @test isequal(fx.data, before)
        end
    end
    for provider in (_rvc_beta, _rvc_scan), n in (0, 16)
        fx = provider(n; singleton = true)
        reference = provider === _rvc_beta ? _rvc_beta_reference : _rvc_scan_reference
        sampler = prepare_sampler(fx.built, fx.bound, fx.u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        ru = Reactant.to_rarray(fx.u)
        primal = Reactant.@compile sampler.kernel(ru)
        reverse = compile_ad_value_and_gradient(sampler.ad, ru)
        value, gradient = reverse(ru)
        _, native = sampler_value_and_gradient!(sampler, similar(fx.u), fx.u)
        @test Float64(primal(ru)) ≈ reference(fx, fx.u)
        @test Float64(value) ≈ reference(fx, fx.u)
        @test Array(gradient) ≈ native
    end
end
