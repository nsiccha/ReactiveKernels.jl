using Reactant

function _compiled_latent_reduction(fx, label)
    sampler = prepare_sampler(fx.built, fx.bound, fx.u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    kernel, ad = sampler.kernel, sampler.ad
    ru = Reactant.to_rarray(fx.u)
    primal = Reactant.@compile kernel(ru)
    reverse = compile_ad_value_and_gradient(ad, ru)
    original = deepcopy(fx.inputs)
    for shift in (0.0, 0.03)
        u = fx.u .+ shift
        native, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        r = Reactant.to_rarray(u)
        value, compiled_gradient = reverse(r)
        @test Float64(primal(r)) ≈ fx.oracle(u) rtol=1e-10
        @test Float64(value) ≈ native rtol=1e-10
        @test Array(compiled_gradient) ≈ gradient rtol=1e-9 atol=1e-9
    end
    @test fx.inputs == original
    if haskey(ENV, "RK_LATENT_REDUCTION_IR_DIR")
        dir = ENV["RK_LATENT_REDUCTION_IR_DIR"]
        mkpath(dir)
        both = reverse.f
        modules = (Reactant.@code_hlo(optimize=false, kernel(ru)),
            Reactant.@code_hlo(kernel(ru)),
            Reactant.@code_hlo(optimize=false, both(ru)),
            Reactant.@code_hlo(both(ru)))
        names = ("primal.raw.mlir", "primal.default.mlir",
            "reverse.raw.mlir", "reverse.default.mlir")
        for (name, mod) in zip(names, modules)
            write(joinpath(dir, "$label-$name"), String(mod))
        end
        write(joinpath(dir, "$label-primal.hlo"),
            repr(only(Reactant.XLA.get_hlo_modules(primal.exec))))
        write(joinpath(dir, "$label-reverse.hlo"),
            repr(only(Reactant.XLA.get_hlo_modules(reverse.exec))))
    end
end

@testset "whole latent plate reductions: default compiled values and reverse" begin
    Reactant.set_default_backend("cpu")
    for n in (3, 9, 19)
        fx = _latent_reduction_fixture(:sum, n)
        Base.invokelatest(_compiled_latent_reduction, fx, "sum-$n")
    end
    for fn in (:mean, :std, :var, :minimum, :maximum, :length)
        fx = _latent_reduction_fixture(fn, 9)
        Base.invokelatest(_compiled_latent_reduction, fx, "$fn-9")
    end
    for position in (:named, :derived)
        fx = _latent_reduction_fixture(:sum, 9; position)
        Base.invokelatest(_compiled_latent_reduction, fx, "sum-$position-9")
    end
end
