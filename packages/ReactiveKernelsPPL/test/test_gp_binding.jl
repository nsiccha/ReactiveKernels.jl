include("gp_binding_fixtures.jl")

@testset "GP spellings follow model bindings and ordinary shapes" begin
    for name in _GPB_NAMES, kind in (:scalar, :whole, :array, :plate, :data, :dotted)
        results = []
        for spelling in (:bare, :qualified, :alias)
            @testset "$name $kind $spelling" begin
                fx = _gpb_build(name, spelling, kind)
                push!(results, _gpb_check(fx))
            end
        end
        @test all(r -> r[1] ≈ results[1][1] && r[2] ≈ results[1][2], results)
    end
    @test all(name -> name ∉ admitted_functions(), _GPB_NAMES)
end

@testset "GP call heads obey Julia name resolution" begin
    # Undefined callees and model values follow the ordinary callable
    # contract; a GP identifier grants neither a binding nor a shape.
    for name in _GPB_NAMES[1:3]
        ast = quote
            s ~ Normal(0, 1)
            f = $(Expr(:call, name, :s))
            y .~ Normal.(f, 0.7)
        end
        # Refused: calling an undefined binding violates Julia name resolution.
        @test_throws SurfaceLoweringError lower_rkppl(ast, (:y,);
            mod = GPBindingUndefined, conditioned = (:y,))
        ast = quote
            $name ~ Normal(0, 1)
            f = $(Expr(:call, name, 0.1))
            y .~ Normal.(f, 0.7)
        end
        # Refused: a sampled model value is not a callable (functions-as-values contract).
        @test_throws SurfaceLoweringError lower_rkppl(ast, (:y,);
            mod = GPBindingModels, conditioned = (:y,))
    end
end
