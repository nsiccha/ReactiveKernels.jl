using Reactant

@testset "Reactant: selected prior values and ordinary reverse" begin
    for prior in (:data, :active, :mapped), (n, nobs) in ((0, 4), (1, 4), (6, 4), (18, 11))
        fx = _prior_selection_fixture(n, nobs; prior)
        saved = deepcopy(fx.inputs)
        sampler = prepare_sampler(fx.built, fx.bound, fx.u;
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        kernel = sampler.kernel
        reverse(v) = only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(kernel), v))
        ru = Reactant.to_rarray(fx.u)
        for (direction, fn) in ((:primal, kernel), (:reverse, reverse))
            mlir = repr(Reactant.@code_hlo optimize=:all fn(ru))
            compiled = Reactant.@compile optimize=:all fn(ru)
            hlo = repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
            for shift in (0.0, 0.03)
                u = fx.u .+ shift
                input = Reactant.to_rarray(u)
                if direction === :primal
                    @test Float64(compiled(input)) ≈ fx.oracle(u)
                else
                    _, expected = sampler_value_and_gradient!(sampler, similar(u), u)
                    @test Array(compiled(input)) ≈ expected rtol=1e-8 atol=1e-9
                end
                @test Array(input) == u
            end
            println("PRIOR_SELECTION_IR prior=", prior, " n=", n, " nobs=", nobs,
                " direction=", direction)
            if haskey(ENV, "RKPPL_PRIOR_SELECTION_IR_DIR")
                dir = ENV["RKPPL_PRIOR_SELECTION_IR_DIR"]
                mkpath(dir)
                write(joinpath(dir, "$prior-$n-$nobs-$direction.mlir"), mlir)
                write(joinpath(dir, "$prior-$n-$nobs-$direction.hlo"), hlo)
            end
        end
        @test Array(ru) == fx.u
        @test fx.inputs == saved
    end
end
