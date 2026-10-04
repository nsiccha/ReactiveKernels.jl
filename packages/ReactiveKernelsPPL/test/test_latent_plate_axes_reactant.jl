using Reactant

function _latent_axis_inventory(text)
    ops = [m.match for m in eachmatch(
        r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|cf|tensor|math|linalg|memref)\.\w+", text)]
    return Dict(op => count(==(op), ops) for op in unique(ops))
end

@testset "Reactant: independent latent plate axis values and reverse" begin
    ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsReactantExt)
    retained = ext._rk_reactant_pipeline_no_slice_slice()
    # The documented default optimizer expands small constant loop bounds.
    # Verify that size with the existing retained-loop pipeline; inspect both
    # pipelines at larger sizes instead of treating small unrolling as support.
    for (pipeline, optimize, n) in ((:retained, retained, 3),
        (:retained, retained, 9), (:retained, retained, 17),
        (:default, :all, 9), (:default, :all, 17))
        fx = _latent_axis_fixture(n)
        saved = deepcopy(fx.data)
        ru = Reactant.to_rarray(fx.u)
        kernel = fx.sampler.kernel
        reverse(v) = only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(kernel), v))
        for (direction, fn) in ((:primal, kernel), (:reverse, reverse))
            mlir = repr(Reactant.@code_hlo optimize=optimize fn(ru))
            compiled = Reactant.@compile optimize=optimize fn(ru)
            hlo = repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
            if direction === :primal
                @test Float64(compiled(ru)) ≈ fx.oracle(fx.u)
            else
                _, expected = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
                @test Array(compiled(ru)) ≈ expected rtol=1e-8
            end
            # Complete inventories are diagnostics: shape specialization and
            # reduction stages may change without replicating authored bodies.
            println("LATENT_AXIS_IR pipeline=", pipeline, " n=", n, " direction=", direction,
                " operations=", sort!(collect(_latent_axis_inventory(mlir)); by=first))
            if haskey(ENV, "RKPPL_LATENT_AXIS_IR_DIR")
                dir = ENV["RKPPL_LATENT_AXIS_IR_DIR"]
                mkpath(dir)
                write(joinpath(dir, "$pipeline-$n-$direction.mlir"), mlir)
                write(joinpath(dir, "$pipeline-$n-$direction.hlo"), hlo)
            end
        end
        @test Array(ru) == fx.u
        @test fx.data == saved
    end
end
