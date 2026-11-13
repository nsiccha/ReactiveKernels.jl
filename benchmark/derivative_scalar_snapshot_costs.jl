# Ordinary scalar Reverse exercises adapter retention, unlike explicit stage loops.
include(joinpath(@__DIR__, "derivative_joint_staging_tradeoffs.jl"))

function scalar_reverse(f::F, x) where {F}
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, Enzyme.Const(f),
        Enzyme.Active, Enzyme.Active(x))
end

function run_scalar_adapter_costs()
    independent = make_rule(scalar_graph(10))
    shared = make_rule(scalar_graph(5))
    for x in (-2.0, 0.0, 0.7, 2.0), f in (independent, shared, exp)
        dx, y = scalar_reverse(f, x)
        @test y ≈ exp(x)
        @test only(dx) ≈ exp(x)
    end
    x = 0.7
    arms = ((:independent, () -> scalar_reverse(independent, x)),
            (:shared, () -> scalar_reverse(shared, x)),
            (:ordinary, () -> scalar_reverse(exp, x)))
    record_trials(:scalar_adapter, 1, arms)
end

println((; julia=VERSION, rk=pkgversion(ReactiveKernels), enzyme=pkgversion(Enzyme),
    library=Enzyme.Compiler.Enzyme_jll.libEnzyme_path))
run_scalar_adapter_costs()
println("SCALAR_SNAPSHOT_ADAPTER_COMPLETE")
