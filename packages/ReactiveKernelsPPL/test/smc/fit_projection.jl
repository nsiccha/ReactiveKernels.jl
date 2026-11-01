include("projection_fixture.jl")
using .SMCProjectionFixture
using ReactiveKernelsPPL, Random, LinearAlgebra, WarmupHMC, Test
const P = SMCProjectionFixture
const F = P.F
@testset "Fitted static carry from latent calibration" begin
    fit = P.completed()
    carried = carry_smc(fit;parameters=filter_parameter_names(P.model()),
        calibration_id="fitted-static",basis=:fit)
    mean = fit.positions[1:2,:]*fit.weights
    residual = fit.positions[1:2,:].-mean
    covariance = (residual.*fit.weights')*residual'
    L = hcat((scale_from_data(carried.reference.scale)*Float64.(1:2 .== i) for i in 1:2)...)
    @test carried.reference.location ≈ mean
    @test L*L' ≈ covariance rtol=1e-7
    @test length(carried.reference.location) == 2
    @test isempty(carried.frame.parameters.layout.bound_values)
    @test carried.fit_diagnostics.reflections == 1
    target = P.learning(carried)
    s = initialize_filter_smc(target;rng=Xoshiro(11),members=16,particles=16,
        window_length=2,ess_fraction=0.8,
        options=SMCOptions(moves=1,stepsize=0.15,max_tree_depth=3))
    s = filter_step(s,(;y=0.9))
    cp = filter_checkpoint(s)
    a = filter_step(s,(;y=1.1))
    b = filter_step(resume_filter(P.learning(carried),cp),(;y=1.1))
    @test F.equal_signatures(F.signature(a),F.signature(b))
    # Rollover replaces calibration moments and must also replay exactly.
    cp = filter_checkpoint(a)
    a = filter_step(a,(;y=1.0))
    b = filter_step(resume_filter(P.learning(carried),cp),(;y=1.0))
    @test F.equal_signatures(F.signature(a),F.signature(b))
end
