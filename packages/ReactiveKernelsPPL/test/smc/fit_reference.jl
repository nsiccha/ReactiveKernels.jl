include("window_fixture.jl")
using .SMCWindowFixture
using ReactiveKernelsPPL, Random, LinearAlgebra, Test, Serialization, WarmupHMC
const W = SMCWindowFixture
const OPTIONS = SMCOptions(metric=W.SCALE, stepsize=0.3, moves=2, max_tree_depth=5)
# Dense covariance is a two-dimensional test oracle only.
factor(data, d=2) = hcat((scale_from_data(data)*Float64.(1:d .== i) for i in 1:d)...)

@testset "Carried posterior learns weighted correlation" begin
    fit = smc(W.window(); rng=Xoshiro(402), particles=1600, options=OPTIONS)
    fit.weights .= 1:length(fit.weights)
    fit.weights ./= sum(fit.weights)
    before = deepcopy(W.signature(fit))
    caller = copy(Random.default_rng())
    carried = carry_smc(fit; calibration_id="fitted", basis=:fit)
    center = fit.positions*fit.weights
    residual = fit.positions.-center
    covariance = (residual.*fit.weights')*residual'
    L = factor(carried.reference.scale)
    @test carried.reference.location ≈ center
    @test L*L' ≈ covariance rtol=1e-7
    @test carried.fit_diagnostics.effective_sample_size ≈ inv(sum(abs2,fit.weights))
    @test carried.fit_diagnostics.reflections == 1
    @test all(carried.fit_diagnostics.converged)
    @test all(0 .< carried.fit_diagnostics.iterations .<= 100)
    @test W.signature(fit) == before
    @test rand(copy(Random.default_rng()),UInt64,8) == rand(caller,UInt64,8)
    again = carry_smc(fit; calibration_id="fitted", basis=:fit)
    @test W.EXT._same(carried,again)

    diagonal = carry_smc(fit; calibration_id="diagonal", basis=:fit, max_reflections=0)
    marginal = carry_smc(fit; calibration_id="marginal")
    D = factor(diagonal.reference.scale)
    @test D*D' ≈ Diagonal(abs2.(marginal.reference.scale))
    @test marginal.fit_diagnostics === nothing
    @test diagonal.fit_diagnostics.reflections == 0
    limited = carry_smc(fit; calibration_id="bounded", basis=:fit, maxiter=1)
    @test all(limited.fit_diagnostics.iterations .<= 1)
    @test_throws ArgumentError carry_smc(fit;calibration_id="bad",basis=:unknown)
    @test_throws ArgumentError carry_smc(fit;calibration_id="bad",basis=:fit,maxiter=0)
    @test_throws ArgumentError carry_smc(fit;calibration_id="bad",basis=:fit,max_reflections=-1)
    @test_throws ArgumentError carry_smc(fit;calibration_id="bad",basis=:fit,scale=W.SCALE)

    selected = carry_smc(fit; parameters=(:a,), calibration_id="selected", basis=:fit)
    @test length(selected.reference.location) == 1
    @test only(selected.reference.location) ≈ center[1]
    S = factor(selected.reference.scale,1)
    @test only(S*S') ≈ covariance[1,1]
    @test smc_parameters(selected,[0.3]) == (;a=0.3)
    io = IOBuffer(); serialize(io,carried); seekstart(io)
    @test W.EXT._same(deserialize(io),carried)
    frozen = deepcopy(carried)
    fit.positions .= 1000
    fit.weights .= inv(length(fit.weights))
    @test W.EXT._same(carried,frozen)
    @test Base.summarysize(carried) < 10000
end

@testset "Sequential fitted carry against full Gaussian posterior" begin
    previous = W.calibration()
    all_y = Float64[]
    H = factor((;W.SCALE...,diagonal=ones(2)))
    for (i,y) in enumerate(([0.9,1.2,0.8],[1.4,0.6,1.1],[0.8,1.0,1.3]))
        append!(all_y,y)
        target = W.window(y;previous,binding_id="fitted-window-$i")
        fit = smc(target;rng=Xoshiro(401+i),particles=1600,options=OPTIONS)
        # The carry learns from the completed draws, with no oracle basis.
        previous = carry_smc(fit;calibration_id="fitted-through-$i",basis=:fit)
        h = ones(2)
        covariance = inv(inv(W.COVARIANCE)+length(all_y)*(h*h'))
        location = covariance*(W.COVARIANCE\W.LOCATION+h*sum(all_y))
        L = factor(previous.reference.scale)
        variance = diag(H'*covariance*H)
        errors = (H'*(previous.reference.location-location))./sqrt.(variance)
        ratios = diag(H'*(L*L')*H)./variance
        @info "Fitted-carry Gaussian oracle" window=i standardized_mean_errors=errors variance_ratios=ratios
        @test maximum(abs.(errors)) < 0.2
        @test maximum(abs.(ratios.-1)) < 0.2
    end
end
