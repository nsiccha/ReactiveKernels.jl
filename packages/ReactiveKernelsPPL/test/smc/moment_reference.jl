include("window_fixture.jl")
using .SMCWindowFixture
using ReactiveKernelsPPL, Random, LinearAlgebra, Statistics, Test, Serialization
using WarmupHMC
const W=SMCWindowFixture
const OPTIONS=SMCOptions(metric=W.SCALE,stepsize=0.3,moves=2,max_tree_depth=5)
inflated=(;W.SCALE...,diagonal=W.SCALE.diagonal.*[9.0,0.03])
# Test-only dense oracle assembled through the supported vector action.
dense_scale(data)=hcat((scale_from_data(data)*Float64.(1:2 .== i) for i in 1:2)...)

@testset "Posterior moments in a bounded learned basis" begin
    fit=smc(W.window();rng=Xoshiro(402),particles=1600,options=OPTIONS)
    before=deepcopy(W.signature(fit))
    basis_before=deepcopy(inflated)
    caller=copy(Random.default_rng())
    carried=carry_smc(fit;calibration_id="moments",basis=inflated)
    nominal=carry_smc(fit;calibration_id="moments",basis=W.SCALE)
    @test carried.reference.scale.diagonal≈nominal.reference.scale.diagonal rtol=1e-12
    @test carried.reference.scale.indices==inflated.indices
    @test carried.reference.scale.reflections==inflated.reflections
    @test inflated==basis_before
    @test W.signature(fit)==before
    @test rand(copy(Random.default_rng()),UInt64,8)==rand(caller,UInt64,8)
    @test_throws ArgumentError carry_smc(fit;calibration_id="both",basis=inflated,scale=W.SCALE)
    @test_throws DimensionMismatch carry_smc(fit;calibration_id="bad",basis=ones(3))
    explicit=carry_smc(fit;calibration_id="as-given",scale=inflated)
    @test explicit.reference.scale==inflated
    diagonal=carry_smc(fit;calibration_id="diagonal",basis=[13.0,0.02])
    default=carry_smc(fit;calibration_id="diagonal")
    @test diagonal.reference.scale≈default.reference.scale rtol=1e-12
    selected=carry_smc(fit;calibration_id="selected",parameters=(:a,:b),basis=inflated)
    @test selected.reference.scale.diagonal≈carried.reference.scale.diagonal
    @test smc_parameters(selected,[0.2,-0.1])==(;a=0.2,b=-0.1)

    # Nonuniform weights test the weighted moment contract. Only this 2D
    # oracle constructs a covariance; production fits through vector solves.
    fit.weights .= collect(1:length(fit.weights))
    fit.weights ./= sum(fit.weights)
    weighted=carry_smc(fit;calibration_id="weighted",basis=inflated)
    center=fit.positions*fit.weights
    residual=fit.positions.-center
    covariance=(residual.*fit.weights')*residual'
    H=dense_scale((;W.SCALE...,diagonal=ones(2)))
    expected=H*Diagonal(diag(H'*covariance*H))*H'
    actual=dense_scale(weighted.reference.scale)
    @test weighted.reference.location≈center
    @test actual*actual'≈expected rtol=1e-12
    @test Base.summarysize(weighted.reference)==Base.summarysize(nominal.reference)
    io=IOBuffer();serialize(io,weighted);seekstart(io)
    restored=deserialize(io)
    @test W.EXT._same(restored.frame,weighted.frame)
    @test restored.reference.scale==weighted.reference.scale
end

@testset "Sequential correlated moment carry against Gaussian posterior" begin
    previous=W.calibration()
    all_y=Float64[]
    H=dense_scale((;W.SCALE...,diagonal=ones(2)))
    for (i,y) in enumerate(([0.9,1.2,0.8],[1.4,0.6,1.1],[0.8,1.0,1.3]))
        append!(all_y,y)
        target=W.window(y;previous,binding_id="window-$i")
        fit=smc(target;rng=Xoshiro(401+i),particles=1600,options=OPTIONS)
        previous=carry_smc(fit;calibration_id="through-$i",basis=inflated)
        h=ones(2)
        covariance=inv(inv(W.COVARIANCE)+length(all_y)*(h*h'))
        location=covariance*(W.COVARIANCE\W.LOCATION+h*sum(all_y))
        scale=dense_scale(previous.reference.scale)
        variance=diag(H'*covariance*H)
        errors=(H'*(previous.reference.location-location))./sqrt.(variance)
        ratios=diag(H'*(scale*scale')*H)./variance
        @info "Moment-carried Gaussian oracle" window=i standardized_mean_errors=errors variance_ratios=ratios
        @test maximum(abs.(errors))<0.2
        @test maximum(abs.(ratios.-1))<0.2
    end
end
