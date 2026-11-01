module SMCProjectionFixture
include("filter_learning_fixture.jl")
using .FilterLearningFixture
using ReactiveKernels, ReactiveKernelsPPL, Random, Serialization
const F=FilterLearningFixture
const BATCH_ID="batch-source-and-training-v1"
const FILTER_ID="stream-source-frozen-config-v1"
function model()
    @rkppl begin
        theta ~ Normal()
        sigma ~ LogNormal(0,0.3)
        @scan begin
            h[1] ~ Normal(theta,sigma)
            for t in 2:T
                h[t] ~ Normal(0.6h[t-1]+0.4theta,sigma)
            end
        end
        y .~ Normal.(h,0.5)
    end
end
function full(n=3)
    plan=model()(;T=n)|(;y=fill(0.8,n))
    built=build_kernel(plan)
    target=prepare_smc(built,plan,zeros(built.layout.total);backend=F.BACKEND,
        model_id=BATCH_ID,binding_id="training-$n")
    (;built,plan,target)
end
function completed()
    smc(full().target;rng=Xoshiro(416),particles=64,
        options=SMCOptions(stepsize=0.15,moves=1,max_tree_depth=3))
end
static_filter(;model_id=FILTER_ID) =
    prepare_filter(model();model_id,parameters=(;theta=0.0,sigma=1.0))
learning(carried;model_id=FILTER_ID) = prepare_filter_smc(static_filter(;model_id),carried;
    backend=F.BACKEND,calibration_model_id=BATCH_ID)
end
