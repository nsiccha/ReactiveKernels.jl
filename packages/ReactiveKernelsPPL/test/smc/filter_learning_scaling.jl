include("filter_learning_fixture.jl")
using .FilterLearningFixture
using ReactiveKernelsPPL, Random, Statistics, Serialization, Test
const F=FilterLearningFixture
const OPT=SMCOptions(moves=1,stepsize=0.35,max_tree_depth=2)
bytes(x)=(io=IOBuffer();serialize(io,x);position(io))
function measure(state,observation)
    filter_step(state,observation)
    samples=[@elapsed(filter_step(state,observation)) for _ in 1:5]
    output=filter_step(state,observation)
    checkpoint=filter_checkpoint(output)
    (;seconds=median(samples),evaluations=output.history[end].likelihood_evaluations,
        reference_evaluations=output.history[end].reference_evaluations,
        bytes=bytes(checkpoint),retained_bytes=Base.summarysize(checkpoint),
        replenished=output.history[end].replenished)
end

@testset "Bounded time, members, particles and groups" begin
    for (members,particles,groups) in ((8,16,1),(16,16,1),(8,32,1),(8,16,2),(8,16,4))
        target=F.target(group_count=groups)
        s=initialize_filter_smc(target;rng=Xoshiro(21),members,particles,
            window_length=4,history_limit=2,ess_fraction=1e-6,options=OPT)
        early=nothing
        for t in 1:240
            s=filter_step(s,(;))
            t==24 && (early=s)
        end
        a,b=measure(early,(;)),measure(s,(;))
        @test a.evaluations==b.evaluations==members*particles
        # Serialization encodes small clock integers more compactly (e.g.
        # 25 versus 241). Live retained storage must stay exactly constant;
        # report serialized bytes too, without mistaking that tag for history.
        @test a.retained_bytes==b.retained_bytes
        @test !a.replenished && !b.replenished
        @info "filter scaling at window boundary" members particles groups elapsed=(24,240) early=a late=b
    end
    for members in (8,16),particles in (8,16)
        target=F.target()
        s=initialize_filter_smc(target;rng=Xoshiro(56),members,particles,
            window_length=3,history_limit=2,ess_fraction=1.0,options=OPT)
        s=filter_stream(s,F.records([0.9,1.1]))
        measured=measure(s,(;y=0.8))
        @test measured.replenished
        @test measured.evaluations==members*particles*(1+3)
        @info "filter replenishment scaling" members particles window=3 measured
    end
end
