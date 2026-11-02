@traceable function filter_source_recurrence(x::AbstractVector, gain::Real)
    T = promote_type(eltype(x), typeof(gain))
    output = Vector{T}(undef, length(x))
    level, slope = zero(T), one(T)
    for t in eachindex(x)
        slope = 0.7slope + gain*x[t]
        level = level + slope
        output[t] = level
    end
    return output
end

@traceable function filter_source_branch(x::AbstractVector)
    output = similar(x)
    carry = 0.0
    for t in eachindex(x)
        if x[t] > 0
            carry = carry + log(x[t])
        else
            carry = carry + sqrt(-x[t])
        end
        output[t] = carry
    end
    return output
end

@traceable function filter_source_history(x::AbstractVector)
    output = similar(x)
    carry = sum(x)
    for t in eachindex(x)
        carry = carry + x[t]
        output[t] = carry
    end
    return output
end

@testset "Helper projection preserves branches and whole vector events" begin
    model = @rkppl begin
        loc = filter_source_branch(x)
        y .~ Normal.(loc,0.5)
    end
    x = [0.2,-0.4,0.0,1.2]
    y = [0.1,0.2,0.3,0.4]
    path = filter_source_branch(x)
    target = prepare_filter(model;model_id="helper-lazy-branches",inputs=(:x,))
    state = initialize_filter(target;rng=Xoshiro(8),particles=2)
    for t in eachindex(x)
        state = filter_step(state,(;y=y[t]);inputs=(;x=x[t]))
        @test state.predictive_logscore ≈ logpdf(Normal(path[t],0.5),y[t])
    end
    history = @rkppl begin
        loc = filter_source_history(x)
        y .~ Normal.(loc,0.5)
    end
    @test_throws ArgumentError prepare_filter(history;model_id="future-dependent-seed",inputs=(:x,))
    event = @rkppl begin
        loc = filter_source_branch(x)
        y ~ MvNormal(loc,covariance)
    end
    covariance = [i==j ? 1.0 : 0.0 for i in 1:4, j in 1:4]
    target = prepare_filter(event;model_id="whole-helper-event",inputs=(:x,),data=(;covariance))
    @test isempty(target.prototype)
    state = initialize_filter(target;rng=Xoshiro(9),particles=2)
    @test filter_step(state,(;y);inputs=(;x)).predictive_logscore ≈ logpdf(MvNormal(path,covariance),y)
end

@testset "Recorded helper recurrence, grouped gathers and exact continuation" begin
    model = @rkppl begin
        gain ~ Normal()
        offset ~ Normal()
        @plate for g in eachindex(y)
            path = filter_source_recurrence(x[g], gain)
            loc[g] = offset .+ path[index[g]]
            # An unrequested diagnostic must not consume absent observations.
            diagnostic[g] = sum(y[g])
            y[g] .~ Normal.(loc[g], 0.5)
        end
    end
    x = [[0.1,0.3,-0.2,0.7],[-0.2,0.4,0.5,0.1]]
    y = [[0.2,0.1,-0.1,0.3],[0.1,-0.2,0.1,0.4]]
    index = [collect(eachindex(v)) for v in x]
    pars = (;gain=0.4,offset=0.2)
    bound = model(;x,index,pars...) | (;y)
    target = prepare_filter(model,bound;model_id="recorded-helper",parameters=pars,
        inputs=(:x,:index),groups=(;g=1:2))
    @test isempty(target.data)
    @test length(target.prototype) == 2
    state = initialize_filter(target;rng=Xoshiro(6),particles=4,history_limit=2)
    paths = [filter_source_recurrence(v,pars.gain) for v in x]
    for t in 1:4
        state = filter_step(state,(;y=[v[t] for v in y]);
            inputs=(;x=[v[t] for v in x],index=[t,t]))
        @test state.predictive_logscore ≈ sum(logpdf(Normal(pars.offset+paths[g][t],0.5),y[g][t]) for g in 1:2)
    end
    @test state.log_normalizer ≈ Base.invokelatest(
        prepare_query(build_kernel(bound),bound,:likelihood),Float64[])
    checkpoint = roundtrip_filter(filter_checkpoint(state))
    step(s) = filter_step(s,(;y=[0.0,0.3]);inputs=(;x=[0.2,0.4],index=[5,5]))
    @test filter_checkpoint(step(state)) == filter_checkpoint(step(resume_filter(target,checkpoint)))
    @test_throws ArgumentError filter_step(state,(;y=[0.0,0.3]);inputs=(;x=[0.2,0.4],index=[4,5]))
    @test filter_checkpoint(state) == checkpoint

    subset = initialize_filter(target;rng=Xoshiro(7),particles=2)
    for (g,t) in [(2,1),(1,1),(2,2),(2,3),(1,2),(1,3),(2,4),(1,4)]
        inputs = (;x=[v[t] for v in x],index=[t,t])
        subset = filter_step(subset,(;y=[v[t] for v in y]);inputs,groups=(;g=[g]))
        @test subset.predictive_logscore ≈ logpdf(Normal(pars.offset+paths[g][t],0.5),y[g][t])
    end
    @test subset.log_normalizer ≈ state.log_normalizer
    @test subset.group_times.g == [4,4]

    warm = initialize_filter(target;rng=Xoshiro(6),particles=4)
    for t in 1:4
        warm = filter_step(warm,(;y=[v[t] for v in y]);
            inputs=(;x=[v[t] for v in x],index=[t,t]),condition_state_only=true)
    end
    @test warm.log_normalizer == 0.0
    @test step(warm).predictive_logscore ≈ step(state).predictive_logscore
    sizes = Int[]
    for t in 5:128
        state = filter_step(state,(;y=[0.1,0.2]);inputs=(;x=[0.2,0.3],index=[t,t]))
        t in (8,32,128) && push!(sizes,Base.summarysize(state))
    end
    @test all(==(first(sizes)),sizes)
end
