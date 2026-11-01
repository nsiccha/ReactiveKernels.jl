using Test, Random, Distributions, SequentialMonteCarlo, Statistics, Serialization
using ReactiveKernels, ReactiveKernelsPPL

roundtrip_filter(x) = (io=IOBuffer(); serialize(io,x); seekstart(io); deserialize(io))
bytes_filter(x) = (io=IOBuffer(); serialize(io,x); position(io))
current(s,nm;g=1) = [getproperty(c,nm)[end,g] for c in s.carries]
fmean(s,nm;g=1) = sum(current(s,nm;g).*s.weights)
fvar(s,nm;g=1) = sum((current(s,nm;g).-fmean(s,nm;g)).^2 .* s.weights)

@testset "Finite deterministic scan and full replay" begin
    model = @rkppl begin
        gain ~ Normal()
        @scan begin
            h[1] = 0.0
            d[1] = 0.0
            for t in 2:T
                d[t] = 0.6d[t-1]+gain*x[t]
                h[t] = h[t-1]+d[t]
            end
        end
        y .~ Normal.(h,0.5)
    end
    target = prepare_filter(model;model_id="det",parameters=(;gain=0.7),inputs=(:x,))
    rng = Xoshiro(13); copyrng=copy(rng)
    s = initialize_filter(target;rng,particles=8,history_limit=3)
    @test rand(rng) == rand(copyrng)
    x = [sin(t) for t in 1:20]
    y = [cos(t) for t in 1:20]
    h,d,score = 0.0,0.0,0.0
    for t in eachindex(x)
        if t > 1
            d=0.6d+0.7x[t]; h+=d
        end
        score+=logpdf(Normal(h,0.5),y[t])
        old=roundtrip_filter(filter_checkpoint(s))
        s=filter_step(s,(;y=y[t]);inputs=(;x=x[t]))
        @test all(==(h),current(s,:h))
        @test all(==(d),current(s,:d))
        @test s.log_normalizer ≈ score
        @test length(s.history) <= 3
        @test old.time == t-1
    end
    @test all(s.carries[i].h !== s.carries[j].h for i in 1:8 for j in 1:8 if i!=j)
    plan=model(;x,T=length(x),gain=0.7)|(;y)
    built=build_kernel(plan)
    @test Base.invokelatest(prepare_query(built,plan,:likelihood),Float64[]) ≈ s.log_normalizer
    cp=roundtrip_filter(filter_checkpoint(s))
    a=filter_step(s,(;y=0.2);inputs=(;x=0.4))
    b=filter_step(resume_filter(target,cp),(;y=0.2);inputs=(;x=0.4))
    @test filter_checkpoint(a) == filter_checkpoint(b)
end

function gaussian_filter(;phi=0.8,q=0.3,r=0.5)
    model=@rkppl begin
        phi ~ Normal()
        q ~ Exponential()
        r ~ Exponential()
        @scan begin
            h[1] ~ Normal()
            for t in 2:T
                h[t] ~ Normal(phi*h[t-1],q)
            end
        end
        y .~ Normal.(h,r)
    end
    prepare_filter(model;model_id="lg",parameters=(;phi,q,r))
end

function kalman_reference(y;phi=0.8,q=0.3,r=0.5)
    m,v=0.0,1.0
    means,vars,scores=Float64[],Float64[],Float64[]
    for t in eachindex(y)
        if t>1; m*=phi; v=phi^2*v+q^2; end
        push!(scores,logpdf(Normal(m,sqrt(v+r^2)),y[t]))
        gain=v/(v+r^2)
        m+=gain*(y[t]-m); v*=1-gain
        push!(means,m);push!(vars,v)
    end
    (;means,vars,scores)
end

@testset "Latent Gaussian calibration and stochastic convergence" begin
    y=[0.6sin(t/4)+0.1cos(t) for t in 1:28]
    oracle=kalman_reference(y)
    errors=Float64[]
    for n in (128,4096)
        squared=0.0
        for seed in 1:4
            s=initialize_filter(gaussian_filter();rng=Xoshiro(seed),particles=n)
            for t in eachindex(y)
                s=filter_step(s,(;y=y[t]))
                squared+=(fmean(s,:h)-oracle.means[t])^2
                if n==4096
                    @test abs(fmean(s,:h)-oracle.means[t])<0.08
                    @test abs(fvar(s,:h)-oracle.vars[t])<0.035
                    @test abs(s.predictive_logscore-oracle.scores[t])<0.12
                end
            end
        end
        push!(errors,squared)
    end
    @test errors[2]<errors[1]/4
end

@testset "Fixed calibrated ensemble, state-only warm-up, drifting model" begin
    model=@rkppl begin
        theta ~ Normal()
        @scan begin
            h[1]=theta
            for t in 2:T
                h[t]=theta
            end
        end
        y .~ Normal.(h,0.5)
    end
    target=prepare_filter(model;model_id="calibrated",parameters=(;theta=0.0))
    s=initialize_filter(target;rng=Xoshiro(9),particles=8,
        ensemble=[(;theta=-0.5),(;theta=0.5)],weights=[0.2,0.8])
    for _ in 1:8
        s=filter_step(s,(;y=0.8);condition_state_only=true)
        @test s.parameter_mass == [0.2,0.8]
        @test s.log_normalizer == 0.0
        @test last(s.history).condition_state_only
    end
    s=filter_step(s,(;y=-0.3))
    w=[0.2pdf(Normal(-0.5,0.5),-0.3),0.8pdf(Normal(0.5,0.5),-0.3)]
    @test s.parameter_mass ≈ w/sum(w)
    @test s.predictive_logscore ≈ log(sum(w))
    @test s.log_normalizer ≈ log(sum(w))
    dynamic=@rkppl begin
        @scan begin
            theta[1] ~ Normal(0,0.2)
            for t in 2:T
                theta[t] ~ Normal(theta[t-1],0.12)
            end
        end
        y .~ Normal.(theta,0.2)
    end
    target=prepare_filter(dynamic;model_id="drifting",drifting=(:theta,))
    s=initialize_filter(target;rng=Xoshiro(22),particles=2048)
    truth=[0.03t for t in 1:40]
    estimates=Float64[]
    for x in truth
        s=filter_step(s,(;y=x))
        push!(estimates,fmean(s,:theta))
    end
    @test sqrt(mean((estimates.-truth).^2))<0.08
    @test mean((estimates.-truth).^2)<mean(abs2,truth)/10
    @test_throws ArgumentError prepare_filter(model;model_id="bad-drift",parameters=(;theta=0.0),drifting=(:theta,))
end

@testset "Ordinary group semantics and absent observations" begin
    model=@rkppl begin
        gain ~ Normal()
        @plate for g in eachindex(y)
            T=length(y[g])
            @scan begin
                h[1]=g*gain
                for t in 2:T
                    h[t]=h[t-1]+gain
                end
            end
            loc[g]=h
            y[g] .~ Normal.(loc[g],0.5)
        end
    end
    target=prepare_filter(model;model_id="groups",parameters=(;gain=0.5),groups=(;g=1:3),
        data=(;g=zeros(1000)))
    @test isempty(target.data) # a loop binder does not retain a same-named training column
    s=initialize_filter(target;rng=Xoshiro(1),particles=8)
    s=filter_step(s,(;y=[0.5,1.0,1.5]))
    @test s.group_times.g == [1,1,1]
    @test [fmean(s,Symbol("g.h");g) for g in 1:3] == [0.5,1.0,1.5]
    old=deepcopy(s.carries)
    s=filter_step(s,(;y=[missing,1.5,missing]);groups=(;g=[2]))
    @test s.group_times.g == [1,2,1]
    @test [fmean(s,Symbol("g.h");g) for g in 1:3] == [0.5,1.5,1.5]
    @test s.predictive_logscore ≈ logpdf(Normal(1.5,0.5),1.5)
    s=filter_step(s,(;);groups=(;g=[1]))
    @test s.group_times.g == [2,2,1]
    @test fmean(s,Symbol("g.h");g=1)==1.0
    @test s.predictive_logscore ≈ 0.0 atol=1e-12
    @test_throws ArgumentError filter_step(s,(;);groups=(;g=[4]))
    subset=@rkppl begin
        @plate for g in 2:3
            @scan begin
                h[1]=0.0
                for t in 2:T
                    h[t]=h[t-1]+1
                end
            end
            y[g] .~ Normal.(h,1)
        end
    end
    @test_throws ArgumentError prepare_filter(subset;model_id="domain-mismatch",groups=(;g=1:3))
end

@rkppl filter_ar(T) = begin
    phi_raw ~ Normal()
    @scan begin
        state[1] ~ Normal()
        for t in 2:T
            eps ~ Normal()
            state[t]=phi*state[t-1]+eps
        end
    end
    phi=tanh(phi_raw)
    state
end

@testset "Nested AR submodel and observed recurrence seeds" begin
    model=@rkppl begin
        z ~ filter_ar(T)
        y .~ Normal.(z,0.5)
    end
    pars=NamedTuple{(Symbol("z.phi_raw"),)}((0.4,))
    @test filter_parameter_names(model)==(Symbol("z.phi_raw"),)
    draw=(;z=(;phi_raw=0.4,state=zeros(10000),eps=zeros(10000)))
    @test filter_parameters(model,draw)==pars
    @test Base.summarysize(filter_parameters(model,draw))<128
    target=prepare_filter(model;model_id="submodel-ar",parameters=pars)
    s=initialize_filter(target;rng=Xoshiro(12),particles=2048)
    y=[0.2,0.3,-0.1,0.7]
    ref=kalman_reference(y;phi=tanh(0.4),q=1.0)
    for t in eachindex(y)
        s=filter_step(s,(;y=y[t]))
        @test abs(s.predictive_logscore-ref.scores[t])<0.15
    end
    observed=@rkppl begin
        @scan begin
            y[1] ~ Normal(0,1)
            for t in 2:T
                y[t] ~ Normal(0.7y[t-1],0.3)
            end
        end
    end
    target=prepare_filter(observed;model_id="observed-ar")
    s=initialize_filter(target;rng=Xoshiro(1),particles=8)
    s=filter_step(s,(;y=0.4))
    @test s.predictive_logscore≈logpdf(Normal(),0.4)
    s=filter_step(s,(;y=0.2))
    @test s.predictive_logscore≈logpdf(Normal(0.28,0.3),0.2)
    s=filter_step(s,(;y=missing))
    @test s.predictive_logscore≈0.0 atol=1e-12
    @test length(unique(current(s,:y)))>1
end

# Public BRM emitter form: deterministic seeds and a sampled innovation,
# without an independently transcribed streaming transition.
@rkppl filter_dar(T)=begin
    beta ~ truncated(Normal(0.5,0.2),0,1)
    sigma ~ HalfNormal(0.2)
    @scan begin
        level[1]=0.0
        increment[1]=0.0
        for t in 2:T
            innovation ~ Normal(0,1)
            increment[t]=beta*increment[t-1]+sigma*innovation
            level[t]=level[t-1]+increment[t]
        end
    end
    return level
end

@testset "Emitted differenced AR submodel" begin
    model=@rkppl begin
        dar_mu_t_level ~ filter_dar(4)
        mu_Intercept ~ Normal(0,1)
        mu=mu_Intercept .+ dar_mu_t_level
        y .~ Normal.(mu,1.0)
    end
    pars=filter_parameters(model,(;dar_mu_t_level=(;beta=0.7,sigma=0.2,
        level=zeros(100),increment=zeros(100)),mu_Intercept=0.3))
    @test keys(pars)==(Symbol("dar_mu_t_level.beta"),Symbol("dar_mu_t_level.sigma"),:mu_Intercept)
    target=prepare_filter(model;model_id="emitted-dar",parameters=pars)
    s=initialize_filter(target;rng=Xoshiro(8),particles=4096)
    s=filter_step(s,(;y=0.4))
    @test s.predictive_logscore≈logpdf(Normal(0.3,1.0),0.4)
    s=filter_step(s,(;y=0.2))
    @test abs(s.predictive_logscore-logpdf(Normal(0.3,sqrt(1.04)),0.2))<0.02
    @test all(abs.(current(s,Symbol("dar_mu_t_level.level"))) .> 0)
end

@kernel filter_provider(x,gain)=begin
    values=scan(x;init=0.0) do carry,value
        next=0.7carry+gain*value
        (next,next)
    end
    return values
end

@kernel filter_bad_provider(x)=begin
    values=scan(x;init=0.0) do carry,value
        next=carry+value
        (next,next)
    end
    return sum(values)
end

function inferred_filter_transition(target,carry,inputs,observation)
    @inferred target.advance(Xoshiro(1),carry,target.parameters,target.data,inputs,
        observation,2,target.operations,(;),(;))
end

@testset "Authored kernel provider and frozen preprocessing" begin
    model=@rkppl begin
        gain ~ Normal()
        center=mean(x_fit)
        scaled=x .- center
        unused_total=sum(y)
        loc=filter_provider(scaled,gain)
        y .~ Normal.(loc,obs_scale)
    end
    bound=model(;gain=0.6,x=[0.1,0.2],x_fit=[0.1,0.2],obs_scale=0.5)|(;y=[0.0,0.0])
    frozen=frozen_values(bound,(:center,))
    target=prepare_filter(model;model_id="provider",parameters=(;gain=0.6),inputs=(:x,),
        frozen,data=(;obs_scale=0.5))
    @test Base.invokelatest(inferred_filter_transition,target,target.prototype,(;x=0.2),(;y=0.1)) isa Tuple
    @test !haskey(target.data,:x_fit)
    frombound=prepare_filter(model,bound;model_id="provider",parameters=(;gain=0.6),inputs=(:x,),frozen)
    @test frombound.data==target.data
    @test Set(keys(frombound.data))==Set((:center,:obs_scale))
    @test frombound.data.center≈0.15
    @test Base.summarysize(frombound)==Base.summarysize(target)
    large=model(;gain=0.6,x=zeros(10000),x_fit=fill(0.15,10000),obs_scale=0.5)|(;y=zeros(10000))
    fromlarge=prepare_filter(model,large;model_id="provider",parameters=(;gain=0.6),inputs=(:x,),frozen)
    @test fromlarge.signature==frombound.signature
    @test Base.summarysize(fromlarge)==Base.summarysize(frombound)
    changed=prepare_filter(model,bound;model_id="provider",parameters=(;gain=0.6),inputs=(:x,),
        frozen,data=(;obs_scale=0.7))
    @test_throws ArgumentError resume_filter(changed,
        filter_checkpoint(initialize_filter(frombound;rng=Xoshiro(1),particles=8)))
    s=initialize_filter(target;rng=Xoshiro(1),particles=8)
    h=0.0
    for x in [0.2,0.6,-0.4,0.5]
        h=0.7h+0.6(x-0.15)
        s=filter_step(s,(;y=0.1);inputs=(;x))
        @test s.predictive_logscore ≈ logpdf(Normal(h,0.5),0.1)
    end
    a=initialize_filter(frombound;rng=Xoshiro(1),particles=8)
    @test filter_checkpoint(filter_step(a,(;y=0.1);inputs=(;x=0.3)))==
        filter_checkpoint(filter_step(initialize_filter(target;rng=Xoshiro(1),particles=8),(;y=0.1);inputs=(;x=0.3)))
end

@rkppl filter_design_submodel(X,ncoef,loc,scale) = begin
    beta[1:ncoef] .~ Normal.(loc,scale)
    return X * beta
end

@testset "Temporal design rows and zero-state updates" begin
    model=@rkppl begin
        center=mean(x_fit)
        spread=std(x_fit)
        scaled=(x .- center) ./ spread
        X=hcat(ones(length(x)),scaled)
        mu ~ filter_design_submodel(X,2,0.0,1.0)
        y .~ Normal.(mu,0.5)
    end
    x=[0.2,0.6,-0.4,0.5]
    y=[0.1,-0.2,0.3,0.4]
    draw=(;mu=(;beta=[0.3,0.7]))
    pars=filter_parameters(model,draw;inputs=(:x,),data=(;x_fit=x))
    frozen=FrozenValues((;center=0.15,spread=0.4),(;))
    bound=model(;x,x_fit=x)|(;y)
    target=prepare_filter(model,bound;model_id="design",parameters=pars,inputs=(:x,),frozen)
    @test Set(keys(target.data))==Set((:center,:spread))
    @test isempty(target.prototype)
    @test Base.invokelatest(inferred_filter_transition,target,target.prototype,(;x=0.2),(;y=0.1)) isa Tuple
    s=initialize_filter(target;rng=Xoshiro(13),particles=4)
    for i in eachindex(x)
        s=filter_step(s,(;y=y[i]);inputs=(;x=x[i]))
        @test s.predictive_logscore≈logpdf(Normal(0.3+0.7*(x[i]-0.15)/0.4,0.5),y[i])
    end
    large=model(;x=zeros(10000),x_fit=zeros(10000))|(;y=zeros(10000))
    other=prepare_filter(model,large;model_id="design",parameters=pars,inputs=(:x,),frozen)
    @test Base.summarysize(other)==Base.summarysize(target)
    @test other.signature==target.signature
    @test filter_checkpoint(filter_step(resume_filter(other,roundtrip_filter(filter_checkpoint(s))),
        (;y=0.2);inputs=(;x=0.8)))==filter_checkpoint(filter_step(s,(;y=0.2);inputs=(;x=0.8)))
end

const filter_work=Ref(0)
count_filter_step(x)=(filter_work[]+=1; x+1.0)

@testset "Bounded retention, ownership, errors and cooperative stop" begin
    model=@rkppl begin
        @scan begin
            h[1]=0.0
            for t in 2:T
                h[t]=count_filter_step(h[t-1])
            end
        end
        y .~ Normal.(h,1.0)
    end
    target=prepare_filter(model;model_id="bounded")
    s=initialize_filter(target;rng=Xoshiro(3),particles=8,history_limit=4)
    sizes=Int[]; checkpoints=Int[]; work=Int[]
    for t in 1:512
        filter_work[]=0
        s=filter_step(s,(;y=Float64(t-1)))
        if t in (8,64,512)
            push!(sizes,Base.summarysize(s));push!(checkpoints,bytes_filter(filter_checkpoint(s)))
            push!(work,filter_work[])
        end
    end
    @test maximum(sizes)==minimum(sizes)
    @test maximum(checkpoints)-minimum(checkpoints)<=32
    @test work==[8,8,8]
    println("Retained bytes at 8/64/512 events: ",sizes,"; checkpoint bytes: ",checkpoints,"; transition calls: ",work)
    cp=filter_checkpoint(s)
    @test_throws ArgumentError filter_step(s,(;y=Inf))
    @test filter_checkpoint(s)==cp
    @test_throws ArgumentError resume_filter(target,merge(cp,(;time=-1)))
    records=Iterators.Stateful(((;observation=(;y=Float64(t))) for t in 1:10))
    fresh=initialize_filter(target;rng=Xoshiro(3),particles=8)
    stopped=filter_stream(fresh,records;stop=s->s.time==3)
    @test stopped.time==3
    @test length(records)==7
    @test filter_stream(stopped,records).time==10
    history=@rkppl begin
        @scan begin
            h[1]=0.0
            for t in 2:T
                h[t]=sum(h[j] for j in 1:t-1)
            end
        end
        y .~ Normal.(h,1.0)
    end
    @test_throws ArgumentError prepare_filter(history;model_id="history")
    reduction=@rkppl begin
        @scan begin
            h[1]=0.0
            for t in 2:T
                h[t]=h[t-1]+1
            end
        end
        total=sum(h)
        y .~ Normal.(total,1.0)
    end
    @test_throws ArgumentError prepare_filter(reduction;model_id="whole-history-reduction")
    opaque=@rkppl begin
        loc=filter_bad_provider(x)
        y .~ Normal.(loc,1.0)
    end
    @test_throws ArgumentError prepare_filter(opaque;model_id="provider-reduction",inputs=(:x,))
    single=initialize_filter(target;rng=Xoshiro(3),particles=1)
    @test filter_step(single,(;y=0.0)).predictive_logscore≈logpdf(Normal(),0.0)
end
