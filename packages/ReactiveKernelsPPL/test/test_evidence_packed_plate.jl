using Test, ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
import Distributions as PackedEvidenceD

function _packed_evidence_check(plan, u, expected)
    validate_plan(plan)
    built=build_kernel(plan)
    k=prepare_query(built,plan,:likelihood)
    Base.invokelatest() do
        @test k(u)≈expected atol=2e-12
        ad=prepare_ad(k,AutoEnzyme(;mode=Enzyme.Reverse),u;active=:unconstrained)
        value,grad=ad_value_and_gradient!(ad,similar(u),u)
        fd=map(eachindex(u)) do i
            up,down=copy(u),copy(u);up[i]+=1e-5;down[i]-=1e-5
            (k(up)-k(down))/2e-5
        end
        @test value≈expected atol=2e-12
        @test grad≈fd atol=1e-7 rtol=1e-6
    end
end

function _packed_wrapped_logpdf(kind,d,y,lo,hi)
    kind===:interval_censored && return log(PackedEvidenceD.cdf(d,hi)-PackedEvidenceD.cdf(d,y))
    return PackedEvidenceD.logpdf(kind===:censored ? PackedEvidenceD.censored(d,lo,hi) : PackedEvidenceD.truncated(d,lo,hi),y)
end

@testset "packed missing-response evidence gathers full-row bounds" begin
    for kind in (:truncated,:censored,:interval_censored), selected_range in (nothing, 2:4)
        x=[-1.,.5,2.,.25];jobs=[1,3];lo=[-1.,-2.,-.6,-3.];hi=[2.,3.,1.8,4.]
        ys=kind===:censored ? [lo[1],hi[3]] : [.2,.4]
        cols=Dict{Symbol,AbstractVector}(:y=>ys,:x=>x,:Jobs=>jobs,:lo=>lo,:hi=>hi)
        terms=TermSpec[TermSpec(InterceptTerm,ColumnRef[],NamedTuple(),:Intercept,:intercept),
            TermSpec(ContinuousTerm,[:x],NamedTuple(),:x,:x_term)]
        response=LikelihoodSpec(GaussianFam,IdentityLink,:y,:mu,:sigma,nothing,
            ResponseEvidence(kind,kind===:interval_censored ? nothing : :lo,:hi),
            :y_resp,nothing,selected_range;mi_jobs=:Jobs)
        plan=StructuralPlan([response],[PredictorSpec(:mu,IdentityLink,terms,:mu)],
            PopulationPrior[PopulationPrior(:mu,:Intercept,0.,1.),PopulationPrior(:mu,:x,0.,2.)],
            SampledParameter[SampledParameter(:sigma,:exponential,(arg1=1.,),nothing,:sigma)],
            AssignmentSpec[],cols,4)
        u=[.2,-.1,log(1.2)]
        expected=sum(_packed_wrapped_logpdf(kind,PackedEvidenceD.Normal(.2-.1*x[j],1.2),y,lo[j],hi[j])
            for (j,y) in zip(jobs,ys) if selected_range === nothing || j in selected_range)
        _packed_evidence_check(plan,u,expected)
    end
end

@testset "packed evidence gathers live derived vector bounds" begin
    for kind in (:truncated,:censored,:interval_censored)
        x=[-1.,.5,2.,.25];jobs=[1,3];lo=[-1.,-2.,-.6,-3.];hi=[2.,3.,1.8,4.]
        ys=[.2,.4]
        cols=Dict{Symbol,AbstractVector}(:y=>ys,:x=>x,:Jobs=>jobs,:lo=>lo,:hi=>hi)
        terms=TermSpec[TermSpec(InterceptTerm,ColumnRef[],NamedTuple(),:Intercept,:intercept),
            TermSpec(ContinuousTerm,[:x],NamedTuple(),:x,:x_term)]
        response=LikelihoodSpec(GaussianFam,IdentityLink,:y,:mu,:sigma,nothing,
            ResponseEvidence(kind,kind===:interval_censored ? nothing : :live_lo,:live_hi),
            :y_resp,nothing,nothing;mi_jobs=:Jobs)
        plan=StructuralPlan([response],[PredictorSpec(:mu,IdentityLink,terms,:mu)],
            PopulationPrior[PopulationPrior(:mu,:Intercept,0.,1.),PopulationPrior(:mu,:x,0.,2.)],
            SampledParameter[SampledParameter(:sigma,:exponential,(arg1=1.,),nothing,:sigma)],
            AssignmentSpec[],cols,4;derived=VectorAssignmentSpec[
                VectorAssignmentSpec(:live_lo,:(lo .+ sigma .- 1.2)),
                VectorAssignmentSpec(:live_hi,:(hi .+ sigma .- 1.2))])
        u=[.2,-.1,log(1.2)]
        expected=sum(_packed_wrapped_logpdf(kind,PackedEvidenceD.Normal(.2-.1*x[j],1.2),y,lo[j],hi[j]) for (j,y) in zip(jobs,ys))
        _packed_evidence_check(plan,u,expected)
    end
end

@testset "direct plate-mean evidence" begin
    for kind in (:truncated,:censored,:interval_censored)
        lo,hi=-1.,2.;ys=kind===:censored ? [lo,.5,hi] : [.2,.5,1.]
        response=LikelihoodSpec(GaussianFam,IdentityLink,:y,:latent,.7,nothing,
            ResponseEvidence(kind,kind===:interval_censored ? nothing : lo,hi),:y_resp)
        plan=StructuralPlan([response],PredictorSpec[],PopulationPrior[],SampledParameter[],AssignmentSpec[],
            Dict{Symbol,AbstractVector}(:y=>ys),3;
            plate_parameters=PlateParameter[PlateParameter(:latent,:normal,(arg1=0.,arg2=1.),nothing)])
        u=[.1,-.2,.3]
        expected=sum(_packed_wrapped_logpdf(kind,PackedEvidenceD.Normal(m,.7),y,lo,hi) for (m,y) in zip(u,ys))
        _packed_evidence_check(plan,u,expected)
    end
end

@testset "packed GLM evidence selects rows and bounds once" begin
    for family in (:NormalIDGLM, :BernoulliLogitGLM, :PoissonLogGLM),
            kind in (:truncated, :censored, :interval_censored)
        x = [-1., .5, 2., .25]
        jobs = [1, 3]
        lo = family === :NormalIDGLM ? [-.5, -.2, .2, -.4] : [.2, .3, .4, .1]
        hi = family === :BernoulliLogitGLM ? ones(4) : [2., 3., 2.8, 4.]
        ys = kind === :censored ? [lo[1], hi[3]] :
            kind === :interval_censored ? [.3, .5] : [1., 1.]
        data = Dict{Symbol,Any}(:y=>ys, :x=>x, :Jobs=>jobs, :lo=>lo, :hi=>hi)
        ctor = family === :NormalIDGLM ? :(NormalIDGLM(X,a,b,1.2)) :
            Expr(:call, family, :X, :a, :b)
        wrap = kind === :interval_censored ? Expr(:call,kind,ctor,:hi) :
            Expr(:call,kind,ctor,:lo,:hi)
        expr = quote
            X=hcat(x); a ~ Normal(0,1); b[axes(X,2)] .~ Normal.(0,1)
            y ~ $wrap
        end
        plan = lower_rkppl(expr, keys(data); conditioned=keys(data))
        r = ReactiveKernelsPPL._with(only(plan.responses); mi_jobs=:Jobs)
        plan = bind_data(ReactiveKernelsPPL._with(plan; responses=[r]), data)
        positions = eachindex(jobs)
        law(eta) = family === :NormalIDGLM ? PackedEvidenceD.Normal(eta,1.2) :
            family === :BernoulliLogitGLM ? PackedEvidenceD.Bernoulli(1/(1+exp(-eta))) :
            PackedEvidenceD.Poisson(exp(eta))
        expected = sum(_packed_wrapped_logpdf(kind,law(.2-.1*x[jobs[k]]),
            ys[k],lo[jobs[k]],hi[jobs[k]]) for k in positions)
        _packed_evidence_check(plan, [.2,-.1], expected)
    end
end
