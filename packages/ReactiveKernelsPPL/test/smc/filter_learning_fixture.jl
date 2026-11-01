module FilterLearningFixture
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using Distributions, SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random
using Statistics, Serialization
const BACKEND = AutoEnzyme(mode=Enzyme.Reverse)
const ID = "filter-learning-v1"
const PARAMETER_ID = "theta-source-coordinate-v1"
const Y = [0.8,1.2,0.9,1.4,0.7,1.0,1.1,0.6]
const EXT = Base.get_extension(ReactiveKernelsPPL,:ReactiveKernelsPPLSMCExt)

function calibration(;model_id=ID,calibration_id="calibration",location=0.0,scale=1.0)
    data = (; y=Float64[])
    plan = bind_data(lower_rkppl(quote
        theta ~ Normal()
        y .~ Normal.(theta,1)
    end,data;conditioned=(:y,)),data)
    target = prepare_smc(build_kernel(plan),plan,[0.0];backend=BACKEND,
        model_id,binding_id="calibration-empty")
    carry_smc(target,SMCReference([location],[scale]);calibration_id)
end

parameters(u) = (;theta=only(u))
function target(;model_id=ID,parameter_id=PARAMETER_ID,carried=calibration(),group_count=nothing)
    model = if group_count === nothing
        @rkppl begin
            theta ~ Normal()
            @scan begin
                h[1] ~ Normal(theta,0.6)
                for t in 2:T
                    h[t] ~ Normal(0.6h[t-1]+0.4theta,0.3)
                end
            end
            y .~ Normal.(h,0.5)
        end
    else
        @rkppl begin
            theta ~ Normal()
            @plate for g in eachindex(y)
                T=length(y[g])
                @scan begin
                    h[1] ~ Normal(theta,0.6)
                    for t in 2:T
                        h[t] ~ Normal(0.6h[t-1]+0.4theta,0.3)
                    end
                end
                loc[g]=h
                y[g] .~ Normal.(loc[g],0.5)
            end
        end
    end
    groups = group_count === nothing ? (;) : (;g=1:group_count)
    filter = prepare_filter(model;model_id,parameters=parameters([0.0]),groups)
    prepare_filter_smc(filter,carried;parameters,parameter_id,backend=BACKEND)
end

function support_target()
    model=@rkppl begin
        theta ~ Normal()
        @scan begin
            h[1]=theta
            for t in 2:T
                h[t]=theta
            end
        end
        y .~ Uniform.(h,h .+ 1)
    end
    filter=prepare_filter(model;model_id=ID,parameters=parameters([0.0]))
    prepare_filter_smc(filter,calibration();parameters,parameter_id=PARAMETER_ID,backend=BACKEND)
end

function kalman_score(y,theta)
    m,v,total = theta,0.6^2,0.0
    for t in eachindex(y)
        if t>1
            m=0.6m+0.4theta
            v=0.6^2*v+0.3^2
        end
        total+=logpdf(Normal(m,sqrt(v+0.5^2)),y[t])
        gain=v/(v+0.5^2)
        m+=gain*(y[t]-m)
        v*=1-gain
    end
    total
end

function oracle(y=Y)
    f(theta) = logpdf(Normal(),theta)+kalman_score(y,theta)
    a,b,c = f(-1.0),f(0.0),f(1.0)
    variance = -1/(a+c-2b)
    (;mean=(c-a)*variance/2,variance)
end

moments(s) = (m=only(s.positions*s.parameter_mass);
    (;mean=m,variance=sum(s.parameter_mass.*(vec(s.positions).-m).^2)))
records(y=Y) = [(;observation=(;y=x)) for x in y]
signature(s) = (;positions=s.positions,parameter_mass=s.parameter_mass,
    calibration=(;s.calibration.reference.location,s.calibration.reference.scale,
        s.calibration.frame,s.calibration.calibration_id),
    log_parameter_mass=s.log_parameter_mass,boundary=s.boundary,records=s.records,
    members=map(filter_checkpoint,s.members),time=s.time,window=s.window,
    history=s.history,score=s.predictive_logscore,log_normalizer=s.log_normalizer,
    rng_draws=rand(copy(s.rng),UInt64,16))

function equal_signatures(a,b)
    # Layout metadata uses NaN for absent bounds. isequal matches those
    # sentinels while retaining exact comparisons, including signed zeros.
    keys(a)==keys(b) || return false
    different=Tuple(k for k in keys(a) if !isequal(a[k],b[k]))
    isempty(different) || @error "Filter replay fields differ" fields=different
    isempty(different)
end
end
