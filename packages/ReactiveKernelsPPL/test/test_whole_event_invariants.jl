module WholeEventInvariantTests
using Test, ReactiveKernels, ReactiveKernelsPPL
using Distributions: Uniform, Normal, Exponential
using LogExpFunctions: logistic, logit, log1pexp, logaddexp
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    bernoulli_product_mixture_logdensity, bernoulli_logit_product_mixture_logdensity
import DifferentiationInterface as DI
import Enzyme

function call_depths!(depths, expression, target, depth=0)
    expression isa Expr || return depths
    expression.head in (:quote,:inert) && return depths
    if expression.head === :call &&
            any(x -> x isa GlobalRef && isdefined(x.mod,x.name) &&
                getglobal(x.mod,x.name) === target, expression.args[1:min(2,end)])
        push!(depths, depth)
    end
    foreach(x -> call_depths!(depths,x,target,depth + (expression.head in (:for,:while))),
            expression.args)
    depths
end

# Independent Julia likelihood: a structural-zero component can explain only
# an entirely absent event. Its impossible branch contributes no derivative.
function event_likelihood(logits, weight, events, offsets, scale)
    total = 0.0
    for i in eachindex(events)
        present = log(weight)
        for j in eachindex(events[i])
            eta = logits[j] + scale * offsets[i]
            present += events[i][j] ? -log1pexp(-eta) : -log1pexp(eta)
        end
        total += any(events[i]) ? present : logaddexp(present,log1p(-weight))
    end
    total
end
function reference(raw, events, offsets, kind)
    if kind === :event_dependent
        # RKPPL packs scalar draws before the vector draw, independently of
        # the vector declaration's position in this authored model.
        scale, weight = exp(raw[1]), logistic(raw[2])
        location = logistic.(raw[3:end])
        logits = logit.(location)
        prior = -scale + raw[1]
        for i in 2:length(raw)
            prior += -log1pexp(-raw[i]) - log1pexp(raw[i])
        end
    else
        weight = logistic(raw[1])
        scale = 0.0
        prior = -log1pexp(-raw[1]) - log1pexp(raw[1])
        if kind === :probability
            location = logistic.(raw[2:end])
            logits = logit.(location)
            for i in 2:length(raw)
                prior += -log1pexp(-raw[i]) - log1pexp(raw[i])
            end
        else
            logits = raw[2:end]
            for x in logits
                prior += -(log(2pi)+x*x)/2
            end
        end
    end
    prior + event_likelihood(logits,weight,events,offsets,scale)
end

function make_sampler(kind, events, offsets, raw)
    T = maximum(length,events; init=3)
    absent = kind === :probability ? zeros(T) : fill(-Inf,T)
    data = (; events,offsets,T,absent)
    source = if kind === :probability
        quote
            weight ~ Uniform(0,1)
            location[1:T] .~ Uniform.(0,1)
            @plate for i in eachindex(events)
                events[i] ~ LogDensity(bernoulli_product_mixture_logdensity,
                    hcat(location,absent),[weight,1-weight])
            end
        end
    elseif kind === :logit
        quote
            weight ~ Uniform(0,1)
            location[1:T] .~ Normal.(0,1)
            @plate for i in eachindex(events)
                events[i] ~ LogDensity(bernoulli_logit_product_mixture_logdensity,
                    hcat(location,absent),[weight,1-weight])
            end
        end
    else
        quote
            location[1:T] .~ Uniform.(0,1)
            scale ~ Exponential(1)
            weight ~ Uniform(0,1)
            @plate for i in eachindex(events)
                eta = logit.(location) .+ scale .* offsets[i]
                events[i] ~ LogDensity(bernoulli_logit_product_mixture_logdensity,
                    hcat(eta,absent),[weight,1-weight])
            end
        end
    end
    bound = bind_data(lower_rkppl(source,data;mod=@__MODULE__,conditioned=(:events,)),data)
    prepare_sampler(build_kernel(bound),bound,raw;
        backend=DI.AutoEnzyme(;mode=Enzyme.Reverse),native_forward_batch=false)
end

@testset "natural whole-event invariant expressions" begin
    events = [Bool[true,false,true],Bool[false,false,false],Bool[false,true,false],
              Bool[true,true,false],Bool[false,false,false]]
    offsets = [-0.3,0.2,0.0,0.4,-0.1]
    for kind in (:probability,:logit,:event_dependent)
        raw = kind === :event_dependent ? [0.1,-0.3,0.2,0.3,-0.2] : [-0.2,0.1,-0.3,0.2]
        sampler = make_sampler(kind,events,offsets,raw)
        ast = code_expr(sampler.kernel)
        if kind === :event_dependent
            @test call_depths!(Int[],ast,logit) == [0]
            @test isempty(call_depths!(Int[],ast,ReactiveKernels._indexed_map))
        else
            @test call_depths!(Int[],ast,ReactiveKernels._indexed_map) == [0,0]
            @test ReactiveKernels._indexed_data_safe((sampler.kernel.constants,raw))
        end
        for values in (raw,raw .+ 0.15)
            expected_gradient = zero(values)
            expected = Enzyme.autodiff(Enzyme.ReverseWithPrimal,reference,Enzyme.Active,
                Enzyme.Duplicated(values,expected_gradient),Enzyme.Const(events),
                Enzyme.Const(offsets),Enzyme.Const(kind))[2]
            actual, gradient = sampler_value_and_gradient!(sampler,similar(values),values)
            @test actual ≈ expected rtol=1e-12
            @test gradient ≈ expected_gradient rtol=1e-11 atol=1e-12
        end
        empty_sampler = make_sampler(kind,Vector{Bool}[],Float64[],raw)
        actual, gradient = sampler_value_and_gradient!(empty_sampler,similar(raw),raw)
        expected_gradient = zero(raw)
        expected = Enzyme.autodiff(Enzyme.ReverseWithPrimal,reference,Enzyme.Active,
            Enzyme.Duplicated(raw,expected_gradient),Enzyme.Const(Vector{Bool}[]),
            Enzyme.Const(Float64[]),Enzyme.Const(kind))[2]
        @test actual ≈ expected
        @test gradient ≈ expected_gradient
    end
end
end
