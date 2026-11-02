module PreparedSMCFixture
using ReactiveKernelsPPL, DifferentiationInterface, Enzyme, LogDensityProblems
using SequentialMonteCarlo, WarmupHMC, Random

const BACKEND = AutoEnzyme(mode=Enzyme.Reverse)

mutable struct GaussianProblem
    location::Vector{Float64}
    scale::Vector{Float64}
    logmass::Float64
    gradient::Vector{Float64}
    evaluations::Int
    gradients::Int
    busy::Bool
    fault::Symbol
end
LogDensityProblems.capabilities(::Type{GaussianProblem}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::GaussianProblem) = length(p.location)
Base.deepcopy_internal(::GaussianProblem, ::IdDict) = error("rebuild this solver workspace through its factory")

function value(p, u)
    p.fault === :density && return NaN
    p.fault === :support && return -Inf
    return p.logmass - length(u)*log(2pi)/2 - sum(log, p.scale) -
        sum(abs2, (u .- p.location) ./ p.scale)/2
end
function LogDensityProblems.logdensity(p::GaussianProblem, u)
    p.evaluations += 1
    return value(p, u)
end
function LogDensityProblems.logdensity_and_gradient(p::GaussianProblem, u)
    p.busy && error("concurrent use of one problem workspace")
    p.busy = true
    try
        p.gradients += 1
        p.gradient .= .-(u .- p.location) ./ p.scale.^2
        yield() # Make aliased worker scratch observable, even on one thread.
        gradient = p.fault === :gradient ? fill(NaN, length(u)) :
            p.fault === :shape ? zeros(length(u)+1) :
            p.fault === :support ? nothing : p.gradient
        return value(p, u), gradient
    finally
        p.busy = false
    end
end

struct GaussianFactory
    location::Vector{Float64}
    scale::Vector{Float64}
    logmass::Float64
    fault::Symbol
    instances::Vector{GaussianProblem}
end
GaussianFactory(; location=[0.4,-0.6], scale=[0.5,0.8], logmass=0.7, fault=:none) =
    GaussianFactory(location, scale, logmass, fault, GaussianProblem[])
function (f::GaussianFactory)()
    p = GaussianProblem(f.location, f.scale, f.logmass, zeros(length(f.location)),
        0, 0, false, f.fault)
    push!(f.instances, p)
    return p
end
Base.deepcopy_internal(::GaussianFactory, ::IdDict) = error("do not copy a problem factory")

function target(f=GaussianFactory(); model_id="prepared-gaussian-v1",
        binding_id="data-v1", frame_id="ordered-x-y-v1", u0=zeros(length(f.location)))
    return prepare_smc(f, u0; backend=BACKEND, model_id, binding_id, frame_id)
end
options(nthreads=1) = SMCOptions(; nthreads, stepsize=0.2, moves=2, max_tree_depth=5)
snapshot(s) = (; s.positions, s.weights, s.temperature, s.log_normalizer,
    s.ancestors, s.lineage, s.history, rng=rand(copy(s.rng), UInt64, 8))
end
