using DifferentiationInterface: AutoEnzyme
using Distributions: Normal, logcdf, logpdf
using ReactiveKernels, ReactiveKernelsPPL, Test
import Enzyme

# A dose-free grouped plate: per-assay scales gathered from a literal vector
# of model scalars feed `CensoredAddpropnormal` (log-CDF at/below LLOQ,
# density above, additive/proportional scale).
function _censored_addprop_ast(source=:([a1,a2]))
    return quote
        a1 ~ Exponential(1.)
        a2 ~ Exponential(1.)
        p1 ~ Exponential(1.)
        p2 ~ Exponential(1.)
        a ~ Normal(0.,1.)
        mu_lp = a
        sched = linear_pk_schedule(obs=(:subj,:time),dose=(:dsubj,:dtime,:damt))
        @plate result for s in 1:kernel_nsub_result
            mu = mu_lp[subj]
            add = $source[assay]
            prop = [p1,p2][assay]
            dv .~ CensoredAddpropnormal.(mu,add,prop,lloq)
            mu
        end
    end
end

_censored_addprop_fd(f, x; h = 1e-6) = [(f(x .+ h .* (eachindex(x) .== i)) -
    f(x .- h .* (eachindex(x) .== i))) / (2h) for i in eachindex(x)]

@testset "assay scale gather and raw censoring ordinary reverse" begin
    cols = Dict{Symbol,AbstractVector}(:subj=>[1,1,1,2,2],:time=>[0.,1.,2.,0.,1.],
        :dsubj=>Int[],:dtime=>Float64[],:damt=>Float64[],
        :location=>[.8,.9,1.2,.7,1.1],:assay=>[1,2,1,2,1],
        :dv=>[.3,.5,1.3,.2,1.4],:lloq=>fill(.5,5))
    plan = lower_rkppl(_censored_addprop_ast(),Set(keys(cols)); conditioned = Set(keys(cols)))
    bound = bind_data(plan,cols;dims=Dict(:kernel_nsub_result=>2))
    built = build_kernel(bound)
    names = coordinate_names(built.layout)
    values = Dict(:a1=>log(.4),:a2=>log(.6),:p1=>log(.15),:p2=>log(.2),
        :a=>.9)
    u = [values[n] for n in names]
    function oracle(p)
        coords = Dict(zip(names,p))
        scales = Dict(n=>exp(coords[n]) for n in (:a1,:a2,:p1,:p2))
        prior = sum(-v for v in Base.values(scales))+
            sum(coords[n] for n in keys(scales))+logpdf(Normal(0.,1.),coords[:a])
        likelihood = 0.
        for i in eachindex(cols[:dv])
            j = cols[:assay][i]
            mu = coords[:a]
            sd = hypot(scales[Symbol(:a,j)],mu*scales[Symbol(:p,j)])
            likelihood += cols[:dv][i] <= cols[:lloq][i] ?
                logcdf(Normal(mu,sd),cols[:lloq][i]) : logpdf(Normal(mu,sd),cols[:dv][i])
        end
        return prior+likelihood
    end
    query = prepare_query(built,bound,:sampler)
    @test query(u) ≈ oracle(u) rtol=2e-13
    sampler = prepare_sampler(built,bound,u;backend=AutoEnzyme(;mode=Enzyme.Reverse))
    g = zeros(length(u))
    value,_ = sampler_value_and_gradient!(sampler,g,u)
    @test value ≈ oracle(u) rtol=2e-13
    @test g ≈ _censored_addprop_fd(oracle,u) rtol=2e-6 atol=2e-8
    for bad in ([0,2,1,2,1],[1,3,1,2,1],[1.,2.,1.,2.,1.])
        # refused: gather indices must be integers within a nonempty source axis (Julia indexing, P3)
        @test_throws ContractValidationError bind_data(plan,merge(cols,Dict(:assay=>bad));
            dims=Dict(:kernel_nsub_result=>2))
    end
    # refused: gather indices must be integers within a nonempty source axis (Julia indexing, P3)
    @test_throws "empty" lower_rkppl(_censored_addprop_ast(:([])),Set(keys(cols)); conditioned = Set(keys(cols)))
    # capability: a gathered vector mixing a data vector and a model scalar (ordinary values, P3/P10a 0dejlw1) (todo `1qlbn5b`)
    @test_broken (lower_rkppl(_censored_addprop_ast(:([location,a1])),Set(keys(cols)); conditioned = Set(keys(cols))); true)
end
