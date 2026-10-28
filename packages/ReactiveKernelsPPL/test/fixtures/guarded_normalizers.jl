module GuardedNormalizers

using Distributions
using LogExpFunctions: logistic, logaddexp
using ReactiveKernels
using ReactiveKernelsPPL
using ReactiveKernelsDistributionKernels.DistributionKernelSources: loggamma

function fixture(family, n; raw_nuisance = false)
    repeat_to(v) = repeat(v, cld(n, length(v)))[1:n]
    x = repeat_to([-1.2, -0.4, 0.1, 0.6, 1.3])
    counts = repeat_to([0, 2, 5, 1, 3])
    trials = repeat_to([3, 5, 8, 2, 7])
    df = repeat_to([3.0, 4.0, 7.0, 5.0, 9.0])
    positive = repeat_to([0.2, 1.4, 2.7, 0.6, 3.1])
    unit = repeat_to([0.1, 0.3, 0.8, 0.6, 0.2])
    y = family === :Gamma ? positive : family === :Beta ? unit :
        family === :StudentT ? positive : counts
    scale = raw_nuisance ? :c : :(exp(c))
    probability = raw_nuisance ? :c : :(1 / (1 + exp(-c)))
    response = if family === :NB2
        :(y .~ NegativeBinomial2.(exp.(eta), $scale))
    elseif family === :NB1
        :(y .~ NegativeBinomial.(exp.(eta), $probability))
    elseif family === :Gamma
        :(y .~ Gamma.($scale, exp.(eta) ./ $scale))
    elseif family === :Beta
        :(y .~ Beta.(logistic.(eta) .* $scale, (1 .- logistic.(eta)) .* $scale))
    elseif family === :StudentT
        :(y .~ StudentT.(df, eta, $scale))
    elseif family === :ZIP
        :(y .~ ZeroInflatedPoisson.(exp.(eta), $probability))
    elseif family === :Hurdle
        :(y .~ HurdlePoisson.(exp.(eta), $probability))
    elseif family === :BetaBinomial
        :(y .~ BetaBinomial2.(trials, logistic.(eta), $scale))
    else
        error("unknown fixture $family")
    end
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        c ~ Normal(0, 1)
        eta = a .+ b .* x
        $response
    end
    data = (; x, y, trials, df)
    oracle = u -> begin
        eta = u[1] .+ u[2] .* x
        s, p = raw_nuisance ? (u[3], u[3]) : (exp(u[3]), logistic(u[3]))
        pointwise = if family === :NB2
            logpdf.(NegativeBinomial.(s, s ./ (s .+ exp.(eta))), y)
        elseif family === :NB1
            logpdf.(NegativeBinomial.(exp.(eta), p), y)
        elseif family === :Gamma
            logpdf.(Gamma.(s, exp.(eta) ./ s), y)
        elseif family === :Beta
            mu = logistic.(eta)
            logpdf.(Beta.(mu .* s, (1 .- mu) .* s), y)
        elseif family === :StudentT
            logpdf.(LocationScale.(eta, s, TDist.(df)), y)
        elseif family === :ZIP
            map(y, exp.(eta)) do count, rate
                count == 0 ? logaddexp(log(p), log1p(-p) - rate) :
                    log1p(-p) + logpdf(Poisson(rate), count)
            end
        elseif family === :Hurdle
            map(y, exp.(eta)) do count, rate
                count == 0 ? log(p) :
                    log1p(-p) + logpdf(Poisson(rate), count) - log(-expm1(-rate))
            end
        else
            mu = logistic.(eta)
            logpdf.(BetaBinomial.(trials, mu .* s, (1 .- mu) .* s), y)
        end
        sum(pointwise) + sum(logpdf.(Normal(), u))
    end
    (; ast, data, oracle)
end

function build(fixture)
    values = Dict{Symbol,Any}(pairs(fixture.data))
    plan = bind_data(lower_rkppl(fixture.ast, values; conditioned = (:y,)), values)
    built = build_kernel(plan)
    prepare_query(built, plan, :sampler)
end

function findiff(f, u)
    h = cbrt(eps(Float64))
    [(f(u + h .* (eachindex(u) .== i)) - f(u - h .* (eachindex(u) .== i))) / (2h)
     for i in eachindex(u)]
end

end
