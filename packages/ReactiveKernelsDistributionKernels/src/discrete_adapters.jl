@kernel _categorical_distribution(probabilities) = begin
    checked = _checked_categorical_probabilities(probabilities)
    logpdf(observed)::Float64 =
        _integer_observation(observed) && (1 <= observed) && (observed <= length(checked)) ?
            log(checked[Int(observed)]) : -Inf
end

@kernel _binomial_distribution(n::Int, p::Float64) = begin
    logpdf(observed)::Float64 = _integer_observation(observed) && (0 <= observed) && (observed <= n) ?
        binomial(n, p).logpdf(Int(observed)) : -Inf
end

@kernel _bernoulli_distribution(p::Float64) = begin
    logpdf(observed)::Float64 = observed == 0 || observed == 1 ?
        bernoulli(p).logpdf(observed == 1) : -Inf
end
