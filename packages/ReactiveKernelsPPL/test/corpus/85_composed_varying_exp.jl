# data: y person item
# IRT 2PL (brms `theta ~ 0 + (1 | person)`, `log_a ~ 0 + (1 | item)`,
# `b ~ 1 + (1 | item)`): varying-effect sub-predictors + a dotted
# `exp.` map in the composition.
begin
    r_t ~ varying_effect(person, [1]; eta = 1.0, sd = Cauchy(0, 2))
    r_a ~ varying_effect(item, [1]; eta = 1.0, sd = Cauchy(0, 2))
    r_b ~ varying_effect(item, [1]; eta = 1.0, sd = Cauchy(0, 2))
    b0 ~ Normal(0, 5)
    th = r_t
    la = r_a
    b = b0 .+ r_b
    eta = exp.(la) .* (th .- b)
    y .~ Bernoulli.(logistic.(eta))
end
