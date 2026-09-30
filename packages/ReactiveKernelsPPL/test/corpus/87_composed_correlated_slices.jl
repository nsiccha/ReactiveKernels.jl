# data: y person item
# hier_2pl: correlated item draws sliced into two sub-predictors
# (discrimination + difficulty), per-person abilities as a factor sub.
begin
    c_th[levels(person)] .~ Normal.(0, 1)
    x1 ~ Normal(0, 1)
    x2 ~ Normal(0, 5)
    dx ~ varying_draws(item, [1, 1]; eta = 4.0, sd = Exponential(10))
    r1 ~ varying_slice(dx, 1)
    r2 ~ varying_slice(dx, 2)
    th = c_th[person]
    xi1 = x1 .+ r1
    xi2 = x2 .+ r2
    eta = exp.(xi1) .* (th .- xi2)
    y .~ Bernoulli.(logistic.(eta))
end
