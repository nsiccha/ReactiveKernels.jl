# data: y x g
# A grouped HSGP: one curve per level of `g`, with per-group length scales
# and marginal scales from log-linear hyper-predictors written as plain
# statements in the shipped `hsgp_grouped_effect` body (synthetic toy
# program).
begin
    a ~ Normal(0, 1)
    (PHI, lambda) = hsgp_basis(x; k = 5, by = g)
    f ~ hsgp_grouped_effect(PHI, lambda, g)
    mu = a .+ f
    y .~ Normal.(mu, 1.5)
end
