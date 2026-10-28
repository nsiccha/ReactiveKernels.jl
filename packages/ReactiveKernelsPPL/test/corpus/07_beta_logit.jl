# data: p x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    kappa ~ Gamma(2.0, 1000.0)
    mu = a .+ b .* x
    p .~ Beta.(logistic.(mu) .* kappa, (1 .- logistic.(mu)) .* kappa)
end
