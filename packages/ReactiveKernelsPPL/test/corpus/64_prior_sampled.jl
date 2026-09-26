# data: y x
begin
    mu = a .+ b .* x
    y .~ Normal.(mu, s)
    t ~ StudentT(3, 1, 2)
    l ~ Laplace(0, 1.5)
    g ~ Logistic(2, 0.5)
    u ~ Uniform(-1, 2)
    h ~ truncated(StudentT(3, 0, 2), 0, Inf)
    s ~ Exponential(1)
end
