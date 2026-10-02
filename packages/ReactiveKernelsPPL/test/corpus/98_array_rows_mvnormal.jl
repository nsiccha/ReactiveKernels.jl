# data: y x g
begin
    L ~ LKJCholesky(2, 2.0)
    sd[1:2] .~ HalfNormal.(1)
    F = sd .* L
    eachrow(B[levels(g), 1:2]) .~ MvNormalCholesky(zeros(2), F)
    a ~ Normal(0, 1)
    sigma ~ Exponential(1)
    r = B[g, 1] .+ B[g, 2] .* x
    mu = a .+ r
    y .~ Normal.(mu, sigma)
end
