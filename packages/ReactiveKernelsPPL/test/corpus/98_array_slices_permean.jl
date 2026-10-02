# data: y x g
begin
    L ~ LKJCholesky(2, 2.0)
    sd[1:2] .~ HalfNormal.(1)
    F = sd .* L
    S = F * F'
    m[1:2] ~ MvNormal(zeros(2), S)
    M[levels(g), 1:2] .~ Normal.(0, 1)
    eachrow(B[levels(g), 1:2]) .~ MvNormalCholesky.(eachrow(M), Ref(F))
    a ~ Normal(0, 1)
    sigma ~ Exponential(1)
    r = B[g, 1] .+ B[g, 2] .* x .+ m[1]
    mu = a .+ r
    y .~ Normal.(mu, sigma)
end
