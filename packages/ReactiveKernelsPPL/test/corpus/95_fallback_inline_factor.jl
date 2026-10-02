# data: y x g
begin
    a ~ Normal(0, 5)
    b ~ Normal(0, 1)
    sg ~ HalfNormal(1)
    st ~ HalfNormal(1)
    z[levels(g)] .~ Normal.(0, 1)
    c[levels(g)] .~ Normal.(0, st)
    sigma ~ Exponential(1)
    mu = a .+ sg .* z[g] .+ b .* x .+ c[g] .* x
    y .~ Normal.(mu, sigma)
end
