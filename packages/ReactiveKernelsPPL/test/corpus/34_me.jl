# data: y x_obs
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    sigma ~ Exponential(1)
    mu = a .+ b .* x_true
    @plate for i in eachindex(x_obs)
        x_true[i] ~ Normal(0.5, 1.5)
    end
    y .~ Normal.(mu, sigma)
    x_obs .~ Normal.(x_true, 0.5)
end
