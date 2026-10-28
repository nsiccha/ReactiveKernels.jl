# data: y x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    s ~ Exponential(1)
    mu = a .+ b .* x
    @plate for i in eachindex(y)
        y[i] ~ Normal.(mu[i], s)
    end
end
