# data: y x1 x2
begin
    b1 ~ Horseshoe()
    b2 ~ Horseshoe(local_scale = 0.5, global_scale = 0.25)
    mu = a .+ b1 .* x1 .+ b2 .* x2
    sigma ~ Exponential(1.0)
    y .~ Normal.(mu, sigma)
end
