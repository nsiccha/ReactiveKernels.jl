# data: y x hi
begin
    mu = a .+ b .* x
    y .~ interval_censored.(Normal.(mu, s), hi)
    s ~ Exponential(1)
end
