# data: y x lo hi
begin
    mu = a .+ b .* x
    y .~ censored.(Normal.(mu, s), lo, hi)
    s ~ Exponential(1)
end
