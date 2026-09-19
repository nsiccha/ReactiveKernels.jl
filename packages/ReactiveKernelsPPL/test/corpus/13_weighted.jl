# data: y x w
begin
    s ~ Exponential(1)
    mu = a .+ b .* x
    y .~ weighted.(Normal.(mu, s), w)
end
