# data: y z
begin
    mu = z
    s ~ Exponential(1.0)
    y .~ Normal.(mu, s)
end
