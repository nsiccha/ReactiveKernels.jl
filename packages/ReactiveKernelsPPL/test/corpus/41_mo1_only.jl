# data: y c
begin
    s ~ Dirichlet(2, 1.0)
    mu = mo1(c, s)
    y .~ Normal.(mu, 1.5)
end
