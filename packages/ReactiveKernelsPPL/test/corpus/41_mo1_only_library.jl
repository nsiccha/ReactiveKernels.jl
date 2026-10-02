# data: y c
begin
    s ~ Dirichlet(2, 1.0)
    m ~ monotonic(c, s)
    y .~ Normal.(m, 1.5)
end
