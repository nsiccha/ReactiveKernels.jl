# data: c1 c2 c3 N
begin
    s ~ Dirichlet([2.0, 2.0, 2.0])
    c1 .~ Multinomial.(N, s, c2, c3)
end
