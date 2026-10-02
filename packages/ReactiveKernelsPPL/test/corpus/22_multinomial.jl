# data: c1 c2 c3 N
begin
    s ~ Dirichlet([2.0, 2.0, 2.0])
    eachrow(hcat(c1, c2, c3)) .~ Multinomial.(N, Ref(s))
end
