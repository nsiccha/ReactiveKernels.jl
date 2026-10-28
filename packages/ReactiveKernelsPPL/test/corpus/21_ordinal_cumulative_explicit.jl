# data: y x
begin
    b ~ Normal(0, 2)
    c ~ Ordered(Normal(0.5, 2), 3)
    eta = b .* x
    y .~ Ordinal.(Cumulative(), CloglogLink(), eta, Ref(c))
end
