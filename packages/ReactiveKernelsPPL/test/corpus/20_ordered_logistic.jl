# data: y x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    eta = a .+ b .* x
    y_cutpoints ~ Ordered(Normal(0, 1), length(levels(y)) - 1)
    y .~ OrderedLogistic.(eta, Ref(y_cutpoints))
end
