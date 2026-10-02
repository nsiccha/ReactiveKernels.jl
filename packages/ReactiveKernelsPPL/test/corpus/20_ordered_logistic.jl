# data: y x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    eta = a .+ b .* x
    y .~ OrderedLogistic.(eta)
end
