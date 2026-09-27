@rkppl begin
    a ~ Normal(0.0, 1.0)
    b ~ Normal(0.0, 1.0)
    eta = a .+ b .* x
    y .~ Poisson.(exp.(eta))
end
