# data: y x z
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    c ~ Normal(0, 1)
    d ~ Normal(0, 1)
    eta = a .+ b .* x
    ls = c .+ d .* z
    y .~ InverseGaussian.(exp.(eta), exp.(ls))
end
