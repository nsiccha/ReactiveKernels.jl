# data: y x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    lam ~ LogNormal(-0.3, 1.0)
    eta = a .+ b .* x
    y .~ InverseGaussian.(exp.(eta), lam)
end
