# data: y x
# 30_hsgp_1d through the library: a data-side HSGP basis and the shipped
# `hsgp_effect` submodel (the length scale's validity floor is a statement).
begin
    a ~ Normal(0, 1)
    (PHI, lambda) = hsgp_basis(x; k = 4)
    f ~ hsgp_effect(PHI, lambda)
    mu = a .+ f
    y .~ Normal.(mu, 1.5)
end
