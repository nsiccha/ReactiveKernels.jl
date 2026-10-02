# data: y x
# 68_hsgp_periodic through the library: a data-side periodic basis and the
# shipped `hsgp_periodic_effect` submodel.
begin
    a ~ Normal(0, 1)
    (PHI, harmonics) = hsgp_periodic_basis(x; k = 4, period = 2.0)
    f ~ hsgp_periodic_effect(PHI, harmonics)
    mu = a .+ f
    y .~ Normal.(mu, 1.5)
end
