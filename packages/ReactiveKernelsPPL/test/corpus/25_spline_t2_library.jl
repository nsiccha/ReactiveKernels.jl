# data: y x z
# 25_spline_t2 through the library: a data-side t2 basis and the shipped
# `t2_smooth` submodel.
begin
    (Xt, Zrr, Zrn, Znr) = t2_basis(x, z)
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    f ~ t2_smooth(Xt, Zrr, Zrn, Znr)
    mu = a .+ f
    y .~ Normal.(mu, sigma)
end
