# data: y x
# 24_spline_s through the library: a data-side thin-plate basis and the
# shipped `penalized_smooth` submodel (every prior a statement in its body).
begin
    (Xf, Zp) = tps_basis(x; k = 4)
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    f ~ penalized_smooth(Xf, Zp)
    mu = a .+ f
    y .~ Normal.(mu, sigma)
end
