# data: y xd xc
# bruno gp_effectiveness: 2-D anisotropic HSGP on a fixed domain (SB
# `hsgp(...; domain=...)`) with a bounding Uniform length-scale prior
# (SB prior-bound intersection) and a Stan-kernel Normal sd.
begin
    b0 ~ Normal(0, 5)
    mu = b0 .+ hsgp(:h)
    hsgp_basis(:h, xd, xc; k = (8, 8), iso = false,
        domain = ((-1.5, 1.5), (-1.5, 1.5)),
        length_scale = Uniform(0.5163616086861821, 2), sd = Normal(0, 1))
    sigma ~ Exponential(1)
    y .~ Normal.(mu, sigma)
end
