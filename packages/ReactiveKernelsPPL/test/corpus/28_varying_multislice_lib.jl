# data: y1 y2 x g
# The body of `d ~ varying_coefs_correlated(g, 2)` written out, with the
# LKJ shape 2.0 instead of the shipped 1.0.
begin
    a1 ~ Normal(0, 1)
    a2 ~ Normal(0, 1)
    s ~ Exponential(1)
    d_sd[1:2] .~ HalfNormal.(1)
    d_L ~ LKJCholesky(2, 2.0)
    d_z[levels(g), 1:2] .~ Normal.(0, 1)
    d = d_z * (d_sd .* d_L)'
    mu1 = a1 .+ d[g, 1]
    mu2 = a2 .+ x .* d[g, 2]
    y1 .~ Normal.(mu1, s)
    y2 .~ Normal.(mu2, s)
end
