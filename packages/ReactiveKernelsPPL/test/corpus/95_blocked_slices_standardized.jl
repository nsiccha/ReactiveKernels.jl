# data: y1 y2 y3 x1 x2 b g grp
# Data-only standardized covariates, reference-coded level priors
# (`levels(g)[2:end]`), a standalone upper-only truncated latent, and two
# independent correlated-draw blocks on one grouping, each sliced into
# sub-predictors.
begin
    z1 = (x1 .- mean(x1)) ./ std(x1)
    z2 = (x2 .- mean(x2)) ./ std(x2)
    a1 ~ Normal(0.0, 1.0)
    b1_z1 ~ Normal(0.0, 0.5)
    b1_b ~ Normal(0.0, 0.5)
    c1[levels(g)[2:end]] .~ Normal.(0.0, 1.0)
    a2 ~ Normal(1.0, 2.0)
    b2_z2 ~ Normal(0.0, 0.5)
    c2[levels(g)[2:end]] .~ Normal.(0.0, 1.0)
    a3 ~ Normal(0.0, 1.0)
    t ~ truncated(Normal(-1.0, 1.0), -Inf, -0.5)
    s1 ~ Exponential(1.0)
    s2 ~ LogNormal(0.0, 0.5)
    d12 ~ varying_draws(grp, [1, 1]; eta = 2.0)
    r1 ~ varying_slice(d12, 1)
    r2 ~ varying_slice(d12, 2)
    d3 ~ varying_draws(grp, [1]; eta = 1.0)
    r3 ~ varying_slice(d3, 1)
    mu1 = a1 .+ b1_z1 .* z1 .+ b1_b .* b .+ c1[g] .+ r1
    mu2 = a2 .+ b2_z2 .* z2 .+ c2[g] .+ r2
    mu3 = a3 .+ r3
    y1 .~ Normal.(mu1, s1)
    y2 .~ Normal.(mu2, s2)
    y3 .~ Normal.(mu3, 1.5)
end
