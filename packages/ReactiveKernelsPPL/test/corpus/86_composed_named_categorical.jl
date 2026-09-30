# data: y person item w1
# GPCM-shaped IRT: a name bound to a composition (`d`) inlines into two
# CategoricalLogit etas; latent-regression ability sub-predictor.
begin
    t0 ~ StudentT(3, 0, 1)
    bw1 ~ Normal(0, 1)
    r_t ~ varying_effect(person, [1]; eta = 1.0, sd = Exponential(1))
    c_la[levels(item)] .~ Normal.(1, 1)
    c_s1[levels(item)] .~ Normal.(0, 3)
    c_s2[levels(item)] .~ Normal.(0, 3)
    th = t0 .+ bw1 .* w1 .+ r_t
    la = c_la[item]
    s1 = c_s1[item]
    s2 = c_s2[item]
    d = exp.(la) .* th
    eta1 = d .- s1
    eta2 = d .+ d .- s1 .- s2
    y .~ CategoricalLogit.(eta1, eta2)
end
