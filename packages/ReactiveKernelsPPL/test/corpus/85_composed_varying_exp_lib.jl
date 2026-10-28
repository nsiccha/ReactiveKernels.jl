# data: y person item
# IRT 2PL with three varying intercepts: the body of `varying_coefs`
# written out for each, with half-Cauchy sds instead of the shipped
# half-Normal.
begin
    r_t_sd ~ HalfCauchy(2)
    r_t_z[levels(person)] .~ Normal.(0, 1)
    r_t = r_t_sd .* r_t_z
    r_a_sd ~ HalfCauchy(2)
    r_a_z[levels(item)] .~ Normal.(0, 1)
    r_a = r_a_sd .* r_a_z
    r_b_sd ~ HalfCauchy(2)
    r_b_z[levels(item)] .~ Normal.(0, 1)
    r_b = r_b_sd .* r_b_z
    b0 ~ Normal(0, 5)
    th = r_t[person]
    la = r_a[item]
    b = b0 .+ r_b[item]
    eta = exp.(la) .* (th .- b)
    y .~ Bernoulli.(logistic.(eta))
end
