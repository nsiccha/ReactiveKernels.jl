# data: y person item
# hier_2pl: the body of `dx ~ varying_coefs_correlated(item, 2)` written
# out, with Exponential(10) sds and LKJ shape 4.0 instead of the shipped
# defaults; both margins are intercepts.
begin
    c_th[levels(person)] .~ Normal.(0, 1)
    x1 ~ Normal(0, 1)
    x2 ~ Normal(0, 5)
    dx_sd[1:2] .~ Exponential.(10)
    dx_L ~ LKJCholesky(2, 4.0)
    dx_z[levels(item), 1:2] .~ Normal.(0, 1)
    dx = dx_z * (dx_sd .* dx_L)'
    th = c_th[person]
    xi1 = x1 .+ dx[item, 1]
    xi2 = x2 .+ dx[item, 2]
    eta = exp.(xi1) .* (th .- xi2)
    y .~ Bernoulli.(logistic.(eta))
end
