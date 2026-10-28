# data: y xs
begin
    a_th ~ Normal(0, 1)
    b_th ~ Normal(0, 1)
    a_al ~ Normal(0, 1)
    b_al ~ Normal(0, 1)
    th = a_th .+ b_th .* xs
    al = a_al .+ b_al .* xs
    be ~ Normal(0.0, 100.0)
    eta = be .* (th .- al)
    y .~ Bernoulli.(logistic.(eta))
end
