# data: y xs
begin
    th = a_th .+ b_th .* xs
    al = a_al .+ b_al .* xs
    be ~ Normal(0.0, 100.0)
    eta = be .* (th .- al)
    y .~ Bernoulli.(logistic.(eta))
end
