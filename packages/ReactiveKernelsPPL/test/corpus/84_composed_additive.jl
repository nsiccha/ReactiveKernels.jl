# data: y xs
begin
    th = a_th .+ b_th .* xs
    al = a_al .+ b_al .* xs
    eta = th .+ al
    y .~ Bernoulli.(logistic.(eta))
end
