# data: y student question
# LSAT (Rasch with a shared discrimination): a scalar-by-factor product
# minus a factor sub composes (an operand that composes on its own
# makes the whole sum composed).
begin
    c_th[levels(student)] .~ Normal.(0, 1)
    c_al[levels(question)] .~ Normal.(0, 100)
    th = c_th[student]
    al = c_al[question]
    be ~ Normal(0, 100)
    eta = be .* th .- al
    y .~ Bernoulli.(logistic.(eta))
end
