# data: prop x z
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    c ~ Normal(0, 1)
    d ~ Normal(0, 1)
    mu = a .+ b .* x
    lk = c .+ d .* z
    prop .~ Beta.(logistic.(mu) .* exp.(lk), (1 .- logistic.(mu)) .* exp.(lk))
end
