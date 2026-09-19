# data: y x
begin
    m ~ Normal(0, 1)
    s ~ Exponential(m)
    t ~ Flat()
    h ~ truncated(Normal(0, 2), 0, Inf)
    h2 ~ HalfNormal(3)
    half_n = length(x) / 2
    s2 = s
    k = 2
    mu = a .+ b .* x
    y .~ Normal.(mu, s2)
end
