# data: earn x
begin
    b1 ~ Flat()
    b2 ~ Flat()
    s ~ Exponential(1)
    ly = log.(earn)
    mu = b1 .+ b2 .* x
    ly .~ Normal.(mu, s)
end
