# data: y1 y2 x g
begin
    a1 ~ Normal(0, 1)
    a2 ~ Normal(0, 1)
    s ~ Exponential(1)
    mu1 = a1 .+ ranef(:ID, g)
    mu2 = a2 .+ ranef(:ID, g)
    y1 .~ Normal.(mu1, s)
    y2 .~ Normal.(mu2, s)
    ranef_bucket(:ID, g; eta = 2.0) do
        mu1 => [1]
        mu2 => [x]
    end
end
