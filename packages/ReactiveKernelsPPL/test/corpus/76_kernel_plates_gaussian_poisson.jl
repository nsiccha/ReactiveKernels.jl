# data: x1 y1 x2 y2
begin
    b0 ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    pred1 ~ plate(x1, y1; subjects = kernel_nsub_pred1) do xx1, yy1
        mu1 = b0 .* xx1
        yy1 .~ Normal.(mu1, sigma)
        mu1
    end
    pred2 ~ plate(x2, y2; subjects = kernel_nsub_pred2) do xx2, yy2
        mu2 = exp.(b0 .* xx2)
        yy2 .~ Poisson.(mu2)
        mu2
    end
end
