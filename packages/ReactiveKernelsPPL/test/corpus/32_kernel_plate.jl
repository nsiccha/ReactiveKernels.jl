# data: t dose obs
begin
    sigma ~ Exponential(1.0)
    b0 ~ Normal(0.0, 1.0)
    pred ~ plate(t, dose, obs; subjects = kernel_nsub_pred) do ts, d, yy
        mu = (b0 .* d) .* ts
        yy .~ Normal.(mu, sigma)
        mu
    end
end
