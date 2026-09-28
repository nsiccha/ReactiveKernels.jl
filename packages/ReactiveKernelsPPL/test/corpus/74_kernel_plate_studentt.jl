# data: t dose obs
begin
    b0 ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    nu ~ Gamma(2.0, 0.1)
    pred ~ plate(t, dose, obs; subjects = kernel_nsub_pred) do ts, d, yy
        mu = (b0 .* d) .* ts
        yy .~ StudentT.(nu, mu, sigma)
        mu
    end
end
