# data: t dose obs
begin
    b0 ~ Normal(0.0, 1.0)
    alpha ~ Exponential(1.0)
    pred ~ plate(t, dose, obs; subjects = kernel_nsub_pred) do ts, d, yy
        mu = exp.(b0 .* d .* ts)
        sc = mu ./ alpha
        yy .~ Gamma.(alpha, sc)
        mu
    end
end
