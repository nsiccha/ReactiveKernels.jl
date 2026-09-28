# data: t dose obs
begin
    b0 ~ Normal(0.0, 1.0)
    kappa ~ Gamma(2.0, 1000.0)
    pred ~ plate(t, dose, obs; subjects = kernel_nsub_pred) do ts, d, yy
        mu = 1 ./ (1 .+ exp.(-b0 .* d .* ts))
        a = mu .* kappa
        b = (1 .- mu) .* kappa
        yy .~ Beta.(a, b)
        mu
    end
end
