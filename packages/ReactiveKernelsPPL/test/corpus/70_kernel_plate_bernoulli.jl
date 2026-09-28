# data: t dose obs
begin
    b0 ~ Normal(0.0, 1.0)
    pred ~ plate(t, dose, obs; subjects = kernel_nsub_pred) do ts, d, yy
        eta = b0 .* d .* ts
        p = 1 ./ (1 .+ exp.(-eta))
        yy .~ Bernoulli.(p)
        p
    end
end
