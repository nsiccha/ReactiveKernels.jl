# data: dose dv ls
begin
    sigma ~ Exponential(1.0)
    pred ~ plate(dose, dv, ls; subjects = kernel_nsub_pred) do dd, yy, lsi
        mu = (dd ./ 10.0) .* exp.(lsi)
        yy .~ Normal.(mu, sigma)
        mu
    end
end
