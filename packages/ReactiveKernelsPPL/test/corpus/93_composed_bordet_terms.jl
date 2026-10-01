# data: log_obs log_time log_dose series lloq uloq
# Bordet `brm_hierarchical_terms`: per-series correlated parametric
# curves (transient bump x saturating dose response) composed over data
# leaves with `logistic.` maps (composed predictors v3).
begin
    b0 ~ Normal(0, 1)
    sigma ~ Exponential(1)
    r_base ~ varying_effect(series, [1])
    base = b0 .+ r_base
    dt ~ varying_draws(series, [1, 1, 1])
    t1 ~ varying_slice(dt, 1)
    t2 ~ varying_slice(dt, 2)
    t3 ~ varying_slice(dt, 3)
    ds ~ varying_draws(series, [1, 1])
    s1 ~ varying_slice(ds, 1)
    s2 ~ varying_slice(ds, 2)
    tl = t1
    tls = t2
    tm = t3
    dl = s1
    dls = s2
    xi = (log_time .- tl) .* exp.(tls)
    bump = logistic.(xi) .* logistic.(.-xi) .* tm
    resp = logistic.((log_dose .- dl) .* exp.(dls))
    mu = base .+ bump .* resp
    log_obs .~ censored.(Normal.(mu, sigma), lloq, uloq)
end
