# data: school_id school_idx time dsubj dtime damt yy se
begin
    r ~ varying_effect(school_id, [1])
    mu ~ Normal(0.0, 10.0)
    theta = mu .+ r
    es = linear_pk_schedule(obs = (:school_idx, :time),
        dose = (:dsubj, :dtime, :damt))
    @plate es8 for s in 1:8
        m = theta[school_idx]
        yy .~ Normal.(m, se)
        m
    end
end
