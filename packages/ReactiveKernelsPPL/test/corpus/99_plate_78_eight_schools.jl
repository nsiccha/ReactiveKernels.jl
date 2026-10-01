# data: school_idx yy se
begin
    r ~ varying_effect(school_idx, [1])
    mu ~ Normal(0.0, 10.0)
    theta = mu .+ r
    @plate for i in eachindex(yy)
        yy[i] ~ Normal(theta[i], se[i])
    end
end
