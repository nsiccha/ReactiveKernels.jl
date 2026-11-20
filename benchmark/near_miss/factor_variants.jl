# Matched models for checking the contribution of a factor to a predictor.
# The bound-matrix and hcat cases use exactly the same two columns.
function factor_case(law, form, x, z, groups, response)
    axis = form === :reference ? :(levels(g)[2:end]) : :(levels(g))
    statements = Any[:(m ~ Normal(0, 5)), :(t ~ Exponential(1.0)),
        :(alpha[$axis] .~ Normal.(m, t))]
    g = form === :missing ? replace(groups, 1 => missing) : groups
    inputs = (; g, x)
    obs = (; y = response)
    if form in (:bound_matrix, :hcat)
        if form === :hcat
            push!(statements, :(X = hcat(x, z)))
            inputs = (; g, x, z)
        else
            inputs = (; g, X = hcat(x, z))
        end
        push!(statements, :(beta[axes(X, 2)] .~ Normal.(0, 2)))
        eta = :(alpha[g] .+ X * beta)
    else
        push!(statements, :(b ~ Normal(0, 2)))
        eta = :(alpha[g] .+ b .* x)
    end
    if form === :scan
        push!(statements, :(@scan begin
            s[1] ~ Normal(0, 1)
            for j in 2:T
                s[j] ~ Normal(s[j - 1], 1)
            end
        end))
        inputs = (; inputs..., T = 3)
        obs = (; obs..., s = [0.5, -0.25, 0.1])
    end
    if law === :normal
        push!(statements, :(sigma ~ Exponential(1.0)), :(y .~ Normal.($eta, sigma)))
    elseif law === :bernoulli
        push!(statements, :(y .~ BernoulliLogit.($eta)))
    elseif law === :poisson
        push!(statements, :(y .~ Poisson.(exp.($eta))))
    else
        error("unknown factor probe law $law")
    end
    return ReactiveKernelsPPL.RKPPLModel(Expr(:block, statements...), @__MODULE__), inputs, obs
end
