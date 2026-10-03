using Statistics

# Scalar margins use ordinary multiplication of explicit coefficients
# (user decisions 1cmodra / 10ldrvz; follow-up 07svprw).
function _sm_build(kind, n, G; labels = false, offset = 1.2)
    body = kind === :library ? quote
        r ~ varying_coefs(g)
    end : quote
        r_sd ~ HalfNormal(1)
        r_z[levels(g)] .~ Normal.(0, 1)
        r = r_sd .* r_z
    end
    head = quote
        a ~ Normal(0, 5)
        m = mean(x)
    end
    tail = quote
        mu = a .+ r[g] .* m
        y .~ Normal.(mu, m)
    end
    ast = Expr(:block, head.args..., body.args..., tail.args...)
    levels = labels ? ["z", "a", "m", "b", "q"][1:G] : collect(1:G)
    data = Dict{Symbol,Any}(
        :x => [offset + 0.08 * sin(i) for i in 1:n],
        :y => [0.2 * cos(i) for i in 1:n],
        :g => [levels[mod1(i, G)] for i in 1:n])
    bound = bind_data(lower_rkppl(ast, data; conditioned = (:y,)), data)
    return bound, build_kernel(bound), data
end

# Independently transform packed coordinates, construct each group effect,
# and evaluate normalized Distributions densities. No generated recipe or
# PPL transform contributes to this reference.
function _sm_oracle(built, data, kind, u)
    values = Dict(zip(coordinate_names(built.layout), u))
    logsd = values[kind === :library ? Symbol("r.sd") : :r_sd]
    sd = exp(logsd)
    levels = sort!(unique(data[:g]))
    z = [values[Symbol(kind === :library ? "r.z." : "r_z.", j)]
        for j in eachindex(levels)]
    a = values[:a]
    m = sum(data[:x]) / length(data[:x])
    prior = logpdf(Normal(0, 5), a) +
        logpdf(truncated(Normal(), 0, Inf), sd) + sum(logpdf.(Normal(), z))
    likelihood = sum(eachindex(data[:y])) do i
        j = findfirst(==(data[:g][i]), levels)
        logpdf(Normal(a + m * sd * z[j], m), data[:y][i])
    end
    return (; prior, likelihood, jacobian = logsd,
        posterior = prior + likelihood + logsd, m)
end

@testset "Varying scalar margin: explicit library and written body" begin
    for kind in (:library, :body), (n, G, labels) in ((7, 3, false),
            (19, 5, false), (7, 3, true))
        bound, built, data = _sm_build(kind, n, G; labels)
        @test built.layout.total == G + 2
        @test isempty(bound.varying_draws)
        @test bound.columns[:m] isa Number
        @test bound.columns[:m] ≈ sum(data[:x]) / n
        for scale in (0.2, 0.6)
            u = [scale * sin(i) for i in 1:built.layout.total]
            ref = _sm_oracle(built, data, kind, u)
            @test _query(built.spec, bound, :m, u) ≈ ref.m
            @test _query(built.spec, bound, :prior, u) ≈ ref.prior atol = 1e-10
            @test _query(built.spec, bound, :likelihood, u) ≈ ref.likelihood atol = 1e-10
            @test _query(built.spec, bound, :log_jacobian, u) ≈ ref.jacobian atol = 1e-12
            @test _query(built.spec, bound, :posterior, u) ≈ ref.posterior atol = 1e-10
            @test unconstrain(built.layout, constrain(built.layout, u)) ≈ u
            oracle(w) = _sm_oracle(built, data, kind, w).posterior
            gradient = _check_gradient(built.spec, bound, u)
            @test gradient ≈ _findiff_grad(oracle, u) rtol = 1e-5 atol = 1e-7
        end
    end
end

@testset "Varying scalar margin: invalid reduced scale" begin
    for kind in (:library, :body), offset in (0.0, -1.2, Inf, NaN)
        # With a nonpositive or nonfinite mean, this same authored response
        # cannot define a positive-scale Gaussian density.
        data = Dict{Symbol,Any}(:x => fill(offset, 7),
            :y => [0.2 * cos(i) for i in 1:7], :g => [mod1(i, 3) for i in 1:7])
        ast = kind === :library ? quote
            a ~ Normal(0, 5)
            m = mean(x)
            r ~ varying_coefs(g)
            mu = a .+ r[g] .* m
            y .~ Normal.(mu, m)
        end : quote
            a ~ Normal(0, 5)
            m = mean(x)
            r_sd ~ HalfNormal(1)
            r_z[levels(g)] .~ Normal.(0, 1)
            r = r_sd .* r_z
            mu = a .+ r[g] .* m
            y .~ Normal.(mu, m)
        end
        plan = lower_rkppl(ast, data; conditioned = (:y,))
        # refused: scale must be strictly positive (Gaussian density domain).
        err = try
            bind_data(plan, data)
            nothing
        catch e
            e
        end
        @test err isa ContractValidationError
        @test occursin("scale m must be finite positive numerics", sprint(showerror, err))
        positive = merge(data, Dict{Symbol,Any}(:x => fill(1.2, 7)))
        bound = bind_data(plan, positive)
        built = build_kernel(bound)
        u = [0.2 * sin(i) for i in 1:built.layout.total]
        ref = _sm_oracle(built, positive, kind, u)
        @test bound.columns[:m] ≈ 1.2
        @test _query(built.spec, bound, :posterior, u) ≈ ref.posterior
    end
end
