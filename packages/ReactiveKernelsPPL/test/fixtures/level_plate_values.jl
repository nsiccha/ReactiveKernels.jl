# Synthetic level-plate models and an independent Distributions oracle.
# Levels need not be contiguous positions or observation-aligned.
function _plv_program(kind)
    cells = kind === :varying ? quote
        loc = a + m[k]
        c[k] ~ Normal(loc, exp(z[k]))
    end : kind === :latent ? quote
        c[k] ~ Normal(a + z[k], 0.7)
    end : kind === :deterministic ? quote
        t = m[k] + z[k]
        c[k] = a + 2t
    end : kind === :constant ? quote
        c[k] = 2.0
    end : error("unknown level-plate fixture $kind")
    return quote
        a ~ Normal(0, 2)
        z[levels(g)] .~ Normal.(0, 1)
        @plate for k in levels(g)
            $(cells.args...)
        end
        mu = a .+ c[g]
        y .~ Normal.(mu, 0.8)
    end
end

function _plv_columns(n, S; labels = 2 .* collect(1:S))
    return Dict{Symbol,Any}(:g => [labels[mod1(i + 1, S)] for i in 1:n],
        :m => [0.4sin(0.7i) for i in 1:2S],
        :y => [0.3cos(0.4i) for i in 1:n])
end

function _plv_oracle(built, cols, kind, u)
    nt = constrain(built.layout, u)
    lv = sort(unique(cols[:g]))
    c = kind in (:varying, :latent) ? nt.c :
        kind === :constant ? fill(2.0, length(lv)) :
        [nt.a + 2(cols[:m][k] + nt.z[j]) for (j, k) in enumerate(lv)]
    prior = logpdf(Normal(0, 2), nt.a) + sum(logpdf.(Normal(), nt.z))
    if kind === :varying
        prior += sum(logpdf(Normal(nt.a + cols[:m][k], exp(nt.z[j])), c[j])
            for (j, k) in enumerate(lv))
    elseif kind === :latent
        prior += sum(logpdf.(Normal.(nt.a .+ nt.z, 0.7), c))
    end
    mu = [nt.a + c[findfirst(==(k), lv)] for k in cols[:g]]
    value = prior + sum(logpdf.(Normal.(mu, 0.8), cols[:y])) +
        logjac(built.layout, u)
    return value, c
end

function _plv_build(kind, n, S; kwargs...)
    cols = _plv_columns(n, S; kwargs...)
    kind in (:latent, :constant) && delete!(cols, :m)
    bound = bind_data(lower_rkppl(_plv_program(kind), keys(cols)), cols)
    built = build_kernel(bound)
    u = [0.27sin(0.9i + 0.2) for i in 1:built.layout.total]
    return built, bound, cols, u
end

function _plv_lazy_build()
    prog = quote
        a ~ Normal(0, 2)
        @plate for k in levels(g)
            c[k] = k == 2 ? a : m[100]
        end
        mu = c[g]
        y .~ Normal.(mu, 0.8)
    end
    cols = Dict(:g => [2, 2, 2], :m => [0.1], :y => [0.2, -0.1, 0.7])
    bound = bind_data(lower_rkppl(prog, (:y, :g, :m)), cols)
    built = build_kernel(bound)
    u = [0.3]
    oracle(w) = logpdf(Normal(0, 2), w[1]) +
        sum(logpdf.(Normal(w[1], 0.8), cols[:y]))
    return built, bound, u, oracle
end
