# Synthetic level-plate models and an independent Distributions oracle.
# Levels need not be contiguous positions or observation-aligned.
using CategoricalArrays
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
    bound = bind_data(lower_rkppl(_plv_program(kind), keys(cols);
        conditioned = (:y,)), cols)
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
    bound = bind_data(lower_rkppl(prog, (:y, :g, :m); conditioned = (:y,)), cols)
    built = build_kernel(bound)
    u = [0.3]
    oracle(w) = logpdf(Normal(0, 2), w[1]) +
        sum(logpdf.(Normal(w[1], 0.8), cols[:y]))
    return built, bound, u, oracle
end

# Declared coordinates and the loop's lanes differ: a selected axis omits
# its first label, while another full axis has an extra label before g's.
function _plv_axis_build(kind, n, S; labels = 2 .* collect(1:S), pooled = false)
    other = kind in (:other, :other_selected, :matrix_other)
    matrix = kind in (:matrix_selected, :matrix_other)
    axis = other ? :h : :g
    dim = kind === :reordered ? :(levels($axis)[[3, 1]]) :
        kind in (:selected, :matrix_selected, :varying, :lazy, :live,
            :live_oob, :other_selected) ?
        :(levels($axis)[2:end]) : :(levels($axis))
    declaration = matrix ? :(z[$dim, 1:2] .~ Normal.(0, 1)) :
        :(z[$dim] .~ Normal.(0, 1))
    cell = kind === :varying ? :(c[k] ~ Normal(a + z[k], exp(z[k]))) :
        kind === :lazy ? :(c[k] = k > 0 ? a + 2z[k] : m[100]) :
        kind === :live ? :(c[k] = a > 0 ? a + 2z[k] : sqrt(-a)) :
        kind === :live_oob ? :(c[k] = a > 0 ? a + 2z[k] : m[100]) :
        kind === :matrix_selected ? :(c[k] = a + dot(z[k, :], [1.0, 0.5])) :
        matrix ? :(c[k] = a + z[k, 1] + 0.5z[k, 2]) :
        :(c[k] = a + 2z[k])
    prog = quote
        a ~ Normal(0, 2)
        $declaration
        @plate for k in levels(g)
            $cell
        end
        y .~ Normal.(c[g], 0.8)
    end
    active = pooled ? min(3, S) : S
    cols = Dict{Symbol,Any}(:g => [labels[mod1(i + 1, active)] for i in 1:n],
        :y => [0.3cos(0.4i) for i in 1:n])
    if other
        hlv = vcat(labels[1] isa String ? "0" : 0, reverse(labels))
        cols[:h] = [hlv[mod1(i, length(hlv))] for i in 1:n]
    end
    if pooled
        cols[:g] = categorical(cols[:g]; levels = labels)
        other && (cols[:h] = categorical(cols[:h]; levels = hlv))
    end
    kind in (:lazy, :live_oob) && (cols[:m] = [0.1])
    bound = bind_data(lower_rkppl(prog, keys(cols); conditioned = (:y,)), cols)
    built = build_kernel(bound)
    u = [0.27sin(0.9i + 0.2) for i in 1:built.layout.total]
    function oracle(w)
        nt = constrain(built.layout, w)
        glv = pooled ? labels : sort(unique(cols[:g]))
        zlv = pooled ? (other ? hlv : labels) : sort(unique(cols[axis]))
        dim.head === :ref && (zlv = kind === :reordered ? zlv[[3, 1]] : zlv[2:end])
        zv(k) = (j = findfirst(==(k), zlv);
            j === nothing ? 0.0 : matrix ? nt.z[j, 1] + 0.5nt.z[j, 2] : nt.z[j])
        c = kind === :varying ? nt.c : [nt.a + (matrix ? 1 : 2) * zv(k) for k in glv]
        prior = logpdf(Normal(0, 2), nt.a) + sum(logpdf.(Normal(), nt.z))
        if kind === :varying
            prior += sum(logpdf(Normal(nt.a + zv(k), exp(zv(k))), c[j])
                for (j, k) in enumerate(glv))
        end
        mu = [c[findfirst(==(k), glv)] for k in cols[:g]]
        return prior + sum(logpdf.(Normal.(mu, 0.8), cols[:y])) +
            logjac(built.layout, w), c
    end
    return built, bound, cols, u, oracle
end

function _plv_stack_build(n, S)
    labels = 2 .* collect(1:S)
    prog = quote
        a ~ Normal(0, 2)
        @plate for j in levels(h)
            L[j] ~ LKJCholesky(2, 1.0)
        end
        @plate for k in levels(g)
            c[k] = a + L[k][2, 1]
        end
        y .~ Normal.(c[g], 0.8)
    end
    hlv = vcat(0, reverse(labels))
    cols = Dict(:g => [labels[mod1(i + 1, S)] for i in 1:n],
        :h => [hlv[mod1(i, length(hlv))] for i in 1:n],
        :y => [0.3cos(0.4i) for i in 1:n])
    bound = bind_data(lower_rkppl(prog, keys(cols); conditioned = (:y,)), cols)
    built = build_kernel(bound)
    u = [0.27sin(0.9i + 0.2) for i in 1:built.layout.total]
    function oracle(w)
        nt = constrain(built.layout, w)
        glv, hlv = sort(unique(cols[:g])), sort(unique(cols[:h]))
        c = [nt.a + nt.L[2, 1, findfirst(==(k), hlv)] for k in glv]
        prior = logpdf(Normal(0, 2), nt.a) +
            sum(logpdf(LKJCholesky(2, 1.0),
                Cholesky(LowerTriangular(nt.L[:, :, j]))) for j in eachindex(hlv))
        mu = [c[findfirst(==(k), glv)] for k in cols[:g]]
        return prior + sum(logpdf.(Normal.(mu, 0.8), cols[:y])) +
            logjac(built.layout, w), c
    end
    return built, bound, cols, u, oracle
end
