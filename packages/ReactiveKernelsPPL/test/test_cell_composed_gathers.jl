using DifferentiationInterface, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

# A group cell gathering its rows of per-row values: a location that adds a
# random intercept to a covariate slope, and a scale whose log is linear in
# the same covariate. Native lowering keeps such a per-row value as its
# arithmetic (no stored vector), and each group's gather `mu[rows[i]]` reads
# that group's rows. Before, every gather computed the whole per-row value,
# so a density cost a full pass per group, and ordinary Enzyme Reverse
# (0.13.216) refused the gradient with `EnzymeRuntimeActivityError` (snag
# `logdensityproble-ed55d946`).

const _CCG_OUTPUTS = @rkppl begin
    b[1:ngroup] .~ Normal.(0, 1)
    a ~ Normal(0, 1)
    s0 ~ Normal(0, 1)
    s1 ~ Normal(0, 1)
    mu = a .+ b[group]
    sigma = exp.(s0 .+ s1 .* x)
    @plate for i = eachindex(rows)
        loc[i] = mu[rows[i]]
        scale[i] = sigma[rows[i]]
    end
    @plate for i = eachindex(y)
        y[i] .~ Normal.(loc[i], scale[i])
    end
end

const _CCG_LOCAL = @rkppl begin
    b[1:ngroup] .~ Normal.(0, 1)
    a ~ Normal(0, 1)
    t ~ Normal(0, 1)
    s0 ~ Normal(0, 1)
    s1 ~ Normal(0, 1)
    mu = a .+ b[group]
    tr = t .* x
    sigma = exp.(s0 .+ s1 .* x)
    @plate for i = eachindex(rows)
        m = mu[rows[i]] .+ tr[rows[i]]
        loc[i] = m
        scale[i] = sigma[rows[i]]
    end
    @plate for i = eachindex(y)
        y[i] .~ Normal.(loc[i], scale[i])
    end
end

# Rows interleaved across groups, as long-format data are.
function _ccg_data(ngroup, n)
    group = [mod1(r, ngroup) for r in 1:n]
    rows = [findall(==(g), group) for g in 1:ngroup]
    x = [cos(2pi * r / 7) for r in 1:n]
    yrow = [0.3 + 0.5 * x[r] + 0.2 * sin(3r) for r in 1:n]
    return (; ngroup, group, rows, x, y = [yrow[r] for r in rows])
end

# Density and gradient by coordinate name, from the model's mathematics.
function _ccg_oracle(q, d)
    value = sum(v -> -log(2pi) / 2 - v^2 / 2, values(q))
    gradient = Dict(name => -v for (name, v) in q)
    t = get(q, :t, 0.0)
    for (g, rows) in enumerate(d.rows), (k, r) in enumerate(rows)
        mu = q[:a] + q[Symbol("b.$(d.group[r])")] + t * d.x[r]
        sigma = exp(q[:s0] + q[:s1] * d.x[r])
        z = (d.y[g][k] - mu) / sigma
        value += -log(2pi) / 2 - log(sigma) - z^2 / 2
        for (name, dmu) in ((:a, 1.0), (Symbol("b.$(d.group[r])"), 1.0), (:t, d.x[r]))
            haskey(gradient, name) && (gradient[name] += dmu * z / sigma)
        end
        gradient[:s0] += z^2 - 1
        gradient[:s1] += d.x[r] * (z^2 - 1)
    end
    return value, gradient
end

function _ccg_prepare(model, d)
    bound = model(; ngroup = d.ngroup, group = d.group, rows = d.rows, x = d.x) |
        (; y = d.y)
    built = build_kernel(bound)
    names = coordinate_names(built.layout)
    u = collect(range(-0.6, 0.5; length = built.layout.total))
    sampler = prepare_sampler(built, bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    gradient = similar(u)
    sampler(u)
    sampler_value_and_gradient!(sampler, gradient, u)
    return (; bound, built, names, sampler, u, gradient)
end

@testset "group cells gather composed per-row values: values and gradients" begin
    d = _ccg_data(6, 90)
    saved = deepcopy(d)
    for model in (_CCG_OUTPUTS, _CCG_LOCAL)
        fx = _ccg_prepare(model, d)
        native = prepare_query(fx.built, fx.bound, :sampler)
        @test occursin("_cell_array_broadcast", string(code_expr(native)))
        for shift in (0.0, 0.35)
            u = fx.u .+ shift
            value, gradient = sampler_value_and_gradient!(fx.sampler, similar(u), u)
            ref_value, ref_gradient = _ccg_oracle(Dict(zip(fx.names, u)), d)
            @test value ≈ ref_value rtol = 1e-12
            @test gradient ≈ [ref_gradient[name] for name in fx.names] rtol = 1e-11
        end
    end
    @test isequal(d, saved)
end

@testset "group cells gather composed per-row values: a density reads each row once" begin
    # The same rows split into 6 or 60 groups. Before, each group computed
    # every row's location and scale: 60 groups allocated about ten times
    # what 6 did.
    for model in (_CCG_OUTPUTS, _CCG_LOCAL)
        bytes = map((6, 60)) do ngroup
            fx = _ccg_prepare(model, _ccg_data(ngroup, 2400))
            fx.sampler(fx.u)
            @allocated fx.sampler(fx.u)
        end
        @test bytes[2] < 2bytes[1]
    end
end
