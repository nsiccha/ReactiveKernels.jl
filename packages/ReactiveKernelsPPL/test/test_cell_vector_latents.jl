using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels,
    ReactiveKernelsPPL, Test

# A dotted cell latent, `v[i] .~ D.(args…)` in a `@plate` cell, states one
# iteration of its Julia loop: `v[i]` holds the broadcast of `D` over the
# arguments' values at index `i`. That is one value per element of a
# per-index array whose length may vary by index, or a number when the
# arguments are numbers there. Its coordinates are one array parameter in
# plate order, and draws show one array per index under the authored name.

_cvl_fd(f, u; h = cbrt(eps(Float64))) = [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end for i in eachindex(u)]

_cvl_q(built, bound, preset, u) = Base.invokelatest(prepare_query(built, bound, preset), u)

_cvl_structure(program) = [(e.kind, e.depth) for e in recipe_inventory(program)
    if e.kind !== :ordinary]

# One latent log total per active district in each sector; the observed
# totals and a few observed districts read them.
const _CVL_TOTALS = quote
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(total)
        loc = log.(mu[i])
        log_T[i] .~ Normal.(loc, 0.5)
        T = exp.(log_T[i])
        y[i] .~ Normal.(T[obs[i]], sigma)
        total[i] ~ Normal(sum(T), 0.01 * total[i])
    end
end

const _CVL_DATA = (; mu = [[10.0, 20.0, 5.0], [7.0, 3.0], [4.0]], obs = [[1, 2], [1], [1]],
    y = [[11.0, 19.0], [6.5], [4.2]], total = [36.0, 10.5, 4.1])

# `u` packs `sigma` on the log scale, then each sector's log totals.
function _cvl_totals_oracle(u, d)
    sigma = exp(u[1])
    lp = logpdf(Exponential(1.0), sigma) + u[1]
    k = 1
    for i in eachindex(d.total)
        lt = u[(k + 1):(k + length(d.mu[i]))]
        k += length(d.mu[i])
        lp += sum(logpdf.(Normal.(log.(d.mu[i]), 0.5), lt))
        T = exp.(lt)
        lp += sum(logpdf.(Normal.(T[d.obs[i]], sigma), d.y[i]))
        lp += logpdf(Normal(sum(T), 0.01 * d.total[i]), d.total[i])
    end
    return lp
end

@testset "a dotted cell latent holds one value per element of its arguments" begin
    d = _CVL_DATA
    saved = deepcopy(d)
    data = (; mu = d.mu, obs = d.obs, y = d.y, total = d.total)
    from_values = bind_data(lower_rkppl(_CVL_TOTALS, data; conditioned = (:y, :total)), data)
    from_names = bind_data(lower_rkppl(_CVL_TOTALS, keys(data); conditioned = (:y, :total)), data)
    for bound in (from_values, from_names)
        built = build_kernel(bound)
        @test coordinate_names(built.layout) == [:sigma, Symbol("log_T.1.1"),
            Symbol("log_T.1.2"), Symbol("log_T.1.3"), Symbol("log_T.2.1"),
            Symbol("log_T.2.2"), Symbol("log_T.3.1")]
        u = [0.1, 2.3, 2.9, 1.7, 1.9, 1.2, 1.4]
        nt = constrain(built.layout, u)
        @test nt.log_T == [[2.3, 2.9, 1.7], [1.9, 1.2], [1.4]]
        @test unconstrain(built.layout, nt) ≈ u
        oracle = v -> _cvl_totals_oracle(v, d)
        sampler = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
        for v in (u, [-0.4, 2.0, 3.1, 1.5, 2.1, 1.0, 1.3], [0.6, 2.5, 2.8, 1.9, 1.8, 1.3, 1.5])
            value, gradient = sampler_value_and_gradient!(sampler, similar(v), v)
            @test value ≈ oracle(v) rtol = 1e-12
            @test gradient ≈ _cvl_fd(oracle, v) rtol = 1e-6 atol = 1e-8
            # The latent's density is a prior: the likelihood holds the observations.
            lt = constrain(built.layout, v).log_T
            @test _cvl_q(built, bound, :prior, v) ≈ logpdf(Exponential(1.0), exp(v[1])) +
                sum(sum(logpdf.(Normal.(log.(d.mu[i]), 0.5), lt[i])) for i in 1:3) rtol = 1e-12
        end
        draws = restore_draws(built.layout, hcat(u, 2 .* u))
        @test draws.log_T == [[2.3 4.6; 2.9 5.8; 1.7 3.4], [1.9 3.8; 1.2 2.4], [1.4 2.8]]
        @test size.(restore_draws(built.layout, zeros(7, 0)).log_T) == [(3, 0), (2, 0), (1, 0)]
    end
    @test d == saved
end

@testset "cell latents: sampled arguments, constrained families and bounds" begin
    mu = [[10.0, 20.0, 5.0], [7.0, 3.0]]
    lognormal = quote
        b ~ Normal(0, 1)
        s ~ Exponential(1)
        @plate for i in eachindex(mu)
            m = log.(mu[i]) .+ b
            w[i] .~ LogNormal.(m, s)
        end
    end
    function lognormal_oracle(v)
        b, s = v[1], exp(v[2])
        locs = vcat(log.(mu[1]) .+ b, log.(mu[2]) .+ b)
        return logpdf(Normal(), b) + logpdf(Exponential(1), s) + v[2] +
            sum(logpdf.(LogNormal.(locs, s), exp.(v[3:7]))) + sum(v[3:7])
    end
    truncated_form = quote
        c ~ Normal(0, 1)
        @plate for i in eachindex(mu)
            h[i] .~ truncated.(Normal.(log.(mu[i]) .* c, 1.0), 0.0, Inf)
        end
    end
    function truncated_oracle(v)
        locs = vcat(log.(mu[1]), log.(mu[2])) .* v[1]
        return logpdf(Normal(), v[1]) +
            sum(logpdf.(truncated.(Normal.(locs, 1.0), 0.0, Inf), exp.(v[2:6]))) + sum(v[2:6])
    end
    # Positive families and `restricted` keep the original density, with the
    # transform's Jacobian.
    positive = quote
        c ~ Normal(0, 1)
        @plate for i in eachindex(mu)
            e[i] .~ Exponential.(mu[i] .* exp(c))
            r[i] .~ restricted.(Normal.(log.(mu[i]), 1.0), 0.0, Inf)
        end
    end
    function positive_oracle(v)
        flat = vcat(mu[1], mu[2])
        e, r = exp.(v[2:6]), exp.(v[7:11])
        return logpdf(Normal(), v[1]) + sum(logpdf.(Exponential.(flat .* exp(v[1])), e)) +
            sum(logpdf.(Normal.(log.(flat), 1.0), r)) + sum(v[2:11])
    end
    for (ast, oracle, u) in ((lognormal, lognormal_oracle, [0.2, -0.3, 0.1, 0.5, -0.2, 0.4, 0.0]),
            (truncated_form, truncated_oracle, [0.3, 0.1, -0.5, 0.2, 0.4, -0.1]),
            (positive, positive_oracle, [0.1, 2.0, 2.5, 1.2, 1.8, 0.9, 2.2, 2.9, 1.6, 1.9, 1.1]))
        bound = bind_data(lower_rkppl(ast, (; mu)), (; mu))
        built = build_kernel(bound)
        @test length(coordinate_names(built.layout)) == length(u)
        sampler = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
        for v in (u, u .- 0.3)
            value, gradient = sampler_value_and_gradient!(sampler, similar(v), v)
            @test value ≈ oracle(v) rtol = 1e-12
            @test gradient ≈ _cvl_fd(oracle, v) rtol = 1e-6 atol = 1e-8
        end
    end
end

@testset "cell latents: numbers, shared arrays, matrices and empty cells" begin
    # Numbers at every index: one scalar per index, as `th[i] ~ Normal(m[i], 1)`.
    m = [0.5, -1.0, 2.0]
    dotted = quote
        @plate for i in eachindex(m)
            th[i] .~ Normal.(m[i], 1)
        end
    end
    undotted = quote
        @plate for i in eachindex(m)
            th[i] ~ Normal(m[i], 1)
        end
    end
    u = [0.1, 0.2, 0.3]
    for ast in (dotted, undotted), source in ((; m), (:m,))
        bound = bind_data(lower_rkppl(ast, source), (; m))
        built = build_kernel(bound)
        @test coordinate_names(built.layout) == [Symbol("th.1"), Symbol("th.2"), Symbol("th.3")]
        @test constrain(built.layout, u).th == u
        @test _cvl_q(built, bound, :prior, u) ≈ sum(logpdf.(Normal.(m, 1), u)) rtol = 1e-12
    end
    # A shared vector gives every index its entries.
    shared = quote
        @plate for i in eachindex(ys)
            z[i] .~ Normal.(w, 1)
            ys[i] ~ Normal(sum(z[i]), 1)
        end
    end
    data = (; w = [0.1, 0.2], ys = [1.0, 2.0, 3.0])
    bound = bind_data(lower_rkppl(shared, data; conditioned = (:ys,)), data)
    built = build_kernel(bound)
    @test coordinate_names(built.layout) == [Symbol("z.$i.$j") for i in 1:3 for j in 1:2]
    v = collect(0.1:0.1:0.6)
    @test _cvl_q(built, bound, :prior, v) ≈ sum(logpdf.(Normal.(repeat(data.w, 3), 1), v)) rtol = 1e-12
    @test _cvl_q(built, bound, :likelihood, v) ≈
        sum(logpdf(Normal(v[2i - 1] + v[2i], 1), data.ys[i]) for i in 1:3) rtol = 1e-12
    # Matrices keep their dimensions, column-major.
    A = [[1.0 2.0; 3.0 4.0], [5.0 6.0 7.0]]
    matrices = quote
        @plate for i in eachindex(A)
            M[i] .~ Normal.(A[i], 1)
        end
    end
    bound = bind_data(lower_rkppl(matrices, (; A)), (; A))
    built = build_kernel(bound)
    @test coordinate_names(built.layout) == Symbol.(["M.1.1.1", "M.1.2.1", "M.1.1.2",
        "M.1.2.2", "M.2.1.1", "M.2.1.2", "M.2.1.3"])
    v = collect(0.1:0.1:0.7)
    nt = constrain(built.layout, v)
    @test nt.M == [[0.1 0.3; 0.2 0.4], [0.5 0.6 0.7]]
    @test unconstrain(built.layout, nt) ≈ v
    @test _cvl_q(built, bound, :prior, v) ≈ sum(logpdf.(Normal.(A[1], 1), nt.M[1])) +
        sum(logpdf.(Normal.(A[2], 1), nt.M[2])) rtol = 1e-12
    # An empty cell has no coordinates; an empty domain has none at all.
    empty_cells = quote
        @plate for i in eachindex(e)
            v[i] .~ Normal.(e[i], 1)
        end
    end
    e = [[1.0, 2.0], Float64[], [3.0]]
    bound = bind_data(lower_rkppl(empty_cells, (; e)), (; e))
    built = build_kernel(bound)
    @test constrain(built.layout, [0.3, 0.4, 0.5]).v == [[0.3, 0.4], Float64[], [0.5]]
    @test _cvl_q(built, bound, :prior, [0.3, 0.4, 0.5]) ≈
        sum(logpdf.(Normal.([1.0, 2.0, 3.0], 1), [0.3, 0.4, 0.5])) rtol = 1e-12
    e = Vector{Float64}[]
    bound = bind_data(lower_rkppl(empty_cells, (; e)), (; e))
    built = build_kernel(bound)
    @test built.layout.total == 0
    @test _cvl_q(built, bound, :prior, Float64[]) == 0.0
end

@testset "cell latents: a later latent, readers outside the plate and rebinding" begin
    mu = [[1.0, 2.0], [3.0]]
    chained = quote
        @plate for i in eachindex(mu)
            a[i] .~ Normal.(mu[i], 1)
            b[i] .~ Normal.(a[i], 0.5)
        end
        s = sum(sum.(b))
        ys .~ Normal.(s, 1)
    end
    data = (; mu, ys = [2.0])
    bound = bind_data(lower_rkppl(chained, data; conditioned = (:ys,)), data)
    built = build_kernel(bound)
    @test coordinate_names(built.layout) == Symbol.(["a.1.1", "a.1.2", "a.2.1",
        "b.1.1", "b.1.2", "b.2.1"])
    oracle = v -> sum(logpdf.(Normal.([1.0, 2.0, 3.0], 1), v[1:3])) +
        sum(logpdf.(Normal.(v[1:3], 0.5), v[4:6])) + logpdf(Normal(sum(v[4:6]), 1), 2.0)
    u = [0.2, 1.1, 2.5, 0.4, 1.3, 2.0]
    sampler = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
    value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ oracle(u) rtol = 1e-12
    @test gradient ≈ _cvl_fd(oracle, u) rtol = 1e-6 atol = 1e-8
    # One lowering binds other lengths; the program keeps its retained plates.
    plan = lower_rkppl(_CVL_TOTALS, (:mu, :obs, :y, :total); conditioned = (:y, :total))
    structures = map(([[2.0, 3.0], [1.0]],
            [[1.0], [2.0, 3.0], [4.0, 5.0, 6.0], [0.5], [7.0, 8.0]])) do mus
        d = (; mu = mus, obs = [[1] for m in mus], y = [[m[1] + 0.1] for m in mus],
            total = sum.(mus) .+ 0.2)
        bound = bind_data(plan, d)
        built = build_kernel(bound)
        n = sum(length, mus)
        @test built.layout.total == 1 + n
        v = vcat(0.2, log.(reduce(vcat, mus)) .+ 0.05)
        @test _cvl_q(built, bound, :sampler, v) ≈ _cvl_totals_oracle(v, d) rtol = 1e-12
        _cvl_structure(built.spec)
    end
    @test structures[1] == structures[2]
    @test !isempty(structures[1])
end

# One program from statement blocks, as written one after the other.
_cvl_join(blocks...) = Expr(:block, (a for b in blocks
    for a in (Meta.isexpr(b, :block) ? b.args : Any[b]))...)

# A model value holding numbers, read at an index, is a number there, as in
# Julia: a declared vector, a dotted definition over scalars and data, its
# cell local, a gather, a design product, a slice array's rows, an earlier
# latent plate and a scan trajectory. Each `v[i]` then has `mu[i]`'s shape.
@testset "cell latents: model values read at the index" begin
    mu = [[10.0, 20.0, 5.0], [7.0, 3.0]]
    lm = [log.(c) for c in mu]
    x, g, gl, X = [0.3, -0.4], [3, 1], [20, 10], [1.0 0.3; 1.0 -0.4]
    rows = [[1, 2, 2], [3, 1]]
    cells(u) = [u[1:3], u[4:5]]
    shifted(m) = [l .+ mk for (l, mk) in zip(lm, m)]
    vlp(locs, v, s) = sum(sum(logpdf.(Normal.(l, s), w)) for (l, w) in zip(locs, v))
    latent(read) = :(@plate for i in eachindex(mu)
        v[i] .~ Normal.(log.(mu[i]) .+ $read, 0.5)
    end)
    affine = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        m = a .+ b .* x
    end
    # The two leading coordinates locate the two cells by `m`.
    two(m, u) = sum(logpdf.(Normal(), u[1:2])) + vlp(shifted(m), cells(u[3:7]), 0.5)
    cases = [
        # A declared vector, as `m ~ filldist(Normal(0, 1), 2)` reads in Turing.
        (_cvl_join(:(m[1:2] .~ Normal.(0, 1)), latent(:(m[i]))), (; mu),
            u -> two(u[1:2], u), [:m, :m]),
        (_cvl_join(affine, latent(:(m[i]))), (; mu, x),
            u -> two(u[1] .+ u[2] .* x, u), [:a, :b]),
        (_cvl_join(affine, quote
            @plate for i in eachindex(mu)
                mi = m[i]
                v[i] .~ Normal.(log.(mu[i]) .+ mi, 0.5)
            end
        end), (; mu, x), u -> two(u[1] .+ u[2] .* x, u), [:a, :b]),
        (_cvl_join(:(m[1:3] .~ Normal.(0, 1)), latent(:(m[g[i]]))), (; mu, g),
            u -> sum(logpdf.(Normal(), u[1:3])) + vlp(shifted(u[g]), cells(u[4:8]), 0.5),
            [:m, :m, :m]),
        # Labels of a `levels(…)` axis are looked up, as in any other read.
        (_cvl_join(:(m[levels(gl)] .~ Normal.(0, 1)), latent(:(m[gl[i]]))), (; mu, gl),
            u -> two(u[[2, 1]], u), [:m, :m]),
        (_cvl_join(:(b[axes(X, 2)] .~ Normal.(0, 1)), :(m = X * b), latent(:(m[i]))),
            (; mu, X), u -> two(X * u[1:2], u), [:b, :b]),
        # Rows of a slice array; its coordinates are `B.i.j`, column-major.
        (_cvl_join(:(eachrow(B[1:2, 1:2]) .~ MvNormal(zeros(2), [1.0 0.5; 0.5 2.0])),
            latent(:(B[i, 2]))), (; mu),
            u -> sum(logpdf(MvNormal(zeros(2), [1.0 0.5; 0.5 2.0]), u[[k, k + 2]]) for k in 1:2) +
                vlp(shifted(u[3:4]), cells(u[5:9]), 0.5), [:B, :B, :B, :B]),
        (_cvl_join(quote
            @plate for j in eachindex(mu)
                z[j] ~ Normal(0, 1)
            end
        end, latent(:(z[i]))), (; mu), u -> two(u[1:2], u), [:z, :z]),
        (_cvl_join(quote
            @scan begin
                h[1] ~ Normal(0, 1)
                for t in 2:T
                    h[t] ~ Normal(h[t - 1], 0.3)
                end
            end
        end, latent(:(h[i]))), (; mu, T = 2), u -> logpdf(Normal(), u[1]) +
            logpdf(Normal(u[1], 0.3), u[2]) + vlp(shifted(u[1:2]), cells(u[3:7]), 0.5),
            [:h, :h]),
        # An index holding an array gathers that many entries.
        (_cvl_join(:(m[1:3] .~ Normal.(0, 1)), latent(:(m[rows[i]]))), (; mu, rows),
            u -> sum(logpdf.(Normal(), u[1:3])) +
                vlp([l .+ u[r] for (l, r) in zip(lm, rows)], cells(u[4:8]), 0.5),
            [:m, :m, :m]),
        # A scale read at the index.
        (_cvl_join(:(m[1:2] .~ Normal.(0, 1)), quote
            @plate for i in eachindex(mu)
                v[i] .~ Normal.(log.(mu[i]), exp(m[i]))
            end
        end), (; mu), u -> sum(logpdf.(Normal(), u[1:2])) +
            sum(sum(logpdf.(Normal.(lm[k], exp(u[k])), w)) for (k, w) in enumerate(cells(u[3:7]))),
            [:m, :m]),
    ]
    for (ast, data, oracle, heads) in cases
        saved = deepcopy(data)
        n = length(heads) + 5
        for source in (data, keys(data))
            bound = bind_data(lower_rkppl(ast, source), data)
            built = build_kernel(bound)
            names = coordinate_names(built.layout)
            @test [Symbol(first(split(String(s), "."))) for s in names[1:(n - 5)]] == heads
            @test names[(n - 4):end] == Symbol.(["v.1.1", "v.1.2", "v.1.3", "v.2.1", "v.2.2"])
            u = collect(range(-0.4, 2.1; length = n))
            @test constrain(built.layout, u).v == cells(u[(n - 4):end])
            sampler = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
            for w in (u, reverse(u))
                value, gradient = sampler_value_and_gradient!(sampler, similar(w), w)
                @test value ≈ oracle(w) rtol = 1e-12
                @test gradient ≈ _cvl_fd(oracle, w) rtol = 1e-6 atol = 1e-8
            end
        end
        @test data == saved
    end
    # Rebinding sizes each index again from the data.
    plan = lower_rkppl(cases[2][1], (:mu, :x))
    mus = [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]
    xs = [0.1, 0.2, 0.3]
    bound = bind_data(plan, (; mu = mus, x = xs))
    built = build_kernel(bound)
    @test built.layout.total == 2 + 6
    u = collect(range(0.1, 0.8; length = 8))
    lt = constrain(built.layout, u).v
    @test lt == [u[3:4], u[5:5], u[6:8]]
    m = u[1] .+ u[2] .* xs
    @test _cvl_q(built, bound, :prior, u) ≈ sum(logpdf.(Normal(), u[1:2])) +
        sum(sum(logpdf.(Normal.(log.(mus[k]) .+ m[k], 0.5), lt[k])) for k in 1:3) rtol = 1e-12
end

@testset "cell latents: dimensions that differ by index, and shapes not built yet" begin
    # Valid Julia, but each cell then reads a value of another type: warned,
    # not refused (user feedback on `0027dq6`, review `1dvac2l`).
    mixed = quote
        @plate for i in eachindex(x)
            v[i] .~ Normal.(x[i], 1)
        end
    end
    x = Any[[1.0, 2.0], [1.0 2.0; 3.0 4.0]]
    bound = @test_logs (:warn, r"different dimensions") match_mode = :any bind_data(
        lower_rkppl(mixed, (:x,)), (; x))
    built = build_kernel(bound)
    u = collect(0.1:0.1:0.6)
    @test constrain(built.layout, u).v == [[0.1, 0.2], [0.3 0.5; 0.4 0.6]]
    @test _cvl_q(built, bound, :prior, u) ≈
        sum(logpdf.(Normal.([1.0, 2.0, 1.0, 3.0, 2.0, 4.0], 1), u)) rtol = 1e-12
    # Not built yet: an argument whose shape a call of sampled values decides,
    # and array latents over `levels(…)`.
    @eval _cvl_pair(x) = [x, x]
    by_call = quote
        b ~ Normal(0, 1)
        @plate for i in eachindex(x)
            v[i] .~ Normal.(_cvl_pair(b), 1)
        end
    end
    gap = try
        lower_rkppl(by_call, (; x = [1.0, 2.0]); mod = @__MODULE__)
        false
    catch err
        err isa SurfaceLoweringError &&
            occursin("must follow from data", sprint(showerror, err)) || rethrow()
        true
    end
    @test_broken !gap
    by_level = quote
        @plate for k in levels(g)
            v[k] .~ Normal.(m[k], 1)
        end
    end
    gap = try
        lower_rkppl(by_level, (; g = [1, 2], m = [[1.0], [2.0, 3.0]]))
        false
    catch err
        err isa SurfaceLoweringError &&
            occursin("not built yet", sprint(showerror, err)) || rethrow()
        true
    end
    @test_broken !gap
    # Not built yet either: a model value read with `:` (its extent is the
    # declaration's), an earlier plate's outputs, and a definition holding
    # one array per index (`b .* w` over arrays) read at the index.
    mu = [[10.0, 20.0, 5.0], [7.0, 3.0]]
    by_colon = quote
        B[1:2, 1:3] .~ Normal.(0, 1)
        @plate for i in eachindex(mu)
            v[i] .~ Normal.(B[i, :], 0.5)
        end
    end
    by_output = quote
        a ~ Normal(0, 1)
        @plate for j in eachindex(mu)
            r[j] = a + j
        end
        @plate for i in eachindex(mu)
            v[i] .~ Normal.(log.(mu[i]) .+ r[i], 0.5)
        end
    end
    for ast in (by_colon, by_output)
        gap = try
            lower_rkppl(ast, (; mu))
            false
        catch err
            err isa SurfaceLoweringError &&
                occursin("must follow from data", sprint(showerror, err)) || rethrow()
            true
        end
        @test_broken !gap
    end
    by_arrays = quote
        b ~ Normal(0, 1)
        m = b .* w
        @plate for i in eachindex(mu)
            v[i] .~ Normal.(log.(mu[i]) .+ m[i], 0.5)
        end
    end
    data = (; mu, w = [[1.0, 2.0, 3.0], [4.0, 5.0]])
    gap = try
        bind_data(lower_rkppl(by_arrays, data), data)
        false
    catch err
        err isa ContractValidationError &&
            occursin("holds numbers", sprint(showerror, err)) || rethrow()
        true
    end
    @test_broken !gap
end
