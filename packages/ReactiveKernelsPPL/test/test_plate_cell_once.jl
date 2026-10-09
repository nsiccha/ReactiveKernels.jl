using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

# A `@plate` cell states one iteration of its Julia loop, so it runs once per
# index however many per-index values it defines: several indexed outputs
# sharing cell locals, the components of a row output, computed observation
# arguments, and level outputs reading one another. The live cell function
# below counts its calls; data-only cell work counts separately and runs only
# at preparation.

const _PCO_LIVE = Ref(0)
const _PCO_DATA = Ref(0)
_pco_live(t, rate) = (_PCO_LIVE[] += 1; t .* exp(rate))
_pco_scalar(w, rate) = (_PCO_LIVE[] += 1; w * exp(rate))
_pco_row(w, rate) = (_PCO_LIVE[] += 1; [w * exp(rate), w + rate])
_pco_positions(kinds) = (_PCO_DATA[] += 1; findall(isone, kinds))
_pco_flat(cells) = reduce(vcat, cells; init = Float64[])

function _pco_data(n)
    t = [Float64[0.2 + 0.1 * i + 0.3 * j for j in 1:(1 + mod(i, 3))] for i in 1:n]
    kinds = [[j == 1 || mod(i + j, 3) != 0 ? 1 : 2 for j in eachindex(ti)]
        for (i, ti) in enumerate(t)]
    x = collect(range(-0.4, 0.6; length = n))
    w = collect(range(0.3, 1.1; length = n))
    g = [mod1(i, 3) for i in 1:n]
    return (; t, kinds, x, w, g)
end

_pco_obs(v::AbstractVector{<:Real}, k) = v .+ 0.1 .* sin.(k .+ eachindex(v))
_pco_obs(v::AbstractVector{<:AbstractVector}, k) =
    [vi .+ 0.1 .* sin.(k + i .+ eachindex(vi)) for (i, vi) in enumerate(v)]
_pco_lik(obs, loc, s) = sum(sum(logpdf.(Normal.(l, s), o)) for (o, l) in zip(obs, loc))

# Each case: the plate and the statements observing it, the live cells per
# evaluation, and the per-index locations `(d, q) -> (first, second)` the two
# observations read.
function _pco_case(kind)
    kind === :arrays_two_outputs && return (quote
        @plate for i in eachindex(t)
            m = _pco_live(t[i], la[i])
            c[i] = m
            r[i] = 2 .* m
        end
        @plate for i in eachindex(y1)
            y1[i] .~ Normal.(c[i], s)
        end
        @plate for i in eachindex(y2)
            y2[i] .~ Normal.(r[i], s)
        end
    end, d -> length(d.t), function (d, q)
        m = [ti .* exp(q.a + xi) for (ti, xi) in zip(d.t, d.x)]
        (m, [2 .* mi for mi in m])
    end)
    kind === :scalar_two_outputs && return (quote
        @plate for i in eachindex(w)
            m = _pco_scalar(w[i], la[i])
            c[i] = m
            r[i] = 2 * m
        end
        y1 .~ Normal.(c, s)
        y2 .~ Normal.(r, s)
    end, d -> length(d.w), function (d, q)
        m = d.w .* exp.(q.a .+ d.x)
        (m, 2 .* m)
    end)
    kind === :cell_observations && return (quote
        @plate for i in eachindex(y1)
            m = _pco_scalar(w[i], la[i])
            y1[i] ~ Normal(m, s)
            y2[i] ~ Normal(2 * m, s)
        end
    end, d -> length(d.w), function (d, q)
        m = d.w .* exp.(q.a .+ d.x)
        (m, 2 .* m)
    end)
    kind === :output_and_observation && return (quote
        @plate for i in eachindex(y1)
            m = _pco_scalar(w[i], la[i])
            c[i] = m
            y1[i] ~ Normal(m + lb[i], s)
        end
        y2 .~ Normal.(2 .* c, s)
    end, d -> length(d.w), function (d, q)
        m = d.w .* exp.(q.a .+ d.x)
        (m .+ q.b .* d.x, 2 .* m)
    end)
    kind === :row_output && return (quote
        @plate for i in eachindex(w)
            row = _pco_row(w[i], la[i])
            B[i, 1:2] = row
        end
        y1 .~ Normal.(B[:, 1], s)
        y2 .~ Normal.(B[:, 2], s)
    end, d -> length(d.w), function (d, q)
        (d.w .* exp.(q.a .+ d.x), d.w .+ q.a .+ d.x)
    end)
    # Level cells: a later output reads an earlier one at the loop index.
    kind === :level_outputs && return (quote
        u[levels(g)] .~ Normal.(0, 1)
        @plate for k in levels(g)
            m = _pco_scalar(1.5, u[k])
            c[k] = m + a
            e[k] = 2 * c[k]
        end
        y1 .~ Normal.(c[g], s)
        y2 .~ Normal.(e[g], s)
    end, d -> length(unique(d.g)), function (d, q)
        c = 1.5 .* exp.(q.u) .+ q.a
        (c[d.g], 2 .* c[d.g])
    end)
    # Data-only cell work and a data-only output stay prepared once beside
    # two live outputs; the plate iterates a value whose shape data establish.
    kind === :data_only_beside_live && return (quote
        @plate for i in eachindex(la)
            p = _pco_positions(kinds[i])
            tp = t[i][p]
            c[i] = _pco_live(tp, la[i])
            r[i] = tp .* exp(lb[i])
            h[i] = 2 .* tp
        end
        @plate for i in eachindex(y1)
            y1[i] .~ Normal.(c[i] .+ h[i], s)
        end
        @plate for i in eachindex(y2)
            y2[i] .~ Normal.(r[i], s)
        end
    end, d -> length(d.t), function (d, q)
        tp = [ti[findall(isone, k)] for (ti, k) in zip(d.t, d.kinds)]
        ([p .* exp(q.a + xi) .+ 2 .* p for (p, xi) in zip(tp, d.x)],
            [p .* exp(q.b * xi) for (p, xi) in zip(tp, d.x)])
    end)
    error("unknown case $kind")
end

function _pco_fixture(kind, n)
    d = _pco_data(n)
    plates, cells, oracle_loc = _pco_case(kind)
    q0 = (; a = 0.2, b = -0.3, s = 0.9, u = [0.1, -0.2, 0.3])
    loc1, loc2 = oracle_loc(d, q0)
    y1, y2 = _pco_obs(loc1, 1), _pco_obs(loc2, 2)
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        s ~ Exponential(1.0)
        la = a .+ x
        lb = b .* x
        $plates
    end
    ast = Expr(:block, Iterators.flatten(Meta.isexpr(a, :block) ? a.args : (a,)
        for a in ast.args)...)
    data = Dict{Symbol,Any}(:t => d.t, :kinds => d.kinds, :x => d.x, :w => d.w,
        :g => d.g, :y1 => y1, :y2 => y2)
    bound = bind_data(lower_rkppl(ast, data; conditioned = (:y1, :y2),
        mod = @__MODULE__), data)
    built = build_kernel(bound)
    q = kind === :level_outputs ? q0 : Base.structdiff(q0, NamedTuple{(:u,)})
    u = unconstrain(built.layout, q)
    function oracle(v)
        q = constrain(built.layout, v)
        l1, l2 = oracle_loc(d, q)
        prior = logpdf(Normal(), q.a) + logpdf(Normal(), q.b) +
            logpdf(Exponential(1.0), q.s) +
            (haskey(q, :u) ? sum(logpdf.(Normal(), q.u)) : 0.0)
        prior + logjac(built.layout, v) + _pco_lik(y1, l1, q.s) + _pco_lik(y2, l2, q.s)
    end
    return (; data, bound, built, u, oracle, cells = cells(d))
end

function _pco_fd(f, u)
    h = cbrt(eps(Float64))
    [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end for i in eachindex(u)]
end

_pco_structure(program) = [(e.kind, e.depth) for e in recipe_inventory(program)
    if e.kind !== :ordinary]

const _PCO_KINDS = (:arrays_two_outputs, :scalar_two_outputs, :cell_observations,
    :output_and_observation, :row_output, :level_outputs, :data_only_beside_live)

@testset "a @plate cell runs once per index however many values it defines" begin
    for kind in _PCO_KINDS
        structures = map((4, 11)) do n
            fx = _pco_fixture(kind, n)
            saved = deepcopy(fx.data)
            sampler = prepare_sampler(fx.built, fx.bound, fx.u;
                backend = AutoEnzyme(; mode = Enzyme.Reverse))
            g = similar(fx.u)
            for shift in (0.0, 0.11)
                u = fx.u .+ shift
                value, gradient = sampler_value_and_gradient!(sampler, g, u)
                @test value ≈ fx.oracle(u) rtol = 1e-10
                @test gradient ≈ _pco_fd(fx.oracle, u) rtol = 1e-6 atol = 1e-8
            end
            _PCO_LIVE[] = 0
            sampler(fx.u)
            @test _PCO_LIVE[] == fx.cells
            _PCO_LIVE[] = 0
            sampler_value_and_gradient!(sampler, g, fx.u)
            @test _PCO_LIVE[] == fx.cells
            @test isequal(fx.data, saved)
            _pco_structure(sampler.kernel)
        end
        # The retained plates do not multiply with the index count.
        @test structures[1] == structures[2]
    end
end

@testset "a shared @plate cell keeps data-only cell work at preparation" begin
    fx = _pco_fixture(:data_only_beside_live, 6)
    _PCO_DATA[] = 0
    sampler = prepare_sampler(fx.built, fx.bound, fx.u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    prepared = _PCO_DATA[]
    g = similar(fx.u)
    for _ in 1:3
        sampler_value_and_gradient!(sampler, g, fx.u)
        sampler(fx.u)
    end
    @test prepared > 0
    @test _PCO_DATA[] == prepared
end

# One cell of the observations a shared cell feeds runs that cell at its index
# only: each observation's cell composes the shared plate, as it composed each
# output's plate before (`plate_cell`).
@testset "a cell query of a shared @plate cell runs it at that index" begin
    for kind in (:arrays_two_outputs, :cell_observations)
        fx = _pco_fixture(kind, 9)
        q = prepare_cell_query(fx.built, fx.bound, (:y1, :y2))
        pointwise = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), fx.u)
        for i in (1, 5, 9)
            _PCO_LIVE[] = 0
            value = q(fx.u, i)
            if kind === :cell_observations
                # An in-cell observation's computed argument reaches its cell
                # through a range selection, so the query runs the argument's
                # plate at every index on main 1169aec3, shared or not
                # (snag prepare-cell-que-30f4f1bf).
                @test_broken _PCO_LIVE[] == 2
            else
                @test _PCO_LIVE[] == 2
            end
            @test value ≈ sum(pointwise.y1[i]) + sum(pointwise.y2[i]) rtol = 1e-12
        end
        sampler = prepare_cell_sampler(fx.built, fx.bound, (:y1, :y2), fx.u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        for i in (2, 7)
            value, gradient = cell_value_and_gradient!(sampler, similar(fx.u), fx.u, i)
            @test value ≈ q(fx.u, i) rtol = 1e-12
            @test gradient ≈ _pco_fd(u -> q(u, i), fx.u) rtol = 1e-6 atol = 1e-8
        end
    end
end

# A data-only output of a cell whose other outputs read parameters is a
# data-only definition: it reads only data, so `weighted` accepts it as
# weights (co-report on snag rkppl-plate-for-5f5a19db, BRM's in-cell
# weighted observations).
@testset "a data-only output of a shared @plate cell stays data-only" begin
    x = [[0.5, 1.0], [0.7], [0.4, 0.9, 1.3]]
    w = [[1.0, 2.0], [0.5], [1.5, 1.0, 2.5]]
    y = [[0.6, 1.4], [0.9], [0.5, 1.2, 1.9]]
    data = Dict{Symbol,Any}(:x => x, :w => w, :y => y)
    ast = quote
        s ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(x)
            m = s .* x[i]
            v = 0.5 .* w[i]
            c[i] = m
            r[i] = 2 .* m
            wt[i] = v .+ 0.25
        end
        @plate for i in eachindex(y)
            y[i] .~ weighted.(Normal.(c[i] .+ r[i], sigma), wt[i])
        end
    end
    bound = bind_data(lower_rkppl(ast, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(bound)
    saved = deepcopy(data)
    function oracle(v)
        q = constrain(built.layout, v)
        logpdf(Normal(), q.s) + logpdf(Exponential(1.0), q.sigma) + logjac(built.layout, v) +
            sum(sum((0.5 .* wi .+ 0.25) .* logpdf.(Normal.(3 .* q.s .* xi, q.sigma), yi))
                for (xi, wi, yi) in zip(x, w, y))
    end
    sampler = prepare_sampler(built, bound, [0.2, 0.1];
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    for u in ([0.2, 0.1], [-0.3, 0.4])
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ oracle(u) rtol = 1e-10
        @test gradient ≈ _pco_fd(oracle, u) rtol = 1e-6 atol = 1e-8
    end
    @test isequal(data, saved)
end
