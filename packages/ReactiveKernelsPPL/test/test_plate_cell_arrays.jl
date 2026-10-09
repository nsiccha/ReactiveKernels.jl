using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

# A `@plate` cell states one iteration of its Julia loop, including when a
# per-index value is an array (`t[i]` holding one vector per subject). Such
# cells run as retained RK plate cells: authored broadcasts, chained and
# local indexing, vector-valued gathers, reductions and function-shaped
# kernel callees mean what the Julia loop means, and cell locals stay in
# their cell.

const _PCA_CALLS = Ref(0)
_pca_flat(cells) = reduce(vcat, cells; init = Float64[])
_pca_positions(kinds) = (_PCA_CALLS[] += 1; findall(isone, kinds))
_pca_scaled(t, rate) = t .* exp(rate)
# A module call: its result's shape is unknown until it runs.
_pca_opaque(a, x) = a .+ x
ReactiveKernels.@kernel _pca_klive(t, rate) = begin
    return t .* exp(rate)
end
ReactiveKernels.@kernel _pca_kcell(kinds, picks, t, rate) = begin
    p = _pca_positions(kinds)
    sel = p[picks]
    return t[sel] .* exp(rate)
end

function _pca_data(n)
    t = [Float64[0.2 + 0.1 * i + 0.3 * j for j in 1:(1 + mod(i, 3))] for i in 1:n]
    kinds = [[mod(i + j, 3) == 0 ? 2 : 1 for j in eachindex(ti)] for (i, ti) in enumerate(t)]
    picks = [collect(1:min(2, count(isone, k))) for k in kinds]
    lengths = cumsum(length.(t))
    rows = [collect((l - length(ti) + 1):l) for (l, ti) in zip(lengths, t)]
    v = reduce(vcat, t)
    x = collect(range(-0.4, 0.6; length = n))
    picked = [ti[findall(isone, k)[p]] for (ti, k, p) in zip(t, kinds, picks)]
    M = [hcat(ti, ti .^ 2) for ti in t]
    return (; x, t, kinds, picks, rows, v, picked, M)
end

# Each case: the plate cells, the response it feeds, and the per-subject
# location oracle `(d, q) -> vector of vectors` (or of numbers).
function _pca_case(kind)
    kind === :kernel_callee && return (
        :(@plate for i in eachindex(la)
            c[i] = _pca_klive(t[i], la[i])
        end), :(_pca_flat(c)), (d, q) -> [ti .* exp(q.a + xi) for (ti, xi) in zip(d.t, d.x)])
    kind === :kernel_callee_data && return (
        :(@plate for i in eachindex(kinds)
            c[i] = _pca_kcell(kinds[i], picks[i], t[i], la[i] - lb[i])
        end), :(_pca_flat(c)),
        (d, q) -> [pk .* exp(q.a + xi - q.b * xi) for (pk, xi) in zip(d.picked, d.x)])
    kind === :function_callee && return (
        :(@plate for i in eachindex(la)
            p = _pca_positions(kinds[i])
            c[i] = _pca_scaled(t[i][p[picks[i]]], la[i] - lb[i])
        end), :(_pca_flat(c)),
        (d, q) -> [pk .* exp(q.a + xi - q.b * xi) for (pk, xi) in zip(d.picked, d.x)])
    kind === :dotted && return (
        :(@plate for i in eachindex(t)
            m = exp(la[i]) .* t[i]
            c[i] = m .* 2 .+ lb[i]
        end), :(_pca_flat(c)),
        (d, q) -> [2 .* exp(q.a + xi) .* ti .+ q.b * xi for (ti, xi) in zip(d.t, d.x)])
    kind === :chained && return (
        :(@plate for i in eachindex(t)
            c[i] = t[i][picks[i]] .* exp(a)
        end), :(_pca_flat(c)),
        (d, q) -> [ti[p] .* exp(q.a) for (ti, p) in zip(d.t, d.picks)])
    kind === :local_index && return (
        :(@plate for i in eachindex(t)
            p = _pca_positions(kinds[i])
            sel = p[picks[i]]
            ti = t[i]
            c[i] = ti[sel] .* exp(la[i])
        end), :(_pca_flat(c)),
        (d, q) -> [pk .* exp(q.a + xi) for (pk, xi) in zip(d.picked, d.x)])
    kind === :ragged_gather && return (
        :(@plate for i in eachindex(rows)
            c[i] = v[rows[i]] .* exp(la[i])
        end), :(_pca_flat(c)),
        (d, q) -> [d.v[r] .* exp(q.a + xi) for (r, xi) in zip(d.rows, d.x)])
    kind === :ragged_reduction && return (
        :(@plate for i in eachindex(rows)
            c[i] = sum(v[rows[i]]) * exp(la[i])
        end), :c,
        (d, q) -> [sum(d.v[r]) * exp(q.a + xi) for (r, xi) in zip(d.rows, d.x)])
    # The same local in two plates: each plate's `m` is its own.
    kind === :two_plates && return (
        quote
            @plate for i in eachindex(la)
                m = _pca_scaled(t[i], la[i])
                c[i] = m
            end
            @plate for i in eachindex(la)
                m = _pca_scaled(t[i], lb[i])
                e[i] = m .* 2
            end
        end, :(_pca_flat(c) .+ _pca_flat(e)),
        (d, q) -> [ti .* exp(q.a + xi) .+ 2 .* ti .* exp(q.b * xi) for (ti, xi) in zip(d.t, d.x)])
    # A plate over an opaque value reads its live iterator.
    kind === :opaque_iterator && return (
        quote
            r = _pca_opaque(a, x)
            @plate for i in eachindex(r)
                p = _pca_positions(kinds[i])
                c[i] = _pca_scaled(t[i][p[picks[i]]], r[i])
            end
        end, :(_pca_flat(c)),
        (d, q) -> [pk .* exp(q.a + xi) for (pk, xi) in zip(d.picked, d.x)])
    # Undotted Julia arithmetic on a per-index array: binding shows that `t`
    # holds one array per index, so the cell is one iteration of the loop
    # whatever operators it applies.
    kind === :undotted_scale && return (
        :(@plate for i in eachindex(t)
            c[i] = t[i] * exp(la[i])
        end), :(_pca_flat(c)), (d, q) -> [ti .* exp(q.a + xi) for (ti, xi) in zip(d.t, d.x)])
    kind === :undotted_left && return (
        :(@plate for i in eachindex(t)
            c[i] = exp(lb[i]) * t[i]
        end), :(_pca_flat(c)), (d, q) -> [exp(q.b * xi) .* ti for (ti, xi) in zip(d.t, d.x)])
    kind === :undotted_divide && return (
        :(@plate for i in eachindex(t)
            c[i] = t[i] / exp(a)
        end), :(_pca_flat(c)), (d, q) -> [ti ./ exp(q.a) for ti in d.t])
    kind === :undotted_negate && return (
        :(@plate for i in eachindex(t)
            c[i] = -t[i] * exp(lb[i])
        end), :(_pca_flat(c)), (d, q) -> [-ti .* exp(q.b * xi) for (ti, xi) in zip(d.t, d.x)])
    kind === :undotted_nary && return (
        :(@plate for i in eachindex(t)
            c[i] = 2 * t[i] * exp(a)
        end), :(_pca_flat(c)), (d, q) -> [2 .* ti .* exp(q.a) for ti in d.t])
    kind === :undotted_sum && return (
        :(@plate for i in eachindex(t)
            c[i] = t[i] * exp(la[i]) - b * t[i]
        end), :(_pca_flat(c)), (d, q) -> [ti .* exp(q.a + xi) .- q.b .* ti for (ti, xi) in zip(d.t, d.x)])
    kind === :matrix_product && return (
        :(@plate for i in eachindex(M)
            c[i] = M[i] * [a, b]
        end), :(_pca_flat(c)), (d, q) -> [Mi * [q.a, q.b] for Mi in d.M])
    # A later plate reads the arrays an earlier plate computed.
    kind === :chained_plates && return (
        quote
            @plate for i in eachindex(t)
                c[i] = t[i] * exp(a)
            end
            @plate for i in eachindex(t)
                e[i] = 2 * c[i] - lb[i] * t[i]
            end
        end, :(_pca_flat(e)),
        (d, q) -> [2 .* ti .* exp(q.a) .- q.b * xi .* ti for (ti, xi) in zip(d.t, d.x)])
    # A definition computed from `t` holds one array per index too.
    kind === :derived_value && return (
        quote
            tr = map(reverse, t)
            @plate for i in eachindex(t)
                c[i] = tr[i] * exp(la[i])
            end
        end, :(_pca_flat(c)), (d, q) -> [reverse(ti) .* exp(q.a + xi) for (ti, xi) in zip(d.t, d.x)])
    error("unknown case $kind")
end

function _pca_fixture(kind, n)
    d = _pca_data(n)
    plates, location, oracle_loc = _pca_case(kind)
    q0 = (; a = 0.2, b = -0.3, s = 0.9)
    y = _pca_flat(oracle_loc(d, q0)) .+ 0.1 .* sin.(1:length(_pca_flat(oracle_loc(d, q0))))
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        s ~ Exponential(1.0)
        la = a .+ x
        lb = b .* x
        $plates
        loc = $location
        y .~ Normal.(loc, s)
    end
    ast = Expr(:block, Iterators.flatten(Meta.isexpr(a, :block) ? a.args : (a,)
        for a in ast.args)...)
    data = Dict{Symbol,Any}(:x => d.x, :t => d.t, :kinds => d.kinds, :picks => d.picks,
        :rows => d.rows, :v => d.v, :M => d.M, :y => y)
    unbound = lower_rkppl(ast, data; conditioned = (:y,), mod = @__MODULE__)
    bound = bind_data(unbound, data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, q0)
    function oracle(v)
        q = constrain(built.layout, v)
        loc = _pca_flat(oracle_loc(d, q))
        logpdf(Normal(), q.a) + logpdf(Normal(), q.b) + logpdf(Exponential(1.0), q.s) +
            logjac(built.layout, v) + sum(logpdf.(Normal.(loc, q.s), y))
    end
    return (; data, bound, built, u, oracle)
end

function _pca_fd(f, u)
    h = cbrt(eps(Float64))
    [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end for i in eachindex(u)]
end

_pca_structure(program) = [(e.kind, e.depth) for e in recipe_inventory(program)
    if e.kind !== :ordinary]

const _PCA_KINDS = (:kernel_callee, :kernel_callee_data, :function_callee, :dotted,
    :chained, :local_index, :ragged_gather, :ragged_reduction, :two_plates,
    :opaque_iterator)

const _PCA_UNDOTTED_KINDS = (:undotted_scale, :undotted_left, :undotted_divide,
    :undotted_negate, :undotted_nary, :undotted_sum, :matrix_product,
    :chained_plates, :derived_value)

function _pca_check(kind)
    structures = map((4, 11)) do n
        fx = _pca_fixture(kind, n)
        saved = deepcopy(fx.data)
        sampler = prepare_sampler(fx.built, fx.bound, fx.u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        for shift in (0.0, 0.11)
            u = fx.u .+ shift
            value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
            @test value ≈ fx.oracle(u) rtol = 1e-10
            @test gradient ≈ _pca_fd(fx.oracle, u) rtol = 1e-6 atol = 1e-8
        end
        @test isequal(fx.data, saved)
        _pca_structure(sampler.kernel)
    end
    # The retained plates do not multiply with the subject count.
    @test structures[1] == structures[2]
    @test count(e -> first(e) === :plate, structures[1]) >= 1
end

@testset "array-valued @plate cells: native values and Enzyme Reverse gradients" begin
    foreach(_pca_check, _PCA_KINDS)
end

@testset "array-valued @plate cells: undotted Julia arithmetic" begin
    foreach(_pca_check, _PCA_UNDOTTED_KINDS)
end

# The reported shape: a definition plate over one array per index, observed
# by a dotted cell over a response that holds one array per index (empty
# groups included).
@testset "undotted arithmetic feeding a dotted cell over arrays per index" begin
    t = [[0.1, 0.7], Float64[], [0.2, 0.8, 1.3], [0.4]]
    y = [[0.2, -0.1], Float64[], [0.1, 0.3, -0.2], [0.5]]
    data = Dict{Symbol,Any}(:t => t, :y => y)
    cases = (
        :(c[i] = t[i] * a) => (a, ti) -> ti .* a,
        :(c[i] = a * t[i]) => (a, ti) -> a .* ti,
        :(c[i] = t[i] / exp(a)) => (a, ti) -> ti ./ exp(a),
        :(c[i] = -t[i]) => (a, ti) -> -ti,
        :(begin m = t[i]; c[i] = m * a end) => (a, ti) -> ti .* a,
    )
    for (cell, loc) in cases
        body = Meta.isexpr(cell, :block) ? cell.args : Any[cell]
        ast = quote
            a ~ Normal(0, 1)
            s ~ Exponential(1.0)
            @plate for i in eachindex(t)
                $(body...)
            end
            @plate for i in eachindex(y)
                y[i] .~ Normal.(c[i], s)
            end
        end
        saved = deepcopy(data)
        bound = bind_data(lower_rkppl(ast, data; conditioned = (:y,), mod = @__MODULE__), data)
        built = build_kernel(bound)
        function oracle(v)
            q = constrain(built.layout, v)
            logpdf(Normal(), q.a) + logpdf(Exponential(1.0), q.s) + logjac(built.layout, v) +
                sum(sum(logpdf.(Normal.(loc(q.a, ti), q.s), yi); init = 0.0) for (ti, yi) in zip(t, y))
        end
        sampler = prepare_sampler(built, bound, zeros(2); backend = AutoEnzyme(; mode = Enzyme.Reverse))
        u = [0.3, -0.2]
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ oracle(u) rtol = 1e-12
        @test gradient ≈ _pca_fd(oracle, u) rtol = 1e-6 atol = 1e-8
        @test isequal(data, saved)
    end
    # Not built: a plan lowered from data names alone cannot see that `t`
    # holds arrays, so its cell keeps whole-column evaluation and binding
    # refuses the arrays, naming the value-aware lowering.
    ast = quote
        a ~ Normal(0, 1)
        s ~ Exponential(1.0)
        @plate for i in eachindex(t)
            c[i] = t[i] * a
        end
        @plate for i in eachindex(y)
            y[i] .~ Normal.(c[i], s)
        end
    end
    err = try
        bind_data(lower_rkppl(ast, (:t, :y); conditioned = (:y,), mod = @__MODULE__), data)
        nothing
    catch e
        e
    end
    @test_broken err === nothing
    @test occursin("lowered with the data values", sprint(showerror, err))
end

# Over data (`eachindex(kinds)`) and over a computed value whose shape bound
# data establish (`eachindex(la)`, `la = a .+ x`), data-only cell work runs
# once, at preparation.
@testset "array-valued @plate cells: data-only cell work runs at preparation" begin
    for kind in (:kernel_callee_data, :function_callee, :local_index)
        fx = _pca_fixture(kind, 6)
        _PCA_CALLS[] = 0
        sampler = prepare_sampler(fx.built, fx.bound, fx.u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        prepared = _PCA_CALLS[]
        g = similar(fx.u)
        for _ in 1:3
            sampler_value_and_gradient!(sampler, g, fx.u)
            sampler(fx.u)
        end
        @test prepared > 0
        @test _PCA_CALLS[] == prepared
    end
end

@testset "@plate cell locals stay in their cell" begin
    d = _pca_data(3)
    data = Dict{Symbol,Any}(:x => d.x, :y => [0.1, 0.2, 0.3])
    # Whole-column scalar cells: the same local in two plates, and a model
    # statement that reads a cell local.
    twice = quote
        a ~ Normal(0, 1)
        s ~ Exponential(1.0)
        @plate for i in eachindex(x)
            m = a + x[i]
            c[i] = 2 * m
        end
        @plate for i in eachindex(x)
            m = a - x[i]
            e[i] = 3 * m
        end
        y .~ Normal.(c .+ e, s)
    end
    bound = bind_data(lower_rkppl(twice, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(bound)
    q = (; a = 0.3, s = 0.7)
    u = unconstrain(built.layout, q)
    loc = 2 .* (q.a .+ d.x) .+ 3 .* (q.a .- d.x)
    expected = logpdf(Normal(), q.a) + logpdf(Exponential(1.0), q.s) +
        logjac(built.layout, u) + sum(logpdf.(Normal.(loc, q.s), data[:y]))
    @test Base.invokelatest(prepare_query(built, bound, :sampler), u) ≈ expected rtol = 1e-12
    outside = quote
        a ~ Normal(0, 1)
        s ~ Exponential(1.0)
        @plate for i in eachindex(x)
            m = a + x[i]
            c[i] = 2 * m
        end
        y .~ Normal.(m, s)
    end
    err = try
        lower_rkppl(outside, data; conditioned = (:y,), mod = @__MODULE__)
        nothing
    catch e
        e
    end
    @test err isa ReactiveKernelsPPL.SurfaceLoweringError
    @test occursin("`m` is a cell local of the `@plate`", sprint(showerror, err))
end
