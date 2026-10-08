using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

# A dotted `@plate` cell broadcasts over its own iteration's values, as the
# Julia loop it states does. A response holding one array per index (ragged,
# possibly empty) is observed entry by entry inside a retained group plate.

ReactiveKernels.@kernel _cba_reader(t, dose, log_k) = begin
    cells = ReactiveKernels.plate(t, dose, Ref(log_k)) do ti, di, lk
        di .* exp.(-exp(lk) .* ti)
    end
    return cells
end

# A visible scalar law, an ordinary Julia law and a data-only grouping helper
# for the sampling-RHS and derived-response cells.
ReactiveKernels.@kernel _cba_score(value, mu, sigma) = begin
    r = (value - mu) / sigma
    return -0.5 * log(2pi) - log(sigma) - 0.5 * r * r
end
_cba_normal_lpdf(value, mu, sigma) = -0.5 * log(2pi) - log(sigma) - 0.5 * ((value - mu) / sigma)^2
_cba_gather(raw, rows) = [raw[r] for r in rows]
_cba_reader_value(d, q, i) = d.dose[i] .* exp.(-exp(q.a) .* d.x[i])

function _cba_data(sizes)
    n = length(sizes)
    x = [Float64[0.1 * i + 0.3 * j for j in 1:k] for (i, k) in enumerate(sizes)]
    y = [0.3 .+ 0.5 .* sin.(3 .* xi .+ i) for (i, xi) in enumerate(x)]
    counts = [[mod(3i + j, 4) for j in 1:k] for (i, k) in enumerate(sizes)]
    bin = [[mod(i + 2j, 3) == 0 ? 1 : 0 for j in 1:k] for (i, k) in enumerate(sizes)]
    w = [Float64[0.5 + 0.25 * j for j in 1:k] for k in sizes]
    dose = collect(range(1.0, 2.0; length = n))
    mu = collect(range(-0.3, 0.4; length = n))
    # `y` flattened, and the rows of each index in the flat vector.
    yraw = reduce(vcat, y; init = Float64[])
    rows = [collect(sum(sizes[1:i-1]; init = 0) .+ (1:k)) for (i, k) in enumerate(sizes)]
    return (; x, y, counts, bin, w, dose, mu, yraw, rows)
end

_cba_response(kind) = kind in (:poisson, :poisson_reader, :nb2_scaled) ? :counts :
    kind === :bernoulli_reader ? :bin : kind === :derived ? :yd : :y

# The observation cell and its per-entry oracle for each kind; parameters
# `a`, `b`, `sigma` are declared in every program.
function _cba_case(kind)
    kind === :reader && return (
        :(loc = _cba_reader(x, dose, a)),
        :(y[i] .~ Normal.(loc[i], sigma)),
        (d, q, i) -> logpdf.(Normal.(d.dose[i] .* exp.(-exp(q.a) .* d.x[i]), q.sigma), d.y[i]))
    kind === :data && return (nothing,
        :(y[i] .~ Normal.(x[i], sigma)),
        (d, q, i) -> logpdf.(Normal.(d.x[i], q.sigma), d.y[i]))
    kind === :computed && return (nothing,
        :(y[i] .~ Normal.(a .+ b .* x[i], sigma)),
        (d, q, i) -> logpdf.(Normal.(q.a .+ q.b .* d.x[i], q.sigma), d.y[i]))
    kind === :local && return (nothing,
        Expr(:block, :(m = a .+ b .* x[i]), :(y[i] .~ Normal.(m, sigma))),
        (d, q, i) -> logpdf.(Normal.(q.a .+ q.b .* d.x[i], q.sigma), d.y[i]))
    kind === :group_scalar && return (nothing,
        :(y[i] .~ Normal.(mu[i] + a, sigma)),
        (d, q, i) -> logpdf.(Normal(d.mu[i] + q.a, q.sigma), d.y[i]))
    kind === :shared && return (nothing,
        :(y[i] .~ Normal.(a, sigma)),
        (d, q, i) -> logpdf.(Normal(q.a, q.sigma), d.y[i]))
    kind === :poisson && return (nothing,
        :(counts[i] .~ Poisson.(exp.(a .+ b .* x[i]))),
        (d, q, i) -> logpdf.(Poisson.(exp.(q.a .+ q.b .* d.x[i])), d.counts[i]))
    kind === :weighted && return (nothing,
        :(y[i] .~ weighted.(Normal.(a .+ x[i], sigma), w[i])),
        (d, q, i) -> d.w[i] .* logpdf.(Normal.(q.a .+ d.x[i], q.sigma), d.y[i]))
    kind === :censored && return (nothing,
        :(y[i] .~ censored.(Normal.(a .+ b .* x[i], sigma), -0.1, 0.7)),
        (d, q, i) -> logpdf.(censored.(Normal.(q.a .+ q.b .* d.x[i], q.sigma), -0.1, 0.7),
            clamp.(d.y[i], -0.1, 0.7)))
    # A visible KernelSpec law and an ordinary Julia law (rkppl-use §2).
    kind === :sampling_kernel && return (
        :(loc = _cba_reader(x, dose, a)),
        :(y[i] .~ LogDensity.(_cba_score, loc[i], sigma)),
        (d, q, i) -> logpdf.(Normal.(_cba_reader_value(d, q, i), q.sigma), d.y[i]))
    kind === :sampling_function && return (nothing,
        :(y[i] .~ LogDensity.(_cba_normal_lpdf, a .+ b .* x[i], sigma)),
        (d, q, i) -> logpdf.(Normal.(q.a .+ q.b .* d.x[i], q.sigma), d.y[i]))
    # Link families located by a composed reader's cells.
    kind === :bernoulli_reader && return (
        :(eta = _cba_reader(x, dose, a)),
        :(bin[i] .~ BernoulliLogit.(eta[i])),
        (d, q, i) -> logpdf.(Bernoulli.(1 ./ (1 .+ exp.(-_cba_reader_value(d, q, i)))), d.bin[i]))
    kind === :poisson_reader && return (
        :(eta = _cba_reader(x, dose, a)),
        :(counts[i] .~ Poisson.(exp.(eta[i]))),
        (d, q, i) -> logpdf.(Poisson.(exp.(_cba_reader_value(d, q, i))), d.counts[i]))
    # Per-index scalar arithmetic and per-index array scales.
    kind === :scaled && return (nothing,
        :(y[i] .~ Normal.(a .+ b .* x[i], dose[i] * sigma)),
        (d, q, i) -> logpdf.(Normal.(q.a .+ q.b .* d.x[i], d.dose[i] * q.sigma), d.y[i]))
    kind === :scaled_named && return (
        :(sc = dose .* sigma),
        :(y[i] .~ Normal.(a .+ b .* x[i], sc[i])),
        (d, q, i) -> logpdf.(Normal.(q.a .+ q.b .* d.x[i], d.dose[i] * q.sigma), d.y[i]))
    kind === :array_scale && return (nothing,
        :(y[i] .~ Normal.(a, sigma .* exp.(b .* x[i]))),
        (d, q, i) -> logpdf.(Normal.(q.a, q.sigma .* exp.(q.b .* d.x[i])), d.y[i]))
    kind === :nb2_scaled && return (nothing,
        :(counts[i] .~ NegativeBinomial2.(exp.(a .+ b .* x[i]), dose[i] * sigma)),
        function (d, q, i)
            m, phi = exp.(q.a .+ q.b .* d.x[i]), d.dose[i] * q.sigma
            logpdf.(NegativeBinomial.(phi, phi ./ (phi .+ m)), d.counts[i])
        end)
    # A data-only definition holding one array per index as the response.
    kind === :derived && return (
        :(yd = _cba_gather(yraw, rows)),
        :(yd[i] .~ Normal.(a .+ b .* x[i], sigma)),
        (d, q, i) -> logpdf.(Normal.(q.a .+ q.b .* d.x[i], q.sigma), d.y[i]))
    error("unknown kind $kind")
end

function _cba_fixture(kind, sizes; unbound = nothing)
    d = _cba_data(sizes)
    kind === :censored && (d = merge(d, (; y = [clamp.(v, -0.1, 0.7) for v in d.y])))
    definition, cell, entry = _cba_case(kind)
    response = _cba_response(kind)
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
    end
    definition === nothing || push!(ast.args, definition)
    push!(ast.args, Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
        Expr(:for, :(i = eachindex($response)), cell isa Expr && cell.head === :block ?
            cell : Expr(:block, cell))))
    data = Dict(pairs(d))
    for r in (:y, :counts, :bin)
        r === response || delete!(data, r)
    end
    response === :yd || (delete!(data, :yraw); delete!(data, :rows))
    unbound === nothing &&
        (unbound = lower_rkppl(ast, data; conditioned = (response,), mod = @__MODULE__))
    bound = bind_data(unbound, data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; a = 0.25, b = -0.3, sigma = 0.8))
    pointwise(v) = [entry(d, constrain(built.layout, v), i) for i in eachindex(sizes)]
    function oracle(v)
        q = constrain(built.layout, v)
        logpdf(Normal(), q.a) + logpdf(Normal(), q.b) + logpdf(Exponential(1.0), q.sigma) +
            logjac(built.layout, v) + sum(sum(pw; init = 0.0) for pw in pointwise(v); init = 0.0)
    end
    return (; kind, response, data, unbound, bound, built, u, oracle, pointwise)
end

function _cba_fd(f, u)
    h = cbrt(eps(Float64))
    [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end for i in eachindex(u)]
end

_cba_structure(program) = [(e.kind, e.depth) for e in recipe_inventory(program)
    if e.kind !== :ordinary]

function _cba_check(kind, sizes)
    fx = _cba_fixture(kind, sizes)
    saved = deepcopy(fx.data)
    sampler = prepare_sampler(fx.built, fx.bound, fx.u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    for shift in (0.0, 0.07)
        u = fx.u .+ shift
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ fx.oracle(u) rtol = 1e-10
        @test gradient ≈ _cba_fd(fx.oracle, u) rtol = 1e-5 atol = 1e-7
        pw = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), u)
        expected = fx.pointwise(u)
        # The pointwise result keeps the response's shape: one array of
        # observation densities per index.
        @test length(pw[fx.response]) == length(sizes)
        @test all(length.(pw[fx.response]) .== collect(sizes))
        @test all(pw[fx.response] .≈ expected)
        lik = Base.invokelatest(prepare_query(fx.built, fx.bound, :likelihood), u)
        @test lik ≈ sum(sum(p; init = 0.0) for p in expected) rtol = 1e-10
    end
    @test isequal(fx.data, saved)
end

const _CBA_SIZES = ((3, 2, 0, 1), (0, 4, 1, 5, 2, 3, 0, 2))

@testset "dotted plate cells over one array per index: native values and gradients" begin
    for kind in (:reader, :data, :computed, :local, :group_scalar, :shared,
            :poisson, :weighted, :censored), sizes in _CBA_SIZES
        _cba_check(kind, sizes)
    end
end

# Each response family and sampling law of a flat observation runs on each
# index's entries: caller-owned laws (a visible kernel and an ordinary
# function), link families over a composed reader, per-index scalar and
# array arguments, and a data-only definition as the response.
@testset "dotted plate cells over arrays: sampling laws, links, per-index arguments, derived responses" begin
    for kind in (:sampling_kernel, :sampling_function, :bernoulli_reader,
            :poisson_reader, :scaled, :scaled_named, :array_scale, :nb2_scaled, :derived),
            sizes in _CBA_SIZES
        _cba_check(kind, sizes)
    end
end

# The same observation over the flattened response and covariates is the
# flat equivalent: one density per entry, summed in the same order.
_cba_strip_index(ex) = ex isa Expr ? (Meta.isexpr(ex, :ref, 2) && ex.args[2] === :i ?
    ex.args[1] : Expr(ex.head, map(_cba_strip_index, ex.args)...)) : ex

@testset "dotted plate cells over arrays match the flat response" begin
    sizes = (3, 2, 0, 1)
    d = _cba_data(sizes)
    cat3 = [[1 + mod(i + j, 3) for j in 1:k] for (i, k) in enumerate(sizes)]
    trials = [[4 + j for j in 1:k] for k in sizes]
    nested = Dict{Symbol,Any}(:x => d.x, :y => d.y, :cat3 => cat3,
        :counts => d.counts, :trials => trials)
    flat = Dict{Symbol,Any}(k => reduce(vcat, v; init = eltype(eltype(v))[]) for (k, v) in nested)
    declarations = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
        c ~ Ordered(Normal(0, 1), 2)
        nu ~ Exponential(0.2)
    end
    function sampler(ast, data, response, u)
        data = Dict(k => v for (k, v) in data if k === response || k ∉ (:y, :cat3, :counts))
        bound = bind_data(lower_rkppl(ast, data; conditioned = (response,),
            mod = @__MODULE__), data)
        return prepare_sampler(build_kernel(bound), bound, u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
    end
    u = 0.1 .+ 0.05 .* (1:6)
    for (response, cell) in (
            (:y, :(y[i] .~ StudentT.(1 + nu, a .+ b .* x[i], sigma))),
            (:y, :(y[i] .~ truncated.(Normal.(a .+ b .* x[i], sigma), -1.0, 1.0))),
            (:counts, :(counts[i] .~ Binomial.(trials[i], logistic.(a .+ b .* x[i])))),
            (:counts, :(counts[i] .~ ZeroInflatedPoisson.(exp.(a .+ b .* x[i]), logistic.(b)))),
            (:cat3, :(cat3[i] .~ OrderedLogistic.(a .+ b .* x[i], Ref(c)))),
            (:cat3, :(cat3[i] .~ Ordinal.(StoppingRatio(), LogitLink(), a .+ b .* x[i], Ref(c)))),
            (:cat3, :(cat3[i] .~ CategoricalLogit.(a .+ x[i], b .* x[i]))))
        cells = Expr(:block, declarations.args..., Expr(:macrocall, Symbol("@plate"),
            LineNumberNode(1), Expr(:for, :(i = eachindex($response)), Expr(:block, cell))))
        whole = Expr(:block, declarations.args..., _cba_strip_index(cell))
        g, gflat = similar(u), similar(u)
        value, _ = sampler_value_and_gradient!(sampler(cells, nested, response, u), g, u)
        expected, _ = sampler_value_and_gradient!(sampler(whole, flat, response, u), gflat, u)
        @test value ≈ expected rtol = 1e-12
        @test g ≈ gflat rtol = 1e-9 atol = 1e-12
    end
end

@testset "dotted plate cells over arrays retain a nested group plate" begin
    for kind in (:computed, :sampling_kernel, :bernoulli_reader, :scaled, :derived)
        small = _cba_fixture(kind, (3, 2, 0, 1))
        large = _cba_fixture(kind, (0, 4, 1, 5, 2, 3, 0, 2, 6, 1, 1, 3))
        # One observation plate per index, nested in the group plate (RK native
        # nested plates); the computed location is its own retained cell.
        @test (:plate, 1) in _cba_structure(small.built.spec)
        @test _cba_structure(small.built.spec) == _cba_structure(large.built.spec)
        query(fx) = prepare_query(fx.built, fx.bound, :likelihood)
        @test _cba_structure(query(small)) == _cba_structure(query(large))
        @test count(==((:plate, 1)), _cba_structure(query(small))) == 1
    end
end

@testset "dotted plate cells over arrays: empty domains and rebinding" begin
    m = @rkppl begin
        a ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(y)
            y[i] .~ Normal.(a .+ x[i], sigma)
        end
    end
    law = @rkppl begin
        a ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(y)
            y[i] .~ LogDensity.(_cba_score, a .+ x[i], sigma)
        end
    end
    u = [0.2, log(0.8)]
    for model in (m, law), (x, y) in (([Float64[], Float64[]], [Float64[], Float64[]]),
            (Vector{Float64}[], Vector{Float64}[]))
        plan = model(; x) | (; y)
        built = build_kernel(plan)
        @test Base.invokelatest(prepare_query(built, plan, :likelihood), u) == 0.0
    end
    # A literal loop over every index observes each index's whole array.
    literal = @rkppl begin
        a ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
        @plate for i in 1:4
            y[i] .~ Normal.(a .+ x[i], sigma)
        end
    end
    d = _cba_data((3, 2, 0, 1))
    plan = literal(; x = d.x) | (; y = d.y)
    built = build_kernel(plan)
    @test Base.invokelatest(prepare_query(built, plan, :likelihood), u) ≈
        sum(sum(logpdf.(Normal.(0.2 .+ d.x[i], 0.8), d.y[i]); init = 0.0) for i in 1:4)
    # A names-only plan binds arrays per index of any lengths.
    fx = _cba_fixture(:computed, (3, 2, 0, 1))
    other = _cba_fixture(:computed, (1, 0, 4); unbound = fx.unbound)
    lik = Base.invokelatest(prepare_query(other.built, other.bound, :likelihood), other.u)
    @test lik ≈ sum(sum(p; init = 0.0) for p in other.pointwise(other.u)) rtol = 1e-10
end

@testset "observing arrays per index outside a dotted cell follows Julia" begin
    y = [[0.7, 0.3, 0.1], [1.4, 0.6], Float64[]]
    whole = @rkppl begin
        a ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
        y .~ Normal.(a, sigma)
    end
    # refused: `y .~ D.(…)` broadcasts a univariate distribution over the
    # arrays of `y`, which takes no array argument in Julia (standard-Julia
    # semantics, standing @rkppl principle 3)
    @test_throws ContractValidationError whole() | (; y)
    @test_throws "dotted `@plate` cell" whole() | (; y)
    undotted = @rkppl begin
        a ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(y)
            y[i] ~ Normal(a, sigma)
        end
    end
    # refused: an undotted cell observes `y[i]` itself, an array, with a
    # univariate distribution (standard-Julia semantics, principle 3)
    @test_throws "`y[i] .~ D.(…)`" undotted() | (; y)
    data_location = @rkppl begin
        sigma ~ Exponential(1.0)
        y .~ Normal.(x, sigma)
    end
    # refused: the same Julia broadcast over a data operand holding arrays
    @test_throws "holds one array per entry" data_location(; x = y) | (; y)
    mismatched = @rkppl begin
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(y)
            y[i] .~ Normal.(x[i], sigma)
        end
    end
    # refused: `x[i]` and `y[i]` must broadcast together, as Julia requires
    # (DimensionMismatch)
    @test_throws "does not broadcast" mismatched(; x = [[0.1, 0.2], [0.3, 0.4], Float64[]]) | (; y)
end
