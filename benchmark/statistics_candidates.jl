# Bounded experiment: discover numerical alternatives BEFORE plan selection.
# No production registration, public API, probability names, or per-binding
# cost calibration. A cost profile is fixed while the prepared graph is rebound.
# Run in the consumer environment used by statistics_alternatives.jl.
module StatisticsCandidates
using ReactiveKernels, LinearAlgebra, Random, Test
import Enzyme
const RK = ReactiveKernels
BLAS.set_num_threads(1)

@kernel plain_rss(X::Matrix{Float64}, y::Vector{Float64}, beta::Vector{Float64}) = begin
    mu::Vector{Float64} = X * beta
    residual::Vector{Float64} = y - mu
    total::Float64 = sum(abs2, residual)
    return total
end
@kernel converted_rss(X::Matrix{Float64}, y::Vector{Float64}, beta::Vector{Float64}) = begin
    mu::Vector{Float32} = X * beta
    residual = y - mu
    total::Float64 = sum(abs2, residual)
    return total
end
@kernel nonlinear_rss(X::Matrix{Float64}, y::Vector{Float64}, beta::Vector{Float64}) = begin
    mu = sin.(X * beta)
    residual = y - mu
    total::Float64 = sum(abs2, residual)
    return total
end

row_rss(X, y, beta) = sum(abs2, y - X * beta)
function candidate_statistics(X, y)
    # Invalid row counts still fail in the authored runtime arithmetic.
    size(X, 1) == length(y) || return RK._ls_unusable(size(X, 2), 1, Int[], size(X, 1))
    RK._ls_statistics_dense(X, y)
end
guarded_rss(s, X, y, beta) = s.usable ? RK._ls_rss_dense(s, (beta,)) : row_rss(X, y, beta)

# This recognizer inspects every authored recipe, without calling plan. It
# supports exactly one Float64 matrix product, subtraction and sum(abs2, r).
# Other identities, alternative producers, effects, checks and conversions are
# left intact. These are scope limits of the experiment, not input restrictions.
function source_call(r, f, nargs)
    r.effectful && return nothing
    length(r.outputs) == 1 || return nothing
    src = RK._ls_recipe_source(r)
    src === nothing && return nothing
    names, ex = src
    RK._ls_is_call(ex, f, nargs) || return nothing
    (; names, ex, env = Dict(zip(names, r.inputs)))
end
function sole_producer(g, v)
    ids = get(g.producers, RK.canon_id(g, v.id), Int[])
    length(ids) == 1 ? g.recipes[only(ids)] : nothing
end
function discover(g, bound)
    isempty(g.checks) && !any(r -> r.effectful, g.recipes) || return NamedTuple[]
    ids = Set(RK.canon_id(g, v.id) for v in bound)
    found = NamedTuple[]
    for total in g.recipes
        src = source_call(total, sum, 2)
        src === nothing && continue
        mod = total.op isa RK._KernelSourceOp ? total.op.mod : @__MODULE__
        RK._rw_resolve(src.ex.args[2], mod) === abs2 || continue
        residual = get(src.env, src.ex.args[3], nothing)
        residual === nothing && continue
        subtract = sole_producer(g, residual)
        subtract === nothing && continue
        sub = source_call(subtract, -, 2)
        sub === nothing && continue
        y = get(sub.env, sub.ex.args[2], nothing)
        mu = get(sub.env, sub.ex.args[3], nothing)
        (y === nothing || mu === nothing) && continue
        product = sole_producer(g, mu)
        product === nothing && continue
        mul = source_call(product, *, 2)
        mul === nothing && continue
        X = get(mul.env, mul.ex.args[2], nothing)
        beta = get(mul.env, mul.ex.args[3], nothing)
        (X === nothing || beta === nothing) && continue
        RK.valtype(X) === Matrix{Float64} && RK.valtype(y) === Vector{Float64} &&
            RK.valtype(beta) === Vector{Float64} && RK.valtype(mu) === Vector{Float64} &&
            RK.valtype(residual) === Vector{Float64} &&
            RK.valtype(only(total.outputs)) === Float64 || continue
        RK.canon_id(g, X.id) in ids && RK.canon_id(g, y.id) in ids &&
            !(RK.canon_id(g, beta.id) in ids) || continue
        push!(found, (; X, y, beta, residual, total = only(total.outputs),
                        row_recipes = (product.id, subtract.id, total.id)))
    end
    found
end

function expose(spec; evaluations, row_cost, preparation_cost, statistics_cost)
    evaluations > 0 || throw(ArgumentError("positive evaluation count required"))
    g = compose(spec.graph)
    found = discover(g, (spec.X, spec.y))
    # A single measured row-gradient cost is assigned to its shared row work.
    # The reduction weight is zero: deliberately approximate marginal costs.
    # This makes the joint-output limitation explicit instead of claiming an
    # operation-wise AD cost decomposition that we have not measured.
    for c in found
        for (i, r) in enumerate(g.recipes)
            r.id in c.row_recipes || continue
            cost = r.id == first(c.row_recipes) ? row_cost : 0.0
            g.recipes[i] = RK.Recipe(r.id, r.inputs, r.outputs, r.op, cost,
                r.cse_key, r.effectful, r.source)
        end
        stats = value!(g, Symbol(c.total.name, :_statistics), RK._LSStatistics)
        add!(g, (c.X, c.y) => stats, candidate_statistics; cost = preparation_cost / evaluations)
        add!(g, (stats, c.X, c.y, c.beta) => c.total, guarded_rss; cost = statistics_cost)
    end
    (; g, found)
end

function median_ns(f; samples = 11, iterations = 500)
    f()
    times = Float64[]
    for _ in 1:samples
        start = time_ns()
        for _ in 1:iterations
            f()
        end
        push!(times, (time_ns() - start) / iterations)
    end
    sort!(times)[(samples + 1) ÷ 2]
end
function reverse!(f, gradient, beta)
    fill!(gradient, 0)
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, Enzyme.Const(f), Enzyme.Active,
        Enzyme.Duplicated(beta, gradient))[2]
end
structure(x) = (1, 0, 0)
function structure(x::Expr)
    parts = map(structure, x.args)
    (1 + sum(first, parts; init = 0),
     (x.head in (:for, :while)) + sum(p -> p[2], parts; init = 0),
     (x.head === :if) + sum(last, parts; init = 0))
end
function data(n, k; deficient = false)
    rng = Xoshiro(711 + n + k)
    X = hcat(ones(n), randn(rng, n, k - 1))
    deficient && (X[:, end] .= X[:, 1])
    y = randn(rng, n)
    (; X, y, beta = fill(0.2, k))
end
have(spec) = (spec.X, spec.y, spec.beta)
function prepared(g, spec, d; joint = false)
    want = joint ? (spec.total, spec.residual) : (spec.total,)
    prepare(g; have = have(spec), want, bound = (spec.X => d.X, spec.y => d.y),
        bound_statistics = false)
end

println((stage = :environment, julia = VERSION, enzyme = pkgversion(Enzyme),
    enzyme_source = pkgdir(Enzyme), library = Enzyme.Compiler.Enzyme_jll.libEnzyme_path))
@testset "discovery before planning" begin
    @test length(discover(plain_rss.graph, (plain_rss.X, plain_rss.y))) == 1
    @test isempty(discover(plain_rss.graph, (plain_rss.X,)))
    @test isempty(discover(converted_rss.graph, (converted_rss.X, converted_rss.y)))
    @test isempty(discover(nonlinear_rss.graph, (nonlinear_rss.X, nonlinear_rss.y)))
end

@testset "fixed candidate graph, runtime eligibility" begin
    a = expose(plain_rss; evaluations = 100, row_cost = 10.0,
        preparation_cost = 20.0, statistics_cost = 2.0)
    @test length(a.found) == 1
    @test length(plain_rss.graph.recipes) == 3  # discovery leaves source unchanged
    kernels, structures = [], []
    for (n, k, deficient) in ((0, 3, false), (5, 3, false), (6, 3, false),
            (40, 3, false), (400, 3, true), (4000, 3, false), (40, 16, false))
        d = data(n, k; deficient)
        original = deepcopy((d.X, d.y, d.beta))
        q = prepared(a.g, plain_rss, d)
        f = b -> first(q(b))
        g = zeros(k)
        value = reverse!(f, g, d.beta)
        @test value ≈ row_rss(d.X, d.y, d.beta) atol = 1e-13 rtol = 1e-12
        @test g ≈ 2d.X' * (d.X * d.beta - d.y) atol = 1e-11 rtol = 1e-11
        @test (d.X, d.y, d.beta) == original
        @test_throws DimensionMismatch q(zeros(k + 1))
        push!(kernels, q)
        push!(structures, structure(code_expr(q)))
        s = only(filter(x -> x isa RK._LSStatistics, q.constants))
        @test s.usable == (n >= 2k && !deficient)
        println((stage = :binding, n, k, deficient, usable = s.usable,
            structure = last(structures)))
    end
    @test allequal(typeof.(kernels))
    @test allequal(structures)
    d = data(40, 3)
    joint = prepared(a.g, plain_rss, d; joint = true)
    @test !any(r -> r.op === guarded_rss, joint.plan.recipes)
    @test last(joint(d.beta)) ≈ d.y - d.X * d.beta
    bad = (; X = ones(5, 3), y = ones(4), beta = ones(3))
    q = prepared(a.g, plain_rss, bad)
    @test_throws DimensionMismatch q(bad.beta)
    # Print the actual generated expression. Its calls retain runtime work:
    # guarded_rss -> _ls_rss_dense/_ls_block_norm loops over s.k, or row_rss
    # invokes runtime matrix multiplication, subtraction and reduction.
    println("Prepared candidate expression (runtime helper calls retained):")
    println(code_expr(first(kernels)))
end

# Calibration is native ordinary Reverse plus its primal, and is performed once
# per explicit experiment profile. It includes Julia allocations; it does not
# predict a nonallocating backend, cold compilation or every parameter point.
@testset "measured objective and amortization" begin
    for (n, k) in ((40, 3), (4000, 3), (40, 16))
        d = data(n, k)
        rows = prepared(plain_rss.graph, plain_rss, d)
        rowf = b -> first(rows(b))
        forced = expose(plain_rss; evaluations = 100, row_cost = 10.0,
            preparation_cost = 20.0, statistics_cost = 2.0)
        statistics = prepared(forced.g, plain_rss, d)
        statf = b -> first(statistics(b))
        grad = zeros(k)
        row_ns = median_ns(() -> reverse!(rowf, grad, d.beta))
        stat_ns = median_ns(() -> reverse!(statf, grad, d.beta))
        prep_ns = median_ns(() -> candidate_statistics(d.X, d.y); iterations = 3)
        crossing = row_ns > stat_ns ? prep_ns / (row_ns - stat_ns) : Inf
        horizons = isfinite(crossing) ? (1, max(2, ceil(Int, 2crossing))) : (1, 1000)
        for evaluations in horizons, joint in (false, true)
            a = expose(plain_rss; evaluations, row_cost = row_ns,
                preparation_cost = prep_ns, statistics_cost = stat_ns)
            want = joint ? (plain_rss.total, plain_rss.residual) : (plain_rss.total,)
            p = plan(a.g; have = have(plain_rss), want)
            selected = any(r -> r.op === guarded_rss, p.recipes)
            @test selected == (!joint && stat_ns + prep_ns / evaluations < row_ns)
            q = prepared(a.g, plain_rss, d; joint)
            f = b -> first(q(b))
            @test reverse!(f, grad, d.beta) ≈ row_rss(d.X, d.y, d.beta) rtol = 1e-12
            @test grad ≈ 2d.X' * (d.X * d.beta - d.y) rtol = 1e-11
            println((stage = :measured_selection, n, k, evaluations, joint,
                selected_statistics = selected, row_ns, stat_ns, prep_ns, crossing,
                declared_cost_ns = p.cost,
                plan_ns = median_ns(() -> plan(a.g; have = have(plain_rss), want)),
                bind_ns = median_ns(() -> prepared(a.g, plain_rss, d; joint); iterations = 3),
                actual_reverse_ns = median_ns(() -> reverse!(f, grad, d.beta)),
                actual_primal_ns = median_ns(() -> q(d.beta)),
                julia_bytes = @allocated(reverse!(f, grad, d.beta))))
        end
    end
end
end # module
