using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels,
    ReactiveKernelsPPL, Test
using LinearAlgebra: det

# An undotted cell latent `c[i] ~ F(args…)` whose caller-owned geometry is not
# one scalar coordinate states one draw of `F(args…)` at each index: a value of
# the geometry's shape there (which may follow from the index's data), under
# ONE density of the whole value. Each index packs the geometry's coordinates,
# and its cell constrains them and adds the density and log-Jacobian.

module _CJLFixtures
using ReactiveKernelsPPL
import Distributions
using LogExpFunctions: logsumexp

# Stan's isometric sum-to-zero basis: K × (K - 1), orthonormal columns
# orthogonal to the ones vector.
function s2z_basis(K::Integer)
    V = zeros(K, max(K - 1, 0))
    for j in 1:(K - 1)
        a = 1 / sqrt(j * (j + 1))
        V[1:j, j] .= a
        V[j + 1, j] = -j * a
    end
    return V
end

# The values with log total `c[1]` and split `c[2:K]`.
function total_values(c::AbstractVector)
    K = length(c)
    K == 0 && return zeros(eltype(c), 0)
    z = s2z_basis(K) * c[2:K]
    return c[1] - logsumexp(z) .+ z
end
function total_coordinates(eta::AbstractVector)
    isempty(eta) && return zeros(eltype(eta), 0)
    return vcat(logsumexp(eta), s2z_basis(length(eta))' * (eta .- sum(eta) / length(eta)))
end

# The reporter's spelling: the free coordinates are the latent (identity
# geometry), and the law reads the values they give. The family is a type
# parameter, so the density is concretely typed (a `family::F` field holding
# `Normal`, a `UnionAll`, is not, and Enzyme Reverse then refuses the cell).
struct TotalLogSumExp{F,L,S}
    loc::L
    scale::S
end
TotalLogSumExp(::Type{F}, loc, scale) where {F} =
    TotalLogSumExp{F,typeof(loc),typeof(scale)}(loc, scale)
ReactiveKernelsPPL.sampling_logdensity(d::TotalLogSumExp{F}, c) where {F} =
    sum(Distributions.logpdf.(F.(d.loc, d.scale), total_values(c)); init = 0.0)
ReactiveKernelsPPL.sampling_geometry(::Type{TotalLogSumExp}, args, shape) =
    ParameterGeometry((:(length($(args[2]))),), :(length($(args[2]))); support = :real,
        constrain = (u, shape, args...) -> collect(u),
        unconstrain = (c, shape, args...) -> collect(c),
        logjac = (u, shape, args...) -> 0.0)

# The latent is the values themselves, sampled in total coordinates: a
# constant Jacobian, √K per index. As in the docs ("One joint latent per
# index").
struct TotalCoordinates{L,S}
    loc::L
    scale::S
end
ReactiveKernelsPPL.sampling_logdensity(d::TotalCoordinates, eta) =
    sum(Distributions.logpdf.(Distributions.Normal.(d.loc, d.scale), eta); init = 0.0)
ReactiveKernelsPPL.sampling_geometry(::Type{TotalCoordinates}, args, shape) =
    ParameterGeometry((:(length($(args[1]))),), :(length($(args[1]))); support = :real,
        constrain = (u, shape, args...) -> total_values(u),
        unconstrain = (eta, shape, args...) -> total_coordinates(eta),
        logjac = (u, shape, args...) -> isempty(u) ? 0.0 : log(length(u)) / 2)

# A vector summing to zero: K values from K - 1 coordinates.
struct ZeroSum{L,S}
    loc::L
    scale::S
end
ReactiveKernelsPPL.sampling_logdensity(d::ZeroSum, x) =
    sum(Distributions.logpdf.(Distributions.Normal.(d.loc, d.scale), x); init = 0.0)
ReactiveKernelsPPL.sampling_geometry(::Type{ZeroSum}, args, shape) =
    ParameterGeometry((:(length($(args[1]))),), :(length($(args[1])) - 1); support = :zero_sum,
        constrain = (u, shape, args...) -> s2z_basis(only(shape)) * u,
        unconstrain = (x, shape, args...) -> s2z_basis(only(shape))' * x,
        logjac = (u, shape, args...) -> 0.0)

# An m × 2 block of values above a live lower bound, at a rate given as a
# keyword: a matrix per index, a live parent and a keyword in every endpoint.
struct ShiftedBlock{M,L,R}
    m::M
    lower::L
    rate::R
end
ShiftedBlock(m, lower; rate = 1.0) = ShiftedBlock(m, lower, rate)
ReactiveKernelsPPL.sampling_logdensity(d::ShiftedBlock, x) =
    sum(log(d.rate) - d.rate * (v - d.lower) for v in x; init = 0.0)
# The argument tuple keeps Julia's leading `Expr(:parameters, …)` keyword entry.
_positional(args) = filter(a -> !Meta.isexpr(a, :parameters), collect(args))
ReactiveKernelsPPL.sampling_geometry(::Type{ShiftedBlock}, args, shape) =
    ParameterGeometry((_positional(args)[1], 2), :(2 * $(_positional(args)[1]));
        support = :lower_bounded,
        constrain = (u, shape, m, lower; rate = 1.0) -> lower .+ reshape(exp.(u), shape) ./ rate,
        unconstrain = (x, shape, m, lower; rate = 1.0) -> vec(log.(rate .* (x .- lower))),
        logjac = (u, shape, m, lower; rate = 1.0) -> sum(u; init = 0.0) - length(u) * log(rate))

# A geometry whose shape reads a sampled value: not data, refused.
struct LiveShape{N}
    n::N
end
ReactiveKernelsPPL.sampling_logdensity(::LiveShape, x) = 0.0
ReactiveKernelsPPL.sampling_geometry(::Type{LiveShape}, args, shape) =
    ParameterGeometry((args[1],), args[1]; support = :real,
        constrain = (u, shape, n) -> collect(u), unconstrain = (x, shape, n) -> collect(x),
        logjac = (u, shape, n) -> 0.0)
end

_cjl_fd(f, u; h = cbrt(eps(Float64))) = [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end for i in eachindex(u)]

_cjl_q(built, bound, preset, u) = Base.invokelatest(prepare_query(built, bound, preset), u)

_cjl_structure(program) = [(e.kind, e.depth) for e in recipe_inventory(program)
    if e.kind !== :ordinary]

_cjl_blocks(u, lengths) = [u[(sum(lengths[1:(i - 1)]; init = 0) + 1):sum(lengths[1:i])]
    for i in eachindex(lengths)]

function _cjl_check(built, bound, u, oracle)
    sampler = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
    value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ oracle(u) rtol = 1e-12
    @test gradient ≈ _cjl_fd(oracle, u) rtol = 1e-6 atol = 1e-8
    return nothing
end

# The reporter's model: one latent vector per sector, of the length of its
# data, whose observed total reads the values its total coordinates give.
const _CJL_TOTALS = quote
    @plate for i in eachindex(total)
        loc = log.(mu[i])
        c[i] ~ _CJLFixtures.TotalLogSumExp(Distributions.Normal, loc, 0.5)
        eta = _CJLFixtures.total_values(c[i])
        total[i] ~ Normal(sum(exp.(eta)), 0.01 * total[i])
    end
end

function _cjl_totals_terms(u, d)
    prior = likelihood = 0.0
    for (i, c) in enumerate(_cjl_blocks(u, length.(d.mu)))
        eta = _CJLFixtures.total_values(c)
        prior += sum(logpdf.(Normal.(log.(d.mu[i]), 0.5), eta); init = 0.0)
        likelihood += logpdf(Normal(sum(exp.(eta); init = 0.0), 0.01 * d.total[i]), d.total[i])
    end
    return prior, likelihood
end

@testset "a joint cell latent: one vector per index under one density" begin
    for d in ((; mu = [[10.0, 20.0, 5.0], [7.0, 3.0]], total = [35.2, 9.9]),
              (; mu = [[4.0], Float64[], [2.0, 6.0, 1.0, 3.0], [8.0, 8.0]],
                 total = [4.1, 0.2, 12.3, 15.6]))
        saved = deepcopy(d)
        n = sum(length, d.mu)
        from_values = bind_data(lower_rkppl(_CJL_TOTALS, d; conditioned = (:total,)), d)
        from_names = bind_data(lower_rkppl(_CJL_TOTALS, keys(d); conditioned = (:total,)), d)
        for bound in (from_values, from_names)
            built = build_kernel(bound)
            @test built.layout.total == n
            @test coordinate_names(built.layout) == [Symbol("c.", i, ".", j)
                for i in eachindex(d.mu) for j in eachindex(d.mu[i])]
            for u in (collect(range(2.4, 1.1; length = n)), [log(sum(m)) + 0.1 * j
                    for m in d.mu for j in eachindex(m)])
                nt = constrain(built.layout, u)
                @test nt.c == _cjl_blocks(u, length.(d.mu))
                @test unconstrain(built.layout, nt) ≈ u
                @test logjac(built.layout, u) == 0
                prior, likelihood = _cjl_totals_terms(u, d)
                @test _cjl_q(built, bound, :prior, u) ≈ prior rtol = 1e-12
                @test _cjl_q(built, bound, :likelihood, u) ≈ likelihood rtol = 1e-12
                @test _cjl_q(built, bound, :log_jacobian, u) == 0
                _cjl_check(built, bound, u, v -> sum(_cjl_totals_terms(v, d)))
            end
            draws = restore_draws(built.layout, hcat(zeros(n), ones(n), fill(2.0, n)))
            @test [size(x) for x in draws.c] == [(length(m), 3) for m in d.mu]
        end
        @test d == saved
    end
end

# The latent is the values, sampled in total coordinates: its geometry's
# constant log-Jacobian, √K per index, is in `log_jacobian`.
const _CJL_COORDINATES = quote
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(total)
        eta[i] ~ _CJLFixtures.TotalCoordinates(log.(mu[i]), 0.5)
        T = exp.(eta[i])
        y[i] .~ Normal.(T[obs[i]], sigma)
        total[i] ~ Normal(sum(T), 0.002 * total[i])
    end
end

@testset "a joint cell latent's geometry constrains its coordinates" begin
    d = (; mu = [[10.0, 20.0, 5.0], [7.0, 3.0], [4.0]], obs = [[1, 2], [1], [1]],
        y = [[11.0, 19.0], [6.5], [4.2]], total = [36.0, 10.5, 4.1])
    lengths = length.(d.mu)
    function terms(u)
        sigma = exp(u[1])
        prior, likelihood = logpdf(Exponential(1.0), sigma), 0.0
        for (i, c) in enumerate(_cjl_blocks(u[2:end], lengths))
            eta = _CJLFixtures.total_values(c)
            prior += sum(logpdf.(Normal.(log.(d.mu[i]), 0.5), eta))
            likelihood += sum(logpdf.(Normal.(exp.(eta[d.obs[i]]), sigma), d.y[i]))
            likelihood += logpdf(Normal(sum(exp.(eta)), 0.002 * d.total[i]), d.total[i])
        end
        return prior, likelihood, u[1] + sum(log(K) / 2 for K in lengths)
    end
    # The Jacobian determinant of the coordinates → values map is √K,
    # independently of the point.
    for K in (1, 3, 4), c in ([0.3, -0.2, 0.5, 0.1][1:K], [1.2, 0.4, -0.7, 0.9][1:K])
        J = _CJLFixtures.total_values(c)
        jac = reduce(hcat, [_cjl_fd(v -> _CJLFixtures.total_values(v)[r], c) for r in 1:K])'
        @test log(abs(det(jac))) ≈ log(K) / 2 atol = 1e-7
    end
    bound = bind_data(lower_rkppl(_CJL_COORDINATES, d; conditioned = (:y, :total)), d)
    built = build_kernel(bound)
    @test built.layout.total == 1 + sum(lengths)
    for u in ([0.1, 3.6, 0.2, -0.5, 2.3, 0.4, 1.4], [-0.3, 3.5, -0.1, 0.3, 2.4, -0.2, 1.4])
        prior, likelihood, jacobian = terms(u)
        nt = constrain(built.layout, u)
        @test nt.eta ≈ [_CJLFixtures.total_values(c) for c in _cjl_blocks(u[2:end], lengths)]
        @test sum.(exp, nt.eta) ≈ [exp(c[1]) for c in _cjl_blocks(u[2:end], lengths)]
        @test unconstrain(built.layout, nt) ≈ u
        @test logjac(built.layout, u) ≈ jacobian
        @test _cjl_q(built, bound, :prior, u) ≈ prior rtol = 1e-12
        @test _cjl_q(built, bound, :likelihood, u) ≈ likelihood rtol = 1e-12
        @test _cjl_q(built, bound, :log_jacobian, u) ≈ jacobian rtol = 1e-12
        _cjl_check(built, bound, u, v -> sum(terms(v)))
    end
end

@testset "a joint cell latent packs other dimensions than its values" begin
    ast = quote
        s ~ Exponential(1.0)
        @plate for i in eachindex(x)
            z[i] ~ _CJLFixtures.ZeroSum(x[i], s)
            y[i] .~ Normal.(z[i], 0.3)
        end
    end
    d = (; x = [[0.1, -0.2, 0.4], [0.3, 0.1], [0.5]], y = [[0.2, -0.4, 0.1], [0.2, -0.1], [0.05]])
    lengths = length.(d.x)
    bound = bind_data(lower_rkppl(ast, d; conditioned = (:y,)), d)
    built = build_kernel(bound)
    @test built.layout.total == 1 + sum(lengths .- 1)
    @test coordinate_names(built.layout) == [:s, Symbol("z.1.1"), Symbol("z.1.2"), Symbol("z.2.1")]
    function oracle(u)
        s = exp(u[1])
        lp = logpdf(Exponential(1.0), s) + u[1]
        for (i, w) in enumerate(_cjl_blocks(u[2:end], lengths .- 1))
            z = _CJLFixtures.s2z_basis(lengths[i]) * w
            lp += sum(logpdf.(Normal.(d.x[i], s), z)) + sum(logpdf.(Normal.(z, 0.3), d.y[i]))
        end
        return lp
    end
    for u in ([0.2, 0.3, -0.1, 0.4], [-0.5, -0.2, 0.6, 0.1])
        nt = constrain(built.layout, u)
        @test length.(nt.z) == lengths
        @test all(z -> abs(sum(z)) < 1e-12, nt.z)
        @test unconstrain(built.layout, nt) ≈ u
        _cjl_check(built, bound, u, oracle)
    end
end

@testset "a joint cell latent: matrices, live parents and keywords" begin
    ast = quote
        lower ~ Normal(0.0, 1.0)
        rate ~ Exponential(1.0)
        @plate for i in eachindex(m)
            B[i] ~ _CJLFixtures.ShiftedBlock(m[i], lower + 0.5; rate = rate)
            y[i] .~ Normal.(B[i][:, 1] .- B[i][:, 2], 1.0)
        end
    end
    d = (; m = [2, 0, 3], y = [[0.3, -0.1], Float64[], [0.2, 0.0, 0.4]])
    bound = bind_data(lower_rkppl(ast, d; conditioned = (:y,)), d)
    built = build_kernel(bound)
    @test built.layout.total == 2 + 2 * sum(d.m)
    function terms(u)
        lower, rate = u[1], exp(u[2])
        prior = logpdf(Normal(0.0, 1.0), lower) + logpdf(Exponential(1.0), rate)
        likelihood, jacobian = 0.0, u[2]
        for (i, w) in enumerate(_cjl_blocks(u[3:end], 2 .* d.m))
            B = lower + 0.5 .+ reshape(exp.(w), d.m[i], 2) ./ rate
            prior += sum(log(rate) - rate * (b - lower - 0.5) for b in B; init = 0.0)
            jacobian += sum(w; init = 0.0) - length(w) * log(rate)
            likelihood += sum(logpdf.(Normal.(B[:, 1] .- B[:, 2], 1.0), d.y[i]); init = 0.0)
        end
        return prior, likelihood, jacobian
    end
    for u in (collect(range(-0.3, 0.4; length = 12)), collect(range(0.5, -0.6; length = 12)))
        prior, likelihood, jacobian = terms(u)
        nt = constrain(built.layout, u)
        @test [size(B) for B in nt.B] == [(2, 2), (0, 2), (3, 2)]
        @test unconstrain(built.layout, nt) ≈ u
        @test logjac(built.layout, u) ≈ jacobian
        @test _cjl_q(built, bound, :prior, u) ≈ prior rtol = 1e-12
        @test _cjl_q(built, bound, :log_jacobian, u) ≈ jacobian rtol = 1e-12
        _cjl_check(built, bound, u, v -> sum(terms(v)))
    end
end

@testset "a joint cell latent: rebinding, dotted readers and retained structure" begin
    small = (; mu = [[10.0, 20.0], [7.0]], total = [30.5, 7.2])
    large = (; mu = [[10.0, 20.0, 5.0], [7.0, 3.0], [1.0, 2.0, 3.0, 4.0], [2.0], [9.0, 1.0]],
        total = [35.2, 9.9, 10.1, 2.1, 10.3])
    plan = lower_rkppl(_CJL_TOTALS, keys(small); conditioned = (:total,))
    structures = map((small, large)) do d
        bound = bind_data(plan, d)
        built = build_kernel(bound)
        u = collect(range(2.2, 0.9; length = sum(length, d.mu)))
        @test built.layout.total == length(u)
        @test _cjl_q(built, bound, :sampler, u) ≈ sum(_cjl_totals_terms(u, d)) rtol = 1e-12
        _cjl_structure(built.spec)
    end
    @test structures[1] == structures[2]
    # A dotted cell latent reads the joint latent's value at the index, of its
    # geometry's shape there.
    ast = quote
        @plate for i in eachindex(x)
            z[i] ~ _CJLFixtures.ZeroSum(x[i], 1.0)
            v[i] .~ Normal.(z[i], 0.5)
        end
    end
    d = (; x = [[0.1, -0.2, 0.4], [0.3, 0.1]])
    bound = bind_data(lower_rkppl(ast, d), d)
    built = build_kernel(bound)
    @test built.layout.total == 3 + 5
    u = [0.3, -0.1, 0.4, 0.2, -0.3, 0.1, 0.5, -0.2]
    nt = constrain(built.layout, u)
    @test length.(nt.v) == [3, 2]
    oracle = w -> begin
        lp = 0.0
        for (i, z) in enumerate([_CJLFixtures.s2z_basis(3) * w[1:2], _CJLFixtures.s2z_basis(2) * w[3:3]])
            vi = i == 1 ? w[4:6] : w[7:8]
            lp += sum(logpdf.(Normal.(d.x[i], 1.0), z)) + sum(logpdf.(Normal.(z, 0.5), vi))
        end
        lp
    end
    _cjl_check(built, bound, u, oracle)
end

@testset "a joint cell latent's shape must follow from data" begin
    # refused: a geometry's shape and packed dimension are data-only
    # expressions evaluated at binding (`ParameterGeometry` contract, rkppl-use §2)
    ast = quote
        n ~ Exponential(1.0)
        @plate for i in eachindex(x)
            w[i] ~ _CJLFixtures.LiveShape(n)
            x[i] ~ Normal(sum(w[i]), 1.0)
        end
    end
    @test_throws ContractValidationError bind_data(lower_rkppl(ast, (; x = [0.1, 0.2]);
        conditioned = (:x,)), (; x = [0.1, 0.2]))
    # Not built yet: joint cell latents over `levels(…)` (refused at lowering,
    # naming the iterators that work).
    levels_ast = quote
        @plate for k in levels(g)
            z[k] ~ _CJLFixtures.ZeroSum(x, 1.0)
        end
    end
    @test_broken (lower_rkppl(levels_ast, (; g = [1, 2], x = [0.1, 0.2])); true)
end
