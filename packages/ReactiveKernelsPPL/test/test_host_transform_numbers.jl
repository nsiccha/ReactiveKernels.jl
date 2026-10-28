using ReactiveKernelsPPL
using Test
using LinearAlgebra

# Host `constrain`, `unconstrain` and `logjac` follow the number type of their
# input (snag rkppl-constrain-aeffb8d7). Plain reals keep `Float64` results;
# wider or number-generic inputs pass through. BigFloat exercises the same
# generic path as dual numbers, and its central differences give a
# high-precision Jacobian without an AD engine.

module HostTransformGeometry
using ReactiveKernelsPPL
struct ShiftedExp{T}
    lower::T
end
ReactiveKernelsPPL.sampling_logdensity(d::ShiftedExp, x) =
    x > d.lower ? -(x - d.lower) : -Inf
ReactiveKernelsPPL.sampling_geometry(::Type{ShiftedExp}, args, shape) =
    ParameterGeometry(shape, 1; support = (:lower, args[1]),
        constrain = (u, shape, lower) -> lower + exp(u[1]),
        unconstrain = (x, shape, lower) -> [log(x - lower)],
        logjac = (u, shape, lower) -> u[1])
end

_host_flat(v::Number) = Any[v]
_host_flat(v::AbstractArray) = reduce(vcat, map(_host_flat, vec(v)); init = Any[])
_host_flat(v::NamedTuple) =
    reduce(vcat, map(_host_flat, collect(values(v))); init = Any[])
_host_values(layout, u) = [x for x in _host_flat(constrain(layout, u))]

# Central differences, column by column; `f` returns a vector.
function _host_fd(f, u, h)
    return reduce(hcat, map(eachindex(u)) do i
        step = h .* (eachindex(u) .== i)
        (f(u .+ step) .- f(u .- step)) ./ (2h)
    end)
end

function _host_built(ast, data)
    plan = bind_data(lower_rkppl(ast, data; mod = @__MODULE__,
        conditioned = (:y,)), data)
    return build_kernel(plan).layout
end

const _HOST_X = [0.1, 0.5, -0.3, 1.2]
const _HOST_Y = [0.3, 0.2, -0.1, 1.0]

# Built layout kinds: scalar supports with literal and sampled bounds,
# elementwise arrays, ordered/simplex vectors, LKJ factors, slice arrays, LKJ
# stacks, scans, plates and caller-owned geometry. Predictor coefficient
# runs are hand-built below.
const _HOST_MODELS = [
    "scalar supports and sampled bounds" => (quote
        a ~ Normal(0, 5)
        sg ~ Exponential(1)
        p ~ Beta(2, 2)
        w ~ Uniform(-1, 3)
        f ~ truncated(LogNormal(0, 1), 0.5, Inf)
        c ~ truncated(Normal(0, 1), -Inf, 2)
        lo ~ Normal(0, 1)
        b ~ Uniform(lo, lo + 3)
        t ~ truncated(Weibull(2 + exp(a), 1), exp(a), 3 + exp(a))
        mu = a .+ w .* x
        y .~ Normal.(mu, sg)
    end, Dict(:x => _HOST_X, :y => _HOST_Y)),
    "arrays, ordered, simplex and LKJ" => (quote
        z[1:3] .~ Exponential.(1)
        Z[1:2, 1:3] .~ Normal.(0, 1)
        q[1:2] .~ Uniform.(-2, 2)
        c ~ Ordered(Normal(0, 1), 3)
        phi ~ Dirichlet([1.0, 2.0, 3.0])
        L ~ LKJCholesky(3, 2.0)
        U ~ LKJCholesky(3, 2.0, 'U')
        sigma ~ Exponential(1)
        y .~ Normal.(0.0, sigma)
    end, Dict(:y => _HOST_Y)),
    "slice arrays and LKJ stacks" => (quote
        eachrow(P[levels(k), 1:3]) .~ Dirichlet([1.0, 2.0, 3.0])
        eachrow(C[levels(k), 1:3]) .~ Ordered(Normal(0, 2), 3)
        @plate for j in levels(s)
            L[j] ~ LKJCholesky(2, 1.0)
        end
        sigma ~ Exponential(1)
        y .~ Normal.(P[k, 1] .+ C[k, 1], sigma)
    end, Dict(:k => [1, 2, 2, 1], :s => [1, 1, 2, 2], :y => _HOST_Y)),
    "scans and plates" => (quote
        sigma ~ Exponential(1)
        @scan begin
            h[1] ~ Exponential(1)
            for t in 2:T
                h[t] ~ Exponential(h[t - 1])
            end
        end
        @plate for i in 1:3
            w[i] ~ Exponential(1)
        end
        y .~ Normal.(h .+ sum(w), sigma)
    end, Dict(:y => _HOST_Y)),
    "caller-owned geometry" => (quote
        a ~ Normal(0, 1)
        lower = a + 0.7
        b ~ HostTransformGeometry.ShiftedExp(lower)
        @plate for i in eachindex(y)
            e[i] ~ HostTransformGeometry.ShiftedExp(a)
        end
        y .~ Normal.(b, 1.0)
    end, Dict(:y => _HOST_Y)),
]

@testset "host transforms follow the input number type: $name" for (name, (ast, data)) in _HOST_MODELS
    layout = _host_built(ast, data)
    n = layout.total
    u = 0.4 .* sin.(1:n)
    before = copy(u)
    # Plain reals keep Float64 values. Package transforms read Float32 input
    # in Float64, as before; caller-owned geometry receives the packed
    # segment unchanged, as before.
    @test all(x -> x isa Float64, _host_values(layout, u))
    @test all(x -> x isa Float64, _host_values(layout, Float32.(u)))
    name == "caller-owned geometry" ||
        @test isequal(_host_values(layout, Float32.(u)),
            _host_values(layout, Float64.(Float32.(u))))
    @test unconstrain(layout, constrain(layout, u)) isa Vector{Float64}
    @test logjac(layout, u) isa Float64
    setprecision(BigFloat, 256) do
        wide = big.(u)
        values = _host_values(layout, wide)
        # Every leaf keeps BigFloat precision through the whole round trip.
        @test all(x -> x isa BigFloat, values)
        @test Float64.(values) ≈ _host_values(layout, u)
        back = unconstrain(layout, constrain(layout, wide))
        @test back isa Vector{BigFloat}
        @test maximum(abs.(back .- wide); init = big(0.0)) < big(1e-60)
        @test logjac(layout, wide) isa BigFloat
        @test Float64(logjac(layout, wide)) ≈ logjac(layout, u)
        # Derivatives of the generic transforms: BigFloat central differences
        # agree with Float64 central differences.
        J = _host_fd(v -> _host_values(layout, v), wide, big(1e-30))
        @test isapprox(Float64.(J), _host_fd(v -> _host_values(layout, v), u, 1e-6);
            atol = 1e-6, rtol = 1e-6)
        dback = _host_fd(v -> unconstrain(layout, constrain(layout, v)), wide, big(1e-30))
        @test maximum(abs.(dback - I); init = big(0.0)) < big(1e-40)
    end
    @test u == before
end

@testset "host logjac is the log-determinant of the constrain Jacobian" begin
    setprecision(BigFloat, 256) do
        h = big(1e-30)
        # Square maps: every constrained value is free.
        layout = _host_built(quote
            a ~ Normal(0, 5)
            sg ~ Exponential(1)
            p ~ Beta(2, 2)
            w ~ Uniform(-1, 3)
            lo ~ Normal(0, 1)
            b ~ Uniform(lo, lo + 3)
            t ~ truncated(Weibull(2 + exp(a), 1), exp(a), 3 + exp(a))
            z[1:3] .~ Exponential.(1)
            c ~ Ordered(Normal(0, 1), 3)
            y .~ Normal.(a, sg)
        end, Dict(:y => _HOST_Y))
        u = big.(0.4 .* cos.(1:layout.total))
        J = _host_fd(v -> _host_values(layout, v), u, h)
        @test abs(logabsdet(J)[1] - logjac(layout, u)) < big(1e-40)
        # Vector and factor transforms over their free coordinates.
        v = big.([0.3, -0.2, 0.5])
        @test abs(logabsdet(_host_fd(ordered_constrain, v, h))[1] - ordered_logjac(v)) < big(1e-40)
        @test abs(logabsdet(_host_fd(x -> simplex_constrain(x)[1:3], v, h))[1] -
            simplex_logjac(v)) < big(1e-40)
        lower(L) = [L[2, 1], L[3, 1], L[3, 2]]
        @test abs(logabsdet(_host_fd(x -> lower(lkj_chol_constrain(x, 3)), v, h))[1] -
            lkj_chol_logjac(v, 3)) < big(1e-40)
        for (c, inv) in ((ordered_constrain, ordered_unconstrain),
                (simplex_constrain, simplex_unconstrain),
                (x -> lkj_chol_constrain(x, 3), L -> lkj_chol_unconstrain(L, 3)))
            back = inv(c(v))
            @test back isa Vector{BigFloat}
            @test maximum(abs.(back .- v)) < big(1e-60)
        end
    end
    # Plain reals keep Float64 results.
    @test ordered_constrain([1, 2]) isa Vector{Float64}
    @test ordered_constrain(Float32[1, 2]) isa Vector{Float64}
    @test simplex_constrain(Float64[]) == [1.0]
    @test lkj_chol_constrain([0.1], 2) isa Matrix{Float64}
end

@testset "predictor coefficient runs follow the input number type" begin
    # One predictor in a single identity block; another whose coefficients
    # split into an identity run and an interval run.
    layout = LayoutTable([
        LayoutEntry(:coefficient, :eta, :eta_coef, [:Intercept, :x], 1, 2, :identity),
        LayoutEntry(:coefficient, :mu, :mu_coef__s1, [:Intercept], 3, 1, :identity),
        LayoutEntry(:coefficient, :mu, :mu_coef__s2, [:x], 4, 1, :interval, -2.0, 3.0),
    ], 4)
    u = [0.1, 0.2, 0.4, -0.7]
    s(v) = 1 / (1 + exp(-v))
    @test constrain(layout, u).eta == [0.1, 0.2]
    @test constrain(layout, u).mu ≈ [0.4, -2 + 5 * s(-0.7)]
    setprecision(BigFloat, 256) do
        wide = constrain(layout, big.(u))
        @test eltype(wide.eta) === BigFloat
        @test eltype(wide.mu) === BigFloat
        @test abs(wide.mu[2] - (-2 + 5 * s(big(-0.7)))) < big(1e-60)
        back = unconstrain(layout, wide)
        @test back isa Vector{BigFloat}
        @test maximum(abs.(back .- big.(u))) < big(1e-60)
        @test abs(logjac(layout, big.(u)) - log(5 * s(big(-0.7)) * (1 - s(big(-0.7))))) <
            big(1e-60)
    end
end
