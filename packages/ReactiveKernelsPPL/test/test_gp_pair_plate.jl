using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Test
import Enzyme
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    gp_exp_quad_cov, gp_periodic_cov, gp_chol_latent, gp_exp_quad_cov_graph, gp_periodic_cov_graph

module GPPairPlateFixtures
using ReactiveKernels, ReactiveKernelsPPL
using Distributions: Normal, LogNormal, logpdf
using LinearAlgebra: Symmetric, cholesky
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    gp_exp_quad_cov, gp_periodic_cov, gp_chol_latent, gp_exp_quad_cov_graph, gp_periodic_cov_graph

# Independent scalar formulas; jitter is positional, including duplicate x.
exp_quad(x, s, r, j) = [s^2 * exp(-(a-b)^2 / (2r^2)) + (i == k ? j : 0.0)
    for (i, a) in enumerate(x), (k, b) in enumerate(x)]
periodic(x, s, r, p, j) = [s^2 * exp(-2sin(pi * abs(a-b) / p)^2 / r^2) + (i == k ? j : 0.0)
    for (i, a) in enumerate(x), (k, b) in enumerate(x)]

function findiff(f, q; h = 1e-6)
    [(f(q + h * (eachindex(q) .== i)) - f(q - h * (eachindex(q) .== i))) / (2h)
        for i in eachindex(q)]
end

exp_loss(q) = sum(gp_exp_quad_cov(q[4:end], q[1], q[2], q[3]))
periodic_loss(q) = sum(gp_periodic_cov(q[5:end], q[1], q[2], q[3], q[4]))

# These are ordinary HAVE cuts of the same production graphs. Validation
# converts public scalar arguments into variance/denominator/jitter; these cuts
# take those values directly, so backend acceptance measures the pair plate
# without claiming that Reactant supports live Julia throwing validators.
const EXP_CUT = prepare(gp_exp_quad_cov_graph;
    have = (:x, :variance, :denominator, :jit), want = :covariance)
const PERIODIC_CUT = prepare(gp_periodic_cov_graph;
    have = (:x, :variance, :squared_width, :per, :jit), want = :covariance)

@kernel exp_cut_loss(q::Vector{Float64}) = begin
    x = q[4:end]
    variance = q[1]
    denominator = q[2]
    jit = q[3]
    covariance = EXP_CUT(x, variance, denominator, jit)
    loss = sum(covariance)
    return loss
end
@kernel periodic_cut_loss(q::Vector{Float64}) = begin
    x = q[5:end]
    variance = q[1]
    squared_width = q[2]
    per = q[3]
    jit = q[4]
    covariance = PERIODIC_CUT(x, variance, squared_width, per, jit)
    loss = sum(covariance)
    return loss
end
@kernel exp_data_cut_loss(x, q::Vector{Float64}) = begin
    variance = q[1]
    denominator = q[2]
    jit = q[3]
    covariance = EXP_CUT(x, variance, denominator, jit)
    loss = sum(covariance)
    return loss
end
@kernel periodic_data_cut_loss(x, q::Vector{Float64}) = begin
    variance = q[1]
    squared_width = q[2]
    per = q[3]
    jit = q[4]
    covariance = PERIODIC_CUT(x, variance, squared_width, per, jit)
    loss = sum(covariance)
    return loss
end

function model(n; periodic = false, live_locations = false)
    x = collect(range(-0.7, 1.1; length = n))
    locations = live_locations ? :(x .* stretch) : :x
    covariance = periodic ? :(gp_periodic_cov($locations, sigma, rho, 1.3, 1e-5)) :
        :(gp_exp_quad_cov($locations, sigma, rho, 1e-5))
    ast = quote
        sigma ~ LogNormal(0, 1)
        rho ~ LogNormal(0, 1)
        stretch ~ Normal(0, 1)
        @plate for i in eachindex(y)
            z[i] ~ Normal(0, 1)
        end
        f = gp_chol_latent($covariance, z)
        y .~ Normal.(f[oi], 0.5)
    end
    data = Dict{Symbol,Any}(:x => x, :y => sin.(x), :oi => collect(1:n))
    bound = bind_data(lower_rkppl(ast, data; mod = @__MODULE__, conditioned = (:y,)), data)
    built = build_kernel(bound)
    bound, built
end

function model_oracle(bound, built, u; periodic, live_locations)
    nt = constrain(built.layout, u)
    x = bound.columns[:x]
    locations = live_locations ? x .* nt.stretch : x
    K = periodic ? GPPairPlateFixtures.periodic(locations, nt.sigma, nt.rho, 1.3, 1e-5) :
        exp_quad(locations, nt.sigma, nt.rho, 1e-5)
    f = cholesky(Symmetric(K)).L * nt.z
    jac = sum(u[i] for (i, name) in enumerate(coordinate_names(built.layout))
        if name in (:sigma, :rho))
    sum(logpdf.(Normal.(f, 0.5), bound.columns[:y])) +
        sum(logpdf.(Normal(), nt.z)) + logpdf(Normal(), nt.stretch) +
        logpdf(LogNormal(), nt.sigma) + logpdf(LogNormal(), nt.rho) + jac
end

plates(p) = filter(r -> recipe_kind(r) === :plate, p.recipes)
names(p) = [only(r.outputs).name for r in p.recipes]
end

@testset "GP pair plates: covariance, axes, jitter and native AD" begin
    F = GPPairPlateFixtures
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    x = [-0.2, 0.3, 0.3, 1.1]
    original = copy(x)
    for locations in (x, view(x, 1:4), range(-0.3, 1.0; length = 4), [0.2],
            Float32.(x), [1, 2, 2, 4])
        K = gp_exp_quad_cov(locations, 1.7, 0.8, 1e-4)
        P = gp_periodic_cov(locations, 1.7, 0.8, 1.3, 1e-4)
        @test K ≈ F.exp_quad(locations, 1.7, 0.8, 1e-4)
        @test P ≈ F.periodic(locations, 1.7, 0.8, 1.3, 1e-4)
        @test axes(K) == axes(P) == (axes(locations, 1), axes(locations, 1))
        @test K ≈ transpose(K)
        @test P ≈ transpose(P)
        @test all(K[i, i] ≈ 1.7^2 + 1e-4 for i in axes(K, 1))
        @test all(P[i, i] ≈ 1.7^2 + 1e-4 for i in axes(P, 1))
    end
    @test gp_exp_quad_cov(x, 1.7, 0.8, 1e-4)[2, 3] ≈ 1.7^2
    @test gp_periodic_cov(x, 1.7, 0.8, 1.3, 1e-4)[2, 3] ≈ 1.7^2
    for (f, q, oracle) in (
            (F.exp_loss, vcat([1.7, 0.8, 1e-4], x), w -> sum(F.exp_quad(w[4:end], w[1], w[2], w[3]))),
            (F.periodic_loss, vcat([1.7, 0.8, 1.3, 1e-4], x), w -> sum(F.periodic(w[5:end], w[1], w[2], w[3], w[4]))))
        before = copy(q)
        @test f(q) ≈ oracle(q)
        @test gradient(f, backend, q) ≈ F.findiff(oracle, q) rtol = 1e-5 atol = 1e-7
        @test q == before
    end
    @test x == original
end

@testset "GP pair plates: data distance cache and live locations" begin
    F = GPPairPlateFixtures
    for (graph, distance) in ((gp_exp_quad_cov_graph, :squared_distance), (gp_periodic_cov_graph, :distance))
        @test distance in F.names(plate_body(only(F.plates(plan(graph)))))
        counts = Int[]
        for n in (8, 32)
            x = collect(range(-0.7, 1.1; length = n))
            bound = prepare(graph; bound = (; x))
            recipe = only(F.plates(bound.plan))
            body = plate_body(recipe)
            @test distance ∉ F.names(body)
            @test any(v -> occursin("bound_plate_", String(v.name)), recipe.inputs)
            push!(counts, length(body.recipes))
        end
        @test counts[1] == counts[2]
    end
    for periodic in (false, true), live_locations in (false, true)
        bound, built = F.model(6; periodic, live_locations)
        cov = only(filter(r -> :distance in F.names(plate_body(r)), F.plates(plan(built.spec))))
        @test length(plate_body(cov).have) >= 7  # both locations and position indices
        kernel = prepare_query(built, bound, :sampler)
        # Data-only geometry disappears from the residual cell; parameter
        # locations keep it even when the source data x is bound.
        bodies = map(plate_body, F.plates(kernel.plan))
        @test any(p -> :distance in F.names(p), bodies) == live_locations
        u = [0.1sin(i) for i in 1:built.layout.total]
        oracle = w -> F.model_oracle(bound, built, w; periodic, live_locations)
        @test kernel(u) ≈ oracle(u) rtol = 1e-10
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        ad = prepare_ad(kernel, backend, u; active = :unconstrained)
        @test ad_gradient(ad, u) ≈ F.findiff(oracle, u) rtol = 1e-5 atol = 1e-7
    end
end
