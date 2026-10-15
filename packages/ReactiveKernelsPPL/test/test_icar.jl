module ICARTests
using Test, ReactiveKernels, ReactiveKernelsPPL
using Distributions: Beta, Normal, Poisson, logpdf, truncated
import DifferentiationInterface as DI
import Enzyme

# A synthetic r × c lattice: one node per cell, one edge per pair of
# horizontal or vertical neighbours, each edge listed once.
function lattice(r, c)
    node(i, j) = (j - 1) * r + i
    node1, node2 = Int[], Int[]
    for j in 1:c, i in 1:r
        i < r && (push!(node1, node(i, j)); push!(node2, node(i + 1, j)))
        j < c && (push!(node1, node(i, j)); push!(node2, node(i, j + 1)))
    end
    return node1, node2
end

# Independent oracle: the ICAR log density is -φ'Qφ/2 with Q the graph
# Laplacian (degrees on the diagonal, -1 per adjacency).
function icar_oracle(phi, node1, node2)
    Q = zeros(length(phi), length(phi))
    for (i, j) in zip(node1, node2)
        Q[i, i] += 1
        Q[j, j] += 1
        Q[i, j] -= 1
        Q[j, i] -= 1
    end
    return -0.5 * phi' * Q * phi
end

structure(program) = [(e.kind, e.depth) for e in recipe_inventory(program)
    if e.kind !== :ordinary]

function fd_check(sampler, u; step = 1e-6)
    gradient = similar(u)
    sampler_value_and_gradient!(sampler, gradient, u)
    for i in eachindex(u)
        plus, minus = copy(u), copy(u)
        plus[i] += step
        minus[i] -= step
        expected = (Base.invokelatest(sampler, plus) -
            Base.invokelatest(sampler, minus)) / (2step)
        @test gradient[i] ≈ expected atol = 1e-5 rtol = 1e-6
    end
end

const BYM2 = quote
    beta0 ~ Normal(0, 1)
    sigma ~ truncated(Normal(0, 1), 0, Inf)
    rho ~ Beta(0.5, 0.5)
    theta[1:N] .~ Normal.(0, 1)
    phi[1:N] ~ ICAR(node1, node2)
    phi_sum = sum(phi)
    phi_sum ~ Normal(0, 0.001 * N)
    convolved_re = sqrt(1 - rho) .* theta .+ sqrt(rho / scaling_factor) .* phi
    log_E = log.(E)
    y .~ Poisson.(exp.(log_E .+ beta0 .+ sigma .* convolved_re))
end

function bym2_data(r, c)
    node1, node2 = lattice(r, c)
    N = r * c
    E = [0.6 + 0.1 * mod(k, 7) for k in 1:N]
    y = [mod(3k, 5) for k in 1:N]
    return (; N, node1, node2, E, y, scaling_factor = 0.45)
end

function bym2_oracle(c, d)
    prior = logpdf(Normal(0, 1), c.beta0) +
        logpdf(truncated(Normal(0, 1), 0, Inf), c.sigma) +
        logpdf(Beta(0.5, 0.5), c.rho) + sum(logpdf.(Normal(0, 1), c.theta)) +
        icar_oracle(c.phi, d.node1, d.node2) +
        logpdf(Normal(0, 0.001 * d.N), sum(c.phi))
    re = sqrt(1 - c.rho) .* c.theta .+ sqrt(c.rho / d.scaling_factor) .* c.phi
    rate = exp.(log.(d.E) .+ c.beta0 .+ c.sigma .* re)
    return prior, sum(logpdf.(Poisson.(rate), d.y))
end

bind_bym2(d) = bind_data(lower_rkppl(BYM2, d; mod = @__MODULE__,
    conditioned = (:y,)), d)

@testset "ICAR prior: the pairwise-difference density as an RK graph law" begin
    d = bym2_data(3, 4)
    bound = bind_bym2(d)
    built = build_kernel(bound)
    names = coordinate_names(built.layout)
    # The field is centered: its entries are its coordinates.
    @test names[end - d.N + 1:end] == [Symbol("phi.", k) for k in 1:d.N]
    code = string(kernel_expr(bound, built.layout))
    @test occursin("icar_logdensity(phi, node1, node2)", code)
    @test !occursin("sampling_logdensity", code)
    prior = prepare_query(built, bound, :prior)
    likelihood = prepare_query(built, bound, :likelihood)
    for u in (collect(range(-0.4, 0.5; length = length(names))),
            [0.3, -0.2, 0.8, sin.(1:d.N)..., 0.01 .* cos.(1:d.N)...])
        c = constrain(built.layout, u)
        @test c.phi == u[end - d.N + 1:end]
        p, l = bym2_oracle(c, d)
        @test Base.invokelatest(prior, u) ≈ p rtol = 1e-12
        @test Base.invokelatest(likelihood, u) ≈ l rtol = 1e-12
    end
end

@testset "ICAR prior: native Enzyme Reverse" begin
    d = bym2_data(3, 3)
    bound = bind_bym2(d)
    built = build_kernel(bound)
    n = length(coordinate_names(built.layout))
    u = [0.2, -0.3, 0.4, 0.3 .* sin.(1:d.N)..., 0.02 .* cos.(1:d.N)...]
    @test length(u) == n
    sampler = prepare_sampler(built, bound, u;
        backend = DI.AutoEnzyme(; mode = Enzyme.Reverse))
    fd_check(sampler, u)
end

@testset "ICAR prior: one retained edge plate as the graph grows" begin
    small, large = bym2_data(2, 3), bym2_data(9, 11)
    shapes = map((small, large)) do d
        bound = bind_bym2(d)
        built = build_kernel(bound)
        u = zeros(length(coordinate_names(built.layout)))
        sampler = prepare_sampler(built, bound, u;
            backend = DI.AutoEnzyme(; mode = Enzyme.Reverse))
        structure(sampler.kernel)
    end
    @test shapes[1] == shapes[2]
    # The edge plate is one of them: a flat field with no ICAR prior has one
    # fewer plate.
    flat = quote
        phi[1:N] .~ Flat.()
        y .~ Poisson.(exp.(log.(E) .+ phi))
    end
    icar = quote
        phi[1:N] ~ ICAR(node1, node2)
        y .~ Poisson.(exp.(log.(E) .+ phi))
    end
    d = small
    plates(ast) = count(e -> e[1] === :plate, structure(build_kernel(bind_data(
        lower_rkppl(ast, d; mod = @__MODULE__, conditioned = (:y,)), d)).spec))
    @test plates(icar) == plates(flat) + 1
end

@testset "ICAR edge lists: data, data-only definitions and literals" begin
    node1, node2 = lattice(2, 3)
    phi0 = [0.3, -0.1, 0.7, 0.2, -0.4, 0.1]
    y = [1, 0, 2, 1, 3, 0]
    function prior_at(ast, data)
        bound = bind_data(lower_rkppl(ast, data; mod = @__MODULE__,
            conditioned = (:y,)), data)
        built = build_kernel(bound)
        q = prepare_query(built, bound, :prior)
        return Base.invokelatest(q, phi0)
    end
    expected = icar_oracle(phi0, node1, node2)
    @test prior_at(quote
        phi[1:6] ~ ICAR(node1, node2)
        y .~ Poisson.(exp.(phi))
    end, (; node1, node2, y)) ≈ expected rtol = 1e-13
    edges = hcat(node1, node2)
    @test prior_at(quote
        phi[1:6] ~ ICAR(edges[:, 1], edges[:, 2])
        y .~ Poisson.(exp.(phi))
    end, (; edges, y)) ≈ expected rtol = 1e-13
    @test prior_at(quote
        phi[1:6] ~ ICAR($(Expr(:vect, node1...)), $(Expr(:vect, node2...)))
        y .~ Poisson.(exp.(phi))
    end, (; y)) ≈ expected rtol = 1e-13
    @test prior_at(quote
        from = edges[:, 1]
        to = edges[:, 2]
        phi[1:6] ~ ICAR(from, to)
        y .~ Poisson.(exp.(phi))
    end, (; edges, y)) ≈ expected rtol = 1e-13
    # The field's length comes from its declaration, not from the edges:
    # nodes on no listed edge add no term.
    @test prior_at(quote
        phi[axes(y, 1)] ~ ICAR(node1[1:4], node2[1:4])
        y .~ Poisson.(exp.(phi))
    end, (; node1, node2, y)) ≈ icar_oracle(phi0, node1[1:4], node2[1:4]) rtol = 1e-13
end

@testset "ICAR edge lists are checked at binding" begin
    y = [1, 0, 2]
    ast = quote
        phi[1:3] ~ ICAR(node1, node2)
        y .~ Poisson.(exp.(phi))
    end
    bind(data) = bind_data(lower_rkppl(ast, data; mod = @__MODULE__,
        conditioned = (:y,)), data)
    @test bind((; node1 = [1, 2], node2 = [2, 3], y)) isa StructuralPlan
    # A node index outside the field.
    @test_throws ContractValidationError bind((; node1 = [1, 2], node2 = [2, 4], y))
    @test_throws ContractValidationError bind((; node1 = [0, 2], node2 = [2, 3], y))
    # One entry per edge in each node vector.
    @test_throws ContractValidationError bind((; node1 = [1, 2], node2 = [2], y))
    # Node indices are integers.
    @test_throws ContractValidationError bind((; node1 = [1.0, 2.0], node2 = [2, 3], y))
end

@testset "ICAR declaration shapes" begin
    lower(ast) = lower_rkppl(ast, (:node1, :node2, :y); mod = @__MODULE__,
        conditioned = (:y,))
    # refused: an ICAR field is one joint draw over its nodes, so `.~` (one
    # draw per element) is a different and wrong statement (`~` vs `.~`, §3).
    @test_throws SurfaceLoweringError lower(quote
        phi[1:3] .~ ICAR(node1, node2)
        y .~ Poisson.(exp.(phi))
    end)
    # refused: the edge list does not determine the number of nodes (a node
    # may lie on no edge), so the field states it, `phi[1:N]`.
    @test_throws SurfaceLoweringError lower(quote
        phi ~ ICAR(node1, node2)
        y .~ Poisson.(exp.(phi))
    end)
    # refused: the edge list is two node vectors (edge e joins node1[e] and
    # node2[e]); one vector does not state the graph.
    @test_throws SurfaceLoweringError lower(quote
        phi[1:3] ~ ICAR(node1)
        y .~ Poisson.(exp.(phi))
    end)
    # Not built yet: rows or columns of independent ICAR fields.
    @test_broken (lower(quote
        eachrow(P[1:2, 1:3]) .~ ICAR(node1, node2)
        y .~ Poisson.(exp.(P[1, :]))
    end); true)
end
end
