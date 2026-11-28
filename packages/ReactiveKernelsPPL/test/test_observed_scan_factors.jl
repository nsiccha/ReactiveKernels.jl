module ObservedScanFactorTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions
import DifferentiationInterface as DI
import Enzyme

# Once every carried array of a `@scan` is observed data, each seed's and
# step's conditional density reads only data, parameters and values computed
# outside the scan. RKPPL states them as plates (rkppl-use §5), so
# `prepare_cell_factors` exposes one term per entry, each reading only what
# that entry reads, and a top-level series' entry cells read only their own
# steps (snag observed-scan-pl-df7dad18, decision 02yqgx5). S series of W
# weeks of H hours; one state per series and week, each hour reads its own
# week's state. Oracles are Distributions.jl recursions.

const S, W, H = 2, 3, 4
const N = W * H
const WEEK = repeat(1:W; inner = H)
const POS = [(g - 1) * W + k for g in 1:S for k in WEEK]
const ROWS = [collect(((g - 1) * N + 1):(g * N)) for g in 1:S]
const Y = [[0.4sin(0.9t + g) + 0.1t for t in 1:N] for g in 1:S]
const TIME = Float64.(repeat(1:N, S))

const LOCAL = @rkppl begin
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    w[1:nw] .~ Normal.(0, 1)
    mu = w[pos]
    @plate for g in eachindex(y)
        m = mu[rows[g]]
        @scan begin
            y[g][1] ~ Normal(m[1], sigma / sqrt(1 - phi^2))
            for t in 2:length(y[g])
                y[g][t] ~ Normal(m[t] + phi * (y[g][t - 1] - m[t - 1]), sigma)
            end
        end
    end
end

# BRM's `AR1Normal` submodel shape: a data bound, a per-step local, gaps.
@rkppl ar1_gaps(m, sigma, phi, time) = begin
    @scan begin
        y[1] ~ Normal(m[1], sigma / sqrt(1 - phi^2))
        for t in 2:length(time)
            k = time[t] - time[t - 1]
            y[t] ~ Normal(m[t] + phi^k * (y[t - 1] - m[t - 1]),
                sigma * sqrt((1 - phi^(2k)) / (1 - phi^2)))
        end
    end
    return y
end

const CALL = @rkppl begin
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    w[1:nw] .~ Normal.(0, 1)
    mu = w[pos]
    @plate for g in eachindex(y)
        y[g] ~ ar1_gaps(mu[rows[g]], sigma, phi, time[rows[g]])
    end
end

const TOP = @rkppl begin
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    w[1:nw] .~ Normal.(0, 1)
    mu = w[pos]
    @scan begin
        y[1] ~ Normal(mu[1], sigma / sqrt(1 - phi^2))
        for t in 2:T
            y[t] ~ Normal(mu[t] + phi * (y[t - 1] - mu[t - 1]), sigma)
        end
    end
end

function parameters(built, u)
    read(name) = (e = only(x for x in built.layout.entries if x.name === name);
        u[e.offset:(e.offset + e.size - 1)])
    (; phi = 2 / (1 + exp(-only(read(:phi)))) - 1, sigma = exp(only(read(:sigma))),
       w = read(:w))
end

# Entry t's conditional density (gaps of one time step).
entries(y, m, phi, sigma) = [logpdf(Normal(m[1], sigma / sqrt(1 - phi^2)), y[1]);
    [logpdf(Normal(m[t] + phi * (y[t - 1] - m[t - 1]), sigma), y[t]) for t in 2:length(y)]]

# Entry t of series g reads phi, sigma, its own week's state and, from the
# second entry on, the previous hour's week's state.
expected_reads(g, t) = Set(["phi", "sigma", "w.$((g - 1) * W + WEEK[t])",
    (t > 1 ? ["w.$((g - 1) * W + WEEK[t - 1])"] : String[])...])

footprint(fp, u, args...) = Base.invokelatest(fp, u, args...).unconstrained

@testset "an observed cell scan gives one factor term per entry ($label)" for (label, model, data) in
        (("cell-local mean", LOCAL, (; nw = S * W, pos = POS, rows = ROWS)),
         ("submodel call", CALL, (; nw = S * W, pos = POS, rows = ROWS, time = TIME)))
    bound = model(; data...) | (; y = Y)
    built = build_kernel(bound)
    names = string.(coordinate_names(built.layout))
    # No sequential loop is left: every entry is a plate element.
    inventory = [e.kind for e in recipe_inventory(built.spec)]
    @test :plate in inventory && !(:scan in inventory)
    f = prepare_cell_factors(built, bound, :y; locality = :error)
    @test cell_reductions(f) == (:_ppl_group_total,)
    term = prepare_cell_factor_query(f, 1)
    outer = prepare_cell_factor_query(f, :outer)
    cell = prepare_cell_query(built, bound, :y; locality = :error)
    tfp = ReactiveKernels.partition_footprint_kernel(term.kernel)
    ofp = ReactiveKernels.partition_footprint_kernel(outer.kernel)
    grad = prepare_cell_factor_ad(prepare_ad, f, 1, zeros(built.layout.total),
        collect(1:built.layout.total), DI.AutoEnzyme(; mode = Enzyme.Reverse))
    for u in ([0.1 + 0.03i for i in 1:built.layout.total],
              [0.2cos(i) - 0.1 for i in 1:built.layout.total])
        p = parameters(built, u)
        for g in 1:S
            @test reduction_rows(f, 1, g) == 1:N
            expected = entries(Y[g], p.w[POS][ROWS[g]], p.phi, p.sigma)
            values = [term(u, g, t) for t in 1:N]
            @test values ≈ expected rtol = 1e-12
            for t in 1:N
                @test Set(names[footprint(tfp, u, g, t)]) == expected_reads(g, t)
            end
            # The outer is the plain sum of the terms and reads no coordinate.
            @test isempty(footprint(ofp, u, [sum(values)], g))
            @test outer(u, [sum(values)], g) ≈ cell(u, g) rtol = 1e-12
            @test cell(u, g) ≈ sum(expected) rtol = 1e-12
            # Each term's gradient against central differences of its oracle.
            t = 5
            reads = footprint(tfp, u, g, t)
            value, gradient = ad_value_and_gradient(grad, u, u, collect(1:built.layout.total), g, t)
            @test value ≈ expected[t] rtol = 1e-12
            h = cbrt(eps(Float64))
            fd = map(1:built.layout.total) do i
                up, dn = copy(u), copy(u)
                up[i] += h
                dn[i] -= h
                pu, pd = parameters(built, up), parameters(built, dn)
                (entries(Y[g], pu.w[POS][ROWS[g]], pu.phi, pu.sigma)[t] -
                 entries(Y[g], pd.w[POS][ROWS[g]], pd.phi, pd.sigma)[t]) / 2h
            end
            @test gradient ≈ fd rtol = 1e-6 atol = 1e-8
            @test all(iszero, gradient[setdiff(1:built.layout.total, reads)])
        end
    end
end

@testset "a top-level observed series' entries read their own steps" begin
    y = Y[1]
    bound = TOP(; nw = W, pos = WEEK) | (; y)
    built = build_kernel(bound)
    names = string.(coordinate_names(built.layout))
    cell = prepare_cell_query(built, bound, :y; locality = :error)
    fp = ReactiveKernels.partition_footprint_kernel(cell.kernel)
    u = [0.1 + 0.03i for i in 1:built.layout.total]
    p = parameters(built, u)
    expected = entries(y, p.w[WEEK], p.phi, p.sigma)
    for t in 1:N
        @test cell(u, t) ≈ expected[t] rtol = 1e-12
        @test Set(names[footprint(fp, u, t)]) == expected_reads(1, t)
    end
end

# A row of `vcat(a, b)` is a row of one operand: the reduction's labels are
# each operand's, in order, and there is one term per entry of every operand.
# Before, the labels were the first operand's alone, so the terms missed `b`'s
# entries and the outer's total was short of them.
const CONCAT = @rkppl begin
    z[1:n] .~ Normal.(0, 1)
    @plate for i in eachindex(y)
        y[i] ~ Normal(sum(vcat(exp.(z[a[i]]), z[b[i]] .^ 2)), 1)
    end
end

@testset "a sum over a vcat has one term per entry of each operand" begin
    a, b = [[1, 3], [2]], [[2, 2, 4], [1, 4]]
    y = [0.3, -0.2]
    bound = CONCAT(; n = 4, a, b) | (; y)
    built = build_kernel(bound)
    f = prepare_cell_factors(built, bound, :y; locality = :error)
    @test length(cell_reductions(f)) == 1
    term = prepare_cell_factor_query(f, 1)
    outer = prepare_cell_factor_query(f, :outer)
    tfp = ReactiveKernels.partition_footprint_kernel(term.kernel)
    u = [0.1, -0.4, 0.25, 0.7]
    for i in eachindex(y)
        @test reduction_rows(f, 1, i) == vcat(a[i], b[i])
        expected = vcat(exp.(u[a[i]]), u[b[i]] .^ 2)
        values = [term(u, i, t) for t in eachindex(expected)]
        @test values ≈ expected rtol = 1e-12
        @test [footprint(tfp, u, i, t) for t in eachindex(expected)] == [[j] for j in vcat(a[i], b[i])]
        @test outer(u, [sum(values)], i) ≈ logpdf(Normal(sum(expected), 1), y[i]) rtol = 1e-12
    end
end

end # module
