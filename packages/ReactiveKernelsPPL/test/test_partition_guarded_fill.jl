module PartitionGuardedFillTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Random, Test
using DifferentiationInterface: AutoEnzyme

# Two completed covariates, as BRM lowers `mi(a) ~ Normal(mu_a, 1)` (its law
# reads the units' effects, so it is non-centered: unit-owned `a.z_mis`) and
# `mi(b) ~ LogNormal(0.5, 0.3)` (centered: shared `b.y_mis`). A local
# partition derivative lowers the observation sums with owner guards; a plate
# whose buffer is allocated before such a guard and filled under it must be
# read after the fill (snag `cell-joined-rows-1ab77c3c`: its readers ran first
# and read undefined rows, `UndefRefError` in `_cell_joined_rows`).
ReactiveKernels.@kernel completed_covariate(observed, missing, order) = begin
    stacked = vcat(observed, missing)
    completed = stacked[order]
    return completed
end
@rkppl group_effects(g) = begin
    tau[1:1] .~ LogNormal.(0, 1)
    z[1:length(levels(g)), 1:1] .~ Normal.(0, 1)
    return z .* tau[1]
end
@rkppl population_effects(X, ncoef, loc, scale) = begin
    beta_pop[1:ncoef] .~ Normal.(loc, scale)
    return X * beta_pop
end
# `mu_a` and `Jmis_a` are the calling program's.
@rkppl noncentered_covariate(observed, rows, order) = begin
    z_mis[axes(rows, 1)] .~ Normal.(0, 1)
    y_mis = mu_a[Jmis_a] .+ 1.0 .* z_mis
    completed = completed_covariate(observed, y_mis, order)
    return completed
end
@rkppl centered_covariate(observed, rows, order) = begin
    y_mis[axes(rows, 1)] .~ LogNormal.(0.5, 0.3)
    completed = completed_covariate(observed, y_mis, order)
    return completed
end
const MODEL = quote
    b_region ~ group_effects(region_levels)
    ranef_mu_a = b_region[region, 1]
    b_region_eta ~ group_effects(region_levels)
    ranef_eta = b_region_eta[region, 1]
    mu_a_Intercept ~ Normal(0.0, 1.0)
    mu_a = mu_a_Intercept .+ ranef_mu_a
    sigma ~ Exponential(1.0)
    a ~ noncentered_covariate(a_observed, Jmis_a, a_order)
    b ~ centered_covariate(b_observed, Jmis_b, b_order)
    lb = log.(b)
    X_eta = hcat(ones(length(region)), a, lb)
    pop_eta ~ population_effects(X_eta, 3, 0.0, 1.0)
    eta = pop_eta .+ ranef_eta
    a_obs .~ Normal.(mu_a[Jobs_a], 1.0)
    b_obs .~ LogNormal.(0.5, 0.3)
    y .~ Normal.(eta, sigma)
end

# A covariate with missing rows: its observed values, the observed and missing
# rows, the observed values at their rows (zero elsewhere), and each row's
# position in `vcat(observed rows' values, missing values)`.
function completion_data(x)
    present = .!ismissing.(x)
    observed = Float64.(x[present])
    Jobs, Jmis = findall(present), findall(!, present)
    n = length(x)
    at_rows = [present[i] ? Float64(x[i]) : 0.0 for i in 1:n]
    order = [present[i] ? i : n + findfirst(==(i), Jmis) for i in 1:n]
    (; observed, Jobs, Jmis, at_rows, order)
end

@testset "a local partition derivative reads a guarded plate after its fill" begin
    rng = Xoshiro(13)
    n = 36
    region = repeat(1:3, inner=12)
    a = Vector{Union{Missing,Float64}}(randn(rng, n))
    b = Vector{Union{Missing,Float64}}(exp.(0.5 .+ 0.3 .* randn(rng, n)))
    y = coalesce.(a, 0.0) .+ log.(coalesce.(b, 1.6)) .+ 0.3 .* randn(rng, n)
    a[rand(rng, n) .< 0.25] .= missing
    b[rand(rng, n) .< 0.25] .= missing
    ca, cb = completion_data(a), completion_data(b)
    data = (; region, region_levels=[1, 2, 3], y, a_obs=ca.observed, Jobs_a=ca.Jobs, Jmis_a=ca.Jmis,
        a_observed=ca.at_rows, a_order=ca.order, b_obs=cb.observed, Jobs_b=cb.Jobs,
        Jmis_b=cb.Jmis, b_observed=cb.at_rows, b_order=cb.order,
        unit_a=region[ca.Jobs], unit_b=region[cb.Jobs], unit_y=region)
    bound = bind_data(lower_rkppl(MODEL, data;
        conditioned=(:a_obs, :b_obs, :y), mod=@__MODULE__), data)
    built = build_kernel(bound)
    # Each region owns its effects and the standardized draws of its missing
    # `a` rows; `b`'s missing values stay shared.
    partitioned = (Symbol("b_region.z") => [1, 2, 3],
                   Symbol("b_region_eta.z") => [1, 2, 3],
                   Symbol("a.z_mis") => region[ca.Jmis])
    observations = (:a_obs, :b_obs, :y)
    by = (; a_obs=:unit_a, b_obs=:unit_b, y=:unit_y)
    u = 0.1 .* randn(Xoshiro(42), built.layout.total)
    query = prepare_partition(built, bound, observations; by, partitioned)
    sampler = prepare_partition_sampler(built, bound, observations, u; by,
        partitioned, compact=true, backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for label in (1, 2, 3)
        fp = partition_footprint(sampler, label)
        local_fp = partition_footprint(sampler, label; shared=false)
        @test !isempty(local_fp)
        full = zeros(length(fp))
        local_gradient = zeros(length(local_fp))
        value, _ = partition_value_and_gradient!(sampler, full, u, label)
        local_value, _ = partition_value_and_gradient!(sampler, local_gradient, u,
            label; shared=false)
        @test local_value ≈ value ≈ query(u, label)
        @test local_gradient ≈ full[indexin(local_fp, fp)]
        for (slot, position) in enumerate(local_fp)
            plus, minus = copy(u), copy(u)
            plus[position] += 1e-6
            minus[position] -= 1e-6
            finite = (query(plus, label) - query(minus, label)) / 2e-6
            @test local_gradient[slot] ≈ finite rtol=1e-6 atol=1e-6
        end
    end
end

end
