module BatchedFieldPlateTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# Snag plate-of-per-row-d1b4ed43: a random-Fourier-feature field per cell, written
# as RK statements, runs as matrix products over tiles of its cells (RK 0.1.127,
# "Plates of products run as matrix products" in the compiler guide). `@rkppl`
# fuses the field with its observations' densities, whose laws are lazy branches:
# those run cell by cell inside each tile. Values and gradients agree with the
# cell loop (`RK_BATCHED_PLATES=0`) to rounding.
function fixture(; nstates = 300, F = 8, D = 3, C = 3, units = 10)
    width = 2F
    data = (; coordinates = [cos.(s .* (1:D) ./ 50) for s in 1:nstates],
        frequency = reshape(sin.(1:(D * F)), D, F), y = sin.(1:nstates),
        unit = repeat(1:units; inner = nstates ÷ units),
        C, width, dimensions = D, ncoef = C * width, features = F)
    model = quote
        z[1:ncoef] .~ Normal.(0.0, 1.0)
        amplitude ~ Exponential(1.0)
        scales[1:dimensions] .~ LogNormal.(0.0, 1.0)
        sigma ~ Exponential(1.0)
        eta[levels(unit)] .~ Normal.(0.0, 1.0)
        Z = reshape(z, C, width)
        Zc = Z[:, 1:2:width]
        Zs = Z[:, 2:2:width]
        W = transpose(frequency ./ scales)
        @plate for s in eachindex(y)
            phase = W * coordinates[s]
            raw = Zc * cos.(phase) .+ Zs * sin.(phase)
            field[s] = sum(raw) * amplitude / sqrt(features)
        end
        @plate for i in eachindex(y)
            y[i] ~ Normal(field[i] + eta[unit[i]], sigma)
        end
    end
    bound = bind_data(lower_rkppl(model, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.3; length = built.layout.total))
    (; bound, built, u)
end
batched(kernel) = occursin("_PLATE_BATCH_TILE", sprint(print, code_expr(kernel)))
cell_loop(f) = withenv(f, "RK_BATCHED_PLATES" => "0")
close_to(a, b; rtol) = all(isapprox.(a, b; rtol, atol = rtol * maximum(abs, b)))

@testset "a sampler's field plate runs as matrix products" begin
    fix = fixture()
    q = prepare_sampler(fix.built, fix.bound, fix.u; backend = BACKEND)
    reference = cell_loop(() -> prepare_sampler(fix.built, fix.bound, fix.u; backend = BACKEND))
    @test batched(q.kernel)
    @test !batched(reference.kernel)
    @test q(fix.u) ≈ reference(fix.u) rtol = 1e-12
    g, g_reference = zero(fix.u), zero(fix.u)
    value, _ = sampler_value_and_gradient!(q, g, fix.u)
    value_reference, _ = sampler_value_and_gradient!(reference, g_reference, fix.u)
    @test value ≈ value_reference rtol = 1e-12
    @test close_to(g, g_reference; rtol = 1e-10)
end

@testset "a partition's unit gradient runs its rows batched" begin
    fix = fixture()
    partition(fix) = prepare_partition_sampler(fix.built, fix.bound, :y, fix.u; by = :unit,
        backend = BACKEND, partitioned = :eta)
    # The preparation plans the partition's plate batched (it logs each plan at
    # the debug level).
    logs, sampler = Test.collect_test_logs(() -> partition(fix);
        min_level = Base.CoreLogging.Debug)
    @test any(log -> log.group === :plate_batching, logs)
    reference = cell_loop(() -> partition(fix))
    for label in (1, 4, 10)
        fp = partition_footprint(sampler, label)
        @test fp == partition_footprint(reference, label)
        g, g_reference = zeros(length(fp)), zeros(length(fp))
        value, _ = partition_value_and_gradient!(sampler, g, fix.u, label)
        value_reference, _ = partition_value_and_gradient!(reference, g_reference, fix.u, label)
        @test value ≈ value_reference rtol = 1e-12
        @test close_to(g, g_reference; rtol = 1e-10)
    end
end
end
