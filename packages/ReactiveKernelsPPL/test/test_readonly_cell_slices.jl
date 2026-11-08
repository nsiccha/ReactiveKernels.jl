module ReadonlyCellSliceTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# Snag plate-cell-row-s-b9c4c730: a field row of a reshaped coefficient array,
# passed to an opaque function, copied its row at every cell, which was the
# largest single item of the gradient. The sampler now reads it in place.
function wave(frequency, scales, amplitude, coefficients, coordinate)
    total = 0.0
    for j in axes(frequency, 2)
        phase = 0.0
        for d in axes(frequency, 1)
            phase += frequency[d, j] * coordinate[d] / scales[d]
        end
        total += coefficients[2j - 1] * cos(phase) + coefficients[2j] * sin(phase)
    end
    amplitude * total / sqrt(size(frequency, 2))
end
writes(frequency, scales, amplitude, coefficients, coordinate) = begin
    coefficients[1] = 0.0
    wave(frequency, scales, amplitude, coefficients, coordinate)
end
branches(frequency, scales, amplitude, coefficients, coordinate) =
    coefficients isa Vector ? wave(frequency, scales, amplitude, coefficients, coordinate) :
    0.0 * amplitude
# Snag readonly-slice-r-cc6a354e: a callee that only indexes the row and hands
# another argument to BLAS (`transpose(frequency) * raw`, a `gemv!`), as a
# channel-mixing field does with its correlation factor.
function mixes(frequency, scales, amplitude, coefficients, coordinate)
    D, F = size(frequency)
    raw = zeros(D)
    for j in 1:F, d in 1:D
        phase = frequency[d, j] * coordinate[d] / scales[d]
        raw[d] += coefficients[2j - 1] * cos(phase) + coefficients[2j] * sin(phase)
    end
    amplitude * sum(transpose(frequency) * raw) / sqrt(F)
end
# The same with a vector-valued cell, the channel field's shape: its result is a
# fresh vector, which holds nothing of the row.
function channels(frequency, scales, amplitude, coefficients, coordinate)
    D, F = size(frequency)
    raw = zeros(D)
    for j in 1:F, d in 1:D
        phase = frequency[d, j] * coordinate[d] / scales[d]
        raw[d] += coefficients[2j - 1] * cos(phase) + coefficients[2j] * sin(phase)
    end
    transpose(frequency) * raw * amplitude / sqrt(F)
end
first_of(v) = v[1]
# Reads the row only through the interface, but a caught exception could hold it.
caught(frequency, scales, amplitude, coefficients, coordinate) =
    try
        wave(frequency, scales, amplitude, coefficients, coordinate)
    catch
        0.0 * amplitude
    end

function fixture(callee; nstates = 40, blocks = 4, F = 6, D = 3,
        read = callee === :channels ? :(first_of(field[i])) : :(field[i]))
    width = 2F
    data = (; block = [mod1(s, blocks) for s in 1:nstates],
        coordinates = [cos.(s .* (1:D) ./ 5) for s in 1:nstates],
        frequency = reshape(sin.(1:(D * F)), D, F),
        y = sin.(1:nstates), blocks, width, dimensions = D, ncoef = blocks * width)
    model = quote
        z[1:ncoef] .~ Normal.(0.0, 1.0)
        amplitude ~ Exponential(1.0)
        scales[1:dimensions] .~ LogNormal.(0.0, 1.0)
        sigma ~ Exponential(1.0)
        coefficients = reshape(z, blocks, width)
        @plate for s in eachindex(block)
            field[s] = $callee(frequency, scales, amplitude, coefficients[block[s], :], coordinates[s])
        end
        @plate for i in eachindex(y)
            y[i] ~ Normal($read, sigma)
        end
    end
    bound = bind_data(lower_rkppl(model, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.3; length = built.layout.total))
    (; bound, built, u)
end
sampler(fix) = prepare_sampler(fix.built, fix.bound, fix.u; backend = BACKEND)
copying(fix) = withenv(() -> sampler(fix), "RK_READONLY_SLICES" => "0")
in_place(q) = occursin("_readonly_slice", sprint(print, code_expr(q.kernel)))
# Values are compared bit for bit. A gradient can differ from the copy's in its
# last bit: the view's adjoint accumulates straight into the parent's shadow,
# where Enzyme's contracted derivative arithmetic (the dogfood default, `nsz` and
# `contract`) may fuse a multiply-add that the copy's shadow rounds twice. Under
# strict derivative arithmetic (`Enzyme.API.fast_math_flags!()`) they are identical.
# A callee that keeps its copy runs the same program, so its gradient is exact.
same_gradient(a, b) = isapprox(a, b; rtol = 1e-13)

@testset "a field row an opaque function only reads is read in place: $callee" for callee in
        (:wave, :mixes, :channels)
    fix = fixture(callee)
    q, reference = sampler(fix), copying(fix)
    @test in_place(q)
    @test !in_place(reference)
    u = copy(fix.u)
    @test q(fix.u) == reference(fix.u)
    g, g_reference = zero(u), zero(u)
    value, _ = sampler_value_and_gradient!(q, g, fix.u)
    value_reference, _ = sampler_value_and_gradient!(reference, g_reference, fix.u)
    @test value == value_reference
    @test same_gradient(g, g_reference)
    @test fix.u == u
    # Against central differences, at another point.
    point = u .+ 0.05 .* sin.(eachindex(u))
    sampler_value_and_gradient!(q, g, point)
    for k in (1, 7, length(u) - 1, length(u))
        plus, minus = copy(point), copy(point)
        plus[k] += 1e-6
        minus[k] -= 1e-6
        @test g[k] ≈ (q(plus) - q(minus)) / 2e-6 atol = 1e-5 rtol = 1e-5
    end
    # The bytes per gradient hold no copy of each row or its shadow.
    bytes(s, gradient) = (sampler_value_and_gradient!(s, gradient, fix.u);
        minimum(@allocated(sampler_value_and_gradient!(s, gradient, fix.u)) for _ in 1:3))
    @test bytes(reference, g_reference) - bytes(q, g) >= 40 * 12 * 8
end

# The path SVI runs: one unit's gradient, over a shared coefficient row.
function partition_fixture(callee; nstates = 400, blocks = 4, F = 32, rows = 20, D = 3)
    width = 2F
    unit = repeat(1:(nstates ÷ rows); inner = rows)
    data = (; block = [mod1(s, blocks) for s in 1:nstates],
        coordinates = [cos.(s .* (1:D) ./ 5) for s in 1:nstates],
        frequency = reshape(sin.(1:(D * F)), D, F), unit, state = collect(1:nstates),
        y = sin.(1:nstates), blocks, width, dimensions = D, ncoef = blocks * width)
    model = quote
        z[1:ncoef] .~ Normal.(0.0, 1.0)
        amplitude ~ Exponential(1.0)
        scales[1:dimensions] .~ LogNormal.(0.0, 1.0)
        eta[levels(unit)] .~ Normal.(0.0, 1.0)
        coefficients = reshape(z, blocks, width)
        @plate for s in eachindex(block)
            field[s] = $callee(frequency, scales, amplitude, coefficients[block[s], :], coordinates[s])
        end
        @plate for i in eachindex(y)
            y[i] ~ Normal(field[state[i]] + eta[unit[i]], 1.0)
        end
    end
    bound = bind_data(lower_rkppl(model, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.3; length = built.layout.total))
    (; bound, built, u)
end
partition(fix) = prepare_partition_sampler(fix.built, fix.bound, :y, fix.u; by = :unit,
    backend = BACKEND, partitioned = :eta, locality = :error)

@testset "a partition's unit gradient reads the row in place" begin
    fix = partition_fixture(:wave)
    sampler = partition(fix)
    reference = withenv(() -> partition(fix), "RK_READONLY_SLICES" => "0")
    @test partition_footprint(sampler, 3) == partition_footprint(reference, 3)
    for label in (1, 3, 20)
        fp = partition_footprint(sampler, label)
        g, g_reference = zeros(length(fp)), zeros(length(fp))
        value, _ = partition_value_and_gradient!(sampler, g, fix.u, label)
        value_reference, _ = partition_value_and_gradient!(reference, g_reference, fix.u, label)
        @test value == value_reference
        @test same_gradient(g, g_reference)
    end
    fp = partition_footprint(sampler, 3)
    g = zeros(length(fp))
    bytes(s) = (partition_value_and_gradient!(s, g, fix.u, 3);
        minimum(@allocated(partition_value_and_gradient!(s, g, fix.u, 3)) for _ in 1:3))
    # 20 rows of 64 coefficients, and the shadow of each.
    @test bytes(reference) - bytes(sampler) >= 20 * 64 * 8

    # A callee that could tell keeps its copy here too.
    fix = partition_fixture(:branches; nstates = 40, F = 4, rows = 10)
    sampler = partition(fix)
    reference = withenv(() -> partition(fix), "RK_READONLY_SLICES" => "0")
    fp = partition_footprint(sampler, 2)
    g, g_reference = zeros(length(fp)), zeros(length(fp))
    @test partition_value_and_gradient!(sampler, g, fix.u, 2)[1] ==
          partition_value_and_gradient!(reference, g_reference, fix.u, 2)[1]
    @test g == g_reference
end

# A consumer can see why a slice keeps its copy: the preparation logs it at the
# debug level (`JULIA_DEBUG=ReactiveKernels`), with the reader and each frame that
# refused the proof.
@testset "the preparation log says why a slice keeps its copy" begin
    # The primal query: the slice is decided there, and no gradient is needed.
    primal(fix) = ReactiveKernels._kernel_latest_prepared(ReactiveKernelsPPL.prepare_query(
        fix.built, fix.bound, :sampler; exemplars = (fix.u,)))
    kept(callee) = begin
        logs, kernel = Test.collect_test_logs(() -> primal(fixture(callee));
            min_level = Base.CoreLogging.Debug)
        (; kernel), filter(log -> log.group === :readonly_slices, logs)
    end
    # A write is not the array interface, so the first gate refuses it.
    q, logs = kept(:writes)
    @test !in_place(q)
    @test length(logs) == 1
    @test occursin("its reader could tell a view from a copy", logs[1].message)
    @test occursin("writes", string(logs[1].kwargs[:reader]))
    # The read-only proof names each frame that refused, and its statement.
    q, logs = kept(:caught)
    @test !in_place(q)
    @test length(logs) == 1
    @test occursin("its reader is not proved to only read it", logs[1].message)
    @test occursin("caught", logs[1].kwargs[:refused])
    q, logs = kept(:wave)
    @test in_place(q)
    @test isempty(logs)
end

@testset "$callee keeps its copy and its value" for callee in (:writes, :branches)
    fix = fixture(callee)
    q, reference = sampler(fix), copying(fix)
    @test !in_place(q)
    u = copy(fix.u)
    @test q(fix.u) == reference(fix.u)
    g, g_reference = zero(u), zero(u)
    @test sampler_value_and_gradient!(q, g, fix.u) ==
          sampler_value_and_gradient!(reference, g_reference, fix.u)
    @test g == g_reference
    @test fix.u == u
end
end
