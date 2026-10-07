module BoundReaderIndexCacheTests
using Test
# Regression: a BRM-style reader whose subject plate iterates a bound `1:n`
# with `Ref` operands derives a per-subject index list from bound data before
# one gather of a parameter-dependent vector. `prepare_sampler` binds the data,
# so the index chain runs once at preparation, never per density or gradient
# evaluation (snag bound-partial-ev-75827c5e).
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, DifferentiationInterface
const calls = Ref(0)
counted_findall(f, x) = (calls[] += 1; findall(f, x))
ReactiveKernels.@kernel subject_reader(subject_count, kinds_by_subject, read_idx, rates) = begin
    cell_values = ReactiveKernels.plate(
            1:subject_count, Ref(kinds_by_subject), Ref(read_idx), Ref(rates)
        ) do subject, kinds_by_subject, read_idx, rates
        kinds = kinds_by_subject[subject]
        read_positions = counted_findall(isone, kinds)
        observation_operations = read_positions[read_idx[subject]]
        rates[observation_operations]
    end
    result = convert(Vector{Float64}, reduce(vcat, cell_values; init=Float64[]))
    return result
end
columns = (
    subject_count = 3,
    kinds_by_subject = [[1, 2, 1, 1, 3, 1], [2, 1, 1, 3, 1, 1], [1, 3, 1, 1, 2, 1]],
    read_idx = [[1, 3], [2, 4], [1, 2]],
    y = [0.9, 1.4, 2.2, 0.3, 1.1, 0.7],
)
model = @rkppl begin
    log_rate[1:6] .~ Normal.(0.0, 1.0)
    rates = exp.(log_rate)
    locations = subject_reader(subject_count, kinds_by_subject, read_idx, rates)
    y .~ Normal.(locations, 0.5)
end

@testset "bound reader index chain is prepared once" begin
    bound = bind_data(lower_rkppl(model, columns; conditioned=(:y,)), columns)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.4; length=6))
    calls[] = 0
    q = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    @test calls[] == columns.subject_count
    @test !occursin("counted_findall", string(readable_code(q.kernel)))

    selected = [1, 4, 3, 6, 1, 3]
    normal_logpdf(x, m, s) = -0.5 * ((x - m) / s)^2 - log(s) - 0.5 * log(2pi)
    for point in (u, reverse(u))
        rates = exp.(point)
        expected = sum(normal_logpdf.(point, 0.0, 1.0)) +
                   sum(normal_logpdf.(columns.y, rates[selected], 0.5))
        expected_gradient = -copy(point)
        for (observation, position) in enumerate(selected)
            expected_gradient[position] +=
                (columns.y[observation] - rates[position]) / 0.25 * rates[position]
        end
        g = similar(point)
        @test Base.invokelatest(q, point) ≈ expected
        value, _ = Base.invokelatest(sampler_value_and_gradient!, q, g, point)
        @test value ≈ expected
        @test g ≈ expected_gradient
    end
    @test calls[] == columns.subject_count
end

# The same reader whose cell also runs a recurrence over each subject's
# operations: the scan reads a parameter, so it stays in the cell, and the index
# chain beside it is still prepared once (snag inner-plate-cach-f920caca).
ReactiveKernels.@kernel scan_reader(subject_count, kinds_by_subject, steps_by_subject,
                                    read_idx, rates) = begin
    cell_values = ReactiveKernels.plate(
            1:subject_count, Ref(kinds_by_subject), Ref(steps_by_subject), Ref(read_idx),
            Ref(rates)
        ) do subject, kinds_by_subject, steps_by_subject, read_idx, rates
        kinds = kinds_by_subject[subject]
        steps = steps_by_subject[subject]
        rate = rates[subject]
        read_positions = counted_findall(isone, kinds)
        observation_operations = read_positions[read_idx[subject]]
        operations = ReactiveKernels.scan(kinds, steps, Ref(rate);
                                          init = 0.0) do level, kind, step, r
            decayed = level * exp(-r * step)
            next = kind == 2 ? decayed + 1.0 : decayed
            (next, next)
        end
        operations[observation_operations]
    end
    result = convert(Vector{Float64}, reduce(vcat, cell_values; init=Float64[]))
    return result
end
scan_columns = (
    subject_count = 2,
    kinds_by_subject = [[1, 2, 1, 1, 2, 1], [2, 2, 1, 2, 1]],
    steps_by_subject = [[0.5, 1.0, 0.25, 2.0, 1.5, 0.75], [0.0, 1.5, 0.5, 1.0, 2.0]],
    read_idx = [[3, 1, 4], [2, 1, 2]],
    y = [0.1, 0.0, 0.9, 1.3, 0.2, 1.1],
)
scan_model = @rkppl begin
    log_rate[1:2] .~ Normal.(0.0, 1.0)
    rates = exp.(log_rate)
    locations = scan_reader(subject_count, kinds_by_subject, steps_by_subject, read_idx, rates)
    y .~ Normal.(locations, 0.5)
end

# An independent hand-written density of the same model.
function scan_reader_density(point)
    normal_logpdf(x, m, s) = -0.5 * ((x - m) / s)^2 - log(s) - 0.5 * log(2pi)
    total = sum(normal_logpdf.(point, 0.0, 1.0))
    observation = 0
    for subject in 1:scan_columns.subject_count
        kinds = scan_columns.kinds_by_subject[subject]
        steps = scan_columns.steps_by_subject[subject]
        rate = exp(point[subject])
        levels = zeros(eltype(point), length(kinds))
        level = zero(eltype(point))
        for i in eachindex(kinds)
            level *= exp(-rate * steps[i])
            kinds[i] == 2 && (level += 1.0)
            levels[i] = level
        end
        for position in findall(isone, kinds)[scan_columns.read_idx[subject]]
            observation += 1
            total += normal_logpdf(scan_columns.y[observation], levels[position], 0.5)
        end
    end
    total
end

@testset "bound reader index chain beside a scan is prepared once" begin
    bound = bind_data(lower_rkppl(scan_model, scan_columns; conditioned=(:y,)), scan_columns)
    built = build_kernel(bound)
    u = [-0.3, 0.4]
    calls[] = 0
    q = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    @test calls[] == scan_columns.subject_count
    code = string(readable_code(q.kernel))
    @test !occursin("counted_findall", code)
    @test occursin("bound_plate_observation_operations", code)
    h = 1e-6
    for point in (u, reverse(u), [0.9, -1.1])
        expected = scan_reader_density(point)
        expected_gradient = [(scan_reader_density(point .+ h .* (1:2 .== i)) -
                              scan_reader_density(point .- h .* (1:2 .== i))) / 2h
                             for i in 1:2]
        g = similar(point)
        @test Base.invokelatest(q, point) ≈ expected
        value, _ = Base.invokelatest(sampler_value_and_gradient!, q, g, point)
        @test value ≈ expected
        @test isapprox(g, expected_gradient; rtol=1e-6, atol=1e-8)
    end
    @test calls[] == scan_columns.subject_count
end

# Regression: a reader plate whose cell result reads only bound data (a
# per-subject vector of observation limits) while it receives the live rates
# atomically. Its whole result is prepared once, so native Reverse never sees
# cached per-subject arrays stored into a plate's output (snag
# native-reverse-r-79fb001d).
ReactiveKernels.@kernel subject_limits(subject_count, limits, read_idx, rates) = begin
    cell_values = ReactiveKernels.plate(
            1:subject_count, Ref(limits), Ref(read_idx), Ref(rates)
        ) do subject, limits, read_idx, rates
        selected = limits[read_idx[subject]]
        ones(length(selected)) .* selected
    end
    result = convert(Vector{Float64}, reduce(vcat, cell_values; init=Float64[]))
    return result
end
limit_columns = (
    subject_count = 3,
    limits = [0.5, 1.0, 2.0, 0.25, 1.5, 4.0],
    read_idx = [[1, 4], [3, 6], [2, 5]],
    y = [0.9, 1.4, 2.2, 0.3, 1.1, 0.7],
)
limit_model = @rkppl begin
    log_rate[1:6] .~ Normal.(0.0, 1.0)
    rates = exp.(log_rate)
    weights = subject_limits(subject_count, limits, read_idx, rates)
    y .~ Normal.(weights .* rates, 0.5)
end

@testset "bound data-only reader result under native Reverse" begin
    bound = bind_data(lower_rkppl(limit_model, limit_columns; conditioned=(:y,)),
                      limit_columns)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.4; length=6))
    q = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    weights = reduce(vcat, [limit_columns.limits[r] for r in limit_columns.read_idx])
    normal_logpdf(x, m, s) = -0.5 * ((x - m) / s)^2 - log(s) - 0.5 * log(2pi)
    for point in (u, reverse(u))
        rates = exp.(point)
        locations = weights .* rates
        expected = sum(normal_logpdf.(point, 0.0, 1.0)) +
                   sum(normal_logpdf.(limit_columns.y, locations, 0.5))
        expected_gradient = -point .+
            (limit_columns.y .- locations) ./ 0.25 .* locations
        g = similar(point)
        @test Base.invokelatest(q, point) ≈ expected
        value, _ = Base.invokelatest(sampler_value_and_gradient!, q, g, point)
        @test value ≈ expected
        @test g ≈ expected_gradient
    end
end
end
