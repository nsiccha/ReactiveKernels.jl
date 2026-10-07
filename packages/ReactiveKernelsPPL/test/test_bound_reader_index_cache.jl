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
end
