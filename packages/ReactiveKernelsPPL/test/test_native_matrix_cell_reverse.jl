using Test, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme

module NativeMatrixCellReverse
using Distributions

cellpath(rain, a) = hcat(a .* rain, (a + 1) .* rain, (a + 2) .* rain)
secondcolumn(paths, shift) = [shift[g] .+ paths[g][:, 2] for g in eachindex(paths)]
sumcolumns(columns) = sum(sum, columns)
allpaths(rain, rain_index, a) =
    [cellpath(rain[rain_index[g]], a[g]) for g in eachindex(rain_index)]

# Independent scalar density and gradient for the data below. Coordinates are
# a[1:2], then shift[1:2]; the two groups have rain sums 0.6 and 0.9.
function reference(u)
    mu = 0.6 * (u[1] + 1) + 0.9 * (u[2] + 1) + 3u[3] + 2u[4]
    residual = 1 - mu
    return -2.5log(2pi) - (sum(abs2, u) + residual^2) / 2,
        -u .+ residual .* [0.6, 0.9, 3, 2]
end
end

@testset "matrix-valued cells through whole-array helpers in native Reverse" begin
    # Enzyme's Julia 1.10 interpreter used to widen the inner sum's keyword
    # wrapper to Any, then reject this concretely inferred prepared model.
    # Enzyme dogfood ae936763 repairs it (static-reverse-l-9e504efb).
    # Keep both original producers and helpers: no activity setting or rule.
    data = (; rain = [0.1, 0.2, 0.3, 0.4, 0.5],
        rain_index = [[1, 2, 3], [4, 5]], y = [1.0])
    snapshot = deepcopy(data)
    for label in (:plate, :helper)
        producer = label === :plate ? :(@plate for g in eachindex(rain_index)
            paths[g] = cellpath(rain[rain_index[g]], a[g])
        end) : :(paths = allpaths(rain, rain_index, a))
        source = quote
            a[1:2] .~ Normal.(0.0, 1.0)
            shift[1:2] .~ Normal.(0.0, 1.0)
            $producer
            columns = secondcolumn(paths, shift)
            y .~ Normal.(sumcolumns(columns), 1.0)
        end
        plan = bind_data(lower_rkppl(source, data;
            conditioned = (:y,), mod = NativeMatrixCellReverse), data)
        built = build_kernel(plan)
        @test built.layout.total == 4
        sampler = prepare_sampler(built, plan, zeros(4);
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        for u in (zeros(4), [0.2, -0.4, 0.1, -0.2], [-0.7, 0.3, -0.2, 0.4])
            saved = copy(u)
            expected_value, expected_gradient = NativeMatrixCellReverse.reference(u)
            @test sampler(u) ≈ expected_value
            gradient = fill(NaN, length(u))
            for _ in 1:2
                value, _ = sampler_value_and_gradient!(sampler, gradient, u)
                @test value ≈ expected_value
                @test gradient ≈ expected_gradient
                @test isequal(u, saved)
                @test isequal(data, snapshot)
            end
            h = 1e-5
            finite_difference = map(eachindex(u)) do j
                up, down = copy(u), copy(u)
                up[j] += h
                down[j] -= h
                (sampler(up) - sampler(down)) / (2h)
            end
            @test gradient ≈ finite_difference atol = 1e-8 rtol = 1e-8
            @test isequal(data, snapshot)
        end
    end
end
