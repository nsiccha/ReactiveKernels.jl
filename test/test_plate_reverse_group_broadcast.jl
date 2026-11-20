module PlateReverseGroupBroadcastTests
using ReactiveKernels, DifferentiationInterface, Enzyme, MutatingFunctions, Test
const RK = ReactiveKernels

@testset "an extracted pointwise plate writes activity into its caller's buffer" begin
    producer = RK._PlateProgram(:(function (input, output, shape_only)
        for i in eachindex(input)
            output[i] = input[i] * length(shape_only)
        end
        nothing
    end), nothing, false)
    call = Expr(:call, producer, :source, :destination, :sizes)
    @test :destination in RK._pr_body_activity(call, Set([:source]))
    @test !(:destination in RK._pr_body_activity(call, Set([:sizes])))
end

@kernel group_bias(rows::Vector{Vector{Float64}}, ids::Vector{Int}, intercept::Float64,
                   effects::Vector{Float64}, weight::Float64) = begin
    offsets = intercept .+ effects
    totals = plate(ids) do i
        row = rows[i]
        offset = offsets[i]
        terms = plate(row) do x
            sin(x + offset)
        end
        subtotal::Float64 = sum(terms)
        subtotal
    end
    result = weight * sum(totals)
    return result
end

@testset "shared and indexed contributions to a nested broadcast" begin
    rows = [[0.1, 0.8], Float64[], [-0.2]]
    ids = [3, 1, 2, 1, 3]
    shift, effects, weight = 0.2, [-0.4, 0.9, 1.1], -1.7
    args = (rows, ids, shift, effects, weight)
    for enabled in (false, true)
        ad = with_plate_reverse(enabled) do
            k = prepare_nonallocating(group_bias, args...)
            prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), args...;
                       active = (:intercept, :effects))
        end
        for order in (ids, [2, 2], Int[]), intercept in (shift, -0.7)
            value = 0.0
            device = zero(effects)
            for i in order, x in rows[i]
                z = x + (intercept + effects[i])
                value += weight * sin(z)
                device[i] += weight * cos(z)
            end
            primal, (shared, indexed) =
                ad_value_and_gradient(ad, rows, order, intercept, effects, weight)
            @test primal ≈ value
            @test shared ≈ sum(device)
            @test indexed ≈ device
        end
    end
end
end
