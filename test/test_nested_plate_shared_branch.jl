module NestedPlateSharedBranchTests
using ReactiveKernels, Enzyme, Test

# A plate nested in another plate's cell receives the outer operands' elements.
# A lazy branch of the nested cell whose condition reads only such a number
# (`scale` below) takes the same arm at every nested cell, so the nested plate
# decides it once per outer cell, as a top-level plate does with a shared
# condition (`_plate_shared_branch`). The counter shows where the condition is
# evaluated; nothing else in these cells has an effect.
const _CONDITIONS = Ref(0)
_counted_positive(s) = (_CONDITIONS[] += 1; s > 0)

@kernel _nested_guarded(groups::Vector{Vector{Float64}}, scales::Vector{Float64},
                        x::Float64) = begin
    terms = plate(groups, scales) do group, scale
        rows = plate(group, scale) do y, s
            _counted_positive(s) ? x * y * log(s) : -Inf
        end
        sum(rows)
    end
    total::Float64 = sum(terms)
    return total
end

function oracle(groups, scales, x)
    total, slope = 0.0, 0.0
    for (group, s) in zip(groups, scales), y in group
        s > 0 || return -Inf, NaN
        total += x * y * log(s)
        slope += y * log(s)
    end
    total, slope
end

gradient(kernel, args...) = last(first(Enzyme.autodiff(Enzyme.Reverse, Const(kernel),
    Active, (Const(a) for a in args[1:end-1])..., Active(last(args)))))

@testset "nested shared branch is decided once per outer cell" begin
    cases = (
        ([[1.0, 2.0, 3.0], Float64[], [0.5, -1.5]], [2.0, -3.0, 0.7]),
        ([[0.25, 4.0], [1.0, 1.0, 1.0, 2.0]], [3.0, 1.5]),
        ([Float64[], Float64[]], [-1.0, 0.0]),
        (Vector{Float64}[], Float64[]),
    )
    for (groups, scales) in cases, bind in (false, true)
        # Bound groups make the outer operand a typed constant; the scales
        # stay a runtime input, so the condition is evaluated at call time.
        kernel = bind ? prepare(_nested_guarded; bound = (; groups)) :
            prepare(_nested_guarded)
        args = bind ? (scales, 0.3) : (groups, scales, 0.3)
        value, slope = oracle(groups, scales, 0.3)
        nonempty = count(!isempty, groups)
        _CONDITIONS[] = 0
        result = kernel(args...)
        @test result === value || result ≈ value
        # Once per non-empty outer cell; never for an empty nested domain, and
        # never the inactive arm's `log` of a non-positive scale.
        @test _CONDITIONS[] == nonempty
        isfinite(value) && @test gradient(kernel, args...) ≈ slope
    end
    # A non-positive scale whose nested domain is non-empty selects the
    # constant arm for every nested cell, without evaluating `log(scale)`.
    groups, scales = [[1.0, 2.0], [3.0]], [2.0, -1.0]
    for bind in (false, true)
        kernel = bind ? prepare(_nested_guarded; bound = (; groups)) :
            prepare(_nested_guarded)
        args = bind ? (scales, 0.3) : (groups, scales, 0.3)
        _CONDITIONS[] = 0
        @test kernel(args...) == -Inf
        @test _CONDITIONS[] == 2
    end
end

end
