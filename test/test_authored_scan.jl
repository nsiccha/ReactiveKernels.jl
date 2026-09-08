using ReactiveKernels
using Test

@kernel authored_scan_arma(q::Vector{Float64}, series::Vector{Float64}) = begin
    mu::Float64 = q[1]
    phi::Float64 = q[2]
    theta::Float64 = q[3]
    errors::Vector{Float64} = scan(series, Ref(mu), Ref(phi), Ref(theta);
            init = (; previous = mu, error = 0.0)) do carry, y, m, f, t
        e = y - (m + f * carry.previous + t * carry.error)
        ((; previous = y, error = e), e)
    end
    pointwise = plate(errors) do e
        -0.5 * e^2
    end
    total::Float64 = sum(pointwise)
    return total
end

function _authored_scan_reference(q, series)
    previous, error = q[1], 0.0
    errors = similar(series)
    for i in eachindex(series)
        error = series[i] - (q[1] + q[2] * previous + q[3] * error)
        previous = series[i]
        errors[i] = error
    end
    errors
end

_authored_scan_allocated(k, args...) = @allocated k(args...)

@testset "authored scan native step lowering" begin
    q = [0.2, 0.7, -0.3]
    for series in ([0.5], sin.(1:200)), bound in ((;), (; series))
        errors = _authored_scan_reference(q, series)
        pointwise = -0.5 .* errors.^2
        args = isempty(bound) ? (q, series) : (q,)
        for (want, expected) in (
                (:errors, errors), (:total, sum(pointwise)),
                ((:errors, :pointwise, :total), (errors, pointwise, sum(pointwise))))
            k = prepare(authored_scan_arma; want, bound)
            actual = k(args...)
            @test want isa Tuple ? all(isapprox.(actual, expected)) : actual ≈ expected
            scan_recipe = only(r for r in k.plan.recipes
                               if r.op isa ReactiveKernels._AuthoredScanOp)
            @test length(scan_body(scan_recipe).want) == 2
        end
    end

    @kernel scalar_carry_scan(xs) = begin
        cumulative = scan(xs; init = 0) do carry, x
            next = carry + x
            (next, next)
        end
        return cumulative
    end
    scalar = prepare(scalar_carry_scan)
    @test scalar([0.5, 1.0, 2.0]) == [0.5, 1.5, 3.5]
    @test scalar([1, 2, 3]) == [1, 3, 6]
    @test_throws ArgumentError scalar(Float64[])

    # Only the returned vector should allocate, never per-step boxed carries.
    series = sin.(1:200)
    k = prepare(authored_scan_arma; want = :errors, bound = (; series))
    k(q)
    _authored_scan_allocated(k, q)
    @test _authored_scan_allocated(k, q) <= sizeof(series) + 256
end
