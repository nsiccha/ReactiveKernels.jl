using ReactiveKernels
using Test

if !isdefined(@__MODULE__, :AuthoredScanFixtures)
    include(joinpath(@__DIR__, "fixtures", "authored_scan.jl"))
end
using .AuthoredScanFixtures: authored_scan_arma, _authored_scan_reference,
    authored_scan_lockstep, _authored_scan_lockstep_reference

_authored_scan_allocated(k, args::Vararg{Any,N}) where {N} = @allocated k(args...)
_authored_scan_mixed(x) = Base.inferencebarrier(x > 2 ? 1.5 : 1)

@testset "authored scan native step lowering" begin
    q = [0.2, 0.7, -0.3]
    for series in ([0.5], sin.(1:200)), bound in ((;), (; series))
        errors = _authored_scan_reference(q, series)
        pointwise = -0.5 .* errors.^2
        args = isempty(bound) ? (q, series) : (q,)
        for (want, expected) in (
                (:errors, errors), (:total, sum(pointwise)),
                ((:pointwise, :total), (pointwise, sum(pointwise))),
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

@testset "authored scan streams into a scalar plate reduction" begin
    q, series = [0.2, 0.7, -0.3], sin.(1:200)
    for bound in ((;), (; series))
        k = prepare(authored_scan_arma; bound)
        args = isempty(bound) ? (q, series) : (q,)
        k(args...)
        _authored_scan_allocated(k, args...)
        @test _authored_scan_allocated(k, args...) == 0
        @test_throws ArgumentError prepare(authored_scan_arma)(q, Float64[])
    end

    @kernel scan_broadcast_consumer(xs, weights, offset::Float64) = begin
        cumulative = scan(xs; init = 0.0) do carry, x
            next = carry + x
            (next, next)
        end
        # This input becomes available after the scan in authored order.
        scale::Float64 = exp(offset)
        pointwise = plate(cumulative, weights, scale) do x, w, s
            x * w / s
        end
        total = sum(pointwise)
        extra = sum(abs2, cumulative)
        return total
    end
    xs = [1.0, 2.0, 3.0]
    for weights in (2.0, [2.0], [2.0, 3.0, 4.0])
        expected = sum(cumsum(xs) .* weights ./ exp(0.5))
        @test prepare(scan_broadcast_consumer)(xs, weights, 0.5) ≈ expected
        # A separate scan consumer keeps its vector and its own result.
        k = prepare(scan_broadcast_consumer; want = (:total, :extra))
        total, extra = k(xs, weights, 0.5)
        @test total ≈ expected
        @test extra == sum(abs2, cumsum(xs))
    end
    @test_throws DimensionMismatch prepare(scan_broadcast_consumer)(xs, ones(2), 0.5)

    @kernel scan_ref_consumer(xs) = begin
        cumulative = scan(xs; init = 0.0) do carry, x
            next = carry + x
            (next, next)
        end
        pointwise = plate(xs, Ref(cumulative)) do x, all_values
            x + sum(all_values)
        end
        return sum(pointwise)
    end
    @test prepare(scan_ref_consumer)(xs) == sum(xs .+ sum(cumsum(xs)))

    @kernel scan_scalar_consumer(xs, offset::Float64) = begin
        cumulative = scan(xs; init = 0.0) do carry, x
            next = carry + x
            (next, next)
        end
        scale::Float64 = exp(offset)
        pointwise = plate(cumulative, scale) do x, s
            log_scale = log(s)
            x / s - log_scale
        end
        return sum(pointwise)
    end
    scalar = prepare(scan_scalar_consumer)
    @test scalar(xs, 0.5) ≈ sum(cumsum(xs) ./ exp(0.5) .- 0.5)
    scalar(xs, 0.5)
    _authored_scan_allocated(scalar, xs, 0.5)
    @test _authored_scan_allocated(scalar, xs, 0.5) == 0

    @kernel scan_inside_plate(xs::Vector{Float64}, shifts::Vector{Float64}) = begin
        totals = plate(Ref(xs), shifts) do data, m
            values = scan(data, Ref(m); init = 0.0) do carry, x, shift
                next = carry + x + shift
                (next, next)
            end
            sum(values)
        end
        return totals
    end
    @test prepare(scan_inside_plate)(xs, [0.0, 1.0]) == [10.0, 16.0]

    @kernel mixed_scan_consumer(xs) = begin
        cumulative = scan(xs; init = 0.0) do carry, x
            next = carry + x
            (next, next)
        end
        pointwise = plate(cumulative) do x
            _authored_scan_mixed(x)
        end
        total = sum(pointwise)
        return total
    end
    fused = prepare(mixed_scan_consumer; want = (:pointwise, :total))(xs)
    materialized = prepare(mixed_scan_consumer;
        want = (:cumulative, :pointwise, :total))(xs)
    @test fused == ([1, 1.5, 1.5], 4.0)
    @test fused == materialized[2:3]
    @test typeof(fused[1]) == typeof(materialized[2])

    @kernel scan_plate_chain(xs, weights) = begin
        cumulative = scan(xs; init = 0.0) do carry, x
            next = carry + x
            (next, next)
        end
        scaled = plate(cumulative, weights) do x, w
            x * w
        end
        pointwise = plate(scaled) do x
            x^2
        end
        return sum(pointwise)
    end
    chain = prepare(scan_plate_chain)
    @test chain(xs, 2.0) == sum(abs2, 2 .* cumsum(xs))
    @test chain(xs, [2.0]) == sum(abs2, 2 .* cumsum(xs))
    @test_throws DimensionMismatch chain(xs, ones(2))
end

@testset "authored scan advances multiple sequences in lockstep" begin
    a, b = [0.9, 0.8, 0.5, -0.2], [1.0, -0.5, 0.2, 0.7]
    seq = _authored_scan_lockstep_reference(a, b)
    pointwise = -0.5 .* seq.^2
    for bound in ((;), (; a), (; a, b))
        args = isempty(bound) ? (a, b) :
               (haskey(bound, :b) ? () : (b,))
        for (want, expected) in (
                (:seq, seq), (:total, sum(pointwise)),
                ((:seq, :total), (seq, sum(pointwise))))
            k = prepare(authored_scan_lockstep; want, bound)
            actual = k(args...)
            @test want isa Tuple ? all(isapprox.(actual, expected)) : actual ≈ expected
        end
    end

    # The concrete posteriordb `prophet` logistic-trend recurrence:
    # m_i = m_{i-1} + (t_change[i] - m_{i-1}) * r[i], with t_change and r both
    # per-step sequences (one data, one parameter-derived).
    @kernel logistic_gamma(tchange::Vector{Float64}, r::Vector{Float64}, m0::Float64) = begin
        m = scan(tchange, r; init = m0) do carry, tc, rr
            next = carry + (tc - carry) * rr
            (next, next)
        end
        return m
    end
    gamma_ref(tchange, r, m0) = begin
        m = similar(tchange); c = m0
        for i in eachindex(tchange, r); c = c + (tchange[i] - c) * r[i]; m[i] = c; end
        m
    end
    tchange, r = [1.0, 2.0, 4.0, 8.0], [0.5, 0.25, 0.5, 0.1]
    kg = prepare(logistic_gamma)
    @test kg(tchange, r, 0.0) ≈ gamma_ref(tchange, r, 0.0)
    @test kg(tchange, r, 3.0) ≈ gamma_ref(tchange, r, 3.0)
    # Iterated sequences must share axes (lockstep): a length mismatch throws.
    @test_throws DimensionMismatch kg(tchange, [0.5, 0.25], 0.0)

    # Two iterated sequences PLUS a Ref-shared scalar gain.
    @kernel linrec(a::Vector{Float64}, b::Vector{Float64}, g::Float64) = begin
        s = scan(a, b, Ref(g); init = 0.0) do carry, ai, bi, gain
            next = gain * (ai * carry + bi)
            (next, next)
        end
        return s
    end
    linrec_ref(a, b, g) = begin
        s = similar(a); c = 0.0
        for i in eachindex(a, b); c = g * (a[i] * c + b[i]); s[i] = c; end
        s
    end
    kl = prepare(linrec)
    @test kl(a, b, 0.5) ≈ linrec_ref(a, b, 0.5)

    # A NamedTuple carry threads two derived quantities across both sequences.
    @kernel two_seq_nt(xs::Vector{Float64}, ys::Vector{Float64}) = begin
        out = scan(xs, ys; init = (; p = 0.0, q = 1.0)) do carry, x, y
            p2 = carry.p + x
            q2 = carry.q * y
            ((; p = p2, q = q2), p2 + q2)
        end
        return out
    end
    nt_ref(xs, ys) = begin
        o = similar(xs); p = 0.0; q = 1.0
        for i in eachindex(xs, ys); p += xs[i]; q *= ys[i]; o[i] = p + q; end
        o
    end
    @test prepare(two_seq_nt)([1.0, 2.0, 3.0], [2.0, 0.5, 4.0]) ≈
        nt_ref([1.0, 2.0, 3.0], [2.0, 0.5, 4.0])

    # An iterated (non-Ref) sequence may not follow a Ref(...) shared operand.
    @test_throws "iterated sequences must precede" (@eval @kernel bad_order(
            xs::Vector{Float64}, g::Float64, ys::Vector{Float64}) = begin
        s = scan(xs, Ref(g), ys; init = 0.0) do carry, x, gg, y
            (carry + x * gg + y, carry)
        end
        return s
    end)
end
