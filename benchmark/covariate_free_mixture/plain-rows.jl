# Backend-only control: nested Bernoulli-logit sums, ordinary Julia primal.
# M names an implementation strategy; data dimensions remain runtime values.
using Enzyme, LogExpFunctions, Test
struct PlainRows{M} end
function (::PlainRows{M})(a, offsets, events) where M
    result = 0.0
    rows = length(a)
    for i in eachindex(events)
        observed = events[i]
        n = length(observed)
        if M === :strict
            rows == n || throw(DimensionMismatch())
        else
            (rows == 1 || rows == n) || throw(DimensionMismatch())
        end
        if M === :stride
            indices = Base.StepRangeLen(1, Int(rows != 1), n)
            for (j, y) in zip(indices, observed)
                eta = a[j] + offsets[i]
                result -= log1pexp(ifelse(y, -eta, eta))
            end
        else
            count = M === :extent ? ifelse(rows == 1, n, rows) : n
            for j in 1:count
                row = M === :strict ? j : ifelse(rows == 1, 1, j)
                eta = a[row] + offsets[i]
                result -= log1pexp(ifelse(observed[j], -eta, eta))
            end
        end
    end
    result
end
function reference(a, offsets, events)
    total = 0.0
    da, doffsets = zero(a), zero(offsets)
    for i in eachindex(events), j in eachindex(events[i])
        k = length(a) == 1 ? 1 : j
        eta = a[k] + offsets[i]
        total += events[i][j] ? -log1pexp(-eta) : -log1pexp(eta)
        slope = events[i][j] - logistic(eta)
        da[k] += slope
        doffsets[i] += slope
    end
    total, da, doffsets
end
function check(f, a, offsets, events; count=false)
    expected, da, doffsets = reference(a, offsets, events)
    ga, goffsets = zero(a), zero(offsets)
    function gradient!()
        fill!(ga, 0.0); fill!(goffsets, 0.0)
        Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active,
            Enzyme.Duplicated(a, ga), Enzyme.Duplicated(offsets, goffsets),
            Enzyme.Const(events))
        nothing
    end
    gradient!(); gradient!()
    @test f(a, offsets, events) ≈ expected rtol=1e-13 atol=1e-13
    @test ga ≈ da rtol=1e-12 atol=1e-11
    @test goffsets ≈ doffsets rtol=1e-12 atol=1e-11
    if count
        ccall(:begin_count, Cvoid, ())
        gradient!()
        packed = ccall(:end_count, UInt64, ())
        println((; mode=typeof(f), N=length(events), rows=length(a),
            native=(Int(packed >> 32), Int(packed & 0xffffffff)),
            julia_bytes=@allocated(gradient!())))
        flush(stdout)
    end
end
println((; julia=VERSION, enzyme=pkgversion(Enzyme),
    library=Enzyme.Compiler.Enzyme_jll.libEnzyme_path))
@testset "ordinary Julia row-loop controls" begin
    for mode in (:observed, :extent, :stride, :strict)
        f = PlainRows{mode}()
        for N in (1, 11, 146)
            events = [Bool[isodd(i+j) for j in 1:5] for i in 1:N]
            offsets = [0.1*cos(i) for i in 1:N]
            a = [0.2*sin(j) for j in 1:5]
            check(f, a, offsets, events; count=true)
        end
        if mode !== :strict
            check(f, [0.3], [0.1, -0.1, 0.5],
                [Bool[], Bool[true], Bool[true, false, true, true]])
            check(f, [0.3], Float64[], Vector{Bool}[])
        end
    end
end
