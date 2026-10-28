using ReactiveKernels, Test

module ErrorPolicyFixture
using ReactiveKernels

@traceable function checked_square(x)
    x > 0 || throw(ArgumentError("positive x required"))
    @assert x != -2 "asserted square"
    x * x
end
@traceable function checked_total(X)
    total = 0.0
    for i in eachindex(X)
        total += checked_square(X[i])
    end
    total
end
@traceable function checked_nested_difference(X, mu)
    Y = X .- mu
    total = 0.0
    for i in eachindex(Y)
        total += Y[i] * Y[i]
        if Y[i] < 0
            Y[i] > -2 || throw(ArgumentError("nested guard"))
        end
    end
    total
end
@kernel score(x) = checked_square(x) + 1
@kernel total(X) = checked_total(X)
@kernel nested(X, mu) = checked_nested_difference(X, mu)
@kernel dotted(X) = sum(checked_square.(X))
@kernel converted(X) = sum(Float64.(view(X, 1:length(X))))
@kernel assertion(x) = begin
    result = begin
        @assert x != 2 "visible assertion"
        x * x
    end
    return result
end
@kernel message(x) = begin
    result = begin
        x > 0 || throw(ArgumentError(error("message must not be evaluated")))
        x * x
    end
    return result
end
@kernel direct(x) = begin
    result = begin
        x > 0 || throw(ArgumentError("direct guard"))
        @assert x != -2 "direct assertion"
        x * x
    end
    return result
end
opaque(x) = (x > 0 || throw(ArgumentError("opaque")); x * x)
@kernel opaque_score(x) = opaque(x)
@kernel bound_score(x, data) = begin
    seed = checked_square(data)
    result = seed + x
    return result
end
end

@testset "opt-in visible throw stripping" begin
    F = ErrorPolicyFixture
    @test_throws ArgumentError F.checked_square(-1.0)
    @test_throws ArgumentError prepare(F.score)(-1.0)
    @test prepare(F.score; on_error = :ignore)(-1.0) == 2.0
    @test prepare(F.score; on_error = :ignore)(-2.0) == 5.0
    @test prepare(F.score)(2.0) == 5.0
    @test_throws ArgumentError prepare(F.score)(-1.0)
    @test_throws ArgumentError prepare(F.direct)(-1.0)
    @test_throws AssertionError prepare(F.assertion)(2.0)
    @test prepare(F.assertion; on_error = :ignore)(2.0) == 4.0
    @test prepare(F.message; on_error = :ignore)(-1.0) == 1.0
    @test prepare(F.direct; on_error = :ignore)(-2.0) == 4.0
    @test prepare(F.total; on_error = :ignore)([2.0, -1.0, 3.0]) == 14.0
    @test prepare(F.dotted; on_error = :ignore)([2.0, -1.0, 3.0]) == 14.0
    @test prepare(F.converted; on_error = :ignore)(Float32[2, -1, 3]) === 4.0
    @test_throws ArgumentError prepare(F.nested)([-3.0, 2.0], 0.5)
    @test prepare(F.nested; on_error = :ignore)([-3.0, 2.0], 0.5) == 14.5
    @test_throws ArgumentError prepare(F.opaque_score; on_error = :ignore)(-1.0)
    @test_throws ArgumentError prepare(F.score; on_error = :anything_else)
    @test_throws ArgumentError prepare(F.bound_score; bound = (; data = -2.0))
    @test prepare(F.bound_score; bound = (; data = -2.0), on_error = :ignore)(3.0) == 7.0
    @test_throws ArgumentError prepare(F.bound_score; bound = (; data = -2.0))
    @test prepare(F.bound_score; bound = (; data = -3.0), on_error = :ignore)(3.0) == 12.0
    cache = PreparationCache()
    @test prepare!(cache, F.score; on_error = :ignore)(-2.0) == 5.0
    @test_throws ArgumentError prepare!(cache, F.score)(-2.0)
    @test prepare!(cache, F.bound_score; bound = (; data = -2.0), on_error = :ignore)(3.0) == 7.0
    @test prepare!(cache, F.bound_score; bound = (; data = -3.0), on_error = :ignore)(3.0) == 12.0
    @test_throws ArgumentError prepare!(cache, F.bound_score; bound = (; data = -2.0))
end
