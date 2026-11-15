module TestShapedWant

# A shaped WANT (a NamedTuple, or a tuple holding one or a tuple) prepares the
# kernel of its distinct leaves and returns each result in the shape, as the
# `js_source` and `jax_source` modules of the same WANT do (snag
# `kernelspec-prepa-ca8878b6`).

using ReactiveKernels
using Test

@kernel shaped_decay(grid::Vector{Float64}, rate::Float64, enabled::Bool) = begin
    curve = exp.(-rate .* grid)
    half = log(2.0) / rate
    root = sqrt(rate)
    n = length(grid)
    return curve
end

@kernel shaped_keywords(x::Float64; scale::Float64 = 2.0) = begin
    y = x * scale
    z = y + 1
    return y
end

@testset "shaped WANT" begin
    have = (:grid, :rate, :enabled)
    args = ([0.0, 0.5, 1.0], 2.0, true)
    flat = prepare(shaped_decay; have, want = (:curve, :half, :root, :n, :enabled))
    curve, half, root, n, enabled = flat(args...)

    @testset "a NamedTuple returns its fields, aliases included" begin
        kernel = prepare(shaped_decay; have,
                         want = (curve = :curve, half = :half, root = :root, count = :n, enabled = :enabled))
        result = @inferred kernel(args...)
        @test result isa NamedTuple{(:curve, :half, :root, :count, :enabled),
                                    Tuple{Vector{Float64},Float64,Float64,Int,Bool}}
        @test result.curve == curve && result.half === half && result.root === root
        @test result.count === n && result.enabled === enabled
        @test inputs(kernel) == inputs(flat)
        @test occursin("ShapedKernel", sprint(show, kernel))
    end

    @testset "shapes nest; a port named twice is computed once" begin
        kernel = prepare(shaped_decay; have, want = (a = (:half, :n), b = (c = :root,)))
        @test @inferred(kernel(args...)) === (a = (half, n), b = (c = root,))
        kernel = prepare(shaped_decay; have, want = (:half, (r = :root,)))
        @test @inferred(kernel(args...)) === (half, (r = root,))
        kernel = prepare(shaped_decay; have, want = (x = :half, y = :half, z = (:half,)))
        @test length(outputs(kernel)) == 1
        @test @inferred(kernel(args...)) === (x = half, y = half, z = (half,))
        kernel = prepare(shaped_decay; have, want = (only = :n,))
        @test @inferred(kernel(args...)) === (only = n,)
    end

    @testset "a flat WANT keeps its positional result" begin
        @test prepare(shaped_decay; have, want = :half)(args...) === half
        @test prepare(shaped_decay; have, want = (:half,))(args...) === half
        @test prepare(shaped_decay; have, want = (:half, :n))(args...) === (half, n)
    end

    @testset "with bound data, a cache and the authored signature" begin
        bound = prepare(shaped_decay; have, want = (count = :n, half = :half),
                        bound = (; grid = args[1]))
        @test @inferred(bound(args[2], args[3])) === (count = n, half = half)
        cache = PreparationCache()
        cached = prepare!(cache, shaped_decay; have, want = (count = :n, half = :half))
        @test @inferred(cached(args...)) === (count = n, half = half)
        @test prepare!(cache, shaped_decay; have, want = (half = :half,),
                       bound = (; grid = args[1]))(args[2], args[3]) === (half = half,)
        keywords = prepare(shaped_keywords; want = (scaled = :y, shifted = :z))
        @test @inferred(keywords(1.5)) === (scaled = 3.0, shifted = 4.0)
        @test @inferred(keywords(1.5; scale = 3.0)) === (scaled = 4.5, shifted = 5.5)
    end

    @testset "malformed shapes and other entry points are refused by name" begin
        @test_throws ArgumentError prepare(shaped_decay; have, want = (a = 1,))
        @test_throws ArgumentError prepare(shaped_decay; have, want = NamedTuple())
        @test_throws KeyError prepare(shaped_decay; have, want = (a = :missing_port,))
        message = try
            plan(shaped_decay; have, want = (count = :n,))
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("names the fields of a prepared result", message)
        @test occursin("`prepare`", message) && occursin("`js_source`", message)
    end
end

end
