module EmptyPlateInvariantTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

@kernel inline_shared(x, scale::Float64) = begin
    cells = plate(x) do xi
        xi * sqrt(scale)
    end
    total = sum(cells)
    materialized = sum(abs2, cells)
end
@kernel named_shared(x, scale::Float64) = begin
    cells = plate(x) do xi
        root = sqrt(scale)
        xi * root
    end
    total = sum(cells)
    materialized = sum(abs2, cells)
end
@kernel atomic_shared(x, scale::Float64) = begin
    cells = plate(x, Ref(scale)) do xi, s
        xi * sqrt(s)
    end
    total = sum(cells)
    materialized = sum(abs2, cells)
end
@kernel outside_shared(x, scale::Float64) = begin
    root = sqrt(scale)
    cells = plate(x) do xi
        xi * root
    end
end
@kernel only_shared(x, scale::Float64) = begin
    cells = plate(x) do xi
        sqrt(scale)
    end
    total = sum(cells)
    materialized = sum(abs2, cells)
end

columns(weights) = vec(sum(weights; dims=1))
@kernel computed_selection(x, weights::Matrix{Float64}, column::Int) = begin
    cells = plate(x) do xi
        xi * columns(weights)[column]
    end
    total = sum(cells)
    materialized = sum(abs2, cells)
end

@kernel outer_shared(groups, scale::Float64) = begin
    cells = plate(groups) do xs
        root = sqrt(scale)
        inner = plate(xs) do xi
            xi * root
        end
        sum(inner)
    end
    total = sum(cells)
    materialized = sum(abs2, cells)
end
@kernel inner_shared(groups, scale::Float64) = begin
    cells = plate(groups) do xs
        inner = plate(xs) do xi
            xi * sqrt(scale)
        end
        sum(inner)
    end
    total = sum(cells)
    materialized = sum(abs2, cells)
end

const CALLS = Ref(0)
counted_root(scale) = (CALLS[] += 1; sqrt(scale))
@kernel counted_flat(x, scale::Float64) = begin
    cells = plate(x) do xi
        xi * counted_root(scale)
    end
    total = sum(cells)
end
@kernel counted_outer(groups, scale::Float64) = begin
    cells = plate(groups) do xs
        root = counted_root(scale)
        inner = plate(xs) do xi
            xi * root
        end
        sum(inner)
    end
    total = sum(cells)
end
@kernel counted_inner(groups, scale::Float64) = begin
    cells = plate(groups) do xs
        inner = plate(xs) do xi
            xi * counted_root(scale)
        end
        sum(inner)
    end
    total = sum(cells)
end

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

function check_scale(spec, port, data, sums; invalid_empty=false)
    saved = deepcopy(data)
    for bound in (false, true)
        fixed = bound ? NamedTuple{(port,)}((data,)) : NamedTuple()
        args(scale) = bound ? (scale,) : (data, scale)
        for want in (:cells, :total, :materialized)
            reader = prepare(spec; have=(port, :scale), want, bound=fixed)
            for scale in (0.25, 4.0)
                expected = want === :cells ? sums .* sqrt(scale) :
                    want === :total ? sum(sums) * sqrt(scale) : sum(abs2, sums) * scale
                @test (@inferred reader(args(scale)...)) ≈ expected
                if want !== :cells
                    derivative = want === :total ? sum(sums) / (2sqrt(scale)) : sum(abs2, sums)
                    ad = prepare_ad(reader, BACKEND, args(scale)...; active=:scale)
                    value, gradient = ad_value_and_gradient(ad, args(scale)...)
                    @test value ≈ expected
                    @test gradient ≈ derivative
                end
            end
            if invalid_empty
                expected = want === :cells ? zero.(sums) : 0.0
                @test (@inferred reader(args(-1.0)...)) == expected
                if want !== :cells
                    ad = prepare_ad(reader, BACKEND, args(-1.0)...; active=:scale)
                    @test ad_value_and_gradient(ad, args(-1.0)...) == (0.0, 0.0)
                end
            else
                # The authored sqrt is evaluated at a visited cell, so its
                # ordinary Julia domain error must remain observable.
                @test_throws DomainError reader(args(-1.0)...)
            end
        end
    end
    @test data == saved
end

@testset "empty plates skip shared work and keep concrete results" begin
    for spec in (inline_shared, named_shared, atomic_shared),
        x in (Float64[], [0.5], [-1.0, 0.5, 2.0], zeros(0, 3), zeros(3, 0),
              reshape(collect(1.0:6.0), 2, 3))
        check_scale(spec, :x, x, x; invalid_empty=isempty(x))
    end
    # The author placed this expression outside the plate, so it still runs
    # for an empty domain and keeps sqrt's ordinary Julia domain error.
    @test_throws DomainError prepare(outside_shared)(Float64[], -1.0)
    for x in (Float64[], [0.5, 2.0], zeros(2, 0))
        check_scale(only_shared, :x, x, one.(x); invalid_empty=isempty(x))
    end
    for spec in (inline_shared, named_shared, atomic_shared, only_shared)
        reader = prepare(spec; want=:cells, bound=(; x=Float64[], scale=-1.0))
        @test (@inferred reader()) == Float64[]
    end
end

@testset "nested empty domains preserve the scope of shared work" begin
    for spec in (outer_shared, inner_shared),
        groups in (Vector{Float64}[], [Float64[]], [Float64[], Float64[]],
                   [[0.5]], [Float64[], [1.0, -0.5], Float64[], [2.0]])
        sums = [sum(xs) for xs in groups]
        invalid_empty = isempty(groups) || (spec === inner_shared && all(isempty, groups))
        check_scale(spec, :groups, groups, sums; invalid_empty)
    end
end

@testset "shared call-result indexing is lazy only over empty domains" begin
    for rows in (2, 5), n in (0, 1, 7), bound in (false, true)
        x = [0.2 + 0.1i for i in 1:n]
        weights = reshape(sin.(1.0:2rows), rows, 2)
        saved = copy(weights)
        for column in (2, 20), want in (:cells, :total, :materialized)
            fixed = bound ? (; x, column) : NamedTuple()
            reader = prepare(computed_selection; have=(:x, :weights, :column), want, bound=fixed)
            args = bound ? (weights,) : (x, weights, column)
            if n > 0 && column == 20
                # A visited cell retains the bounds check of ordinary indexing.
                @test_throws BoundsError reader(args...)
                continue
            end
            selected = n == 0 ? 0.0 : sum(weights[:, column])
            expected = want === :cells ? x .* selected :
                want === :total ? sum(x) * selected : sum(abs2, x) * selected^2
            @test (@inferred reader(args...)) ≈ expected
            if want !== :cells
                ad = prepare_ad(reader, BACKEND, args...; active=:weights)
                value, gradient = ad_value_and_gradient(ad, args...)
                reference = zeros(size(weights))
                if n > 0
                    reference[:, column] .= want === :total ? sum(x) : 2sum(abs2, x)*selected
                end
                @test value ≈ expected
                @test gradient ≈ reference
            end
            @test weights == saved
        end
    end
end

@testset "shared work retains its nonempty-domain reuse" begin
    for (spec, port, data, calls) in
        ((counted_flat, :x, Float64[], 0), (counted_flat, :x, ones(17), 1),
         (counted_outer, :groups, Vector{Float64}[], 0),
         (counted_outer, :groups, [Float64[], Float64[]], 1),
         (counted_outer, :groups, [ones(3) for _ in 1:17], 1),
         (counted_inner, :groups, [Float64[], Float64[]], 0),
         (counted_inner, :groups, [Float64[], ones(3), Float64[], ones(5)], 2))
        for bound in (false, true), want in (:cells, :total)
            fixed = bound ? NamedTuple{(port,)}((data,)) : NamedTuple()
            reader = prepare(spec; have=(port, :scale), want, bound=fixed)
            for scale in (1.0, 4.0)
                CALLS[] = 0
                bound ? reader(scale) : reader(data, scale)
                @test CALLS[] == calls
            end
        end
    end
end
end
