module TestJSReshape

using ReactiveKernels, Test

@kernel reshape_window(xs, lo, hi, rows, cols) = begin
    window = view(xs, lo:hi)
    matrix = reshape(window, rows, cols)
    values = copy(matrix)
    total = sum(matrix)
    return values, total
end

@kernel reshape_vector(xs, rows, cols) = begin
    matrix = reshape(xs, rows, cols)
    values = copy(matrix)
    total = sum(matrix)
    return values, total
end

@kernel gather_column(matrix, indices, column) = begin
    values = matrix[indices, column]
    return values
end

@kernel borrowed_column_plate(matrix, indices, column, offsets) = begin
    selected = ReactiveKernels._readonly_gather(view(matrix, :, column), indices)
    values = plate(selected, offsets) do x, offset
        x + offset
    end
    return values
end

@kernel borrowed_window_plate(xs, lo, hi, indices, offsets) = begin
    selected = ReactiveKernels._readonly_gather(view(xs, lo:hi), indices)
    values = plate(selected, offsets) do x, offset
        x + offset
    end
    return values
end

function borrowed_alias_read(matrix, indices, column, value)
    parent = copy(matrix)
    selected = ReactiveKernels._readonly_gather(view(parent, :, column), indices)
    parent[indices[1], column] = value
    return selected[1]
end

@kernel borrowed_alias(matrix, indices, column, value) = begin
    result = borrowed_alias_read(matrix, indices, column, value)
    return result
end

function guarded_weighted_sum(xs, weights, enabled)
    total = 0.0
    for i in 1:length(weights)
        if enabled
            total += xs[i] * weights[i]
        end
    end
    return total
end

@kernel view_helper(xs, lo, hi, weights, enabled) = begin
    window = view(xs, lo:hi)
    total = guarded_weighted_sum(window, weights, enabled)
    return total
end

function reshape_aliases(xs, lo, hi, rows, cols, value)
    parent = copy(xs)
    matrix = reshape(view(parent, lo:hi), rows, cols)
    matrix[1, 2] = value
    parent[lo] = value + 1
    return [parent[lo + rows], matrix[1, 1], sum(parent)]
end

@kernel reshape_owned(xs, lo, hi, rows, cols, value) = begin
    result = reshape_aliases(xs, lo, hi, rows, cols, value)
    return result
end

function check(native, source, cases)
    for vectors in (:array, :float64array)
        for (args, result) in zip(cases, ReactiveKernels.js_run(source, :reshaped, cases; vectors))
            @test result.error === nothing
            @test !result.inputs_mutated
            if result.error === nothing
                expected = native(args...)
                if expected isa NamedTuple
                    @test result.value.values ≈ expected.values
                    @test result.value.total ≈ expected.total
                else
                    @test result.value ≈ expected
                end
            end
        end
    end
end

@testset "JavaScript matrix reshape shares vector storage" begin
    want = (; values = :values, total = :total)
    have = (:xs, :lo, :hi, :rows, :cols)
    types = (; xs = Vector{Float64}, lo = Int64, hi = Int64, rows = Int64, cols = Int64)
    native = prepare(reshape_window; have, want)
    source = ReactiveKernels.js_source(native; types, name = :reshaped)
    cases = [(Float64[], 1, 0, 0, 0), (Float64[], 1, 0, 0, 3),
        (Float64[], 1, 0, 3, 0), (collect(1.0:8.0), 2, 7, 2, 3),
        (collect(1.0:8.0), 2, 7, 6, 1), (collect(1.0:32.0), 2, 31, 5, 6)]
    node = ReactiveKernels._js_default_node() !== nothing
    node && check(native, source, cases)
    plain = prepare(reshape_vector; have = (:xs, :rows, :cols), want)
    plainsource = ReactiveKernels.js_source(plain;
        types = (; xs = Vector{Float64}, rows = Int64, cols = Int64), name = :reshaped)
    node && check(plain, plainsource, [(Float64[], 0, 3), (Float64[], 3, 0),
        (collect(1.0:6.0), 2, 3), (collect(1.0:30.0), 5, 6)])
    for T in (Int64, Bool)
        typedsource = ReactiveKernels.js_source(plain;
            types = (; xs = Vector{T}, rows = Int64, cols = Int64), name = :reshaped)
        node && check(plain, typedsource, [(T[], 0, 3), (T[0, 1, 1, 0, 0, 1], 2, 3)])
    end

    owned = prepare(reshape_owned; have = (have..., :value), want = :result)
    ownedsource = ReactiveKernels.js_source(owned; types = (; types..., value = Float64), name = :reshaped)
    node && check(owned, ownedsource, [(collect(1.0:8.0), 2, 7, 3, 2, 10.0),
        (collect(1.0:32.0), 2, 31, 5, 6, -2.0)])

    if node
        bad = [(collect(1.0:8.0), 2, 7, 2, 4), (collect(1.0:8.0), 2, 7, -1, 2),
            (collect(1.0:8.0), 2, 9, 2, 4)]
        errors = [DimensionMismatch, DimensionMismatch, BoundsError]
        for (args, error) in zip(bad, errors)
            @test_throws error native(args...)
        end
        for vectors in (:array, :float64array)
            @test [r.error for r in ReactiveKernels.js_run(source, :reshaped, bad; vectors)] == string.(nameof.(errors))
        end
        # Julia's view reshape permits negative axes, unlike an Array. The JS
        # row-array representation cannot hold them; report that boundary.
        @test size(reshape(view(collect(1.0:8.0), 2:7), -2, -3)) == (-2, -3)
        refused = only(ReactiveKernels.js_run(source, :reshaped, [(collect(1.0:8.0), 2, 7, -2, -3)]))
        @test refused.error == "ArgumentError"
        @test occursin("nonnegative matrix dimensions", refused.message)
    end
    sources = [ReactiveKernels.js_source(prepare(reshape_window; have, want, bound = (; xs));
        types = (; lo = Int64, hi = Int64, rows = Int64, cols = Int64), name = :reshaped)
        for xs in (Float64[], collect(1.0:8.0), collect(1.0:32.0))]
    bodies = [replace(s, r"(?m)^const _rk_bound_.*\n\n" => "") for s in sources]
    @test all(==(first(bodies)), bodies)
    @test count("Object.defineProperty", first(bodies)) == 1
end

@testset "JavaScript matrix indexed column" begin
    native = prepare(gather_column; have = (:matrix, :indices, :column), want = :values)
    types = (; matrix = Matrix{Float64}, indices = Vector{Int64}, column = Int64)
    source = ReactiveKernels.js_source(native; types, name = :reshaped)
    if ReactiveKernels._js_default_node() !== nothing
        check(native, source, [(reshape(collect(1.0:6.0), 3, 2), [3, 1, 3], 2),
            (zeros(0, 3), Int[], 2), (zeros(3, 2), Int[], 1)])
        bad = [(zeros(3, 2), [0], 1), (zeros(3, 2), [4], 1),
            (zeros(3, 2), Int[], 3), (zeros(3, 0), Int[], 1)]
        for args in bad
            @test_throws BoundsError native(args...)
        end
        for vectors in (:array, :float64array)
            @test all(r -> r.error == "BoundsError", ReactiveKernels.js_run(source, :reshaped, bad; vectors))
        end
    end
    sources = [ReactiveKernels.js_source(prepare(gather_column;
        have = (:matrix, :indices, :column), want = :values, bound = (; indices));
        types = (; matrix = Matrix{Float64}, column = Int64), name = :reshaped)
        for indices in (Int[], [1, 2, 1], ones(Int, 257))]
    bodies = [replace(s, r"(?m)^const _rk_bound_.*\n\n" => "") for s in sources]
    @test all(==(first(bodies)), bodies)
    @test count("for (let k =", first(bodies)) == 1
end

@testset "JavaScript borrowed gathers retain their parent" begin
    have = (:matrix, :indices, :column, :offsets)
    types = (; matrix = Matrix{Float64}, indices = Vector{Int64}, column = Int64, offsets = Vector{Float64})
    native = prepare(borrowed_column_plate; have, want = :values)
    source = ReactiveKernels.js_source(native; types, name = :reshaped)
    node = ReactiveKernels._js_default_node() !== nothing
    if node
        check(native, source, [(reshape(collect(1.0:6.0), 3, 2), [3, 1, 3], 2, [0.5]),
            (ones(3, 2), [2], 1, [1.0, 2.0, 3.0]), (zeros(0, 2), Int[], 2, Float64[])])
        bad = [(ones(3, 2), [0], 1, [0.0]), (ones(3, 2), [4], 1, [0.0]),
            (ones(3, 2), Int[], 3, Float64[])]
        for args in bad
            @test_throws BoundsError native(args...)
        end
        @test all(r -> r.error == "BoundsError", ReactiveKernels.js_run(source, :reshaped, bad))
    end
    window = prepare(borrowed_window_plate; have = (:xs, :lo, :hi, :indices, :offsets), want = :values)
    windowsource = ReactiveKernels.js_source(window;
        types = (; xs = Vector{Float64}, lo = Int64, hi = Int64, indices = Vector{Int64}, offsets = Vector{Float64}),
        name = :reshaped)
    node && check(window, windowsource, [(collect(1.0:8.0), 2, 7, [6, 1, 6], [0.5]),
        (Float64[], 1, 0, Int[], Float64[])])
    alias = prepare(borrowed_alias; have = (:matrix, :indices, :column, :value), want = :result)
    aliassource = ReactiveKernels.js_source(alias;
        types = (; matrix = Matrix{Float64}, indices = Vector{Int64}, column = Int64, value = Float64), name = :reshaped)
    node && check(alias, aliassource, [(ones(3, 2), [2, 2], 1, 17.0)])
    sources = [ReactiveKernels.js_source(prepare(borrowed_column_plate; have, want = :values, bound = (; indices));
        types = (; matrix = Matrix{Float64}, column = Int64, offsets = Vector{Float64}), name = :reshaped)
        for indices in (Int[], [1, 2, 1], ones(Int, 257))]
    bodies = [replace(s, r"(?m)^const _rk_bound_.*\n\n" => "") for s in sources]
    @test all(==(first(bodies)), bodies)
end

@testset "JavaScript helpers preserve view arguments and lazy reads" begin
    have = (:xs, :lo, :hi, :weights, :enabled)
    types = (; xs = Vector{Float64}, lo = Int64, hi = Int64, weights = Vector{Float64}, enabled = Bool)
    native = prepare(view_helper; have, want = :total)
    source = ReactiveKernels.js_source(native; types, name = :reshaped)
    if ReactiveKernels._js_default_node() !== nothing
        check(native, source, [(collect(1.0:8.0), 2, 4, [0.2, -0.1, 2.0], true),
            (Float64[], 1, 0, Float64[], true), (Float64[], 1, 0, ones(257), false),
            (ones(259), 2, 258, ones(257), true)])
        bad = [(Float64[], 1, 0, [1.0], true)]
        @test_throws BoundsError native(only(bad)...)
        @test only(ReactiveKernels.js_run(source, :reshaped, bad)).error == "BoundsError"
    end
    sources = [ReactiveKernels.js_source(prepare(view_helper; have, want = :total, bound = (; xs));
        types = (; lo = Int64, hi = Int64, weights = Vector{Float64}, enabled = Bool), name = :reshaped)
        for xs in (Float64[], ones(3), ones(257))]
    bodies = [replace(s, r"(?m)^const _rk_bound_.*\n\n" => "") for s in sources]
    @test all(==(first(bodies)), bodies)
end

end
