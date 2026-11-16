module TestJSNestedVectors

using ReactiveKernels
using Test

const V = Vector{Float64}
const VV = Vector{V}

@kernel nested_literal(x) = begin
    y = [x]
    return y
end

@kernel nested_groups(xs, scale, offset) = begin
    y = plate(xs, scale, offset) do row, a, b
        a .* row .+ b
    end
    return y
end
const GROUPS = prepare(nested_groups; have = (:xs, :scale, :offset), want = :y,
                       bound = (; offset = 0.25))
@kernel nested_prepared(x, scale) = begin
    y = GROUPS([x], scale)
    return y
end

@kernel nested_comprehension(xs, scale) = begin
    y = [scale .* row for row in xs if !isempty(row)]
    return y
end

function nested_loop(xs, enabled)
    total = 0.0
    if enabled
        for row in xs
            for value in row
                total += value * value
            end
        end
    end
    return total
end
@kernel nested_guarded(xs, enabled) = begin
    y = nested_loop(xs, enabled)
    return y
end

function nested_alias(x, shift)
    isempty(x) && return 0.0
    owned = copy(x)
    rows = vec([owned, owned])
    rows[1][1] += shift
    return rows[2][1]
end
@kernel nested_owned(x, shift) = begin
    y = nested_alias(x, shift)
    return y
end

@kernel nested_identity(xs) = begin
    y = copy(vec(xs))
    return y
end

@kernel nested_flatten(xs) = begin
    y = reduce(vcat, xs)
    return y
end

@kernel nested_flatten_guarded(xs, enabled) = begin
    y = enabled ? reduce(vcat, xs) : Float64[]
    return y
end

function nested_fill(x, n)
    rows = Vector{Vector{Float64}}(undef, n)
    for i in eachindex(rows)
        rows[i] = x .+ i
    end
    return vcat(rows, [copy(x)])
end
@kernel nested_allocated(x, n) = begin
    y = nested_fill(x, n)
    return y
end

@kernel nested_access(xs, enabled, i) = begin
    y = enabled ? xs[i][1] : 0.0
    return y
end

@kernel nested_undefined(n) = begin
    rows = Vector{Vector{Float64}}(undef, n)
    y = rows[1]
    return y
end

agree(x::Real, y::Real) = isequal(x, y) || isapprox(x, y; rtol = 1e-12, atol = 1e-14)
agree(x::AbstractArray, y::AbstractArray) = size(x) == size(y) && all(agree.(x, y))

function check(native, source, name, cases)
    for vectors in (:array, :float64array)
        for (args, outcome) in zip(cases, ReactiveKernels.js_run(source, name, cases; vectors))
            expected = native(args...)
            @test outcome.error === nothing
            @test !outcome.inputs_mutated
            outcome.error === nothing || continue
            @test typeof(outcome.value) === typeof(expected)
            @test agree(outcome.value, expected)
        end
    end
end

@testset "JavaScript nested vectors" begin
    node = ReactiveKernels._js_default_node() !== nothing
    for (spec, have, types, cases) in (
        (nested_literal, (:x,), (; x = V), [(Float64[],), ([1.0, -0.0, Inf],)]),
        (nested_prepared, (:x, :scale), (; x = V, scale = Float64),
            [(Float64[], 2.0), ([1.0], -0.5), (collect(1.0:37.0), 0.25)]),
        (nested_groups, (:xs, :scale, :offset), (; xs = VV, scale = Float64, offset = Float64),
            [(VV(), 2.0, 1.0), ([Float64[], [1.0, 2.0], [-3.0]], 0.5, -2.0),
             ([[Float64(i + j) for j in 1:i] for i in 1:23], -2.0, 0.5)]),
        (nested_comprehension, (:xs, :scale), (; xs = VV, scale = Float64),
            [(VV(), 2.0), ([Float64[], [1.0], [2.0, 3.0]], 0.5)]),
        (nested_guarded, (:xs, :enabled), (; xs = VV, enabled = Bool),
            [(VV(), true), ([Float64[], [2.0, -3.0]], true), ([[NaN]], false)]),
        (nested_owned, (:x, :shift), (; x = V, shift = Float64),
            [(Float64[], 3.0), ([1.0, 2.0], -4.0)]),
        (nested_identity, (:xs,), (; xs = Vector{VV}),
            [(Vector{VV}(),), ([VV(), [Float64[], [1.0]]],)]),
        (nested_identity, (:xs,), (; xs = Vector{Vector{Bool}}),
            [(Vector{Bool}[],), ([Bool[], [true, false]],)]),
        (nested_identity, (:xs,), (; xs = Vector{Vector{Int64}}),
            [(Vector{Int64}[],), ([Int64[], [1, -2]],)]),
        (nested_allocated, (:x, :n), (; x = V, n = Int64),
            [(Float64[], 0), ([1.0, 2.0], 3)]),
        (nested_access, (:xs, :enabled, :i), (; xs = VV, enabled = Bool, i = Int64),
            [(VV(), false, 10), ([Float64[]], false, 1), ([[2.0]], true, 1)]),
        (nested_flatten, (:xs,), (; xs = VV),
            [([Float64[]],), ([Float64[], [1.0, 2.0], [-3.0]],),
             ([collect(1.0:i) for i in 1:31],)]),
        (nested_flatten, (:xs,), (; xs = Vector{VV}),
            [([VV(), [Float64[], [1.0]]],)]),
    )
        native = prepare(spec; have, want = :y)
        source = ReactiveKernels.js_source(spec; have, want = :y, types, name = :nested)
        if node
            check(native, source, :nested, cases)
        else
            @test_skip "no node on the PATH"
        end
    end

    @testset "bound ragged data retains iteration" begin
        sources = String[]
        flattened = String[]
        for xs in (VV(), [Float64[], [1.0, 2.0]], [collect(1.0:i) for i in 1:31])
            native = prepare(nested_groups; have = (:xs, :scale, :offset), want = :y,
                             bound = (; xs, offset = 0.25))
            source = ReactiveKernels.js_source(native; types = (; scale = Float64), name = :bound)
            push!(sources, source)
            node && check(native, source, :bound, [(0.5,), (-3.0,)])
            flat = prepare(nested_flatten_guarded; have = (:xs, :enabled), want = :y, bound = (; xs))
            flat_source = ReactiveKernels.js_source(flat; types = (; enabled = Bool), name = :flatten)
            push!(flattened, flat_source)
            if node
                check(flat, flat_source, :flatten, isempty(xs) ? [(false,)] : [(false,), (true,)])
                if isempty(xs)
                    native_error = try flat(true) catch err; err end
                    @test native_error isa Exception
                    @test only(ReactiveKernels.js_run(flat_source, :flatten, [(true,)])).error == string(nameof(typeof(native_error)))
                end
            end
        end
        # Data literal storage grows, but executable bodies and loop counts do not.
        for emitted in (sources, flattened)
            bodies = [replace(s, r"(?m)^const _rk_bound_.*\n\n" => "") for s in emitted]
            @test all(==(first(bodies)), bodies)
            @test count("for (let ", first(bodies)) >= 2
        end
    end

    if node
        @testset "nested input checks and reference errors" begin
            source = ReactiveKernels.js_source(nested_access; have = (:xs, :enabled, :i), want = :y,
                types = (; xs = VV, enabled = Bool, i = Int64), name = :access)
            cases = [(VV(), true, 1), ([Float64[]], true, 1), ([1.0], false, 1), ([[true]], false, 1)]
            outcomes = ReactiveKernels.js_run(source, :access, cases)
            @test [o.error for o in outcomes] == ["BoundsError", "BoundsError", "ArgumentError", "ArgumentError"]
            native = prepare(nested_access; have = (:xs, :enabled, :i), want = :y)
            @test_throws BoundsError native(cases[1]...)
            @test_throws BoundsError native(cases[2]...)

            source = ReactiveKernels.js_source(nested_undefined; have = (:n,), want = :y,
                types = (; n = Int64), name = :uninitialized)
            outcomes = ReactiveKernels.js_run(source, :uninitialized, [(1,), (-1,)])
            @test [o.error for o in outcomes] == ["UndefRefError", "ArgumentError"]
            @test_throws UndefRefError prepare(nested_undefined; have = (:n,), want = :y)(1)
            @test_throws ArgumentError prepare(nested_undefined; have = (:n,), want = :y)(-1)

            source = ReactiveKernels.js_source(nested_flatten; have = (:xs,), want = :y,
                types = (; xs = VV), name = :flatten)
            outcome = only(ReactiveKernels.js_run(source, :flatten, [(VV(),)]))
            native_error = try prepare(nested_flatten; have = (:xs,), want = :y)(VV()) catch err; err end
            @test native_error isa Exception
            @test outcome.error == string(nameof(typeof(native_error)))
        end
    end
end

end
