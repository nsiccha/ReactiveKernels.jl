module TestJSPrepared

using ReactiveKernels
using Test

# Prepared kernels in JavaScript modules: a kernel prepared with bound data is
# emitted on its own (`js_source(kernel)`) and called from another kernel's
# recipe, its bound data part of the module (snag emit-an-rk-pictu-b9fa64e8).

const NODE = ReactiveKernels._js_default_node() !== nothing
const V = Vector{Float64}

@kernel js_weighted_residual(x::Vector{Float64}, w::Vector{Float64}, y::Vector{Float64},
                             a::Float64, b::Float64, sigma::Float64) = begin
    mu = muladd.(b, x, a)
    scales = sigma .* w
    cells = plate(y, mu, scales) do yi, mi, si
        si > 0 ? -log(si) - 0.5 * ((yi - mi) / si)^2 : -Inf
    end
    total::Float64 = sum(cells)
    return total
end

@kernel js_weighted(x, scale, weights) = begin
    y = scale .* x .+ weights
    total = sum(y)
    return y
end
const SCALE = 2.0
const WEIGHTS = [0.5, -1.0, 3.0]
const BOUND = (; scale = SCALE, weights = WEIGHTS)
const WEIGHTED = prepare(js_weighted; have = (:x, :scale, :weights), want = :y, bound = BOUND)
const WEIGHTED_PAIR = prepare(js_weighted; have = (:x, :scale, :weights), want = (:y, :total), bound = BOUND)
const WEIGHTED_SHAPED = prepare(js_weighted; have = (:x, :scale, :weights),
                                want = (values = :y, total = :total), bound = BOUND)
const WEIGHTED_EXEMPLAR = prepare(js_weighted; have = (:x, :scale, :weights), want = :total,
                                  bound = BOUND, exemplars = ([1.0, 2.0, 3.0],))

# A bound matrix is a frozen module constant of frozen rows (snag
# js-source-stack-1e1692d1), as a bound vector is.
@kernel js_projected(x, basis) = begin
    y = basis * x
    return y
end
const BASIS = [1.0 0.5; -2.0 0.25; 0.0 3.0]
const PROJECTED = prepare(js_projected; have = (:x, :basis), want = :y, bound = (; basis = BASIS))

# A recipe calling the prepared kernel, twice with the same argument types.
@kernel js_calls_prepared(x, shift) = begin
    y = WEIGHTED(x)
    z = WEIGHTED(x .+ shift) .- y
    return z
end

# An input check (`input_check!`, a function bound to a global name) runs at
# the module function's entry, as at the native kernel's.
function js_check_three(x)
    length(x) == 3 || throw(DimensionMismatch("x needs 3 entries, " * "got $(length(x))"))
    return nothing
end
@kernel js_checked(x) = begin
    y = [v for v in x] .* 2.0
    return y
end
ReactiveKernels.input_check!(js_checked, :x, js_check_three, :x)

function js_check_size(x, n)
    length(x) == n || throw(DimensionMismatch("packed size"))
    nothing
end
@kernel js_bound_window(x, n, window, settings) = begin
    selected = view(x, window)
    y = plate(selected) do value
        settings.gain * value + settings.offset
    end
    return y
end
ReactiveKernels.input_check!(js_bound_window, :x, js_check_size, :x, :n)

# Vector literals and concatenation (RKPPL's affine coefficients are `vcat([e])`).
@kernel js_literals(x, a, b) = begin
    v = vcat([a, b], x, a)
    return v
end

const CASES = [([1.0, 2.0, 3.0],), ([1.0],), ([1.0, 2.0],), (Float64[],), ([NaN, Inf, -0.0],)]

agree(a::Real, b::Real) = (isnan(a) && isnan(b)) || isapprox(a, b; rtol = 1e-12, atol = 1e-15)
agree(a::AbstractVector, b::AbstractVector) = length(a) == length(b) && all(agree.(a, b))
agree(a::Tuple, b::Tuple) = length(a) == length(b) && all(agree.(a, b))
agree(a::NamedTuple, b::NamedTuple) = keys(a) == keys(b) && agree(Tuple(a), Tuple(b))
agree(a, b) = false

# Each case's outcome in Node against the native kernel: the same value and
# type, or the same Julia exception type.
function check_against(native, source, entry, cases)
    for vectors in (:array, :float64array)
        for (c, r) in zip(cases, ReactiveKernels.js_run(source, entry, cases; vectors))
            expected = try
                native(c...)
            catch e
                e
            end
            @test !r.inputs_mutated
            if expected isa Exception
                @test r.julia_error && r.error == string(nameof(typeof(expected)))
            else
                @test r.error === nothing
                r.error === nothing || continue
                @test typeof(r.value) == typeof(expected)
                @test agree(expected, r.value)
            end
        end
    end
end

@testset "js_source of prepared kernels" begin
    @testset "a prepared kernel's module holds its bound data" begin
        source = ReactiveKernels.js_source(WEIGHTED; types = (; x = V), name = :weighted)
        @test occursin("export function weighted(x)", source)
        # Bound arrays and numbers are data stored outside the executable body.
        @test occursin("const _rk_bound_weights = Object.freeze([0.5, -1.0, 3.0]);", source)
        @test occursin("const _rk_bound_scale = 2.0;", source)
        @test occursin("export const INPUTS = [{ name: \"x\", type: \"Vector{Float64}\" }];", source)
        if NODE
            check_against(WEIGHTED, source, :weighted, CASES)
        else
            @test_skip "no node on the PATH"
        end
    end

    @testset "a bound matrix is a frozen module constant" begin
        source = ReactiveKernels.js_source(PROJECTED; types = (; x = V), name = :projected)
        @test occursin("const _rk_bound_basis = Object.freeze([Object.freeze([1.0, 0.5]), " *
                       "Object.freeze([-2.0, 0.25]), Object.freeze([0.0, 3.0])]);", source)
        if NODE
            check_against(PROJECTED, source, :projected, [([1.0, 2.0],), ([1.0],), (Float64[],)])
        else
            @test_skip "no node on the PATH"
        end
    end

    @testset "tuple and shaped WANTs return as the kernel does" begin
        pair = ReactiveKernels.js_source(WEIGHTED_PAIR; types = (; x = V), name = :pair)
        shaped = ReactiveKernels.js_source(WEIGHTED_SHAPED; types = (; x = V), name = :shaped)
        @test occursin("export const RETURNS = { \"values\": \"y\", \"total\": \"total\" };", shaped)
        if NODE
            check_against(WEIGHTED_PAIR, pair, :pair, CASES)
            check_against(WEIGHTED_SHAPED, shaped, :shaped, CASES)
        else
            @test_skip "no node on the PATH"
        end
    end

    @testset "the exemplar types are the default types" begin
        source = ReactiveKernels.js_source(WEIGHTED_EXEMPLAR; name = :exemplar)
        @test occursin("type: \"Vector{Float64}\"", source)
        NODE ? check_against(WEIGHTED_EXEMPLAR, source, :exemplar, CASES) : @test_skip "no node on the PATH"
        # without exemplars, an abstract HAVE type needs `types`
        @test_throws ArgumentError ReactiveKernels.js_source(WEIGHTED)
    end

    @testset "prepared least squares emits its authored rows" begin
        for weights in (repeat([0.5, 1.5], 30), [0.0; ones(59)])
            bound = (; x = sin.(1:60), w = weights, y = cos.(1:60))
            kernel = prepare(js_weighted_residual; bound)
            @test any(s -> s isa ReactiveKernels._LSWeightedStatistics, kernel.constants)
            source = ReactiveKernels.js_source(kernel; types = (Float64, Float64, Float64), name = :residual)
            if NODE
                check_against(kernel, source, :residual, [(0.2, 0.5, 0.8), (0.1, -0.1, 0.0)])
            else
                @test_skip "no node on the PATH"
            end
        end
    end

    @testset "an input check runs in the module" begin
        source = ReactiveKernels.js_source(js_checked; have = (:x,), want = :y, types = (; x = V), name = :checked)
        @test occursin("js_check_three(x);", source)
        if NODE
            native = prepare(js_checked; have = (:x,), want = :y)
            check_against(native, source, :checked, CASES)
        else
            @test_skip "no node on the PATH"
        end
    end

    @testset "vector literals and concatenation" begin
        source = ReactiveKernels.js_source(js_literals; have = (:x, :a, :b), want = :v,
                                           types = (; x = V, a = Float64, b = Float64), name = :literals)
        if NODE
            native = prepare(js_literals; have = (:x, :a, :b), want = :v)
            check_against(native, source, :literals, [([1.0, 2.0], 0.5, -1.0), (Float64[], 3.0, 4.0)])
        else
            @test_skip "no node on the PATH"
        end
    end

    @testset "a recipe calling a prepared kernel" begin
        source = ReactiveKernels.js_source(js_calls_prepared; have = (:x, :shift), want = :z,
                                           types = (; x = V, shift = Float64), name = :calls)
        # one function for the kernel, called twice; its data emitted once
        @test count("function _rk_prepared_y(x) {", source) == 1
        @test count("_rk_prepared_y(", source) == 3
        @test count("Object.freeze(", source) == 1
        if NODE
            native = prepare(js_calls_prepared; have = (:x, :shift), want = :z)
            check_against(native, source, :calls, [(c[1], 0.5) for c in CASES])
        else
            @test_skip "no node on the PATH"
        end
    end

    @testset "bound sizes, ranges and record numbers stay in data storage" begin
        sources = String[]
        for (n, window, gain, offset) in ((0, 1:0, 0.5, -1.0), (3, 1:3, 2.0, 0.1), (17, 2:16, -0.5, 7.0))
            native = prepare(js_bound_window; have = (:x, :n, :window, :settings), want = :y,
                bound = (; n, window, settings = (; gain, offset)))
            source = ReactiveKernels.js_source(native; types = (; x = V), name = :windowed)
            push!(sources, source)
            NODE && check_against(native, source, :windowed, [(collect(1.0:n),), (ones(n + 1),)])
        end
        bodies = [replace(source, r"(?m)^const _rk_bound_.*\n\n" => "") for source in sources]
        # Fresh kernel closure arguments have fresh names, independent of data.
        # Compare their executable structure up to that alpha-renaming only.
        bodies = map(bodies) do body
            names = Dict{String,String}()
            replace(body, r"\bargument_\d+\b" => name -> get!(names, name) do
                "argument_local_$(length(names))"
            end)
        end
        @test all(==(first(bodies)), bodies)
        @test count("for (let ", first(bodies)) >= 1
    end
end

end
