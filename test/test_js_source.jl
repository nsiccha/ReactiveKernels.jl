module TestJSSource

using ReactiveKernels
using Test

include(joinpath(@__DIR__, "js_node.jl"))

# Plain Julia helpers without annotations: branches, early returns and loops
# are translated from their typed code.
js_piecewise(x) = x < 0 ? -x : x <= 1 ? x^2 : 2x - 1
function js_early(x)
    x <= 1 && return log1p(x)
    return sqrt(x) + exp(-x) / x
end
function js_logistic_map(x0, r, n)
    x = x0
    for _ in 1:n
        x = r * x * (1 - x)
    end
    x
end
function js_cube_root(target, lo, hi)
    while hi - lo > 1e-12
        mid = (lo + hi) / 2
        if mid^3 > target
            hi = mid
        else
            lo = mid
        end
    end
    (lo + hi) / 2
end
function js_weighted_total(v)
    s = 0.0
    for i in eachindex(v)
        s += v[i] * i
    end
    s
end
js_squares(v) = [x^2 for x in v]
js_c_sin(x) = ccall(:sin, Float64, (Float64,), x)

@kernel js_scalar(a, b) = begin
    c = a * b + exp(-a) / (1 + b^2)
    return c
end
@kernel js_fused(x, a, b) = begin
    y = a .* x .+ b
    z = exp.(-y) .* sin.(x) ./ (1 .+ x .^ 2)
    return z
end
@kernel js_helpers(x) = begin
    p = js_piecewise.(x)
    e = js_early.(abs.(x) .+ 0.5)
    q = p .+ e
    return q
end
@kernel js_loops(x0, r, n, target) = begin
    y = js_logistic_map(x0, r, n)
    w = js_cube_root(target, 0.0, max(target, 1.0))
    return y
end
@kernel js_reductions(x) = begin
    s = sum(x)
    m = maximum(x)
    t = js_weighted_total(x)
    f = x[1] - x[end]
    return s
end
@kernel js_switch(x, flag, k) = begin
    y = ifelse(flag, x * k, x / k)
    return y
end
@kernel js_difference(x, y) = begin
    d = x .- y
    return d
end
@kernel js_comprehension(x) = begin
    y = js_squares(x)
    return y
end
@kernel js_branch(a, b) = begin
    y = if a > b
        a - b
    else
        log(b - a - 1)
    end
    return y
end
@kernel js_scan(x, decay, initial) = begin
    trajectory = scan(x; init = initial) do previous, observation
        current = decay * previous + observation
        (current, current)
    end
    return trajectory
end
@kernel js_plate(y, mu) = begin
    lp = plate(y, mu) do yi, mi
        -(yi - mi)^2 / 2
    end
    total = sum(lp)
    return total
end
@kernel js_c_call(x) = begin
    y = js_c_sin(x)
    return y
end
@kernel js_fma(a, b) = begin
    y = fma(a, b, 1.0)
    return y
end

const V = Vector{Float64}
const CASES = [
    (:js_scalar, js_scalar, (:a, :b), :c, (; a = Float64, b = Float64),
     [(0.5, 2.0), (-3.0, 0.0), (1e-300, 1e10)]),
    (:js_fused, js_fused, (:x, :a, :b), :z, (; x = V, a = Float64, b = Float64),
     [(collect(range(-3, 3; length = 41)), 0.7, -0.2), (Float64[], 1.0, 2.0), ([1.0], 0.0, 0.0)]),
    (:js_helpers, js_helpers, (:x,), :q, (; x = V), [(collect(range(-2, 3; length = 51)),), (Float64[],)]),
    (:js_loops, js_loops, (:x0, :r, :n, :target), (:y, :w), (; x0 = Float64, r = Float64, n = Int, target = Float64),
     [(0.3, 3.7, 0, 2.0), (0.3, 3.7, 1, 0.5), (0.3, 3.7, 50, 27.0), (0.2, 2.5, -4, 1.0)]),
    (:js_reductions, js_reductions, (:x,), (:s, :m, :t, :f), (; x = V),
     [(collect(range(-2, 3; length = 2001)),), ([4.0],), (Float64[],)]),
    (:js_switch, js_switch, (:x, :flag, :k), :y, (; x = Float64, flag = Bool, k = Int),
     [(3.0, true, 4), (3.0, false, 4), (3.0, false, 0)]),
    (:js_difference, js_difference, (:x, :y), :d, (; x = V, y = V),
     [([1.0, 2.0, 3.0], [0.5, 0.25, 0.125]), ([1.0, 2.0, 3.0], [1.0]), ([2.0], [1.0, 2.0, 3.0]),
      ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])]),
    (:js_comprehension, js_comprehension, (:x,), :y, (; x = V), [([1.0, -2.0, 3.5],), (Float64[],)]),
    # The arm not taken is not evaluated: `log` of a negative number is never
    # computed for a > b.
    (:js_branch, js_branch, (:a, :b), :y, (; a = Float64, b = Float64), [(5.0, 1.0), (1.0, 4.0), (1.0, 1.5)]),
    (:js_scan, js_scan, (:x, :decay, :initial), :trajectory, (; x = V, decay = Float64, initial = Float64),
     [([1.0, 0.0, 0.0, 2.0], 0.5, 0.0), (Float64[], 0.5, 1.0), (collect(range(0, 1; length = 1000)), 0.9, 2.0)]),
    (:js_plate, js_plate, (:y, :mu), :total, (; y = V, mu = V),
     [([1.0, 2.0, 3.0], [0.5, 2.5, 3.0]), ([1.0, 2.0], [0.0]), (Float64[], Float64[]), ([1.0, 2.0, 3.0], [1.0, 2.0])]),
]

agree(a::Real, b::Real) = (isnan(a) && isnan(b)) || isapprox(a, b; rtol = 1e-12, atol = 1e-15)
agree(a::AbstractVector, b::AbstractVector) = length(a) == length(b) && all(agree.(a, b))
agree(a, b) = false

@testset "js_source" begin
    @testset "emits a module for $name" for (name, k, have, want, types, _) in CASES
        source = ReactiveKernels.js_source(k; have, want, types, name)
        @test occursin("export function " * string(name) * "(" * join(string.(have), ", ") * ")", source)
        @test occursin("export const INPUTS", source) && occursin("export class JuliaError", source)
    end

    @testset "Node agrees with native: $name" for (name, k, have, want, types, cases) in CASES
        if !rk_js_node_available()
            @test_skip "no node on the PATH"
            continue
        end
        source = ReactiveKernels.js_source(k; have, want, types, name)
        wants = want isa Symbol ? (want,) : want
        native = prepare(k; have, want = wants)
        results = rk_js_run(source, [name => c for c in cases])
        for (c, r) in zip(cases, results)
            expected = try
                native(c...)
            catch e
                e
            end
            if expected isa Exception
                # The same Julia exception type, thrown by the JavaScript.
                @test haskey(r, :error) && r.error == string(nameof(typeof(expected)))
            else
                @test !haskey(r, :error)
                haskey(r, :error) && continue
                if want isa Symbol
                    @test agree(Float64.(expected), r.value)
                else
                    @test all(agree(Float64.(expected[j]), r.value[j]) for j in eachindex(wants))
                end
            end
        end
    end

    @testset "branching helpers need no annotation" begin
        source = ReactiveKernels.js_source(js_helpers; have = (:x,), want = :q, types = (; x = V))
        @test occursin("function js_piecewise(x) {", source)
        @test occursin("function js_early(x) {", source)
        @test occursin("if (x <= 1) {", source)
    end

    @testset "data-dependent iteration stays a loop" begin
        loops = ReactiveKernels.js_source(js_loops; have = (:x0, :r, :n, :target), want = :y,
                                          types = (; x0 = Float64, r = Float64, n = Int, target = Float64))
        @test occursin("while (true)", loops)
        scan = ReactiveKernels.js_source(js_scan; have = (:x, :decay, :initial), want = :trajectory,
                                         types = (; x = V, decay = Float64, initial = Float64))
        body = scan[findfirst("export function kernel(", scan)[1]:end]
        body = body[1:findfirst("\n}", body)[1]]
        @test count("for (let", body) == 1
        @test occursin("x.length", body)
    end

    @testset "unsupported constructs are named" begin
        e = try
            ReactiveKernels.js_source(js_c_call; have = (:x,), want = :y, types = (; x = Float64))
        catch err
            err
        end
        @test e isa ReactiveKernels.JSEmissionError
        @test occursin("C function `sin`", e.message) && occursin("js_c_sin", e.message)
        e = try
            ReactiveKernels.js_source(js_fma; have = (:a, :b), want = :y, types = (; a = Float64, b = Float64))
        catch err
            err
        end
        @test e isa ReactiveKernels.JSEmissionError
        @test occursin("`fma`", e.message)
    end

    @testset "inputs and names" begin
        @test_throws ArgumentError ReactiveKernels.js_source(js_switch; have = (:x, :flag, :k), want = :y,
            types = (; x = Float64, flag = Bool, k = Int), name = :switch)
        if rk_js_node_available()
            source = ReactiveKernels.js_source(js_scalar; have = (:a, :b), want = :c,
                                               types = (; a = Float64, b = Float64), name = :scalar)
            r = rk_js_call(source, :scalar, [1.0], 2.0)
            @test r.error == "ArgumentError" && occursin("`a` must be a number", r.message)
        else
            @test_skip "no node on the PATH"
        end
    end
end

end
