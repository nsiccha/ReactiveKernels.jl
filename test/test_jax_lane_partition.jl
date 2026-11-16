module JAXLanePartitionTests
using ReactiveKernels, Test
include("jax_python.jl")

# A per-row lower threshold (`-Inf`: none), as RKPPL's evidence laws emit it:
# the parameter's support guard outside, the data branches inside its arm.
# The censored arm stands for an expensive tail (a continued fraction).
@kernel lane_censored(u::Vector{Float64}, y::Vector{Float64}, lo::Vector{Float64}) = begin
    values = plate(y, lo) do yi, li
        s = exp(u[2])
        s > 0 ? (yi < li ? -Inf : (yi <= li ? log1p(-exp(-exp((li - u[1]) / s))) :
                                   -0.5 * ((yi - u[1]) / s)^2 - log(s))) : -Inf
    end
    total = sum(values)
end

# Two independent data branches in one cell, and the plate's value kept.
@kernel lane_two(u::Vector{Float64}, y::Vector{Float64}, w::Vector{Float64}) = begin
    values = plate(y, w) do yi, wi
        a = wi > 0 ? log(wi) * u[1] : 0.0
        b = isfinite(yi) ? -0.5 * (yi - u[2])^2 : u[2]
        a + b
    end
    total = sum(values)
end

const LANE_CENSORED_DATA = (
    mixed = ([0.3, -1.0, 1.2, -0.4, 2.0], [-Inf, -1.0, -Inf, -0.4, -Inf]),
    observed = ([0.3, 1.2, 2.0], [-Inf, -Inf, -Inf]),
    censored = ([-1.0, -0.4], [-1.0, -0.4]),
    empty = (Float64[], Float64[]),
)

@testset "jax_source partitions a plate by its data-only branches" begin
    src = jax_source(lane_censored; want = :total, data = (:y, :lo))
    text = string(src)
    # `prepare_data` returns each pattern's cells and the restoring order.
    for key in ("values_lanes_1", "values_lanes_2", "values_lanes_3", "values_lane_order")
        @test occursin(key, text)
    end
    # One cell per pattern, none deciding a data condition: the only
    # `jax.lax.cond` left is the parameter guard, and the censored arm (a
    # stable `log1p(-x)`) is in one cell alone.
    functions = split(text, "def kernel")[end]
    @test count("def values_cell", functions) == 3
    @test !occursin(r"jax\.lax\.cond\([^,]*\bli", functions)
    @test count("_log1p_neg_of(", functions) == 1
    native = prepare(lane_censored; want = :total)
    if rk_jax_available()
        for (name, (y, lo)) in pairs(LANE_CENSORED_DATA), u in ([0.2, -0.3], [-0.5, 0.4])
            got = rk_jax_evaluate(src, :kernel, (u, y, lo); grad = true, prepare = :arguments)
            @test got.value ≈ native(u, y, lo) rtol = 1e-13
            @test got.grad ≈ rk_fd_gradient(v -> native(v, y, lo), u) rtol = 1e-5 atol = 1e-7
        end
        # A row below its threshold is impossible: the value is -Inf there.
        got = rk_jax_evaluate(src, :kernel, ([0.2, -0.3], [-2.0, 0.5], [-1.0, -Inf]);
                              prepare = :arguments)
        @test got.value == -Inf
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end

@testset "a lane partition's patterns follow the cell, not the data" begin
    src = jax_source(lane_two; want = (:total, :values), data = (:y, :w))
    text = string(src)
    # Two independent conditions: four patterns, each with its plate.
    @test occursin("values_lanes_4", text) && !occursin("values_lanes_5", text)
    native = prepare(lane_two; want = (:total, :values))
    if rk_jax_available()
        for (y, w) in (([0.1, Inf, -0.3, NaN], [2.0, 0.5, -1.0, 0.0]),
                       ([0.1, 0.2], [1.0, 3.0]),
                       (Float64[], Float64[]))
            u = [0.7, -0.2]
            got = rk_jax_evaluate(src, :kernel, (u, y, w); prepare = :arguments)
            total, values = native(u, y, w)
            @test got.value[1] ≈ total
            @test got.value[2] ≈ values
        end
    else
        @test_skip "RK_JAX_PYTHON not set"
    end
end

end
