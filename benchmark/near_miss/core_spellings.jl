# Sibling spellings of native fast paths, on the core package alone: for each
# spelling whether the prepared program holds the fast path's helper, the time
# of one call and the bytes it allocates.
#   julia --project=<benchmark environment> benchmark/near_miss/core_spellings.jl
# A caller that has defined its own `@b` (one that reads CPU time on a loaded
# host, say) is used as it is; otherwise Chairmarks provides it.
using ReactiveKernels, Printf, Random
isdefined(@__MODULE__, Symbol("@b")) || @eval using Chairmarks
const RK = ReactiveKernels
const log1pexp = RK.LogExpFunctions.log1pexp

# --- logarithm of a sum of exponentials in a summed plate (RK 0.1.373)
@kernel logs_nested(x::Vector{Float64}) = begin
    cells = plate(x) do v
        log((exp(v) + exp(-v)) + exp(2.0 * v))
    end
    total::Float64 = sum(cells)
    return total
end
@kernel logs_flat(x::Vector{Float64}) = begin
    cells = plate(x) do v
        log(exp(v) + exp(-v) + exp(2.0 * v))
    end
    total::Float64 = sum(cells)
    return total
end
@kernel logs_right(x::Vector{Float64}) = begin
    cells = plate(x) do v
        log(exp(v) + (exp(-v) + exp(2.0 * v)))
    end
    total::Float64 = sum(cells)
    return total
end
@kernel logs_weighted(x::Vector{Float64}) = begin
    cells = plate(x) do v
        log(0.3 * exp(v) + 0.7 * exp(-v))
    end
    total::Float64 = sum(cells)
    return total
end
@kernel logs_four(x::Vector{Float64}) = begin
    cells = plate(x) do v
        log(((exp(v) + exp(-v)) + exp(2.0 * v)) + exp(0.5 * v))
    end
    total::Float64 = sum(cells)
    return total
end

@kernel logs_five(x::Vector{Float64}) = begin
    cells = plate(x) do v
        log(exp(v) + exp(-v) + exp(2.0 * v) + exp(0.5 * v) + exp(-0.5 * v))
    end
    total::Float64 = sum(cells)
    return total
end
@kernel logs_scaled_right(x::Vector{Float64}) = begin
    cells = plate(x) do v
        log(exp(v) * 2 + 0.5 * (exp(-v) + exp(2.0 * v)))
    end
    total::Float64 = sum(cells)
    return total
end
# A negative factor: the argument can be negative, the written logarithm stays.
@kernel logs_signed(x::Vector{Float64}) = begin
    cells = plate(x) do v
        log(1.5 * exp(v) + -0.25 * exp(-v))
    end
    total::Float64 = sum(cells)
    return total
end

# --- softplus beside another term in a summed plate (RK 0.1.372)
@kernel soft_abs2(x::Vector{Float64}, w::Vector{Float64}) = begin
    cells = plate(x, w) do v, z
        log1pexp(v) + 0.5 * abs2(z)
    end
    total::Float64 = sum(cells)
    return total
end
@kernel soft_square(x::Vector{Float64}, w::Vector{Float64}) = begin
    cells = plate(x, w) do v, z
        log1pexp(v) + 0.5 * z^2
    end
    total::Float64 = sum(cells)
    return total
end
@kernel soft_product(x::Vector{Float64}, w::Vector{Float64}) = begin
    cells = plate(x, w) do v, z
        log1pexp(v) + 0.5 * (z * z)
    end
    total::Float64 = sum(cells)
    return total
end
@kernel soft_spelled(x::Vector{Float64}, w::Vector{Float64}) = begin
    cells = plate(x, w) do v, z
        log1p(exp(v)) + 0.5 * abs2(z)
    end
    total::Float64 = sum(cells)
    return total
end
@kernel soft_sqrt(x::Vector{Float64}, w::Vector{Float64}) = begin
    cells = plate(x, w) do v, z
        log1pexp(v) + sqrt(abs2(z) + 1.0)
    end
    total::Float64 = sum(cells)
    return total
end

function report(group, name, k, helper, args...)
    source = string(code_expr(k))
    value = k(args...)
    trial = @b $k($(args)...) seconds = 0.3
    @printf("CORE\t%s\t%s\t%s\t%.1f\t%d\t%.12g\n", group, name, join((string(occursin(h, source)) for h in helper), "/"),
        trial.time * 1e9, trial.bytes, value isa Number ? value : sum(value))
    flush(stdout)
end

println(join(("CORE", "group", "spelling", "helper present", "ns per call", "bytes per call", "value"), '\t'))
x = sin.(collect(1.0:4096.0))
w = cos.(collect(1.0:4096.0))
for (name, spec) in (("(exp(v) + exp(-v)) + exp(2v)", logs_nested), ("exp(v) + exp(-v) + exp(2v)", logs_flat),
                     ("exp(v) + (exp(-v) + exp(2v))", logs_right), ("0.3 exp(v) + 0.7 exp(-v)", logs_weighted),
                     ("four terms, nested left", logs_four), ("five terms, flat", logs_five),
                     ("exp(v) * 2 + 0.5 (exp(-v) + exp(2v))", logs_scaled_right),
                     ("1.5 exp(v) - 0.25 exp(-v), on abs(v)", logs_signed))
    try
        report("vector log, 4096 cells", name, prepare(spec), ("_log_exp_sum",), spec === logs_signed ? abs.(x) : x)
    catch err
        println("CORE\tvector log\t$name\terror\t", first(sprint(showerror, err), 200))
    end
end
for (name, spec) in (("log1pexp(v) + 0.5 * abs2(z)", soft_abs2), ("log1pexp(v) + 0.5 * z^2", soft_square),
                     ("log1pexp(v) + 0.5 * (z * z)", soft_product), ("log1p(exp(v)) + 0.5 * abs2(z)", soft_spelled),
                     ("log1pexp(v) + sqrt(abs2(z) + 1)", soft_sqrt))
    try
        report("vector softplus, 4096 cells", name, prepare(spec), ("_vector_log1pexp", "_vector_rk_log1pexp"), 3.0 .* x, w)
    catch err
        println("CORE\tvector softplus\t$name\terror\t", first(sprint(showerror, err), 200))
    end
end
println("vector doubles: ", RK._vector_doubles())

# --- single-use gathered rows in a position reduction (a4bf732e1)
@kernel weighted_rows(X, W, index) = begin
    rows = W[index, :]
    value = vec(sum(X .* rows; dims=2))
    return value
end
@kernel weighted_rows_apart(X, W, index) = begin
    rows = W[index, :]
    scaled = X .* rows
    value = vec(sum(scaled; dims=2))
    return value
end
@kernel weighted_rows_dropdims(X, W, index) = begin
    rows = W[index, :]
    value = dropdims(sum(X .* rows; dims=2); dims=2)
    return value
end
rng = MersenneTwister(7)
n, k, m, positions = 2000, 6, 50, 16
X = randn(rng, n, k)
W = randn(rng, m, k, positions)
index = rand(rng, 1:m, n)
for (name, spec) in (("fused spelling", weighted_rows), ("product on its own line", weighted_rows_apart),
                     ("dropdims instead of vec", weighted_rows_dropdims))
    reader = vectorize(prepare(spec); batched = :W, reuse = true)
    for (kind, ix) in (("Vector{Int}", index), ("Vector{Int32}", Int32.(index)), ("Vector{UInt16}", UInt16.(index)),
                       ("view of a Vector{Int}", view(vcat(index, index), 1:n)), ("range 1:1:50", nothing))
        # A range needs as many rows of W as of X: take the first 50 rows of X.
        Xk, ix = ix === nothing ? (X[1:m, :], 1:1:m) : (X, ix)
        try
            report("gathered row sum, $(size(Xk, 1)) rows x 6, 16 positions", "$name, index $kind", reader, ("_lane_gathered_sum",), Xk, W, ix)
        catch err
            println("CORE\tgathered row sum\t$name, index $kind\terror\t", first(sprint(showerror, err), 200))
        end
    end
end
println("CORE-END")
