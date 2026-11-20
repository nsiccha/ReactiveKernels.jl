# Usage: julia --project=<env> factor_scaling.jl [rows] [levels,...] [laws,...] [forms,...] [bound-matrix columns]
include("harness.jl")
include("factor_variants.jl")

function measure_factor(law, form, n, k, matrix_columns)
    rng = MersenneTwister(42)
    groups = [1 + (i - 1) * k ÷ n for i in 1:n]
    x, z = randn(rng, n), randn(rng, n)
    y = law === :normal ? 0.3 .+ 0.7 .* x .+ randn(rng, n) :
        law === :bernoulli ? Int.(rand(rng, n) .< 1 ./ (1 .+ exp.(-x))) : rand(rng, 0:6, n)
    model, inputs, obs = factor_case(law, form, x, z, groups, y)
    if form === :bound_matrix && matrix_columns != size(inputs.X, 2)
        kept = min(matrix_columns, size(inputs.X, 2))
        X = hcat(view(inputs.X, :, 1:kept), randn(rng, n, matrix_columns - kept))
        inputs = (; inputs..., X)
    end
    GC.gc()
    preparation = @timed begin
        plan = model(; inputs...) | obs
        built = build_kernel(plan)
        u = 0.1 .* randn(MersenneTwister(1), built.layout.total)
        q = prepare_sampler(built, plan, u; backend=BACKEND)
        (; q, u)
    end
    q, u = preparation.value
    g = similar(u)
    value, _ = sampler_value_and_gradient!(q, g, u)
    isfinite(value) && all(isfinite, g) || error("nonfinite result for $law $form ($n, $k)")
    density = @b $q($u) seconds=0.3
    gradient = @b sampler_value_and_gradient!($q, $g, $u) seconds=0.3
    @printf("%-10s %-12s %6d %5d %10.4f %12.2f %12.0f %12.0f %10.0f %10.0f\n",
        law, form, n, k, preparation.time, preparation.bytes / 2^20,
        density.time * 1e9, gradient.time * 1e9, density.bytes, gradient.bytes)
    flush(stdout)
end

n = isempty(ARGS) ? 20000 : parse(Int, ARGS[1])
counts = length(ARGS) < 2 ? [50, 500, 5000] : parse.(Int, split(ARGS[2], ','))
laws = length(ARGS) < 3 ? [:normal, :bernoulli] : Symbol.(split(ARGS[3], ','))
forms = length(ARGS) < 4 ? [:full, :reference, :missing, :bound_matrix, :hcat, :scan] :
    Symbol.(split(ARGS[4], ','))
matrix_columns = length(ARGS) < 5 ? 2 : parse(Int, ARGS[5])
matrix_columns > 0 || error("bound-matrix columns must be positive")
println("bound-matrix columns: ", matrix_columns)
println("law        form           rows levels    setup_s    setup_MiB   density_ns  val_grad_ns  density_B val_grad_B")
for law in laws, form in forms
    # A fresh model for each preparation excludes sampler-cache hits. The
    # small case first warms Julia methods and is printed with its row count.
    measure_factor(law, form, 100, 5, matrix_columns)
end
for k in counts, law in laws, form in forms
    measure_factor(law, form, n, k, matrix_columns)
end
