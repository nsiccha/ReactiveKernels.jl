# Emit a generic censored Student-t regression twice: as `jax_source` emits it
# (`partitioned.py`), and with the lane partition switched off in this
# process only (`whole.py`, one `jax.vmap` of the response plate, as before
# RK 0.1.277). Writes the raw data, native log densities and gradients at the
# benchmark points, and the parameter layout to `inputs.json`.
#
#     julia --project=ENV benchmark/jax_censored_lanes/emit.jl OUT [ROWS] [CENSORED]
using ReactiveKernels, ReactiveKernelsPPL, Random
import Enzyme
using DifferentiationInterface: AutoEnzyme, gradient

out = ARGS[1]
rows = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12_245
censored = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 115
mkpath(out)

const MODEL = @rkppl begin
    alpha ~ Normal(0, 2)
    beta ~ Normal(0, 1)
    sigma ~ Exponential(1)
    nu ~ Gamma(2, 5)
    y .~ censored.(StudentT.(nu, alpha .+ beta .* x, sigma), limit, Inf)
end

rng = Xoshiro(20261116)
x = randn(rng, rows)
y = 0.5 .+ 0.8 .* x .+ 0.6 .* (randn(rng, rows) ./ sqrt.(randexp(rng, rows) ./ 1.5))
# A detection limit on `censored` rows, reported at the limit; none elsewhere.
limit = fill(-Inf, rows)
for i in randperm(rng, rows)[1:censored]
    limit[i] = y[i] + 0.3 * rand(rng)
    y[i] = limit[i]
end
data = (; x, limit)
plan = MODEL(; data...) | (; y)
built = build_kernel(plan)

write(joinpath(out, "partitioned.py"), string(jax_source(built, plan)))
@eval ReactiveKernels _jax_lane_partition(g::Graph, r::Recipe, dependent::Set{ValueId}) = nothing
write(joinpath(out, "whole.py"), string(jax_source(built, plan)))

query = prepare_query(built, plan, :sampler)
backend = AutoEnzyme(; mode = Enzyme.Reverse)
points = [unconstrain(built.layout, (; alpha = a, beta = b, sigma = s, nu = n))
          for (a, b, s, n) in ((0.5, 0.8, 0.6, 3.0), (0.3, 0.9, 0.7, 1.7), (0.6, 0.7, 0.5, 8.0),
                               (0.4, 0.8, 0.65, 2.4))]
json(v::AbstractVector) = "[" * join(map(json, v), ", ") * "]"
json(v::Real) = isfinite(v) ? repr(Float64(v)) : (v > 0 ? "\"inf\"" : "\"-inf\"")
open(joinpath(out, "inputs.json"), "w") do io
    print(io, "{\"y\": ", json(y), ", \"x\": ", json(x), ", \"limit\": ", json(limit),
          ", \"points\": [", join((json(p) for p in points), ", "), "]",
          ", \"log_density\": ", json([query(p) for p in points]),
          ", \"gradient\": [", join((json(gradient(query, backend, p)) for p in points), ", "), "]}")
end
println("rows=", rows, " censored=", count(isfinite, limit))
