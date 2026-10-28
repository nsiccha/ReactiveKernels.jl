using JSON, SciMLBase, Enzyme, StanBlocks
include("prototype.jl")
using .NativeBDFPrototype

decay(t, y, rate) = [-rate * y[1]]
ts = collect(0.001:0.001:0.02)
rt, at, limit = 1e-6, 1e-8, 4096
a = interval_bdf(decay, [1.0], 0.0, ts, rt, at, limit, 0.7)
b = continuous_bdf(decay, [1.0], 0.0, ts, rt, at, limit, 0.7)
counts = step_counts(decay, [1.0], 0.0, ts, rt, at, 0.7)

p = Float64[]
layout = NativeBDFPrototype._pack!(p, (0.7,))
prob = ODEProblem(NativeBDFPrototype.OriginalTimeRHS(decay, layout),
    [1.0], (0.0, last(ts)), p)
sol = solve(prob, NativeBDFPrototype.ODEBDF(); saveat = ts,
    save_start = false, save_everystep = false, reltol = rt, abstol = at,
    maxiters = limit + 1, sensealg = NativeBDFPrototype._sensitivity())
continuous_steps = sol.stats.naccept + sol.stats.nreject

budget_limit = maximum(counts)
budget_primal = try
    c = continuous_bdf(decay, [1.0], 0.0, ts, rt, at, budget_limit, 0.7;
        budget = true)
    Dict("status" => "pass", "shape" => collect(size(c)))
catch e
    Dict("status" => "fail", "error" => sprint(showerror, e))
end

report = Dict(
    "scope" => "public primal comparison, not original-model performance acceptance",
    "times" => ts,
    "controls" => [rt, at, limit],
    "interval_steps" => counts,
    "interval_total_steps" => sum(counts),
    "continuous_total_steps" => continuous_steps,
    "max_abs_difference" => maximum(abs.(a .- b)),
    "max_abs_interval_oracle_error" => maximum(abs.(a[:, 1] .- exp.(-0.7 .* ts))),
    "max_abs_continuous_oracle_error" => maximum(abs.(b[:, 1] .- exp.(-0.7 .* ts))),
    "continuous_budget_primal" => budget_primal,
    "comparison_budget_limit" => budget_limit,
    "original_token_native_method_count" => length(methods(StanBlocks.ode_bdf_tol)),
)
println(JSON.json(report))
