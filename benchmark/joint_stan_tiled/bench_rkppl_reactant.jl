# Reactant side of the tiled Stan-vs-RKPPL comparison: full-program
# `Reactant.@compile` of the sampler-cut posterior (primal) and
# `compile_ad_value_and_gradient` (Enzyme-through-Reactant) at tiling K.
#
#   TILE_K=K [RK_GRAD=1] [RK_OPT=default|no_slice_slice] [REPS=50] \
#     julia --project=<env with Reactant+Enzyme+DI> bench_rkppl_reactant.jl
#
# Prints parity against the native kernels at the benchmark's random point
# (seed 20260917) and per-call means, then ONE `RESULT {…}` JSON line whose
# fields merge into a `results.json` cell (`reactant_*`).  Timings are for
# device-resident inputs (`Reactant.to_rarray` once, outside the loop) —
# the shape an HMC loop keeps.
using ReactiveKernels
using ReactiveKernelsPPL
using DifferentiationInterface
using Enzyme
using Reactant
using Printf
using Random

const PPLT = normpath(joinpath(@__DIR__, "..", "..", "packages",
    "ReactiveKernelsPPL", "test"))
include(joinpath(PPLT, "parity", "joint_parity_fixture.jl"))
include(joinpath(PPLT, "parity", "joint_tiling.jl"))

K = parse(Int, get(ENV, "TILE_K", "3"))
DO_GRAD = get(ENV, "RK_GRAD", "1") == "1"
OPT = get(ENV, "RK_OPT", "default")
REPS = parse(Int, get(ENV, "REPS", "50"))

cols = tiled_columns(K)
plan = final_plan("continuous")
t_bind = @elapsed bound = bind_data(plan, cols; dims = Dict(:kernel_nsub_pk_loc => 3K))
t_build = @elapsed k = build_kernel(bound)
post_q = prepare_query(k, bound, :sampler)
u = Vector{Float64}(0.1 .* randn(Xoshiro(20260917), k.layout.total))
native = Base.invokelatest(post_q, u)
@printf("K=%d n_obs=%d u_dim=%d bind=%.1fs build=%.1fs native lp=%.9f\n",
    K, bound.n_obs, k.layout.total, t_bind, t_build, native)
flush(stdout)

res = Dict{String,Any}("n_obs" => bound.n_obs, "unc_dim" => k.layout.total)

# ---- primal ----------------------------------------------------------------
ur = Reactant.to_rarray(u)
t_compile = @elapsed compiled = Reactant.@compile post_q(ur)
got = Float64(compiled(ur))
ok1 = isapprox(got, native; rtol = 1e-9)
u2 = Vector{Float64}(0.1 .* randn(Xoshiro(7), k.layout.total))
ok2 = isapprox(Float64(compiled(Reactant.to_rarray(u2))), Base.invokelatest(post_q, u2); rtol = 1e-9)
@printf("REACTANT K=%d primal compile=%.1fs lp=%.9f match=%s/%s reldiff=%.3e\n",
    K, t_compile, got, ok1, ok2, abs(got - native) / abs(native))
compiled(ur)
t_r = @elapsed for _ in 1:REPS; compiled(ur); end
Base.invokelatest(post_q, u)
t_n = @elapsed for _ in 1:REPS; Base.invokelatest(post_q, u); end
@printf("K=%d eval: reactant=%.4fms native=%.4fms\n", K, 1e3 * t_r / REPS, 1e3 * t_n / REPS)
res["rkppl_eval_ms"] = round(1e3 * t_n / REPS; digits = 4)
res["reactant_eval_ms"] = round(1e3 * t_r / REPS; digits = 4)
res["reactant_compile_s"] = round(t_compile; digits = 1)
res["reactant_primal_parity"] = ok1 && ok2
flush(stdout)

# ---- value + gradient ------------------------------------------------------
if DO_GRAD
    be = AutoEnzyme(; mode = Enzyme.Reverse)
    t_prep = @elapsed q = prepare_sampler(k, bound, u; backend = be)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    t_ng = @elapsed for _ in 1:REPS; sampler_value_and_gradient!(q, g, u); end
    @printf("K=%d native grad: prep=%.1fs val=%.9f gnorm=%.6f mean=%.4fms\n",
        K, t_prep, val, sqrt(sum(abs2, g)), 1e3 * t_ng / REPS)
    flush(stdout)
    t_c = @elapsed cad = OPT == "default" ? compile_ad_value_and_gradient(q.ad, ur) :
        compile_ad_value_and_gradient(q.ad, ur; optimize = Symbol(OPT))
    rval, rgrad = cad(ur)
    rg = Array(rgrad)
    gok = isapprox(rg, g; rtol = 1e-9) && isapprox(Float64(rval), val; rtol = 1e-9)
    @printf("REACTANT AD K=%d opt=%s compile=%.1fs val=%.9f parity(rtol1e-9)=%s maxrel=%.3e\n",
        K, OPT, t_c, Float64(rval), gok, maximum(abs.(rg .- g) ./ max.(abs.(g), 1e-8)))
    t_rg = @elapsed for _ in 1:REPS; cad(ur); end
    @printf("K=%d grad: reactant=%.4fms native=%.4fms\n", K, 1e3 * t_rg / REPS, 1e3 * t_ng / REPS)
    res["rkppl_grad_ms"] = round(1e3 * t_ng / REPS; digits = 4)
    res["reactant_grad_ms"] = round(1e3 * t_rg / REPS; digits = 4)
    res["reactant_ad_compile_s"] = round(t_c; digits = 1)
    res["reactant_ad_opt"] = OPT
    res["reactant_grad_parity"] = gok
end
kv = join(["\"$(k)\":$(v isa String ? "\"$v\"" : v)" for (k, v) in sort!(collect(res); by = first)], ",")
println("RESULT {", kv, "}")
