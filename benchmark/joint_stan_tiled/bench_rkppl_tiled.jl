# RKPPL side of the Stan-vs-RKPPL speed comparison (throwaway probe).
# Tiles the joint fixture K times (subject replicas, same order as tile.py:
# replica-major: S1,S2,S3,S1,S2,S3,...).
using ReactiveKernels
using ReactiveKernelsPPL
using DifferentiationInterface
using Enzyme
using Printf
using Random

const PPLT = normpath(joinpath(@__DIR__, "..", "..", "packages",
    "ReactiveKernelsPPL", "test"))
include(joinpath(PPLT, "parity", "joint_parity_fixture.jl"))
include(joinpath(PPLT, "parity", "joint_tiling.jl"))

K = parse(Int, get(ENV, "TILE_K", "3"))

cols = tiled_columns(K)
plan = final_plan("continuous")
t_bind = @elapsed bound =
    bind_data(plan, cols; dims = Dict(:kernel_nsub_pk_loc => 3K))
lay = assign_layout(bound)
@printf("K=%d n_obs=%d n_subj=%d u_dim=%d\n", K, bound.n_obs, 3K, lay.total)
t_build = @elapsed k = build_kernel(bound)
post_q = prepare_query(k, bound, :sampler)
u = Vector{Float64}(0.1 .* randn(Xoshiro(20260917), lay.total))
Base.invokelatest(post_q, u) # warmup (compile)
lp = Base.invokelatest(post_q, u)
@printf("lp=%.6f finite=%s\n", lp, isfinite(lp))
t_eval = @elapsed for _ in 1:20
    Base.invokelatest(post_q, u)
end
t_eval /= 20
be = AutoEnzyme(; mode = Enzyme.Reverse)
t_prep = @elapsed q = prepare_sampler(k, bound, u; backend = be)
g = similar(u)
sampler_value_and_gradient!(q, g, u) # warmup
@printf("grad_norm=%.4f all_finite=%s\n", sqrt(sum(abs2, g)), all(isfinite, g))
t_grad = @elapsed for _ in 1:10
    sampler_value_and_gradient!(q, g, u)
end
t_grad /= 10
@printf("RKPPL K=%d bind=%.2fs build=%.2fs eval=%.5fs gradprep=%.2fs grad=%.5fs\n",
    K, t_bind, t_build, t_eval, t_prep, t_grad)
