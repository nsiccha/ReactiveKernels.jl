# Stan side of the speed comparison (throwaway probe).
using BridgeStan, Printf, Random

K = parse(Int, get(ENV, "TILE_K", "1"))
ORACLE = "/home/n/scratch/kb-agent-tmp/ReactiveKernels-brm-tgi/joint-oracle/continuous"
data = K == 1 ? joinpath(ORACLE, "stan_data.json") :
    "/tmp/bs-bench/stan_data_tiled$(K).json"
SO = joinpath(ORACLE, "joint_model.so")

t_construct = @elapsed sm = StanModel(SO, data)
n = BridgeStan.param_unc_num(sm)
@printf("K=%d unc_dim=%d construct=%.2fs\n", K, n, t_construct)
z = Vector{Float64}(0.1 .* randn(Xoshiro(20260917), n))
PR = get(ENV, "PROPTO", "1") == "1"
lp, grad = log_density_gradient(sm, z; propto = PR) # warmup
@printf("lp=%.6f finite=%s grad_norm=%.4f all_finite=%s\n",
    lp, isfinite(lp), sqrt(sum(abs2, grad)), all(isfinite, grad))
t_eval = @elapsed for _ in 1:20
    log_density(sm, z; propto = PR)
end
t_eval /= 20
t_grad = @elapsed for _ in 1:20
    log_density_gradient(sm, z; propto = PR)
end
t_grad /= 20
@printf("STAN K=%d propto=%d eval=%.5fs grad=%.5fs\n", K, PR, t_eval, t_grad)
