# Run by runtests.jl after two `pool_writer.jl` processes have each saved one
# cloud's checkpoints. This process pools them having loaded only the SMC
# extension's triggers: it builds no model, prepares no target and loads
# neither Enzyme nor the writers' fixture module.
using ReactiveKernelsPPL, SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random
using Serialization

for kind in (:built, :prepared)
    records = [getfield(deserialize(path), kind) for path in ARGS]
    pooled = pool_smc(records)
    logz = [r.log_normalizer for r in records]
    peak = maximum(logz)
    @assert pooled.log_normalizer == peak + log(sum(exp.(logz .- peak)) / length(logz))
    @assert pooled.positions == reduce(hcat, [r.positions for r in records])
    @assert sum(pooled.weights) ≈ 1
    @assert pool_smc(reverse(records)).log_normalizer ≈ pooled.log_normalizer
end
@assert !any(m -> nameof(m) in (:Enzyme, :PreparedSMCFixture), values(Base.loaded_modules))
println("Fresh-process pooling of separately written checkpoints passed")
