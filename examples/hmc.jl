# One scalar mathematical source of truth: a single-chain HMC transition.
#
# The design goal (user directive, 2026-08-28): there is ONE scalar, as-simple-as-possible
# `@kernel` source of truth, and RK transforms it into the batched (multi-chain) form — the
# author never hand-writes a batched sampler. This module holds that scalar source.
#
# The transition is written as a PURE FUNCTION of (state, injected randomness): the momentum
# `p0` and the Metropolis uniform `u` are HAVE ports, not drawn inside the kernel. That keeps
# the mathematical transition deterministic and backend-neutral, so:
#   * the RNG lives entirely in the driver / a backend adapter (split-key under Reactant);
#   * `ReactiveKernels.replica(hmc_transition; batched = (:q, :p0, :u))` (the compiler-owned
#     vmap, ext/Reactant) batches THIS unchanged kernel over chains — `grad_U`/`pot` stay on
#     the scalar contract (`Vector -> Vector`, `Vector -> scalar`) and are mapped per slice.
#
# It has NO data-dependent control flow: the leapfrog loop is a static count and the
# Metropolis accept is a `select` (ternary), which is exactly why a single scalar HMC source
# auto-batches cleanly (unlike NUTS, whose per-chain divergence would make a ragged batch).
#
# The kernel here is executed on the CPU against a correlated-Gaussian target as a
# correctness check. No throughput, ESS, GPU, or Reactant claim is made in this file; the
# Reactant single-chain `@jit` compilation and the multi-chain `replica` batching are proven
# in the optional Reactant integration (ext/ReactiveKernelsReactantExt.jl + its tests).
module HMCExample

export HMC_SOURCE
export all_sources, evaluate_source, run

const HMC_SOURCE = raw"""
using LinearAlgebra, Random, Statistics

# ONE scalar source of truth — a single-chain HMC transition. `grad_U`/`pot` are the model's
# gradient/potential as ORDINARY scalar closures (Vector -> Vector, Vector -> scalar); nothing
# here is batching-specific. Momentum `p0` and MH uniform `u` are injected (deterministic).
hmc_transition = @kernel hmc_transition(grad_U::Function, pot::Function,
        q::Vector{Float64}, p0::Vector{Float64}, u::Float64,
        stepsize::Float64, L::Int) = begin
    # Leapfrog integrator: a static `L`-step loop (compile-time count) captured as one recipe.
    integrated::Tuple{Vector{Float64},Vector{Float64}} = let e = stepsize
        qq = q
        pp = p0 .- (e / 2) .* grad_U(q)
        for _ in 1:(L - 1)
            qq = qq .+ e .* pp
            pp = pp .- e .* grad_U(qq)
        end
        qq = qq .+ e .* pp
        pp = pp .- (e / 2) .* grad_U(qq)
        (qq, pp)
    end
    qL::Vector{Float64} = integrated[1]
    pL::Vector{Float64} = integrated[2]
    # Hamiltonians (per-chain scalar reductions when this kernel is later vmapped over chains).
    H0::Float64 = pot(q)  + 0.5 * dot(p0, p0)
    HL::Float64 = pot(qL) + 0.5 * dot(pL, pL)
    # Metropolis accept as a SELECT (ternary), not data-dependent control flow.
    accept::Bool = log(u) < (H0 - HL)
    q_next::Vector{Float64} = accept ? qL : q
end

k = prepare(hmc_transition;
            have = (:grad_U, :pot, :q, :p0, :u, :stepsize, :L), want = :q_next)

# Correlated-Gaussian target: pot(q) = ½ qᵀ Σ⁻¹ q, grad_U(q) = Σ⁻¹ q — plain scalar closures.
rng = MersenneTwister(11)
D = 5
A = randn(rng, D, D)
Sigma = A * A' + D * I
Sinv = inv(Sigma)
grad_U = q -> Sinv * q
pot = q -> 0.5 * dot(q, Sinv * q)

# Drive the ONE scalar kernel single-chain; the RNG lives HERE (driver), not in the kernel.
function run_chain(k, grad_U, pot, q0, rng; stepsize, L, warmup, draws)
    q = q0
    d = length(q)
    for _ in 1:warmup
        q = k(grad_U, pot, q, randn(rng, d), rand(rng), stepsize, L)
    end
    samples = Matrix{Float64}(undef, d, draws)
    for i in 1:draws
        q = k(grad_U, pot, q, randn(rng, d), rand(rng), stepsize, L)
        samples[:, i] = q
    end
    samples
end

samples = run_chain(k, grad_U, pot, randn(rng, D), rng;
                    stepsize = 0.2, L = 25, warmup = 2000, draws = 6000)

estimated_cov = cov(samples; dims = 2)
cov_relerror = maximum(abs.(estimated_cov .- Sigma)) / maximum(abs.(Sigma))
mean_abserror = maximum(abs.(vec(mean(samples; dims = 2))))   # target mean is 0
n_recipes = length(plan(hmc_transition;
                        have = (:grad_U, :pot, :q, :p0, :u, :stepsize, :L),
                        want = :q_next).recipes)

# The single-chain sampler recovers the target within a loose sampling tolerance.
@assert cov_relerror < 0.1
@assert mean_abserror < 0.3
@assert size(samples, 1) == D

docs_example = (;
    name = :hmc_transition,
    origin = "scalar single-chain HMC transition (deterministic, injected momentum + MH uniform)",
    kernel = k,
    D,
    samples,
    Sigma,
    estimated_cov,
    cov_relerror,
    mean_abserror,
    n_recipes,
)
"""

all_sources() = (HMC_SOURCE,)

function evaluate_source(source::AbstractString)
    sandbox = Module(gensym(:HMCExample), true, true)
    Core.eval(sandbox, :(using ReactiveKernels))
    parsed = Meta.parseall(source; filename = "hmc-example.jl")
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    for expression in expressions
        expression isa LineNumberNode && continue
        Core.eval(sandbox, expression)
    end
    Core.eval(sandbox, :docs_example)
end

function run(io::IO = stdout)
    artifacts = map(evaluate_source, all_sources())
    for artifact in artifacts
        println(io, artifact.name)
        println(io, "  params: ", artifact.D)
        println(io, "  recipes in prepared kernel: ", artifact.n_recipes)
        println(io, "  relative cov error: ", artifact.cov_relerror)
        println(io, "  |mean| error (target 0): ", artifact.mean_abserror)
    end
    artifacts
end

end # module HMCExample

if abspath(PROGRAM_FILE) == @__FILE__
    HMCExample.run()
end
