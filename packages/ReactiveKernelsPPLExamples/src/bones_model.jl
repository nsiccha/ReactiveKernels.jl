module BonesModelExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export BONES_GRADE, BONES_GAMMA, BONES_DELTA, BONES_NCAT
export BONES_NCHILD, bones_inputs
export build_bones_model_graph, demo
export BONES_SOURCE, evaluate_bones_model_source

# posteriordb `bones_data-bones_model` — the BUGS "bones" grade-of-ossification
# latent-trait model (WinBUGS vol. 1). `nChild = 13` children each have a latent
# skeletal maturity `theta[i] ~ Normal(0, 36)` (36 is the SD); `nInd = 34`
# radiographic indicators, indicator j with `delta[j]` discrimination and
# `ncat[j] ∈ {2,3,4,5}` ordered grades separated by cut points `gamma[j, 1..ncat[j]-1]`.
# The graded response uses the exceedance probabilities
#   Q[i,j,k] = inv_logit(delta[j] * (theta[i] - gamma[j,k])),
# and the category probability of an observed grade g is the cumulative gap
#   p(g) = Q[g-1] - Q[g],   with the conventions Q[0] = 1 and Q[ncat] = 0.
# So p(1) = 1 - Q[1] and p(ncat) = Q[ncat-1] fall out of the one uniform formula.
# The log-likelihood sums log p(grade[i,j]) over the OBSERVED cells (missing grade
# is coded -1 and contributes nothing). `theta` is unconstrained (no Jacobian).
#
# The acceptance entry binds ONLY the RAW data block (`grade`, `gamma`, `delta`,
# `ncat`). Everything else is derived IN the graph over the full nChild×nInd grid:
# the grid coordinate matrices `ROWIDX`/`COLIDX` are built in-graph by `repeat`
# over the raw dimensions; the two bracketing cut indices are computed from `grade`
# by arithmetic, safe-clamped, and used to GATHER `gamma`; `delta`/`theta` are
# gathered by those coordinates; the boundary and missing masks are bound
# comparisons. All of this is data-only bookkeeping (folds under `bound=` partial
# evaluation) except the `theta[ROWIDX]` gather. Real data (full) from posteriordb
# `bones_data-bones_model`.

_bones_int_matrix(x) = x isa AbstractMatrix ? Int.(x) :
    reduce(vcat, [permutedims(Int.(r)) for r in x])
_bones_float_matrix(x) = x isa AbstractMatrix ? Float64.(x) :
    reduce(vcat, [permutedims(Float64.(r)) for r in x])

"""
    bones_inputs(data) -> NamedTuple

Build the bound ports from a loaded posteriordb `bones_data` dict: ONLY the RAW
block (`GRADE` nChild×nInd, `GAMMA` nInd×maxcut, `DELTA`, `NCAT`). Grid
coordinates and the ragged cut selection are all derived in-graph.
"""
function bones_inputs(data)
    (; GRADE = _bones_int_matrix(data["grade"]),
       GAMMA = _bones_float_matrix(data["gamma"]),
       DELTA = Float64.(data["delta"]),
       NCAT = Int.(data["ncat"]))
end

let d = _posteriordb_data("bones_data-bones_model")
    inp = bones_inputs(d)
    global const BONES_GRADE = inp.GRADE
    global const BONES_GAMMA = inp.GAMMA
    global const BONES_DELTA = inp.DELTA
    global const BONES_NCAT = inp.NCAT
    global const BONES_NCHILD = size(inp.GRADE, 1)
end

const BONES_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic

@kernel model(unconstrained::Vector{Float64},
              GRADE::Matrix{Int},
              GAMMA::Matrix{Float64},
              DELTA::Vector{Float64},
              NCAT::Vector{Int}) = begin
    # Stan's unconstrained order is just theta[1..nChild] (unconstrained real);
    # dim = nChild, and there is no constraint Jacobian.
    theta::Vector{Float64} = unconstrained
    log_jacobian::Float64 = 0.0

    parameters = (; theta)
    theta::Vector{Float64} = parameters.theta

    # Prior: thetaᵢ ~ Normal(0, 36).
    theta_pointwise = plate(theta) do t
        normal(0.0, 36.0).logpdf(t)
    end
    prior::Float64 = sum(theta_pointwise)

    # Grid coordinates built in-graph from the raw dimensions (data-only recipes,
    # fold under partial evaluation): ROWIDX[i,j]=i (child), COLIDX[i,j]=j (item).
    n_child::Int = size(GRADE, 1)
    n_ind::Int = size(GAMMA, 1)
    ROWIDX = repeat(1:n_child, 1, n_ind)
    COLIDX = repeat(permutedims(1:n_ind), n_child, 1)

    # Full-grid graded-response likelihood, all derived in-graph from raw data.
    # Bracketing cut indices g-1 and g, safe-clamped so the gather stays in-bounds
    # (invalid/missing cells read a dummy cut and are killed by the masks below).
    max_cut::Int = size(GAMMA, 2)
    lo_cut = clamp.(GRADE .- 1, 1, max_cut)
    hi_cut = clamp.(GRADE, 1, max_cut)
    lo_lin = COLIDX .+ (lo_cut .- 1) .* n_ind          # linear index into GAMMA
    hi_lin = COLIDX .+ (hi_cut .- 1) .* n_ind
    gamma_lo = GAMMA[lo_lin]                            # raw gamma gathered in-graph
    gamma_hi = GAMMA[hi_lin]
    delta_g = DELTA[COLIDX]                             # raw delta gathered by column
    theta_g = theta[ROWIDX]                             # child trait gathered by row

    # Q[0] = 1 (grade 1) and Q[ncat] = 0 (grade ncat) substitutions via masks.
    lo_valid = 1.0 .* (GRADE .> 1)
    hi_valid = 1.0 .* (GRADE .< NCAT[COLIDX])
    obs_mask = 1.0 .* (GRADE .!= -1)                   # missing grade coded -1
    q_lo = lo_valid .* logistic.(delta_g .* (theta_g .- gamma_lo)) .+ (1.0 .- lo_valid)
    q_hi = hi_valid .* logistic.(delta_g .* (theta_g .- gamma_hi))
    # Observed cells: log(Q_lo - Q_hi). Missing cells fold to log(1)=0, killing
    # both value and gradient regardless of their clamped dummy gather.
    arg = obs_mask .* (q_lo .- q_hi) .+ (1.0 .- obs_mask)
    cell = obs_mask .* log.(arg)
    likelihood::Float64 = sum(cell)

    constrained_logdensity::Float64 = prior + likelihood
    posterior::Float64 = constrained_logdensity + log_jacobian

    return posterior
end

q = zeros(size(BONES_GRADE, 1))
GRADE = BONES_GRADE
GAMMA = BONES_GAMMA
DELTA = BONES_DELTA
NCAT = BONES_NCAT

requested_nodes = (:parameters, :log_jacobian, :prior, :likelihood, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :GRADE, :GAMMA, :DELTA, :NCAT),
    want = requested_nodes,
    bound = (; GRADE, GAMMA, DELTA, NCAT))

output = density_kernel(q)
parameters, log_jacobian, prior, likelihood, posterior = output
@assert posterior ≈ prior + likelihood + log_jacobian

docs_example = (;
    name = :bones_model_posterior,
    origin = "posteriordb bones_model — BUGS graded-response latent-trait (ordered ossification grades)",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
)
"""

function evaluate_bones_model_source(; model_only::Bool = false)
    _evaluate_ppl_source(BONES_SOURCE, @__MODULE__; bindings = (
        :BONES_GRADE, :BONES_GAMMA, :BONES_DELTA, :BONES_NCAT,
    ), model_only)
end

const _BONES_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _BONES_GRAPH_TEMPLATE[] = evaluate_bones_model_source(; model_only = true).model
    nothing
end

"""
    build_bones_model_graph()

Build the posteriordb `bones_model` (the BUGS graded-response latent-trait model:
`theta ~ Normal(0, 36)` and, per observed indicator cell, a cumulative-logit
category probability `Q[g-1] - Q[g]` with `Q[k] = inv_logit(delta*(theta -
gamma[k]))`) as a declarative `ReactiveKernels.KernelSpec`. `theta` is
unconstrained (no Jacobian). The acceptance entry is ONLY the RAW `grade`/`gamma`/
`delta`/`ncat` block; grid coordinates and the ragged cut selection are all
derived in-graph over the full grid (in-graph `repeat` coordinates, arithmetic
cut indices, `gamma`/`delta`/`theta` gathers, boundary/missing masks). Prior,
likelihood, and posterior are named nodes.
"""
function build_bones_model_graph()
    compose(_BONES_GRAPH_TEMPLATE[])
end

function demo()
    model = build_bones_model_graph()
    q = zeros(BONES_NCHILD)
    posterior_kernel = prepare(model;
        have = (:unconstrained, :GRADE, :GAMMA, :DELTA, :NCAT),
        want = :posterior,
        bound = (; GRADE = BONES_GRADE, GAMMA = BONES_GAMMA, DELTA = BONES_DELTA,
                   NCAT = BONES_NCAT))
    println("bones unconstrained log posterior = ", posterior_kernel(q))
    nothing
end

end # module BonesModelExample

if abspath(PROGRAM_FILE) == @__FILE__
    BonesModelExample.demo()
end
