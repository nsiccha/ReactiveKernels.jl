# ── Posteriors of summed-out discrete latents ──────────────────────────
#
# Two observation laws sum a discrete latent out of the density:
#
# - a whole-event hidden Markov observation (`y ~ LogDensity(law, …)` with one
#   of DistributionKernels' hidden Markov laws) sums over the state path;
# - a finite-mixture response (`y .~ MixtureModel(…)`) sums over each
#   observation's component.
#
# The graph computes each latent's posterior given the parameters, as a value
# of the parameters like any other: the hidden Markov law's state-posterior
# twin (`hmm_state_posterior_name`) on the same arguments, and each mixture
# observation's normalized weighted component terms. Drawing a latent happens
# at the query boundary in ordinary Julia, as for predictive draws (user
# decision `ReactiveKernels:snag.predictive-rng-f-f51c6561/decisions/2026-10-21T10-08-19-848-05ameuz`):
# no RNG enters an RK kernel, and the sampler's density and gradients are
# untouched, since the latents node is a separate WANT.

_membership_name(label::Symbol) = Symbol(:_ppl_membership_, label)
_hidden_state_name(name::Symbol) = Symbol(:_ppl_hidden_states_, name)

# A mixture response whose observations each have one component posterior:
# every observation is one plain draw from the mixture.
_mixture_memberships(r::LikelihoodSpec, plan::StructuralPlan) =
    r.evidence.kind === :none && r.weights === nothing && r.mi_jobs === nothing &&
    r.range === nothing && !haskey(plan.columns, _observed_mask_name(r.response)) &&
    !_cell_broadcast_response(plan, r)

function _membership_entry!(latents, uncovered, r::LikelihoodSpec, rs::Vector{Expr})
    r.family === MixtureFam || return nothing
    name = _membership_name(r.label)
    if !any(st -> _assigned_name(st) === name, rs)
        push!(uncovered, r.response)
        return nothing
    end
    K = _mixture_component_count(r)
    matrix = GlobalRef(@__MODULE__, :_ppl_membership_matrix)
    push!(latents, r.response =>
        :((; probabilities = $matrix($name, $(GlobalRef(Base, :Val))($K)))))
    return nothing
end

# The state-posterior twin of a hidden Markov observation, called on the
# observation and the law's own arguments; other external observations have
# no summed-out latent.
function _hidden_state_entry!(stmts, latents, uncovered, plan::StructuralPlan, p)
    graph_density = _external_graph_density(p.args)
    graph_density === nothing && return nothing
    endpoint = graph_density.endpoint
    endpoint isa GlobalRef || return nothing
    twin = DistributionKernelSources.hmm_state_posterior_name(
        getglobal(endpoint.mod, endpoint.name))
    twin === nothing && return nothing
    if p.args.broadcast || _cell_broadcast_observation(plan, p.name)
        push!(uncovered, p.name)
        return nothing
    end
    call = _external_graph_density_call(
        (; graph_density..., endpoint = GlobalRef(DistributionKernelSources, twin)),
        Any[p.name, _external_arguments(p.args)[graph_density.first_argument:end]...],
        _external_keywords(p.args))
    name = _hidden_state_name(p.name)
    call[end] = :($name = $(call[end]))
    append!(stmts, call)
    push!(latents, p.name => name)
    return nothing
end

# The `_ppl_latents` node, only in programs with a summed-out latent.
function _latents_statement!(stmts, latents, uncovered)
    isempty(latents) && isempty(uncovered) && return stmts
    value = isempty(uncovered) ?
        Expr(:tuple, (Expr(:(=), name, v) for (name, v) in latents)...) :
        Expr(:call, GlobalRef(@__MODULE__, :_ppl_latents_unsupported),
            Expr(:tuple, QuoteNode.(uncovered)...))
    push!(stmts, Expr(:(=), :_ppl_latents, value))
    return stmts
end

_ppl_latents_unsupported(names::Tuple) = throw(ContractValidationError(
    "[query] discrete-latent posteriors cover whole-event hidden Markov " *
    "observations (`y ~ LogDensity(normal_hmm_logdensity, …)` and the other RK " *
    "hidden Markov laws) and plain finite-mixture responses (`y .~ " *
    "MixtureModel(…)`). This program sums out a latent it cannot recover for " *
    join(names, ", ") * ". Not covered yet: hidden Markov observations " *
    "inside `@plate` cells or dotted, and mixture responses with evidence " *
    "(censoring, truncation), weights, missing entries, ranges, packed rows " *
    "or one array per `@plate` index."))

# One observation's posterior over the mixture's components, from its
# weighted component log densities `log(w_k) + logpdf_k(y)`.
function _ppl_membership_probabilities(scores::NTuple{K,Real}) where {K}
    top = maximum(scores)
    isfinite(top) || return _ppl_membership_nan(Val(K))
    weights = map(s -> exp(Float64(s) - top), scores)
    total = sum(weights)
    return map(w -> w / total, weights)
end
_ppl_membership_nan(::Val{K}) where {K} = ntuple(_ -> NaN, Val(K))

# The membership plate's per-observation tuples as an observations ×
# components matrix, rows in the response's (column-major) order.
function _ppl_membership_matrix(cells::AbstractArray, ::Val{K}) where {K}
    out = Matrix{Float64}(undef, length(cells), K)
    for (i, cell) in enumerate(cells), k in 1:K
        out[i, k] = cell[k]
    end
    return out
end

# ── Query surface ───────────────────────────────────────────────────────

"""
    DiscreteLatentQuery

A prepared query of the posteriors of a program's summed-out discrete latents
(`prepare_discrete_latents`). `q(u)` returns them at the parameters `u`;
[`discrete_latent_draws`](@ref) draws the latents. Not thread-safe: prepare
one per thread.
"""
struct DiscreteLatentQuery{K}
    kernel::K
end

"""
    prepare_discrete_latents(built, plan) -> DiscreteLatentQuery

Prepare the posteriors of the discrete latents the observations of `plan`
sum out, on the graph `built` (`build_kernel` of a binding of the same
program). The query returns a `NamedTuple` keyed by observation name:

- for a whole-event hidden Markov observation (`y ~ LogDensity(law, …)` with
  one of `DistributionKernelSources`' hidden Markov laws), the law's state
  posterior: `probabilities[t, k]`, the probability of state `k` at step `t`
  given every observation; `log_filtered[t, k]`, its log probability given
  the first `t` steps; and the `log_transition` weights the law reads;
- for a plain finite-mixture response (`y .~ MixtureModel(…)`),
  `probabilities[i, k]`, the probability that observation `i` (in the
  response's column-major order) came from component `k`.

At posterior draws of the parameters these are the latents' conditional
posteriors; [`discrete_latent_draws`](@ref) draws the latents themselves, so
parameter and latent draws together are exact draws from the joint
posterior.

Call the prepared query at an unconstrained point to obtain the latent
probabilities. Pass the query, point, and an explicit random-number generator
to [`discrete_latent_draws`](@ref) for a sampled state path.

A program with no summed-out latent has none to prepare, and one whose
latents are not covered (hidden Markov observations in `@plate` cells or
dotted; mixtures with evidence, weights, missing entries, ranges, packed rows
or arrays per index) is refused, naming them.
"""
function prepare_discrete_latents(built, plan::StructuralPlan; on_error = nothing)
    kernel = _prepare_wants(built, plan, discrete_latents_node(built); on_error)
    return DiscreteLatentQuery(kernel)
end

(q::DiscreteLatentQuery)(u::AbstractVector) = Base.invokelatest(q.kernel, u)

"""
    discrete_latents_node(built) -> Symbol

The graph value of `built` that holds the posteriors of its summed-out
discrete latents: the value `prepare_discrete_latents` evaluates. WANT it in
any cut of `built.spec` beside other values, and `vectorize` that cut over
draws; its leaves are numeric arrays, stacked on a trailing draw axis. Draw
the latents of draw `k` with `discrete_latent_draws(value, rng; position = k)`.
"""
function discrete_latents_node(built)
    hasproperty(built, :spec) || throw(ArgumentError("[query] " *
        "discrete_latents_node needs a `build_kernel` result"))
    haskey(built.spec.ports, :_ppl_latents) || throw(ContractValidationError(
        "[query] this program sums out no discrete latent: it has no whole-event " *
        "hidden Markov observation (`y ~ LogDensity(normal_hmm_logdensity, …)` " *
        "and the other RK hidden Markov laws) and no finite-mixture response " *
        "(`y .~ MixtureModel(…)`)"))
    return :_ppl_latents
end

"""
    discrete_latent_draws(q::DiscreteLatentQuery, u, rng) -> NamedTuple
    discrete_latent_draws(value, rng; position = nothing) -> NamedTuple

One exact draw of every summed-out discrete latent from its posterior, keyed
like the posteriors: a hidden Markov observation's state path
(`Vector{Int}`, one state per step; `draw_hmm_states`) and a mixture
response's components (`Vector{Int}`, one per observation, drawn
independently). `value` is the value of [`discrete_latents_node`](@ref) in
any cut; `position = k` selects draw `k` of a vectorized cut. Only `rng`
supplies randomness, in key order, so a seeded `rng` reproduces the draws.
"""
discrete_latent_draws(q::DiscreteLatentQuery, u::AbstractVector, rng) =
    discrete_latent_draws(q(u), rng)
function discrete_latent_draws(value::NamedTuple, rng; position = nothing)
    value = position === nothing ? value : _at_position(value, position)
    return map(posterior -> _latent_draw(rng, posterior), value)
end

_latent_draw(rng, posterior::NamedTuple) = hasproperty(posterior, :log_filtered) ?
    DistributionKernelSources.draw_hmm_states(rng, posterior) :
    _membership_draws(rng, posterior.probabilities)

function _membership_draws(rng, probabilities::AbstractMatrix)
    out = Vector{Int}(undef, size(probabilities, 1))
    for i in eachindex(out)
        u = rand(rng)
        accumulated = 0.0
        out[i] = 0
        for k in axes(probabilities, 2)
            p = probabilities[i, k]
            isfinite(p) && p >= 0 || throw(ArgumentError("[query] observation $i " *
                "has component probability $p: its component posterior is undefined"))
            accumulated += p
            p > 0 && (out[i] = k)
            u < accumulated && p > 0 && break
        end
        out[i] == 0 && throw(ArgumentError("[query] observation $i has no " *
            "component with positive posterior probability"))
    end
    return out
end
