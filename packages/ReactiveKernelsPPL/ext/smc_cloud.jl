# Cloud-fitted stage kernels (SMCOptions(cloud_kernels=J)).
#
# Before a stage moves its particles, fit a diagonal Gaussian mixture with at
# most J components to the stage target's weighted cloud (the pre-resampling
# particles reweighted to the new temperature), choosing the component count by
# BIC. Component j defines a NUTS kernel K_j: its diagonal scales are the
# component's standard deviations and its step size is tuned on trial
# transitions from that component's particles (trial draws are discarded).
#
# Each move selects a component with the mixture responsibilities r(x) at the
# particle's current position, takes one K_j transition to x', and accepts x'
# with probability min(1, r_j(x')/r_j(x)). Every K_j is reversible for the stage
# target, so pi(x) r_j(x) K_j(x,x') min(1, r_j(x')/r_j(x)) =
# pi(x) K_j(x,x') min(r_j(x), r_j(x')) is symmetric: the move is reversible for
# the stage target given the fit. A particle in any mode therefore moves with
# the scales of the cloud's particles near it, not those of one pilot chain.
# J = 1 is one cloud-fitted diagonal metric and needs no selection step.
#
# Work is linear in particles x dimension x components; nothing dense is formed.

const _CLOUD_ACCEPTANCE = 0.8      # target mean NUTS acceptance statistic
const _CLOUD_TRIALS = 16           # trial transitions per component per round
const _CLOUD_ROUNDS = 6            # geometric step-size bisection rounds
const _CLOUD_MIN_COUNT = 10.0      # minimum effective particles per component
const _CLOUD_EM_ITERATIONS = 200
const _CLOUD_SD_FLOOR = 1e-3       # relative to the cloud's own standard deviation

function _cloud_component_logpdf(logweight, mean, sd, x, j)
    s = logweight[j]
    @inbounds for i in eachindex(x)
        z = (x[i] - mean[i, j]) / sd[i, j]
        s -= 0.5 * z * z + log(sd[i, j])
    end
    return s
end

# Selection probabilities at x, written into `r`. Constants common to all
# components cancel.
function _cloud_responsibilities!(r, kernel, x)
    J = kernel.components
    J == 1 && (r[1] = 1.0; return r)
    peak = -Inf
    for j in 1:J
        r[j] = _cloud_component_logpdf(kernel.logweights, kernel.means, kernel.scales, x, j)
        peak = max(peak, r[j])
    end
    total = 0.0
    for j in 1:J
        r[j] = exp(r[j] - peak)
        total += r[j]
    end
    r ./= total
    return r
end

function _cloud_categorical(rng, p)
    u = rand(rng) * sum(p)
    acc = 0.0
    for j in eachindex(p)
        acc += p[j]
        u <= acc && return j
    end
    return length(p)
end

# Weighted EM for a J-component diagonal Gaussian mixture, seeded by weighted
# k-means++ in the cloud's own standard-deviation units.
function _cloud_em(X, w, J, rng, scale, floor)
    d, n = size(X)
    mean = zeros(d, J)
    mean[:, 1] = X[:, _cloud_categorical(rng, w)]
    nearest = fill(Inf, n)
    for j in 2:J
        for i in 1:n
            s = 0.0
            @inbounds for k in 1:d
                s += abs2((X[k, i] - mean[k, j-1]) / scale[k])
            end
            nearest[i] = min(nearest[i], s)
        end
        mean[:, j] = X[:, _cloud_categorical(rng, w .* nearest)]
    end
    sd = repeat(scale, 1, J)
    logweight = fill(-log(J), J)
    R = zeros(J, n)
    loglik = -Inf
    for _ in 1:_CLOUD_EM_ITERATIONS
        total = 0.0
        for i in 1:n
            x = view(X, :, i)
            peak = -Inf
            for j in 1:J
                R[j, i] = _cloud_component_logpdf(logweight, mean, sd, x, j)
                peak = max(peak, R[j, i])
            end
            s = 0.0
            for j in 1:J
                R[j, i] = exp(R[j, i] - peak)
                s += R[j, i]
            end
            for j in 1:J
                R[j, i] /= s
            end
            total += w[i] * (peak + log(s) - 0.5 * d * log(2pi))
        end
        mass = zeros(J)
        fill!(mean, 0.0)
        for i in 1:n, j in 1:J
            a = w[i] * R[j, i]
            mass[j] += a
            @inbounds for k in 1:d
                mean[k, j] += a * X[k, i]
            end
        end
        mass .= max.(mass, eps())
        mean ./= mass'
        fill!(sd, 0.0)
        for i in 1:n, j in 1:J
            a = w[i] * R[j, i]
            @inbounds for k in 1:d
                sd[k, j] += a * abs2(X[k, i] - mean[k, j])
            end
        end
        sd .= max.(sqrt.(sd ./ mass'), floor)
        logweight .= log.(mass ./ sum(mass))
        converged = abs(total - loglik) <= 1e-9 * max(1.0, abs(total))
        loglik = total
        converged && break
    end
    mass = exp.(logweight)
    return (; logweight, mean, sd, loglik, mass)
end

# Choose the component count by BIC on the importance-weighted cloud, keeping
# only fits whose every component holds `_CLOUD_MIN_COUNT` effective particles.
function _cloud_fit(X, w, max_components, rng)
    d, n = size(X)
    effective = 1 / sum(abs2, w)
    location = X * w
    sd = sqrt.(max.((X .- location) .^ 2 * w, 0.0))
    sd .= max.(sd, eps())
    floor = _CLOUD_SD_FLOOR .* sd
    best = nothing
    best_bic = Inf
    for J in 1:max_components
        J > 1 && effective / J < _CLOUD_MIN_COUNT && break
        fit = _cloud_em(X, w, J, rng, sd, floor)
        J > 1 && minimum(fit.mass) * effective < _CLOUD_MIN_COUNT && continue
        bic = -2 * effective * fit.loglik + (J * (2d + 1) - 1) * log(effective)
        if bic < best_bic
            best, best_bic = (; fit..., components=J, bic), bic
        end
    end
    return best
end

# Mean NUTS acceptance statistic of one transition from each start.
function _cloud_trial(target, starts, metric, stepsize, depth, rng)
    acceptance, evaluations = 0.0, 0
    for u in starts
        chain = FixedKernelNUTSState(Xoshiro(rand(rng, UInt64)), u;
            metric, stepsize, max_tree_depth=depth)
        t = fixed_kernel_nuts_step(chain, target)
        acceptance += t.stats.acceptance_rate
        evaluations += t.stats.steps
    end
    return acceptance / length(starts), evaluations
end

# Geometric bisection on the step size, from `initial`, towards the target
# acceptance statistic. Returns the largest tried step meeting it, or the
# smallest tried step when none does.
function _cloud_stepsize(target, starts, metric, initial, depth, rng)
    lo, hi, step = 0.0, Inf, initial
    evaluations = 0
    for _ in 1:_CLOUD_ROUNDS
        acceptance, e = _cloud_trial(target, starts, metric, step, depth, rng)
        evaluations += e
        if acceptance >= _CLOUD_ACCEPTANCE
            lo = step
            step = isfinite(hi) ? sqrt(lo * hi) : 4step
        else
            hi = step
            step = lo > 0 ? sqrt(lo * hi) : step / 4
        end
    end
    return (lo > 0 ? lo : step), evaluations
end

# The stage kernel: mixture fit plus tuned per-component step sizes.
function _cloud_kernel(state, beta, delta, seed, base)
    X = state.positions
    d, n = size(X)
    logw = log.(state.weights) .+ (beta - state.temperature) .* delta
    peak = maximum(logw)
    w = exp.(logw .- peak)
    w ./= sum(w)
    rng = Xoshiro(seed)
    fit = _cloud_fit(X, w, state.options.cloud_kernels, rng)
    J = fit.components
    target = BridgeTarget(state.source, state.target, beta, d)
    stepsizes = zeros(J)
    tuning_evaluations = 0
    kernel = (; kind=:cloud, components=J, logweights=fit.logweight, means=fit.mean,
        scales=fit.sd, stepsizes, moves=base.moves, max_tree_depth=base.max_tree_depth)
    r = zeros(J)
    R = zeros(J, n)
    for i in 1:n
        R[:, i] = _cloud_responsibilities!(r, kernel, view(X, :, i))
    end
    for j in 1:J
        p = w .* view(R, j, :)
        starts = [X[:, _cloud_categorical(rng, p)] for _ in 1:_CLOUD_TRIALS]
        stepsizes[j], e = _cloud_stepsize(target, starts, Diagonal(fit.sd[:, j]),
            base.stepsize, base.max_tree_depth, rng)
        tuning_evaluations += e
    end
    return merge(kernel, (; bic=fit.bic, tuning_evaluations))
end

function _mutate_cloud!(positions, state, ancestors, seeds, target, kernel, indices)
    J = kernel.components
    d = size(positions, 1)
    origin = state.positions[:, ancestors[first(indices)]]
    templates = [FixedKernelNUTSState(Xoshiro(0), origin; metric=Diagonal(kernel.scales[:, j]),
        stepsize=kernel.stepsizes[j], max_tree_depth=kernel.max_tree_depth) for j in 1:J]
    divergences, evaluations, movement, acceptance, rejections = 0, 0, 0.0, 0.0, 0
    r = zeros(J)
    for i in indices
        u = state.positions[:, ancestors[i]]
        rng = Xoshiro(seeds[i])
        for _ in 1:kernel.moves
            j = 1
            if J > 1
                _cloud_responsibilities!(r, kernel, u)
                j = _cloud_categorical(rng, r)
            end
            here = r[j]
            t = fixed_kernel_nuts_step(FixedKernelNUTSState(templates[j]; position=u, rng), target)
            rng = t.state.rng
            divergences += t.divergent
            evaluations += t.stats.steps
            acceptance += t.stats.acceptance_rate
            proposal = t.state.position
            if J == 1
                u = proposal
            else
                there = _cloud_responsibilities!(r, kernel, proposal)[j]
                if rand(rng) * here < there
                    u = proposal
                else
                    rejections += 1
                end
            end
        end
        movement += sum(abs2, u .- view(state.positions, :, ancestors[i]))
        positions[:, i] = u
    end
    return (; divergences, evaluations, movement, acceptance, rejections)
end
