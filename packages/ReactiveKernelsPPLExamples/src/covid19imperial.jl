module Covid19ImperialExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source, _posteriordb_data

export COVID19IMPERIAL_X, COVID19IMPERIAL_EPIDEMICSTART, COVID19IMPERIAL_N,
    COVID19IMPERIAL_DEATHS, COVID19IMPERIAL_SI, COVID19IMPERIAL_F,
    COVID19IMPERIAL_POP, COVID19IMPERIAL_M, COVID19IMPERIAL_P,
    COVID19IMPERIAL_N0, COVID19IMPERIAL_N2
export build_covid19imperial_graph, demo
export COVID19IMPERIAL_SOURCE, evaluate_covid19imperial_source

# posteriordb `covid19imperial` (Imperial/flaxman2020report renewal-equation
# model). ONE faithful translation serves all four posteriors: the bundled
# `covid19imperial_v2.stan` and `covid19imperial_v3.stan` are byte-identical
# (md5 c32cbb4c…), and BridgeStan confirms both expose the same 51 unconstrained
# parameters with identical names (`mu, alpha_hier, kappa, y, phi, tau,
# ifr_noise`); the v3 posterior JSON's extra `lockdown`/`gamma` dimension
# entries are stale metadata that no shipped `.stan` declares. The graph binds
# the ecdc0401 dataset here and stays fully data-generic, so the sibling
# ecdc0501 dataset binds the same ports (same M/P/N0/N2 shapes).
#
# The sequential heart is the per-country renewal recurrence: a shift-register
# `buffer` (last N2 predictions × M countries) is threaded through ONE batched
# scan over the renewal days i = N0+1:N2 with a NamedTuple carry that also
# tracks cumulative infections; each step computes both the SI-convolution for
# `prediction` and the column-weighted f-convolution for `E_deaths`. The
# imputation days are covered exactly without scan-phase tricks: for i <= N0
# every convolution term is an imputed y[m], so E_deaths[i,m] = ifr[m]*y[m]*
# cumsum(f)[i-1,m] — a closed-form block over the data-only f-prefix sums —
# and Stan's day-1 special case E_deaths[1] = 1e-15 * prediction[1] is its own
# node. All raw-data preprocessing (X permutation/reshape, the full observed
# grid mask, the deaths grid, the count log-factorial) is derived in-graph from
# the raw PosteriorDB arrays.

let d = _posteriordb_data("ecdc0401-covid19imperial_v2")
    global const COVID19IMPERIAL_X = Float64.(d["X"])             # M×N2×P (Stan declares X real)
    global const COVID19IMPERIAL_EPIDEMICSTART = Int.(d["EpidemicStart"])
    global const COVID19IMPERIAL_N = Int.(d["N"])
    global const COVID19IMPERIAL_DEATHS = Int.(d["deaths"])       # N2×M
    global const COVID19IMPERIAL_SI = Float64.(d["SI"])           # N2
    global const COVID19IMPERIAL_F = Float64.(d["f"])             # N2×M
    global const COVID19IMPERIAL_POP = Float64.(d["pop"])         # M (Stan declares pop real)
    global const COVID19IMPERIAL_M = Int(d["M"])
    global const COVID19IMPERIAL_P = Int(d["P"])
    global const COVID19IMPERIAL_N0 = Int(d["N0"])
    global const COVID19IMPERIAL_N2 = Int(d["N2"])
end

const COVID19IMPERIAL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, gamma, exponential
using SpecialFunctions: loggamma
using LogExpFunctions: xlogy

@kernel model(unconstrained::Vector{Float64},
              X::Array{Float64,3}, EpidemicStart::Vector{Int}, N::Vector{Int},
              deaths::Matrix{Int}, SI::Vector{Float64}, fmat::Matrix{Float64},
              pop::Vector{Float64}, M::Int, P::Int, N0::Int, N2::Int) = begin
    # Stan's declared unconstrained order (all positive-support, so the only
    # transform is exp with a sum(u) Jacobian):
    # mu[1:M], alpha_hier[1:P], kappa, y[1:M], phi, tau, ifr_noise[1:M].
    u = unconstrained
    mu        = exp.(u[1:M])
    alpha_hier= exp.(u[(M+1):(M+P)])
    kappa     = exp(u[M+P+1])
    y         = exp.(u[(M+P+2):(M+P+1+M)])
    phi       = exp(u[2M+P+2])
    tau       = exp(u[2M+P+3])
    ifr_noise = exp.(u[(2M+P+4):(2M+P+3+M)])
    log_jacobian::Float64 = sum(u)

    # ---- in-graph data-only preprocessing (named all-bound recipes) ----
    XX = reshape(permutedims(X, (2, 1, 3)), N2 * M, P)    # row (m-1)*N2+i = X[m,i,:]
    # The observed grid is the FULL N2×M day range: Stan takes the
    # user-supplied EpidemicStart[m]:N[m] slice, which may legitimately include
    # the imputation days 1:N0 (EpidemicStart is data, not a compiler constant).
    icol = reshape(1:N2, :, 1)                            # absolute day i
    ESrow = reshape(EpidemicStart, 1, :)
    Nrow = reshape(N, 1, :)
    observed_mask = (icol .>= ESrow) .& (icol .<= Nrow)   # N2×M: i in ES[m]:N[m]
    # Raw deaths carry -1 placeholder entries in rows i > N[m] (the .stan
    # comment: "should be ignored"). Only UNOBSERVED cells are replaced by the
    # finite placeholder 0 (their masked contribution is exactly zero); an
    # invalid OBSERVED count is never silently clamped into valid data.
    deaths_grid = ifelse.(observed_mask, Float64.(deaths), 0.0)  # N2×M count grid
    obs_count = sum(observed_mask)
    count_logfactorial = sum(ifelse.(observed_mask, loggamma.(deaths_grid .+ 1.0), 0.0))

    # ---- Rt: hierarchical suppression covariates ----
    alpha = alpha_hier .- (log(1.05)/6.0)
    L = reshape(XX * alpha, N2, M)
    Rt_full = transpose(mu) .* exp.(-L)                  # N2×M

    # ---- E_deaths over the imputation prefix (closed form, exact) ----
    # For i <= N0 every prediction in the convolution is an imputed y[m], so
    # Stan's E_deaths[i,m] = ifr[m] * y[m] * sum_{d=1}^{i-1} f[d,m] — exactly a
    # row-prefix sum of the (data-only) f kernel scaled by the active product
    # ifr .* y. Row r covers day i = r+1.
    f_prefix = cumsum(fmat; dims = 1)                     # data-only named node
    ed_early = transpose(ifr_noise .* y) .* f_prefix[1:(N0 - 1), :]

    # ---- renewal recurrence: one batched scan over days i = N0+1:N2 ----
    # carry.buffer[d, m] = prediction[i-d, m]: a shift register seeded with the
    # N0 imputed infections; carry.cumulative[m] = cumm_sum[i-1, m] = (N0-1)*y.
    # N2 rides as an explicit Ref operand (a `scan` closure resolves free names
    # in its enclosing module, NOT in the @kernel body), and the ACTIVE
    # per-country ifr_noise rides INSIDE the carry — passed through unchanged —
    # because Ref'ing an active vector makes the step body store constant memory
    # into differentiable state under ordinary (unannotated) Enzyme. The xs
    # matrix stays purely param-derived: concatenating a CONST phase column
    # into it triggers the same runtime-activity error, which the closed-form
    # early block above makes unnecessary.
    # One extra padding row keeps the sequence non-empty when N2 = N0 (no
    # renewal days at all): its step output lands in grid row N2+1, which the
    # [1:N2, :] slice below removes, so it never contributes to E_deaths.
    xs = vcat(Rt_full[(N0 + 1):N2, :], Rt_full[1:1, :])
    buffer0 = vcat(permutedims(y) .* ones(N0, M), zeros(N2 - N0, M))
    cum0 = (N0 - 1) .* y

    eds = scan(eachrow(xs), Ref(SI), Ref(fmat), Ref(pop), Ref(N2);
               init = (; buffer = buffer0, cumulative = cum0,
                       ifr = ifr_noise)) do carry, Rt_i, si, ff, pp, n2
        conv = transpose(carry.buffer) * si
        cumulative = carry.cumulative .+ carry.buffer[1, :]
        susceptible = (pp .- cumulative) ./ pp
        pred = susceptible .* Rt_i .* conv
        conv_f = vec(sum(carry.buffer .* ff; dims = 1))
        ed = carry.ifr .* conv_f
        newbuf = vcat(permutedims(pred), carry.buffer[1:(n2 - 1), :])
        ((; buffer = newbuf, cumulative = cumulative, ifr = carry.ifr), ed)
    end
    # Full E_deaths grid: Stan's day-1 special case E_deaths[1,m] =
    # 1e-15 * prediction[1,m] (= 1e-15 * y[m]), the closed-form imputation days
    # 2:N0, then the renewal days. `permutedims(reduce(hcat, eds))` is the
    # materialization that lowers through Reactant; `reduce(vcat,
    # permutedims.(eds))` does not (traced-size typeassert), and `stack(eds)`
    # is transposed (vectors become columns).
    edmat = vcat(permutedims(1e-15 .* y), ed_early,
                 permutedims(reduce(hcat, eds)))[1:N2, :]

    # ---- priors ----
    # NOTE: ReactiveKernels `exponential(theta)` is SCALE-parameterized while
    # Stan `exponential(lambda)` is RATE-parameterized, so Stan's rate enters
    # as `exponential(1/lambda)`.
    prior_tau = exponential(1 / 0.03).logpdf(tau)
    pw_y = plate(y, tau) do ym, t
        exponential(t).logpdf(ym)     # Stan rate 1/tau -> RK scale tau
    end
    prior_y = sum(pw_y)
    prior_phi = normal(0.0, 5.0).logpdf(phi)
    prior_kappa = normal(0.0, 0.5).logpdf(kappa)
    pw_mu = plate(mu, kappa) do mm, kk
        normal(3.28, kk).logpdf(mm)
    end
    prior_mu = sum(pw_mu)
    pw_alpha = plate(alpha_hier) do aa
        gamma(0.1667, 1.0).logpdf(aa)
    end
    prior_alpha = sum(pw_alpha)
    pw_ifr = plate(ifr_noise) do ii
        normal(1.0, 0.1).logpdf(ii)
    end
    prior_ifr = sum(pw_ifr)
    prior::Float64 = prior_tau + prior_y + prior_phi + prior_kappa +
                     prior_mu + prior_alpha + prior_ifr

    # ---- neg_binomial_2 likelihood over the masked observed grid ----
    # The placeholders (0 counts, 1.0 inside the masked log) keep UNOBSERVED
    # cells finite and contribute exactly 0 through the mask; observed cells
    # keep their raw values. xlogy's iszero(x) shortcut makes the masked xlogy
    # term safe even where an unobserved forecast E_deaths is nonpositive.
    lpd = log.(phi .+ ifelse.(observed_mask, edmat, 1.0))
    loglik::Float64 = sum(ifelse.(observed_mask, loggamma.(deaths_grid .+ phi), 0.0)) -
                      obs_count * loggamma(phi) - count_logfactorial +
                      obs_count * phi * log(phi) -
                      phi * sum(ifelse.(observed_mask, lpd, 0.0)) +
                      sum(ifelse.(observed_mask, xlogy.(deaths_grid, edmat), 0.0)) -
                      sum(ifelse.(observed_mask, deaths_grid .* lpd, 0.0))

    posterior::Float64 = prior + log_jacobian + loglik
    return posterior
end

q = zeros(3 * COVID19IMPERIAL_M + COVID19IMPERIAL_P + 3)

requested_nodes = (:prior, :log_jacobian, :loglik, :posterior)
density_kernel = prepare(model;
    have = (:unconstrained, :X, :EpidemicStart, :N, :deaths, :SI, :fmat, :pop,
            :M, :P, :N0, :N2),
    want = requested_nodes,
    bound = (; X = COVID19IMPERIAL_X, EpidemicStart = COVID19IMPERIAL_EPIDEMICSTART,
             N = COVID19IMPERIAL_N, deaths = COVID19IMPERIAL_DEATHS,
             SI = COVID19IMPERIAL_SI, fmat = COVID19IMPERIAL_F,
             pop = COVID19IMPERIAL_POP, M = COVID19IMPERIAL_M,
             P = COVID19IMPERIAL_P, N0 = COVID19IMPERIAL_N0, N2 = COVID19IMPERIAL_N2))

output = density_kernel(q)
prior, logjac, loglik, posterior = output
@assert isfinite(posterior)
@assert posterior ≈ prior + logjac + loglik

docs_example = (;
    name = :covid19imperial_density,
    origin = "Imperial covid19 renewal-equation model — posteriordb covid19imperial_v2/v3",
    inputs = (; q),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    gamma_object = gamma,
    exponential_object = exponential,
)
"""

function evaluate_covid19imperial_source(; model_only::Bool = false)
    # Bind only the RAW posteriordb arrays; every model-specific preprocessing
    # step is a named in-graph node inside the source above.
    _evaluate_ppl_source(COVID19IMPERIAL_SOURCE, @__MODULE__; bindings = (
        :COVID19IMPERIAL_X, :COVID19IMPERIAL_EPIDEMICSTART, :COVID19IMPERIAL_N,
        :COVID19IMPERIAL_DEATHS, :COVID19IMPERIAL_SI, :COVID19IMPERIAL_F,
        :COVID19IMPERIAL_POP, :COVID19IMPERIAL_M, :COVID19IMPERIAL_P,
        :COVID19IMPERIAL_N0, :COVID19IMPERIAL_N2,
    ), model_only)
end

# Evaluate the authored source from `__init__`, after package precompilation has
# closed the module. `build_covid19imperial_graph` clones this runtime template
# so every caller gets an independent mutable graph without crossing a fresh
# `Core.eval` world-age boundary inside its own compiled function.
const _COVID19IMPERIAL_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _COVID19IMPERIAL_GRAPH_TEMPLATE[] = evaluate_covid19imperial_source(; model_only = true).model
    nothing
end

"""
    build_covid19imperial_graph()

Build the posteriordb Imperial covid19 renewal-equation model
(`ecdc0401-covid19imperial_v2`; `covid19imperial_v3` is the same shipped model)
as a declarative `ReactiveKernels.KernelSpec`. The nonlinear susceptible-depletion
renewal recurrence and the `E_deaths` convolution are computed by ONE batched
scan over days i = 2:N2 whose NamedTuple carry threads a per-country
shift-register prediction buffer plus cumulative infections over the renewal
days, an exact closed-form block for the imputation days, and Stan's day-1
`E_deaths` special case as its own node, so the natural sequential semantics
stay ordinary-Enzyme-clean and lower through Reactant.
Raw PosteriorDB arrays (`X`, `EpidemicStart`, `N`, `deaths`, `SI`, `f`, `pop`
and the four dimension scalars) are the only data ports; the covariate reshape,
the full observed-grid mask, the deaths grid, and the count log-factorial are
derived in-graph as named all-bound recipes. The Normal/Gamma/exponential endpoints are reused from
`ReactiveKernelsDistributionKernels` (note the exponential SCALE convention).
"""
function build_covid19imperial_graph()
    compose(_COVID19IMPERIAL_GRAPH_TEMPLATE[])
end

function demo()
    model = build_covid19imperial_graph()
    q = zeros(3 * COVID19IMPERIAL_M + COVID19IMPERIAL_P + 3)

    println("Full unconstrained-space log density (ecdc0401 binding):")
    density_plan = plan(model;
        have = (:unconstrained, :X, :EpidemicStart, :N, :deaths, :SI, :fmat,
                :pop, :M, :P, :N0, :N2),
        want = (:prior, :log_jacobian, :loglik, :posterior))
    println(explain(density_plan))
    prior, log_jacobian, loglik, posterior = prepare(density_plan)(q,
        COVID19IMPERIAL_X, COVID19IMPERIAL_EPIDEMICSTART, COVID19IMPERIAL_N,
        COVID19IMPERIAL_DEATHS, COVID19IMPERIAL_SI, COVID19IMPERIAL_F,
        COVID19IMPERIAL_POP, COVID19IMPERIAL_M, COVID19IMPERIAL_P,
        COVID19IMPERIAL_N0, COVID19IMPERIAL_N2)
    println("log prior + log Jacobian + log likelihood")
    println("= ", prior, " + ", log_jacobian, " + ", loglik)
    println("= log density = ", posterior)

    nothing
end

end # module Covid19ImperialExample

if abspath(PROGRAM_FILE) == @__FILE__
    Covid19ImperialExample.demo()
end
