# Sweep-owned F items (v1 inventory pair 4, RK half): fail-closed battery.
# Each case quotes the natural surface spelling for a sweep F item and pins
# the loud rejection. Previously refused spellings that now lower have
# positive regression tests, including the prophet density oracle below.
using Distributions: Normal, Laplace, truncated, logpdf
using ReactiveKernels
using ReactiveKernelsPPL
using Test

@testset "sweep refusals and capability gaps" begin
    # Each entry: (sweep item, label, program, data names, error type).
    cases = [
        # Rate scalar spelling: the thin layer is column-oriented; scalar
        # data has no admission (rate replicates via 1-element columns).
        # capability: scalar data at every model door (P10a 0dejlw1) (todo `1qlbn5b`).
        ("rate_1", "scalar-data likelihood",
            :(begin
                theta ~ Beta(1.0, 1.0)
                k ~ Binomial(n, theta)
            end), (:k, :n), SurfaceLoweringError),
        # Rate literal prob: a fully fixed Binomial contributes a
        # constant, so literals stay rejected (bare-location form takes
        # a sampled parameter). (The old non-Beta-prob pin is gone with
        # the BinomialProb surface path: bare locations admit any
        # sampled prior, matrix-b.)
        # capability: a parameter-free density statement (10gzbm9 degenerate) (todo `1qlbn5b`).
        ("rate_1", "literal prob rejected",
            :(begin
                k .~ Binomial.(n, 0.3)
            end), (:k, :n), SurfaceLoweringError),
        # m0/mb/mh/mt/mtbh/mth occupancy: custom per-cell branch over
        # capture histories. Predictor-position data-driven ifelse IS
        # admitted (elementwise selection); the response position takes
        # distribution objects only. (m0 itself replicates via the
        # ZeroInflatedBinomial head; this pins the branch spelling's
        # rejection, not the item.)
        # refused: `.~` right-hand side must be a distribution; `ifelse.(…, a, b)` yields reals (P3)
        ("m0", "response-position branch",
            :(begin
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                s .~ ifelse.(s .> 0, a, b)
            end), (:s,), SurfaceLoweringError),
        # survey_model: discrete-n marginal needs data-sized mixture
        # support; mixture weights are literal vectors or simplex names.
        # capability: data-supplied mixture weights (P10a 0dejlw1) (todo `1qlbn5b`).
        ("survey_model", "data-column mixture weights",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                mu2 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.(vcat.(Normal.(mu1, s), Normal.(mu2, s)), Ref(wcol))
            end), (:y, :wcol), ContractValidationError),
        # (The kronecker_gp `E = eigen(K)` case was removed: functions as
        # values admit any function visible in the model module from `=`,
        # a data-only call evaluated once at bind — test_functions_as_values.jl.)
        # bym2_offset_only: ICAR pairwise-difference prior needs
        # sampled-vector gathers plus a custom edge reduction.
        # refused: indexes scalar-declared `phi` with data index vectors (Julia BoundsError, P3); declared-array gathers are admitted
        ("bym2_offset_only", "sampled-vector gather",
            :(begin
                phi ~ Normal(0, 1)
                mu = phi[node1] .- phi[node2]
                y .~ Normal.(mu, 1.0)
            end), (:y, :node1, :node2), SurfaceLoweringError),
        # losscurve_sislob: the ordinary Weibull shape parameter and literal
        # scale are admitted by constructor-value lowering (`1qlbn5b`).
        ("losscurve_sislob", "Weibull response",
            :(begin
                a ~ Normal(0, 1)
                y .~ Weibull.(a, 1.5)
            end), (:y,), SurfaceLoweringError),
        # sum_to_zero: orthonormal-pivot constraint transform; no
        # constraint head in the sampling vocabulary.
        # refused: `sum_to_zero_effect` is not a visible distribution/function (Julia UndefVarError, P3); sum-to-zero becomes a library submodel (P8)
        ("sum_to_zero", "constraint sampling head",
            :(begin
                alpha ~ Normal(0.0, 10.0)
                b1 ~ Normal(0.0, 10.0)
                sigma ~ Exponential(1.0)
                r ~ sum_to_zero_effect(g)
                mu = alpha .+ b1 .* ylag1
                yt .~ Normal.(mu, sigma)
            end), (:yt, :ylag1, :g), SurfaceLoweringError),
        # covid19imperial: renewal-equation convolution recursion; no
        # convolution primitive.
        # refused: calls `conv`, undefined in the model module (Julia UndefVarError, P3)
        ("covid19imperial", "convolution call",
            :(begin
                a ~ Normal(0, 1)
                Rt = conv(prev, wgt)
                y .~ Normal.(a, 1.0)
            end), (:y, :prev, :wgt), SurfaceLoweringError),
        # (The cumulative-simplex `cumsum(w)` and data-indexed gather
        # `cs[gidx]` cases were removed: functions as values admit both —
        # `cumsum(vcat(0.0, zeta))[c]` is pinned against Distributions in
        # test_functions_as_values.jl and corpus 97.)
        # state_space_stochastic: latent random-walk prior over a sampled
        # vector (vectorized reductions over free states); ranges cover
        # eachindex exactly, never a 2:T window.
        # refused: undeclared `mu`/`T` (P6, 05oe96l) and `mu[1]` has no prior (P7, 0d5a67r); random walks spell as @scan
        ("state_space_stochastic", "indexed self-prior",
            :(begin
                s ~ Exponential(1.0)
                mu[2:T] .~ Normal.(mu[1:T-1], s)
                y .~ Normal.(mu, 1.0)
            end), (:y,), SurfaceLoweringError),
        # arma11: ARMA(1,1) sequential error recursion is deterministic
        # given data+params; a @scan seed and step that read the data
        # column `y`, now supported. Density/gradient oracles for both models
        # live in test_scan_recurrence_capabilities.jl.
        ("arma11", "data-varying scan recurrence",
            :(begin
                mu ~ Normal(0.0, 5.0)
                phi ~ Normal(0.0, 1.0)
                the ~ Normal(0.0, 1.0)
                sigma ~ Exponential(1.0)
                @scan begin
                    e[1] = y[1] - mu
                    for t in 2:T
                        e[t] = y[t] - mu - phi * (y[t - 1] - mu) - the * e[t - 1]
                    end
                end
                z .~ Normal.(e, sigma)
            end), (:y, :z), SurfaceLoweringError),
        # Prophet is admitted by ordinary-parameter fallback and checked
        # against an independent density oracle below.
        # garch11: GARCH(1,1) variance recursion is deterministic given
        # data+params, with a retained scan and ordinary indexed data reads.
        ("garch11", "data-varying scan recurrence",
            :(begin
                m ~ Normal(0.0, 1.0)
                a0 ~ Exponential(1.0)
                a1 ~ Beta(1.0, 1.0)
                b1 ~ Beta(1.0, 1.0)
                sigma ~ Exponential(1.0)
                @scan begin
                    s2[1] = a0 + a1 * (y[1] - m)^2 + b1
                    for t in 2:T
                        s2[t] = a0 + a1 * (y[t - 1] - m)^2 + b1 * s2[t - 1]
                    end
                end
                y .~ Normal.(m, sigma)
            end), (:y,), SurfaceLoweringError),
    ]
    capabilities = Set(["scalar-data likelihood", "literal prob rejected"])
    for (item, label, prog, datanames, E) in cases
        @testset "$item: $label" begin
            if label == "data-varying scan recurrence"
                observed = item == "arma11" ? (:z,) : (:y,)
                data = Dict{Symbol,Any}(:y => [0.3sin(t) for t in 1:4])
                item == "arma11" && (data[:z] = zeros(4))
                plan = bind_data(lower_rkppl(prog, datanames; conditioned = observed), data)
                @test build_kernel(plan).spec isa KernelSpec
            elseif label in ("data-column mixture weights", "Weibull response")
                @test lower_rkppl(prog, datanames; conditioned=datanames) isa StructuralPlan
            elseif label in capabilities
                # capability: each entry above names a valid model shape (todo `1qlbn5b`).
                @test_broken (lower_rkppl(prog, datanames; conditioned = datanames); true)
            else
                # refused: remaining entries cite a Julia signature,
                # undeclared name or read-before-write violation (P3/P6).
                @test_throws E lower_rkppl(prog, datanames; conditioned = datanames)
            end
        end
    end
end

# The fallback landing admits this previously refused surface: shared
# seasonality coefficients and changepoint parameters are ordinary sampled
# values wherever the affine coefficient block cannot own them exclusively.
@testset "sweep admitted: prophet shared parameters in a bilinear mean" begin
    prog = quote
        k ~ Normal(0.0, 5.0)
        m ~ Normal(0.0, 5.0)
        d1 ~ Laplace(0.0, 1.0)
        d2 ~ Laplace(0.0, 1.0)
        sigma_obs ~ truncated(Normal(0.0, 0.5), 0.0, Inf)
        b1 ~ Normal(0.0, 1.0)
        b2 ~ Normal(0.0, 0.5)
        Ad = A1 .* d1 .+ A2 .* d2
        Atd = C1 .* d1 .+ C2 .* d2
        trend = (k .+ Ad) .* t .+ (m .- Atd)
        seas_m = X1m .* b1 .+ X2m .* b2
        seas_a = X1a .* b1 .+ X2a .* b2
        mu = trend .* (1.0 .+ seas_m) .+ seas_a
        y .~ Normal.(mu, sigma_obs)
    end
    data = (y = [0.4, -0.2, 0.8, 1.1, -0.5],
        t = [0.0, 0.2, 0.5, 0.8, 1.1],
        A1 = [0.0, 0.0, 1.0, 1.0, 1.0],
        A2 = [0.0, 0.0, 0.0, 1.0, 1.0],
        C1 = [0.0, 0.0, 0.25, 0.25, 0.25],
        C2 = [0.0, 0.0, 0.0, 0.7, 0.7],
        X1m = [0.2, -0.4, 0.8, -0.3, 0.1],
        X2m = [-0.6, 0.1, 0.2, 0.5, -0.2],
        X1a = [0.9, 0.3, -0.2, 0.4, -0.7],
        X2a = [0.1, -0.5, 0.6, -0.1, 0.8])
    probes = (
        (k = 0.7, m = -0.25, d1 = 0.3, d2 = -0.1,
            sigma_obs = 0.6, b1 = 0.4, b2 = -0.2),
        (k = -0.4, m = 0.6, d1 = -0.2, d2 = 0.5,
            sigma_obs = 1.3, b1 = -0.3, b2 = 0.7),
        (k = 1.1, m = -0.8, d1 = 0.6, d2 = -0.4,
            sigma_obs = 0.2, b1 = 0.9, b2 = 0.3))
    for n in (1, 3, 5)
        @testset "$n observations" begin
            cols = Dict{Symbol,AbstractVector}(k => v[1:n] for (k, v) in pairs(data))
            bound = bind_data(lower_rkppl(prog, keys(cols); conditioned = keys(cols)), cols)
            built = build_kernel(bound)
            lay = built.layout
            @test lay.total == 7
            likelihood = prepare_query(built, bound, :likelihood)
            prior = prepare_query(built, bound, :prior)
            posterior = prepare_query(built, bound, :sampler)
            for q in probes
                u = unconstrain(lay, merge(constrain(lay, zeros(lay.total)), q))
                # Independent scalar, per-row calculation; no extracted
                # columns or composed predictors from the lowered plan.
                ll = sum(eachindex(cols[:y])) do i
                    slope = q.k + cols[:A1][i] * q.d1 + cols[:A2][i] * q.d2
                    intercept = q.m - cols[:C1][i] * q.d1 - cols[:C2][i] * q.d2
                    multiplicative = cols[:X1m][i] * q.b1 + cols[:X2m][i] * q.b2
                    additive = cols[:X1a][i] * q.b1 + cols[:X2a][i] * q.b2
                    mu = (slope * cols[:t][i] + intercept) * (1 + multiplicative) + additive
                    logpdf(Normal(mu, q.sigma_obs), cols[:y][i])
                end
                pr = logpdf(Normal(0, 5), q.k) + logpdf(Normal(0, 5), q.m) +
                    logpdf(Laplace(0, 1), q.d1) + logpdf(Laplace(0, 1), q.d2) +
                    logpdf(truncated(Normal(0, 0.5), 0, Inf), q.sigma_obs) +
                    logpdf(Normal(0, 1), q.b1) + logpdf(Normal(0, 0.5), q.b2)
                @test Base.invokelatest(likelihood, u) ≈ ll rtol = 1e-12
                @test Base.invokelatest(prior, u) ≈ pr rtol = 1e-12
                @test Base.invokelatest(posterior, u) ≈ ll + pr + log(q.sigma_obs) rtol = 1e-12
            end
        end
    end
end
