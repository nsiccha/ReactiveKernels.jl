# Sweep-owned F items (v1 inventory pair 4, RK half): fail-closed battery.
# Each case quotes the natural surface spelling for a sweep F item and pins
# the loud rejection. All spellings/error types confirmed against the
# admission probes (probe/probe2/probe3).
using ReactiveKernelsPPL
using Test

@testset "sweep fail-closed battery" begin
    # Each entry: (sweep item, label, program, data names, error type).
    cases = [
        # Rate scalar spelling: the thin layer is column-oriented; scalar
        # data has no admission (rate replicates via 1-element columns).
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
        ("m0", "response-position branch",
            :(begin
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                s .~ ifelse.(s .> 0, a, b)
            end), (:s,), SurfaceLoweringError),
        # survey_model: discrete-n marginal needs data-sized mixture
        # support; mixture weights are literal vectors or simplex names.
        ("survey_model", "data-column mixture weights",
            :(begin
                mu1 ~ Normal(0.0, 5.0)
                mu2 ~ Normal(0.0, 5.0)
                s ~ Exponential(1.0)
                y .~ MixtureModel.([Normal.(mu1, s), Normal.(mu2, s)], wcol)
            end), (:y, :wcol), ContractValidationError),
        # kronecker_gp: Kronecker-eigenspace marginal likelihood needs
        # spectral calls outside the slice-1 value vocabulary.
        ("kronecker_gp", "eigen call in assignments",
            :(begin
                a ~ Normal(0, 1)
                E = eigen(K)
                y .~ Normal.(a, 1.0)
            end), (:y, :K), SurfaceLoweringError),
        # bym2_offset_only: ICAR pairwise-difference prior needs
        # sampled-vector gathers plus a custom edge reduction.
        ("bym2_offset_only", "sampled-vector gather",
            :(begin
                phi ~ Normal(0, 1)
                mu = phi[node1] .- phi[node2]
                y .~ Normal.(mu, 1.0)
            end), (:y, :node1, :node2), SurfaceLoweringError),
        # losscurve_sislob: Weibull/log-logistic CDF growth curve; no
        # Weibull in the response vocabulary.
        ("losscurve_sislob", "Weibull response",
            :(begin
                a ~ Normal(0, 1)
                y .~ Weibull.(a, 1.5)
            end), (:y,), SurfaceLoweringError),
        # sum_to_zero: orthonormal-pivot constraint transform; no
        # constraint head in the sampling vocabulary.
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
        ("covid19imperial", "convolution call",
            :(begin
                a ~ Normal(0, 1)
                Rt = conv(prev, wgt)
                y .~ Normal.(a, 1.0)
            end), (:y, :prev, :wgt), SurfaceLoweringError),
        # bruno qt: indexed cumulative-simplex slope; no cumsum admission.
        ("bruno-qt", "cumsum call",
            :(begin
                w ~ Dirichlet([1.0, 1.0, 1.0])
                cs = cumsum(w)
                y .~ Normal.(a, 1.0)
            end), (:y,), SurfaceLoweringError),
        # bruno qt (second half): data-indexed gather of a slope vector.
        ("bruno-qt", "index gather",
            :(begin
                a ~ Normal(0, 1)
                mu = cs[gidx]
                y .~ Normal.(mu, 1.0)
            end), (:y, :cs, :gidx), SurfaceLoweringError),
        # state_space_stochastic: latent random-walk prior over a sampled
        # vector (vectorized reductions over free states); ranges cover
        # eachindex exactly, never a 2:T window.
        ("state_space_stochastic", "indexed self-prior",
            :(begin
                s ~ Exponential(1.0)
                mu[2:T] .~ Normal.(mu[1:T-1], s)
                y .~ Normal.(mu, 1.0)
            end), (:y,), SurfaceLoweringError),
        # arma11: ARMA(1,1) sequential error recursion is deterministic
        # given data+params, but @scan setup must sample its state.
        ("arma11", "deterministic scan setup",
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
        # prophet (out-of-scope, must still fail loudly): the changepoint
        # trend with continuity correction plus shared-beta
        # multiplicative/additive seasonality forces dual coef/param roles
        # (a predictor coefficient inside an extracted column). The reduced
        # bilinear-with-pure-value-factors shape lowers; the full model does
        # not (sweep probe3 + corpus-71 attempts).
        ("prophet", "dual coef/param roles in bilinear mean",
            :(begin
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
            end), (:y, :t, :A1, :A2, :C1, :C2, :X1m, :X2m, :X1a, :X2a),
            SurfaceLoweringError),
        # bordet-builder7 grouped-hsgp-by probe REMOVED 2026-09-30
        # (R-P1/P2/P3 re-verify): matrix-b landed grouped HSGP
        # (`hsgp_basis(...; by = g)`, commit 6dfdca8d), so the spelling
        # this case pinned as SurfaceLoweringError now lowers. Builder 7
        # (Bordet e800ef8d) is admitted and trio-tested as D7 in
        # test_smooth_sb.jl.
        # garch11: GARCH(1,1) variance recursion is deterministic given
        # data+params; same scan-setup gate.
        ("garch11", "deterministic scan setup",
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
    for (item, label, prog, datanames, E) in cases
        @testset "$item: $label" begin
            @test_throws E lower_rkppl(prog, datanames)
        end
    end
end
