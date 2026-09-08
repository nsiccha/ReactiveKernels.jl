# Density parity between the EXPERIMENTAL `@ppl` front-end and the authoritative
# hand-authored posteriordb example graphs. This is the regression guard for the
# claim that `@ppl` lowers a posteriordb model to a density-equivalent RK kernel
# (coordination with ReactiveKernels:ppl:posteriordb / :benchmark). It compares
# the macro-produced kernel against the real hand-authored graph — not merely a
# reference formula — at several points, including nontrivial constrained values.
#
# `@ppl` is EXPERIMENTAL and deliberately NOT exported; reach it through the
# qualified submodule path.
using ReactiveKernelsPPLExamples.PPLMacro: @ppl
using ReactiveKernelsPPLExamples: EightSchoolsExample, PPLEightSchoolsExample
using ReactiveKernelsPPLExamples.EightSchoolsExample:
    EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA, build_eight_schools_graph
using ReactiveKernelsPPLExamples.PPLEightSchoolsExample: build_ppl_eight_schools
using ReactiveKernelsPPLExamples.KilpisjarviExample:
    KILPISJARVI_X, KILPISJARVI_Y, KILPISJARVI_XPRED,
    KILPISJARVI_PMUALPHA, KILPISJARVI_PSALPHA,
    KILPISJARVI_PMUBETA, KILPISJARVI_PSBETA, build_kilpisjarvi_graph
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, cauchy

@testset "@ppl posteriordb-model parity (experimental)" begin
    @testset "eight_schools (centered) — parity with the hand-authored kernel" begin
        # The whole model, authored in four `~` statements. The centered
        # posteriordb parametrization: μ ~ Normal(0,5), τ ~ HalfCauchy(0,5),
        # θⱼ ~ Normal(μ,τ), yⱼ ~ Normal(θⱼ,σⱼ). `positive(cauchy(…))` supplies the
        # half-Cauchy (+log 2 normalization and log/exp transform with Jacobian
        # log_τ), so the unconstrained packing is q = [μ, log_τ, θ…] — identical
        # to the hand-authored kernel's.
        @ppl eight_schools(y::Vector{Float64}, sigma::Vector{Float64}, J::Int) =
            begin
                mu ~ normal(0.0, 5.0)
                tau ~ positive(cauchy(0.0, 5.0))
                theta::vector[J] ~ normal(mu, tau)
                y ~ normal(theta, sigma)
            end
        @test eight_schools isa KernelSpec

        y = EIGHT_SCHOOLS_Y
        sigma = EIGHT_SCHOOLS_SIGMA
        J = 8
        test_points = (
            [1.5, log(2.0), (0.25 .* (1:8))...],
            [0.0, log(5.0), zeros(8)...],
            [-2.3, log(0.7), collect(range(-1.0, 1.0; length = 8))...],
        )

        for q in test_points
            hand = build_eight_schools_graph()
            hand_post = prepare(hand;
                have = (:unconstrained, :observations, :observation_scales),
                want = :posterior)(q, y, sigma)
            hand_prior, hand_lik = prepare(hand;
                have = (:unconstrained, :observations, :observation_scales),
                want = (:prior, :likelihood))(q, y, sigma)

            ppl_post = prepare(eight_schools;
                have = (:unconstrained, :y, :sigma, :J),
                want = :posterior)(q, y, sigma, J)
            ppl_prior, ppl_lik = prepare(eight_schools;
                have = (:unconstrained, :y, :sigma, :J),
                want = (:prior, :likelihood))(q, y, sigma, J)

            # Density-equivalent to machine precision — same posterior, and the
            # same prior/likelihood decomposition (prior excludes the Jacobian in
            # both; the unconstrained posterior adds log_τ in both).
            @test ppl_post ≈ hand_post rtol = 1e-12
            @test ppl_prior ≈ hand_prior rtol = 1e-12
            @test ppl_lik ≈ hand_lik rtol = 1e-12
        end
    end

    @testset "importable builder — build_ppl_eight_schools()" begin
        # The benchmark 4th side consumes this builder (not test-local code). It
        # returns a KernelSpec with the IDENTICAL query surface as
        # build_eight_schools_graph() — same have/want, same q packing, and NO
        # size port (the effects vector uses the literal size 8).
        y = EIGHT_SCHOOLS_Y
        sigma = EIGHT_SCHOOLS_SIGMA
        builder = build_ppl_eight_schools()
        @test builder isa KernelSpec
        # A fresh, independent graph per call.
        @test build_ppl_eight_schools() !== builder

        for q in ([1.5, log(2.0), (0.25 .* (1:8))...],
                  [-2.3, log(0.7), collect(range(-1.0, 1.0; length = 8))...])
            hand = build_eight_schools_graph()
            hand_post = prepare(hand;
                have = (:unconstrained, :observations, :observation_scales),
                want = :posterior)(q, y, sigma)
            # Note: identical `have` tuple to the hand kernel — no :J.
            ppl_post = prepare(build_ppl_eight_schools();
                have = (:unconstrained, :observations, :observation_scales),
                want = :posterior)(q, y, sigma)
            @test ppl_post ≈ hand_post rtol = 1e-12
        end
    end

    @testset "kilpisjarvi (flat sigma) — parity with the hand-authored kernel" begin
        # posteriordb `kilpisjarvi`: α ~ Normal(pmualpha, psalpha), β ~
        # Normal(pmubeta, psbeta) with data-supplied adjustable hyperparameters,
        # and σ with NO prior — Stan's `real<lower=0> sigma` improper-flat, so
        # only the log/exp transform Jacobian (log σ) contributes. `positive(flat())`
        # supplies exactly that: positive support (Jacobian log σ) with a zero
        # prior term. The unconstrained packing q = [α, β, log σ] is identical to
        # the hand-authored kernel's — the dominant posteriordb coverage gap (todo
        # 0gy1ejg) closed for a real model.
        @ppl kilpisjarvi(y::Vector{Float64}, x::Vector{Float64},
                         pmualpha::Float64, psalpha::Float64,
                         pmubeta::Float64, psbeta::Float64) = begin
            alpha ~ normal(pmualpha, psalpha)
            beta ~ normal(pmubeta, psbeta)
            sigma ~ positive(flat())
            y ~ normal(alpha + beta * x, sigma)
        end
        @test kilpisjarvi isa KernelSpec

        x = KILPISJARVI_X
        y = KILPISJARVI_Y
        pma, psa = KILPISJARVI_PMUALPHA, KILPISJARVI_PSALPHA
        pmb, psb = KILPISJARVI_PMUBETA, KILPISJARVI_PSBETA
        test_points = (
            [9.3, 0.0, log(1.0)],
            [9.0, 0.05, log(0.7)],
            [10.1, -0.02, log(2.3)],
        )
        for q in test_points
            hand = build_kilpisjarvi_graph()
            hand_have = (:unconstrained, :x, :y, :xpred, :pmualpha, :psalpha,
                         :pmubeta, :psbeta)
            hand_post = prepare(hand; have = hand_have, want = :posterior)(
                q, x, y, KILPISJARVI_XPRED, pma, psa, pmb, psb)
            hand_prior, hand_lik = prepare(hand; have = hand_have,
                want = (:log_prior, :likelihood))(
                    q, x, y, KILPISJARVI_XPRED, pma, psa, pmb, psb)

            ppl_have = (:unconstrained, :y, :x, :pmualpha, :psalpha, :pmubeta, :psbeta)
            ppl_post = prepare(kilpisjarvi; have = ppl_have, want = :posterior)(
                q, y, x, pma, psa, pmb, psb)
            ppl_prior, ppl_lik = prepare(kilpisjarvi; have = ppl_have,
                want = (:prior, :likelihood))(q, y, x, pma, psa, pmb, psb)

            # Density-equivalent to machine precision — the flat σ contributes 0 to
            # the prior in both, and both add log σ (the transform Jacobian) to the
            # unconstrained posterior.
            @test ppl_post ≈ hand_post rtol = 1e-12
            @test ppl_prior ≈ hand_prior rtol = 1e-12
            @test ppl_lik ≈ hand_lik rtol = 1e-12
        end
    end
end
