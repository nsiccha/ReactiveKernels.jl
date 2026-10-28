module WholeEventOccupancyTests
using Test, ReactiveKernels, ReactiveKernelsPPL
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    zero_inflated_binomial_logit_product_mixture_logdensity
using Distributions: Beta, MvNormal, Normal, logpdf, truncated
import DifferentiationInterface as DI
import Enzyme

# Data-augmented community occupancy: species i is a member of the community
# with probability omega; a member occupies each site with occupancy log-odds
# `a + u[i, 1]` and is then detected on each of three visits with log-odds
# `b + u[i, 2]`. Rows past the detected species are all-zero augmentation
# rows. Membership is marginalized over a species' whole row and occupancy
# per site, so a row's law is a two-component whole-event mixture whose
# member component is a product of zero-inflated Binomial cells. The
# species effects are correlated through a 2 x 2 covariance.
function occupancy_reference(nt, counts, visits)
    setprecision(BigFloat, 512) do
        a, b, omega, r = big(nt.a), big(nt.b), big(nt.omega), big(nt.r)
        s = big.(nt.s)
        rho = 2r - 1
        log_inv_logit(x) = -log1p(exp(-x))
        logaddexp(x, y) = (top = max(x, y); top == -Inf ? top :
            top + log(exp(x - top) + exp(y - top)))
        likelihood = sum(eachindex(counts); init = big(0.0)) do i
            occupancy, detection = a + big(nt.u[i, 1]), b + big(nt.u[i, 2])
            cell(y) = y > 0 ?
                log_inv_logit(occupancy) + log(binomial(big(visits), big(y))) +
                    y * log_inv_logit(detection) +
                    (visits - y) * log_inv_logit(-detection) :
                logaddexp(log_inv_logit(occupancy) +
                    visits * log_inv_logit(-detection), log_inv_logit(-occupancy))
            member = log(omega) + sum(cell, counts[i]; init = big(0.0))
            any(>(0), counts[i]) ? member : logaddexp(member, log1p(-omega))
        end
        Sigma = [s[1]^2 rho * s[1] * s[2]; rho * s[1] * s[2] s[2]^2]
        effects = sum(eachindex(counts); init = big(0.0)) do i
            v = big.(nt.u[i, :])
            -log(2 * big(pi)) - log(Sigma[1, 1] * Sigma[2, 2] - Sigma[1, 2]^2) / 2 -
                (v' * (Sigma \ v)) / 2
        end
        prior = logpdf(Normal(0, 1), nt.a) + logpdf(Normal(0, 1), nt.b) +
            logpdf(Beta(2, 2), nt.omega) + logpdf(Beta(2, 2), nt.r) +
            sum(logpdf(truncated(Normal(0, 1), 0, Inf), x) for x in nt.s)
        (; likelihood, posterior = prior + effects + likelihood)
    end
end

@testset "whole-event zero-inflated Binomial mixture: data-augmented occupancy" begin
    counts = [[2, 0, 1, 0], [0, 0, 3, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    visits, S = 3, length(counts)
    trials = fill(visits, 4, 2)
    never = fill(-Inf, 4)
    data = (; counts, trials, never, S)
    model = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        omega ~ Beta(2, 2)
        s[1:2] .~ HalfNormal.(1)
        r ~ Beta(2, 2)
        rho = 2 * r - 1
        Sigma = [s[1]^2 rho * s[1] * s[2]; rho * s[1] * s[2] s[2]^2]
        eachrow(u[1:S, 1:2]) .~ MvNormal(zeros(2), Sigma)
        @plate for i in eachindex(counts)
            occupancy = a + u[i, 1]
            detection = b + u[i, 2]
            counts[i] ~ LogDensity(zero_inflated_binomial_logit_product_mixture_logdensity,
                trials, hcat(fill(detection, length(counts[i])), never),
                hcat(fill(-occupancy, length(counts[i])), never), [omega, 1 - omega])
        end
    end
    saved = deepcopy(data)
    bound = bind_data(lower_rkppl(model, data; mod=@__MODULE__,
        conditioned=(:counts,)), data)
    built = build_kernel(bound)
    @test !occursin("sampling_logdensity", string(kernel_expr(bound, built.layout)))
    likelihood = prepare_query(built, bound, :likelihood)
    names = coordinate_names(built.layout)
    position(name) = findfirst(==(name), names)
    base = [0.15 * cos(3i + 0.2) for i in eachindex(names)]
    sampler = prepare_sampler(built, bound, base;
        backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
    oracle(v) = occupancy_reference(constrain(built.layout, v), counts, visits)
    # The last point saturates detection (logit about 40) and occupancy
    # (about -40), where the probability spelling loses its tails.
    saturated = copy(base)
    saturated[position(:b)] = 40.0
    saturated[position(:a)] = -40.0
    for unconstrained in (base, -1.3 .* base, saturated)
        actual, gradient = sampler_value_and_gradient!(sampler,
            similar(unconstrained), unconstrained)
        expected = oracle(unconstrained)
        @test likelihood(unconstrained) ≈ Float64(expected.likelihood) rtol=1e-12
        @test actual ≈ Float64(expected.posterior) +
            logjac(built.layout, unconstrained) rtol=1e-12
        step = 1e-6
        for i in eachindex(unconstrained)
            plus, minus = copy(unconstrained), copy(unconstrained)
            plus[i] += step
            minus[i] -= step
            target = (Float64(oracle(plus).posterior) + logjac(built.layout, plus) -
                Float64(oracle(minus).posterior) - logjac(built.layout, minus)) / (2step)
            @test gradient[i] ≈ target atol=1e-6 rtol=1e-6
        end
    end
    @test data == saved
end
end
