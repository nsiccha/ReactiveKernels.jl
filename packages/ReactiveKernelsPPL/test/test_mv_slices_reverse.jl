using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernels,
    ReactiveKernelsPPL, Test

# The Reverse gradient of a multivariate normal density equals central
# differences of the sampler's own density. RKPPL solves the slices (and
# factors a covariance) into fresh `Float64` storage that a non-inlined helper
# fills. Handed `undef` storage that way, Enzyme returned a wrong gradient,
# often uninitialized memory, for every case below while the value stayed
# right: slice rows beside a `@plate` (alone they were right) and a sampled
# covariance (snag `ran-packages-rea-d8d16659`, Enzyme snag
# `rkppl-mvnormal-s-1df3d9b3`). The destination starts as `NaN`, so an entry
# nothing writes fails too.

_mvr_fd(f, u; h = cbrt(eps(Float64))) = [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end for i in eachindex(u)]

const _MVR_G = [1, 2, 3, 1, 2, 3, 1, 2]
const _MVR_X = [0.1, -0.3, 0.5, 0.9, -1.1, 0.2, 0.4, -0.6]
const _MVR_Y = [0.3, 1.1, -0.4, 0.8, 0.2, -1.0, 0.5, 0.7]
const _MVR_VECTORS = [[0.1, 0.4], [0.9, -0.2], [0.3, 0.3]]
const _MVR_D = [(a - b)^2 for a in (0.1, 0.5, 0.8, 1.6), b in (0.1, 0.5, 0.8, 1.6)]

@testset "multivariate normal Reverse gradients beside a plate" begin
    cases = [
        # A constant covariance's rows; a dotted cell latent reads a column.
        ("MvNormal rows read by a cell latent", quote
            eachrow(B[1:2, 1:2]) .~ MvNormal(zeros(2), [1.0 0.5; 0.5 2.0])
            @plate for i in eachindex(mu)
                v[i] .~ Normal.(log.(mu[i]) .+ B[i, 2], 0.5)
            end
        end, (; mu = [[10.0, 20.0, 5.0], [7.0, 3.0]]), ()),
        # Correlated group effects from a scaled LKJ factor, as BRM writes them.
        ("MvNormalCholesky rows with a sampled factor", quote
            sd[1:2] .~ Exponential.(1.0)
            L ~ LKJCholesky(2, 2.0)
            eachrow(B[levels(g), 1:2]) .~ MvNormalCholesky(zeros(2), sd .* L)
            sigma ~ Exponential(1.0)
            @plate for i in eachindex(y)
                y[i] ~ Normal(B[g[i], 1] + B[g[i], 2] * x[i], sigma)
            end
        end, (; g = _MVR_G, x = _MVR_X, y = _MVR_Y), (:y,)),
        # A sampled covariance: its Cholesky factor is active too.
        ("MvNormal rows with a sampled covariance", quote
            s ~ Exponential(1.0)
            eachrow(B[levels(g), 1:2]) .~ MvNormal(zeros(2), s .* [1.0 0.5; 0.5 2.0])
            @plate for i in eachindex(y)
                y[i] ~ Normal(B[g[i], 1] + B[g[i], 2] * x[i], 1.0)
            end
        end, (; g = _MVR_G, x = _MVR_X, y = _MVR_Y), (:y,)),
        ("observed vectors with a sampled covariance in a plate", quote
            m[1:2] .~ Normal.(0, 1)
            s ~ Exponential(1.0)
            @plate for i in eachindex(Y)
                Y[i] ~ MvNormal(m, s .* [1.0 0.5; 0.5 2.0])
            end
        end, (; Y = _MVR_VECTORS), (:Y,)),
        ("a Gaussian process vector", quote
            ell ~ Gamma(4, 0.5)
            amp ~ Exponential(1.0)
            C = amp^2 .* exp.(-D ./ (2 * ell^2)) + 0.3 * I
            r[axes(D, 1)] ~ MvNormal(zeros(4), C)
        end, (; D = _MVR_D, r = [0.2, -0.1, 0.4, 0.3]), (:r,)),
    ]
    for (label, ast, data, conditioned) in cases
        @testset "$label" begin
            bound = bind_data(lower_rkppl(ast, data; conditioned), data)
            built = build_kernel(bound)
            n = built.layout.total
            q = prepare_sampler(built, bound, zeros(n);
                backend = AutoEnzyme(; mode = Enzyme.Reverse))
            for u in (collect(range(-0.4, 0.9; length = n)),
                      collect(range(0.7, -0.5; length = n)))
                g = fill(NaN, n)
                value, _ = sampler_value_and_gradient!(q, g, u)
                @test value ≈ q(u) rtol = 1e-12
                @test g ≈ _mvr_fd(q, u) rtol = 1e-6 atol = 1e-7
            end
        end
    end
end
