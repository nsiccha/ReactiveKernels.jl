module PPLNamedMatrixDiscriminationTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Test
using Distributions: Normal, cdf, logpdf

module Models
using ReactiveKernelsPPL
@rkppl population(X, loc, scale) = begin
    beta_pop[axes(X, 2)] .~ Normal.(loc, scale)
    return X * beta_pop
end
end

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

function build_case()
    data = (; x=[-1.0, -0.4, 0.2, 0.6, 0.9, 1.3],
        z=[0.1, -0.3, 0.6, -0.2, 0.4, 0.8], y=[1, 2, 3, 2, 1, 3])
    ast = quote
        X_eta = hcat(x)
        pop_eta ~ population(X_eta, 0.0, 1.0)
        eta = pop_eta
        X_log_disc = hcat(ones(length(z)), z)
        pop_log_disc ~ population(X_log_disc, 0.0, 1.0)
        disc = pop_log_disc
        cuts ~ Ordered(Normal(0.0, 1.0), 2)
        y .~ Ordinal.(Cumulative(), ProbitLink(), eta, Ref(cuts), exp.(disc))
    end
    bound = bind_data(lower_rkppl(ast, data; conditioned=(:y,), mod=Models), data)
    return bound, build_kernel(bound), data
end

function reference(layout, data, u)
    q = constrain(layout, u)
    eta = only(q.pop_eta.beta_pop) .* data.x
    d = exp.(q.pop_log_disc.beta_pop[1] .+ q.pop_log_disc.beta_pop[2] .* data.z)
    likelihood = sum(eachindex(data.y)) do i
        y = data.y[i]
        F(j) = cdf(Normal(), d[i] * (q.cuts[j] - eta[i]))
        log((y == 3 ? 1.0 : F(y)) - (y == 1 ? 0.0 : F(y - 1)))
    end
    return likelihood + sum(logpdf.(Normal(), q.pop_eta.beta_pop)) +
        sum(logpdf.(Normal(), q.pop_log_disc.beta_pop)) +
        sum(logpdf.(Normal(), q.cuts)) + logjac(layout, u)
end

function reference_gradient(layout, data, u; h=1e-6)
    return map(eachindex(u)) do i
        hi, lo = copy(u), copy(u)
        hi[i] += h
        lo[i] -= h
        (reference(layout, data, hi) - reference(layout, data, lo)) / (2h)
    end
end

@testset "submodel matrix products compose in ordinal discrimination" begin
    bound, built, data = build_case()
    @test Set(coordinate_names(built.layout)) == Set(Symbol.(
        ["pop_eta.beta_pop.1", "pop_log_disc.beta_pop.1", "pop_log_disc.beta_pop.2",
            "cuts.1", "cuts.2"]))
    original = deepcopy(data)
    u = [0.1 * cos(i) for i in 1:built.layout.total]
    sampler = prepare_sampler(built, bound, u; backend=BACKEND)
    for w in (u, -u)
        original_w = copy(w)
        g = similar(w)
        value, _ = sampler_value_and_gradient!(sampler, g, w)
        @test value ≈ reference(built.layout, data, w) rtol=1e-12
        @test g ≈ reference_gradient(built.layout, data, w) rtol=1e-5 atol=1e-7
        @test w == original_w
        @test data == original
    end
end

end
