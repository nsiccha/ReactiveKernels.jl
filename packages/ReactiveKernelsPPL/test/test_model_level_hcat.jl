module ModelLevelHcatTests
using Test, ReactiveKernels, ReactiveKernelsPPL
using Distributions: Normal, logpdf
import DifferentiationInterface as DI
import Enzyme

# An `hcat` of model-level values (declared arrays, sampled scalars) is a
# model-level matrix, as a declared matrix is. Elementwise maps over it,
# named or inline, and its products with a data matrix keep Julia's meaning
# (snag `rkppl-elementwis-f934772a`: `E = exp.(u * hcat(w1, w2))` failed
# with "derived column is scalar-valued").

function check(ast, data, oracle; points)
    bound = bind_data(lower_rkppl(ast, data; mod = @__MODULE__,
        conditioned = (:y,)), data)
    built = build_kernel(bound)
    likelihood = prepare_query(built, bound, :likelihood)
    sampler = prepare_sampler(built, bound, first(points);
        backend = DI.AutoEnzyme(; mode = Enzyme.Reverse))
    for u in points
        @test Base.invokelatest(likelihood, u) ≈
            oracle(constrain(built.layout, u)) rtol = 1e-12
        gradient = similar(u)
        sampler_value_and_gradient!(sampler, gradient, u)
        step = 1e-6
        for i in eachindex(u)
            plus, minus = copy(u), copy(u)
            plus[i] += step
            minus[i] -= step
            expected = (Base.invokelatest(sampler, plus) -
                Base.invokelatest(sampler, minus)) / (2step)
            @test gradient[i] ≈ expected atol = 1e-6 rtol = 1e-6
        end
    end
    return built
end

softmax_rows(R) = exp.(R) ./ sum(exp.(R); dims = 2)

u = [0.5 1.0; 1.2 -0.3; 0.1 0.4; 2.0 1.1; -0.4 0.2; 0.9 -1.0]
y = [0.3, 0.9, 0.1, 0.7, -0.2, 0.6]
points = ([0.3, -0.4, 0.1, 0.5, -0.2], [-0.7, 0.2, 0.9, -0.3, 0.4])

@testset "elementwise maps over an hcat of declared columns" begin
    # Read whole: the row log-sum-exp of a model-level matrix is a
    # model-level vector, summed into one location.
    oracle(c) = sum(logpdf.(Normal.(
        sum(log.(sum(exp.(hcat([c.a, c.b], c.z)); dims = 2))), 1), y))
    for body in (
        :(begin
            H = hcat([a, b], z)
            E = exp.(H)
            loc = sum(log.(sum.(eachrow(E))))
        end),
        :(begin
            loc = sum(log.(sum.(eachrow(exp.(hcat([a, b], z))))))
        end),
    )
        check(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            z[1:2] .~ Normal.(0, 1)
            $(body.args...)
            y .~ Normal.(loc, 1)
        end, (; y), oracle; points = (points[1][1:4], points[2][1:4]))
    end
end

@testset "a data matrix times an hcat of declared columns, read by column" begin
    # `u * hcat(w1, w2)` has one row per observation, exactly as `u * W`
    # with a declared matrix `W`; its softmax reads per observation.
    oracle(c) = sum(logpdf.(Normal.(softmax_rows(u * hcat(c.w1, c.w2))[:, 2],
        c.s), y))
    for body in (
        :(begin
            R = u * hcat(w1, w2)
            E = exp.(R)
            P = E ./ sum.(eachrow(E))
        end),
        :(begin
            Wm = hcat(w1, w2)
            P = exp.(u * Wm) ./ sum.(eachrow(exp.(u * Wm)))
        end),
        :(begin
            W = hcat(w1, w2)
            R = u * W
            P = exp.(R .- log.(sum.(eachrow(exp.(R)))))
        end),
    )
        check(quote
            w1[axes(u, 2)] .~ Normal.(0, 2)
            w2[axes(u, 2)] .~ Normal.(0, 2)
            s ~ Exponential(1)
            $(body.args...)
            y .~ Normal.(P[:, 2], s)
        end, (; y, u), oracle; points)
    end
end

@testset "hcat columns and a declared matrix give one density" begin
    # `W[axes(u, 2), 1:2]` packs column-major, so its coordinates line up
    # with `w1` then `w2`.
    data = (; y, u)
    densities = map((
        quote
            w1[axes(u, 2)] .~ Normal.(0, 2)
            w2[axes(u, 2)] .~ Normal.(0, 2)
            s ~ Exponential(1)
            E = exp.(u * hcat(w1, w2))
            y .~ Normal.((E ./ sum.(eachrow(E)))[:, 2], s)
        end,
        quote
            W[axes(u, 2), 1:2] .~ Normal.(0, 2)
            s ~ Exponential(1)
            E = exp.(u * W)
            y .~ Normal.((E ./ sum.(eachrow(E)))[:, 2], s)
        end,
    )) do ast
        bound = bind_data(lower_rkppl(ast, data; mod = @__MODULE__,
            conditioned = (:y,)), data)
        built = build_kernel(bound)
        sampler = prepare_query(built, bound, :sampler)
        [Base.invokelatest(sampler, p) for p in points]
    end
    @test densities[1] ≈ densities[2] rtol = 1e-14
end
end
