# Standalone regression: only Enzyme and Test are required.
# A geometry callback accepts live vector and scalar arguments in an unused
# vararg tail. On Julia 1.10.12, registered Enzyme 0.13.216 passes; dogfood
# 3a89b497 with libEnzyme 782c9d95 raises EnzymeInternalError in
# runtime_generic_fwd during Reverse over Forward. First-order Reverse and
# Julia inference pass. Enzyme snag compute-an-ordin-902b6418 repaired this
# in dogfood 0e251cfb with the same libEnzyme build.
using Enzyme
using Test

const geometry = (
    constrain=(u, shape, lo, rest...) -> isempty(shape) ?
        lo + exp(only(u)) : reshape(lo .+ exp.(u), shape),
    logjac=(u, shape, rest...) -> sum(u))

function objective(u, data)
    lo, weights, rows, g = data
    mu = u[1]
    sigma = exp(u[2])
    location = mu .* weights
    x = g.constrain(u[3:end], size(lo), lo, location[rows], sigma)
    r = 0.0
    for j in eachindex(x)
        r += -log(sigma) - (x[j]-location[rows[j]])^2/(2sigma^2)
    end
    return -mu^2/2 - sigma - (1-mu-sum(x))^2/2 + r + u[2] +
        g.logjac(u[3:end], size(lo), lo, location[rows], sigma)
end

directional(u, v, data) =
    only(autodiff(Forward, objective, Duplicated(u, v), Const(data)))

@testset "geometry HVP" begin
    for n in (0, 2)
        @testset "n=$n" begin
            data = (fill(1.0, n), ones(2), collect(1:n), geometry)
            u = zeros(n + 2)
            v = ones(n + 2)
            hu, du = zero(u), zero(u)
            @test @inferred(objective(u, data)) ≈ -1-(1-2n)^2/2-2n
            autodiff(Reverse, objective, Active, Duplicated(u, du), Const(data))
            @test du ≈ vcat(1.0, 3n, fill(-2.0n, n))
            autodiff(Reverse, directional, Active,
                Duplicated(u, hu), Const(v), Const(data))
            @test hu ≈ vcat(-2.0-5n, -1.0-8n, fill(2.0-3n, n))
        end
    end
end
