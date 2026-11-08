using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

# A `@plate` output whose cell reads no per-index value still holds one value
# per index, as the Julia loop leaves it (snag `plate-output-of-c4a0ca87`):
# `c[i] = exp(log_a)` over three indices is three entries, so `c[2]`,
# `length(c)` and `sum(c)` read them. Before, the output was the bare scalar:
# a later plate's `c[2]` raised a `BoundsError`, and `sum(c)` silently gave one
# entry's value. One cell cannot tell the two apart, so every check binds
# several.

const _IPO_T = [[0.5, 1.0, 2.0], [0.5, 3.0], [1.0, 2.0, 4.0, 8.0], [2.0]]
const _IPO_Y = [[18.0, 15.0, 11.0], [20.0, 8.0], [16.0, 12.0, 7.0, 3.0], [9.0]]
const _IPO_REVERSE = AutoEnzyme(; mode = Enzyme.Reverse)

_ipo_value(built, bound, name, u) =
    Base.invokelatest(prepare_query(built, bound, (name,)), u)
_ipo_query(built, bound, what, u) =
    Base.invokelatest(prepare_query(built, bound, what), u)

# The Julia loop's density of the two-plate model, and its gradient.
function _ipo_two_plate_density(u, t, y)
    d, s = exp(u[1]), exp(u[2])
    return sum(sum(logpdf.(Normal.(exp.(-d .* t[i]), s), y[i])) for i in eachindex(t))
end
function _ipo_two_plate_gradient(u, t, y)
    d, s = exp(u[1]), exp(u[2])
    ga = gs = 0.0
    for i in eachindex(t), (tij, yij) in zip(t[i], y[i])
        m = exp(-d * tij)
        r = (yij - m) / s^2
        ga += r * m * (-tij) * d
        gs += -1 + (yij - m)^2 / s^2
    end
    return [ga, gs]
end

const _IPO_TWO_PLATES = @rkppl begin
    log_a ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    @plate for i = eachindex(path)
        decay_cells[i] = exp(log_a)
    end
    @plate for i_ = eachindex(path)
        m_cell = exp.(-(decay_cells[i_]) .* t[i_])
        y[i_] .~ Normal.(m_cell, sigma)
    end
end

@testset "an output that reads no per-index value is read per index by a later plate" begin
    for n in (3, 4, 1)
        t, y = _IPO_T[1:n], _IPO_Y[1:n]
        bound = _IPO_TWO_PLATES(; t, path = collect(1:n)) | (; y)
        built = build_kernel(bound)
        for u in ([0.1, 0.1], [-0.4, 0.3])
            @test _ipo_value(built, bound, :decay_cells, u) == fill(exp(u[1]), n)
            @test _ipo_query(built, bound, :likelihood, u) ≈
                _ipo_two_plate_density(u, t, y) rtol = 1e-12
            pointwise = _ipo_query(built, bound, :pointwise, u).y
            @test length.(pointwise) == length.(y)
            @test sum(sum, pointwise) ≈ _ipo_two_plate_density(u, t, y) rtol = 1e-12
        end
        prior(u) = logpdf(Normal(0.0, 1.0), u[1]) + logpdf(Exponential(1.0), exp(u[2])) + u[2]
        for retain in ((), (:pointwise,))
            sampler = prepare_sampler(built, bound, [0.1, 0.1];
                backend = _IPO_REVERSE, retain)
            for u in ([0.1, 0.1], [-0.4, 0.3])
                g = zeros(2)
                value, _ = sampler_value_and_gradient!(sampler, g, u)
                @test value ≈ prior(u) + _ipo_two_plate_density(u, t, y) rtol = 1e-12
                @test g ≈ [-u[1], 1 - exp(u[2])] .+ _ipo_two_plate_gradient(u, t, y) rtol = 1e-10
            end
        end
    end
end

@testset "reductions and length of an output that reads no per-index value" begin
    model = @rkppl begin
        log_a ~ Normal(0.0, 1.0)
        @plate for i = eachindex(path)
            decay_cells[i] = exp(log_a)
        end
        total = sum(decay_cells)
        n = length(decay_cells)
        z ~ Normal(total, 1.0)
    end
    bound = model(; path = [1, 2, 3]) | (; z = 3.0)
    built = build_kernel(bound)
    for a in (0.1, -0.7)
        @test _ipo_value(built, bound, :n, [a]) == 3
        @test _ipo_value(built, bound, :total, [a]) ≈ 3exp(a) rtol = 1e-14
        @test _ipo_query(built, bound, :likelihood, [a]) ≈
            logpdf(Normal(3exp(a), 1.0), 3.0) rtol = 1e-12
    end
    sampler = prepare_sampler(built, bound, [0.1]; backend = _IPO_REVERSE)
    for a in (0.1, -0.7)
        g = zeros(1)
        sampler_value_and_gradient!(sampler, g, [a])
        @test g[1] ≈ -a + (3.0 - 3exp(a)) * 3exp(a) rtol = 1e-10
    end
end

@testset "invariant outputs beside per-index outputs, locals, literals and latents" begin
    path = [1, 2, 3]
    model = @rkppl begin
        log_a ~ Normal(0.0, 1.0)
        @plate for i = eachindex(path)
            per_cell[i] = 1.0 * path[i]
            invariant[i] = exp(log_a)
            k = 2 * exp(log_a)
            from_local[i] = k
            literal[i] = 0.5
            w[i] ~ Normal(0.0, 1.0)
            r[i] = i * 1.0
        end
        z ~ Normal(sum(invariant) + sum(from_local) + sum(literal) + sum(w) + sum(r), 1.0)
    end
    bound = model(; path) | (; z = 3.0)
    built = build_kernel(bound)
    u = [0.1, 0.2, -0.3, 0.4]
    a = exp(u[1])
    @test _ipo_value(built, bound, :per_cell, u) == [1.0, 2.0, 3.0]
    @test _ipo_value(built, bound, :invariant, u) == fill(a, 3)
    @test _ipo_value(built, bound, :from_local, u) == fill(2a, 3)
    @test _ipo_value(built, bound, :literal, u) == fill(0.5, 3)
    @test _ipo_value(built, bound, :r, u) == [1.0, 2.0, 3.0]
    total = 3a + 6a + 1.5 + sum(u[2:4]) + 6.0
    @test _ipo_query(built, bound, :likelihood, u) ≈ logpdf(Normal(total, 1.0), 3.0) rtol = 1e-12
end
