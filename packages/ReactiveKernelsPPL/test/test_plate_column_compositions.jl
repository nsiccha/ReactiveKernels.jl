using DifferentiationInterface, Distributions, Enzyme, Random, ReactiveKernels,
    ReactiveKernelsPPL, Test

# A `@plate` output holding one number per index is read whole at model scope
# like any other vector (`rkppl-use` §13), whatever its cell reads. A cell
# reading a composed predictor (`f = 0.4 .* mu[main] .+ u[m]`, a product over
# the response location `mu`) made the column itself a composition: its
# arguments are the cell's inputs, and the composition check followed them.
# The observation then failed lowering with "predictor c applies
# `_ppl_plate_column` inside a composition" (snag
# `rkppl-plate-colu-99114109`; BRM emits every totals plate this way).

# `n` cells of two rows each, over `2n` levels of `u`.
function _pcc_data(n)
    data = (; z = collect(range(-1, 1; length = 4n)),
        m = repeat(1:2n, inner = 2), main = repeat(1:2:4n, inner = 2),
        rows = [[4k - 3, 4k - 2] for k in 1:n],
        w = [cos(1.3k) for k in 1:n])
    obs = (; y = collect(range(0, 2; length = 4n)), resid = [mod(5k, 8) for k in 1:n],
        v = [sin(2.1k) for k in 1:n])
    return data, obs
end

const _PCC_LOCATION = @rkppl begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    sigma ~ Exponential(1.0)
    su ~ Exponential(1.0)
    u[levels(m)] .~ Normal.(0, su)
    mu = a .+ b .* z
    y .~ Normal.(mu, sigma)
    f = 0.4 .* mu[main] .+ u[m]
    @plate for i in eachindex(rows)
        lam = exp.(f[rows[i]])
        c[i] = sum(lam)
    end
    resid .~ Poisson.(c)
end

# The same program with an uncomposed `f`, which lowered before.
const _PCC_PLAIN = @rkppl begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    sigma ~ Exponential(1.0)
    su ~ Exponential(1.0)
    u[levels(m)] .~ Normal.(0, su)
    mu = a .+ b .* z
    y .~ Normal.(mu, sigma)
    f = 0.4 .+ u[m]
    @plate for i in eachindex(rows)
        lam = exp.(f[rows[i]])
        c[i] = sum(lam)
    end
    resid .~ Poisson.(c)
end

# The column as a scale, and as an operand of a composition.
const _PCC_SCALE = @rkppl begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    sigma ~ Exponential(1.0)
    su ~ Exponential(1.0)
    u[levels(m)] .~ Normal.(0, su)
    mu = a .+ b .* z
    y .~ Normal.(mu, sigma)
    f = 0.4 .* mu[main] .+ u[m]
    @plate for i in eachindex(rows)
        c[i] = sum(exp.(f[rows[i]]))
    end
    v .~ Normal.(0.0, c)
end

const _PCC_OPERAND = @rkppl begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    d ~ Normal(0, 1)
    sigma ~ Exponential(1.0)
    su ~ Exponential(1.0)
    u[levels(m)] .~ Normal.(0, su)
    mu = a .+ b .* z
    y .~ Normal.(mu, sigma)
    f = 0.4 .* mu[main] .+ u[m]
    @plate for i in eachindex(rows)
        c[i] = sum(exp.(f[rows[i]]))
    end
    g = d .* w
    resid .~ Poisson.(c .* exp.(g))
end

function _pcc_cells(nt, data, composed)
    mu = nt.a .+ nt.b .* data.z
    f = composed ? 0.4 .* mu[data.main] .+ nt.u[data.m] : 0.4 .+ nt.u[data.m]
    return mu, [sum(exp.(f[r])) for r in data.rows]
end

_pcc_y(nt, mu, obs) = sum(logpdf.(Normal.(mu, nt.sigma), obs.y))

# Each program's likelihood from the Julia loop's cells.
function _pcc_location(nt, data, obs; composed = true)
    mu, c = _pcc_cells(nt, data, composed)
    return _pcc_y(nt, mu, obs) + sum(logpdf.(Poisson.(c), obs.resid))
end
function _pcc_scale(nt, data, obs)
    mu, c = _pcc_cells(nt, data, true)
    return _pcc_y(nt, mu, obs) + sum(logpdf.(Normal.(0.0, c), obs.v))
end
function _pcc_operand(nt, data, obs)
    mu, c = _pcc_cells(nt, data, true)
    rate = c .* exp.(nt.d .* data.w)
    return _pcc_y(nt, mu, obs) + sum(logpdf.(Poisson.(rate), obs.resid))
end

_pcc_structure(program) = [(e.kind, e.depth) for e in recipe_inventory(program)
    if e.kind !== :ordinary]

# At 3 and 7 cells: the likelihood against its oracle at two points, and the
# native Enzyme Reverse gradient of the sampler's density against central
# differences. The retained plate does not multiply with the cell count.
function _pcc_check(model, oracle)
    structures = map((3, 7)) do n
        data, obs = _pcc_data(n)
        bound = model(; data...) | obs
        built = build_kernel(bound)
        likelihood = prepare_query(built, bound, :likelihood)
        dim = built.layout.total
        sampler = prepare_sampler(built, bound, zeros(dim);
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        rng = Xoshiro(17 + n)
        for _ in 1:2
            u = 0.3 .* randn(rng, dim)
            @test likelihood(u) ≈ oracle(constrain(built.layout, u), data, obs) rtol = 1e-12
            g = zeros(dim)
            value, _ = sampler_value_and_gradient!(sampler, g, u)
            @test value ≈ sampler(u) rtol = 1e-12
            h = 1e-6
            fd = [(sampler(u .+ h .* (1:dim .== k)) -
                   sampler(u .- h .* (1:dim .== k))) / 2h for k in 1:dim]
            @test g ≈ fd rtol = 1e-6
        end
        _pcc_structure(sampler.kernel)
    end
    @test structures[1] == structures[2]
    @test count(e -> first(e) === :plate, structures[1]) >= 1
end

@testset "a plate column reading a composed predictor" begin
    @testset "as a location" _pcc_check(_PCC_LOCATION, _pcc_location)
    @testset "uncomposed control" _pcc_check(_PCC_PLAIN,
        (nt, data, obs) -> _pcc_location(nt, data, obs; composed = false))
    @testset "as a scale" _pcc_check(_PCC_SCALE, _pcc_scale)
    @testset "as a composition operand" _pcc_check(_PCC_OPERAND, _pcc_operand)
end
