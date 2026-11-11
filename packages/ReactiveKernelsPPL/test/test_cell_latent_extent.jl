module CellLatentExtentTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, DifferentiationInterface, Enzyme, Test
include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

const MODEL = quote
    a ~ Normal(0, 1)
    means = a .+ x
    @plate for i in eachindex(means)
        z[i] .~ Normal.(zeros(n), 1)
    end
    @plate for i in eachindex(y)
        y[i] ~ Normal(means[i] + sum(z[i]), 1)
    end
end

const RAGGED_MODEL = quote
    a ~ Normal(0, 1)
    means = a .+ x
    @plate for i in eachindex(means)
        z[i] .~ Normal.(zeros(n[i]), 1)
    end
    @plate for i in eachindex(y)
        y[i] ~ Normal(means[i] + sum(z[i]), 1)
    end
end

function oracle(u, data)
    lp = logpdf(Normal(), u[1])
    offset = 1
    for i in eachindex(data.x)
        n = data.n isa Integer ? data.n : data.n[i]
        z = u[(offset + 1):(offset + n)]
        offset += n
        lp += sum(logpdf.(Normal(), z))
        lp += logpdf(Normal(u[1] + data.x[i] + sum(z), 1), data.y[i])
    end
    lp
end

function check_bindings(model, cases)
    structure = lower_rkppl(model, (:x, :y, :n); conditioned = (:y,))
    programs = []
    sources = JAXSource[]
    for (rows, n) in cases
        data = (; x = collect(1.0:rows), y = fill(0.5, rows), n)
        bound = bind_data(structure, data)
        built = build_kernel(bound)
        @test built.layout.total == 1 + (n isa Integer ? rows * n : sum(n))
        u = 0.03 .* collect(1:built.layout.total)
        f = prepare_query(built, bound, :sampler)
        @test Base.invokelatest(f, u) ≈ oracle(u, data)
        sampler = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ oracle(u, data)
        h = 1e-5
        expected = [(up = copy(u); down = copy(u); up[i] += h; down[i] -= h;
                     (oracle(up, data) - oracle(down, data)) / (2h)) for i in eachindex(u)]
        @test gradient ≈ expected rtol = 1e-7 atol = 1e-8
        push!(programs, [(r.kind, r.depth) for r in recipe_inventory(built.spec)])
        src = jax_source(built, bound)
        push!(sources, src)
        inputs = only(f for f in first(sources).functions if f.name == "prepare_data").inputs
        @test all(name -> Symbol(name) in keys(data), first.(inputs))
        if rk_jax_available()
            raw = Tuple(getproperty(data, Symbol(name)) for (name, _) in inputs)
            result = rk_jax_evaluate(first(sources), :log_density, (u, raw...);
                                     grad = true, prepare = :arguments)
            @test result.value ≈ oracle(u, data)
            @test result.grad ≈ expected rtol = 1e-7 atol = 1e-8
            @test rk_jax_layout(first(sources), raw).dimension == length(u)
        else
            @test_skip "RK_JAX_PYTHON not set"
        end
    end
    @test programs[1] == programs[2]
    @test sources[1].text == sources[2].text
end

@testset "cell latent extents use the data shape of sampled values" begin
    check_bindings(MODEL, ((2, 3), (5, 7), (2, 0), (0, 3)))
end
@testset "sampled plate extents retain ragged and empty cells" begin
    check_bindings(RAGGED_MODEL, ((3, [2, 0, 4]), (5, [0, 1, 3, 0, 5]), (0, Int[])))
end
end
