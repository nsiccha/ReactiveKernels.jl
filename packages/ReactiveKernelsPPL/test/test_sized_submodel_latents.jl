module SizedSubmodelLatents
using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

@rkppl weights(modes, sd) = begin
    shift ~ Normal(0, sd)
    scale ~ LogNormal(0, 1)
    beta_raw[1:prod(modes)] .~ Normal.(0, 1)
    return shift + scale * sum(beta_raw)
end
@rkppl component(modes, sd) = begin
    basis = copy(modes)
    term ~ weights(basis, sd)
    return term
end
const MODEL = quote
    @plate for i in eachindex(modes)
        channel[i] ~ component(modes[i], sd[i])
        y[i] ~ Normal(channel[i], 1)
    end
end

@testset "a sized prior keeps its submodel scope and uses each cell's extent" begin
    plan = lower_rkppl(MODEL, (:modes, :sd, :y); conditioned = (:y,), mod = @__MODULE__)
    structures = []
    for modes in ([[3], [2, 2]], [[0], [2], [4, 2], [1]], Vector{Int}[])
        n = length(modes)
        data = (; modes, sd = 1 .+ (1:n) .* 0.25, y = (1:n) .* -0.3)
        bound = bind_data(plan, data)
        built = build_kernel(bound)
        sizes = prod.(modes)
        shift = (1:n) .* 0.15
        beta = [collect(1:k) .* (-0.1 * i) for (i, k) in enumerate(sizes)]
        logscale = (1:n) .* -0.2
        scale = exp.(logscale)
        names = coordinate_names(built.layout)
        @test Set(names) == Set(vcat([Symbol("channel.term.shift.", i) for i in 1:n],
            [Symbol("channel.term.scale.", i) for i in 1:n],
            [Symbol("channel.term.beta_raw.", i, ".", j) for i in 1:n for j in 1:sizes[i]]))
        @test built.layout.total == 2n + sum(sizes)
        u = unconstrain(built.layout, (; channel = (; term = (; shift, scale, beta_raw = beta))))
        restored = constrain(built.layout, u)
        @test restored.channel.term.shift == shift
        @test restored.channel.term.scale ≈ scale
        @test restored.channel.term.beta_raw == beta
        oracle = sum(eachindex(modes); init = 0.0) do i
            logpdf(Normal(0, data.sd[i]), shift[i]) + logpdf(Normal(), logscale[i]) +
                sum(x -> logpdf(Normal(), x), beta[i]; init = 0.0) +
                logpdf(Normal(shift[i] + scale[i] * sum(beta[i]), 1), data.y[i])
        end
        q = prepare_query(built, bound, :sampler)
        @test Base.invokelatest(q, u) ≈ oracle
        isempty(u) && continue
        residual = data.y .- shift .- scale .* sum.(beta)
        grad = Dict(Symbol("channel.term.shift.", i) => residual[i] - shift[i] / data.sd[i]^2
            for i in 1:n)
        for i in 1:n
            grad[Symbol("channel.term.scale.", i)] = residual[i] * scale[i] * sum(beta[i]) - logscale[i]
        end
        for i in 1:n, j in 1:sizes[i]
            grad[Symbol("channel.term.beta_raw.", i, ".", j)] = scale[i] * residual[i] - beta[i][j]
        end
        sampler = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ oracle
        @test gradient ≈ [grad[name] for name in names]
        push!(structures, [(r.kind, r.depth) for r in recipe_inventory(built.spec)])
    end
    @test structures[1] == structures[2]
    @test any(r -> first(r) == :plate, first(structures))
end

@testset "the same sized prior at model scope" begin
    data = (; modes = [2, 2], sd = 1.5, y = 0.25)
    model = quote
        channel ~ weights(modes, sd)
        y ~ Normal(channel, 1)
    end
    bound = bind_data(lower_rkppl(model, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(bound)
    @test built.layout.total == 6
    @test length(constrain(built.layout, zeros(6)).channel.beta_raw) == 4
end

@rkppl matrix_weights(n, center) = begin
    w[1:n, 1:2] .~ Normal.(center, 1)
    return w
end

@testset "the destination axes govern broadcasting in a sized cell prior" begin
    model = quote
        @plate for i in eachindex(n)
            group[i] ~ matrix_weights(n[i], center[i])
        end
    end
    function build(n, center)
        d = (; n, center)
        build_kernel(bind_data(lower_rkppl(model, d; mod = @__MODULE__), d))
    end
    built = build([2, 1], [[-0.3, 0.1], [0.5]])
    u = collect(0.1:0.1:0.6)
    @test constrain(built.layout, u).group.w == [reshape(u[1:4], 2, 2), reshape(u[5:6], 1, 2)]
    @test_throws "could not be broadcast" build([2], [[0.1, 0.2, 0.3]])
    @test_throws "nonnegative integer" build([-1], [[0.1]])
end
end
