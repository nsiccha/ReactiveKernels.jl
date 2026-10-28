using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, Random,
    ReactiveKernels, ReactiveKernelsPPL, Test

# A multivariate normal in a `@plate` cell observes that index's vector jointly,
# as the Julia loop it states does: `y[i] ~ MvNormal(mu, Sigma)` with `y[i]` one
# vector per index, its mean and covariance computed in the cell. Each index
# adds one joint density; an empty vector adds zero, as Distributions.jl gives.

_cmv_fd(f, u; h = cbrt(eps(Float64))) = [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end for i in eachindex(u)]

_cmv_q(built, bound, preset, u) = Base.invokelatest(prepare_query(built, bound, preset), u)

_cmv_structure(program) = [(e.kind, e.depth) for e in recipe_inventory(program)
    if e.kind !== :ordinary]

# Units nested in districts, one cell per sector: the sector's total is
# observed first, then its observed district totals jointly given that total.
# The covariance reads the observed vector itself, as a measurement-error term.
const _CMV_AGGREGATE = quote
    a ~ Normal(0, 1)
    lg ~ Normal(0, 1)
    sr ~ Exponential(1)
    @plate for s in eachindex(y)
        M = a .+ 0.5 .* x[s]
        V = exp(lg) .* (1.0 .+ x[s] .^ 2)
        sll = sum(V) + 0.1
        tot[s] ~ Normal(sum(M), sqrt(sll))
        cmean = M .+ V .* (tot[s] - sum(M)) ./ sll
        y[s] ~ MvNormal(cmean, Diagonal(V .+ (sr .* y[s]) .^ 2) .- V * V' ./ sll)
    end
end

function _cmv_aggregate_oracle(u, x, y, tot)
    a, lg, lsr = u
    sr = exp(lsr)
    lp = logpdf(Normal(0, 1), a) + logpdf(Normal(0, 1), lg) +
        logpdf(Exponential(1), sr) + lsr
    for s in eachindex(y)
        M = a .+ 0.5 .* x[s]
        V = exp(lg) .* (1.0 .+ x[s] .^ 2)
        sll = sum(V) + 0.1
        lp += logpdf(Normal(sum(M), sqrt(sll)), tot[s])
        cmean = M .+ V .* (tot[s] - sum(M)) ./ sll
        S = Diagonal(V .+ (sr .* y[s]) .^ 2) .- V * V' ./ sll
        lp += logpdf(MvNormal(cmean, Symmetric(Matrix(S))), y[s])
    end
    return lp
end

function _cmv_aggregate_data(sizes)
    rng = Xoshiro(17)
    x = [randn(rng, n) for n in sizes]
    y = [0.3 .* randn(rng, n) for n in sizes]
    tot = randn(rng, length(sizes))
    return (; x, y, tot)
end

@testset "a cell observes one vector jointly with MvNormal" begin
    data = _cmv_aggregate_data([3, 0, 2, 4])
    saved = deepcopy(data)
    bound = bind_data(lower_rkppl(_CMV_AGGREGATE, data; conditioned = (:y, :tot)), data)
    built = build_kernel(bound)
    @test coordinate_names(built.layout) == [:a, :lg, :sr]
    oracle = u -> _cmv_aggregate_oracle(u, data.x, data.y, data.tot)
    sampler = prepare_sampler(built, bound, [0.2, -0.3, -0.5];
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    for u in ([0.2, -0.3, -0.5], [-0.7, 0.4, 0.1], [1.1, -1.2, 0.3])
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ oracle(u) rtol = 1e-12
        @test gradient ≈ _cmv_fd(oracle, u) rtol = 1e-6 atol = 1e-8
        # One joint density per index; summing them gives the likelihood.
        pw = _cmv_q(built, bound, :pointwise, u)
        @test length(pw.y) == 4
        @test iszero(pw.y[2])
        @test sum(pw.y) + sum(pw.tot) ≈ _cmv_q(built, bound, :likelihood, u) rtol = 1e-12
        sr = exp(u[3])
        s = 3
        M = u[1] .+ 0.5 .* data.x[s]
        V = exp(u[2]) .* (1.0 .+ data.x[s] .^ 2)
        sll = sum(V) + 0.1
        S = Diagonal(V .+ (sr .* data.y[s]) .^ 2) .- V * V' ./ sll
        @test pw.y[s] ≈ logpdf(MvNormal(M .+ V .* (data.tot[s] - sum(M)) ./ sll,
            Symmetric(Matrix(S))), data.y[s]) rtol = 1e-12
    end
    @test data == saved
end

@testset "in-cell MvNormal: retained plates as groups grow" begin
    structures = map(([2, 0, 3], [4, 1, 0, 2, 3, 5, 2, 0, 1, 3])) do sizes
        data = _cmv_aggregate_data(sizes)
        bound = bind_data(lower_rkppl(_CMV_AGGREGATE, data; conditioned = (:y, :tot)), data)
        built = build_kernel(bound)
        u = [0.15, -0.25, -0.4]
        value, gradient = sampler_value_and_gradient!(prepare_sampler(built, bound, u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse)), similar(u), u)
        oracle = v -> _cmv_aggregate_oracle(v, data.x, data.y, data.tot)
        @test value ≈ oracle(u) rtol = 1e-12
        @test gradient ≈ _cmv_fd(oracle, u) rtol = 1e-6 atol = 1e-8
        _cmv_structure(built.spec)
    end
    @test structures[1] == structures[2]
    @test !isempty(structures[1])
end

@testset "in-cell MvNormal: shared covariance, named and inline arguments" begin
    y = [[0.4, -0.1], [1.2, 0.3], [-0.5, 0.8]]
    g = [[1.0, 2.0], [0.0, 1.0], [2.0, -1.0]]
    data = (; y, g)
    inline = quote
        a ~ Normal(0, 1)
        ls ~ Normal(0, 1)
        rho ~ Uniform(-0.9, 0.9)
        S = exp(2ls) .* [1.0 rho; rho 1.0]
        @plate for i in eachindex(y)
            y[i] ~ MvNormal(a .* g[i], S)
        end
    end
    named = quote
        a ~ Normal(0, 1)
        ls ~ Normal(0, 1)
        rho ~ Uniform(-0.9, 0.9)
        S = exp(2ls) .* [1.0 rho; rho 1.0]
        @plate for i in eachindex(y)
            m = a .* g[i]
            y[i] ~ MvNormal(m, S)
        end
    end
    # The same law through the covariance's lower Cholesky factor.
    cholesky_form = quote
        a ~ Normal(0, 1)
        ls ~ Normal(0, 1)
        rho ~ Uniform(-0.9, 0.9)
        L = exp(ls) .* [1.0 0.0; rho sqrt(1 - rho^2)]
        @plate for i in eachindex(y)
            y[i] ~ MvNormalCholesky(a .* g[i], L)
        end
    end
    u = [0.3, -0.2, 0.4]
    function oracle(v)
        a, ls = v[1], v[2]
        rho = -0.9 + 1.8 / (1 + exp(-v[3]))
        S = exp(2ls) .* [1.0 rho; rho 1.0]
        logpdf(Normal(), a) + logpdf(Normal(), ls) + logpdf(Uniform(-0.9, 0.9), rho) +
            log(1.8) - log1p(exp(-v[3])) - log1p(exp(v[3])) +
            sum(logpdf(MvNormal(a .* g[i], S), y[i]) for i in eachindex(y))
    end
    values = map((inline, named, cholesky_form)) do ast
        bound = bind_data(lower_rkppl(ast, data; conditioned = (:y,)), data)
        built = build_kernel(bound)
        sampler = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ oracle(u) rtol = 1e-12
        @test gradient ≈ _cmv_fd(oracle, u) rtol = 1e-6 atol = 1e-8
        value
    end
    @test values[1] ≈ values[2] rtol = 1e-12
    @test values[1] ≈ values[3] rtol = 1e-12
end

@testset "in-cell MvNormal: a prediction design leaves the response absent" begin
    ast = quote
        a ~ Normal(0, 1)
        @plate for i in eachindex(x)
            y[i] ~ MvNormal(a .+ x[i], Diagonal(1.0 .+ x[i] .^ 2))
        end
    end
    fit_data = (; x = [[1.0, 2.0], [0.5]], y = [[0.9, 2.2], [0.1]])
    fit = bind_data(lower_rkppl(ast, fit_data; conditioned = (:y,)), fit_data)
    built = build_kernel(fit)
    design = bind_data(fit, (; x = [[0.0, 1.0], [2.0]]))
    @test _cmv_q(built, design, :prior, [0.3]) ≈ logpdf(Normal(), 0.3)
    @test_throws ContractValidationError _cmv_q(built, design, :likelihood, [0.3])
end

@testset "in-cell MvNormal: argument and observation errors" begin
    # A mean whose length is not the observed vector's raises as Distributions.jl does.
    ast = quote
        a ~ Normal(0, 1)
        @plate for i in eachindex(y)
            y[i] ~ MvNormal(fill(a, 2), Matrix(1.0I, 2, 2))
        end
    end
    data = (; y = [[0.1, 0.2], [0.3, 0.4, 0.5]])
    bound = bind_data(lower_rkppl(ast, data; conditioned = (:y,)), data)
    built = build_kernel(bound)
    err = try
        _cmv_q(built, bound, :likelihood, [0.1]); nothing
    catch e
        e
    end
    @test err isa DimensionMismatch && occursin("length 3", sprint(showerror, err))
    # A covariance that is not positive definite has no density.
    ast = quote
        a ~ Normal(0, 1)
        @plate for i in eachindex(y)
            y[i] ~ MvNormal(fill(a, 2), [1.0 2.0; 2.0 1.0])
        end
    end
    data = (; y = [[0.1, 0.2]])
    bound = bind_data(lower_rkppl(ast, data; conditioned = (:y,)), data)
    built = build_kernel(bound)
    @test_throws ArgumentError _cmv_q(built, bound, :likelihood, [0.1])
    # A number per index is not a vector to observe jointly.
    ast = quote
        a ~ Normal(0, 1)
        @plate for i in eachindex(y)
            y[i] ~ MvNormal([a], ones(1, 1))
        end
    end
    data = (; y = [0.1, 0.2])
    bound = bind_data(lower_rkppl(ast, data; conditioned = (:y,)), data)
    built = build_kernel(bound)
    @test_throws ArgumentError _cmv_q(built, bound, :likelihood, [0.1])
end

@testset "in-cell MvNormal: per-entry spellings are refused" begin
    data = (; y = [[0.1, 0.2], [0.3]], x = [[1.0, 2.0], [0.5]])
    # refused: a dotted cell broadcasts `MvNormal` over the cell's entries,
    # which has no meaning for a law of whole vectors (Julia's own broadcast
    # of `MvNormal.(scalars)` fails the same way).
    dotted = quote
        a ~ Normal(0, 1)
        @plate for i in eachindex(y)
            y[i] .~ MvNormal.(a .+ x[i], Ref(Diagonal(ones(length(x[i])))))
        end
    end
    err = try
        lower_rkppl(dotted, data; conditioned = (:y,)); nothing
    catch e
        e
    end
    @test err isa ReactiveKernelsPPL.SurfaceLoweringError
    @test occursin("undotted", sprint(showerror, err))
    # refused: the same per-entry broadcast over a whole column.
    flat = quote
        a ~ Normal(0, 1)
        yf .~ MvNormal.(a, 1.0)
    end
    err = try
        lower_rkppl(flat, (; yf = [0.1, 0.2]); conditioned = (:yf,)); nothing
    catch e
        e
    end
    @test err isa ReactiveKernelsPPL.SurfaceLoweringError
    @test occursin("@plate", sprint(showerror, err))
end

@testset "in-cell MvNormal: predictive draws (not built yet)" begin
    ast = quote
        a ~ Normal(0, 1)
        @plate for i in eachindex(y)
            y[i] ~ MvNormal(a .+ x[i], Diagonal(1.0 .+ x[i] .^ 2))
        end
    end
    data = (; x = [[1.0, 2.0], [0.5]], y = [[0.9, 2.2], [0.1]])
    bound = bind_data(lower_rkppl(ast, data; conditioned = (:y,)), data)
    built = build_kernel(bound)
    # Predictive draws hold one law per response entry; a joint law of a whole
    # vector per index is a capability gap tracked on the ReactiveKernels queue.
    @test_broken try
        draws = prepare_predictive(built, bound)([0.3], Xoshiro(1))
        length.(draws.y) == [2, 1]
    catch
        false
    end
end
