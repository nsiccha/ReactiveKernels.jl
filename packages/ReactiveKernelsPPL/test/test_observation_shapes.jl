using Test, Distributions, ReactiveKernels, ReactiveKernelsPPL
import DifferentiationInterface, Enzyme

const _OS_BACKEND = DifferentiationInterface.AutoEnzyme(; mode = Enzyme.Reverse)
const _OS_U = [0.3, -0.2]

function _os_model()
    @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 0.7)
    end
end

function _os_oracle(x, y, u)
    mu = u[1] .+ u[2] .* x
    residual = y .- mu
    value = sum(logpdf.(Normal.(mu, 0.7), y)) +
        sum(logpdf.(Normal(), u))
    gradient = [sum(residual) / 0.7^2 - u[1],
        sum(residual .* x) / 0.7^2 - u[2]]
    return value, gradient
end

function _os_check(bound, expected, gradient, u = _OS_U)
    built = build_kernel(bound)
    post = prepare_query(built, bound, :sampler)
    @test Base.invokelatest(post, u) ≈ expected rtol = 1e-12
    sampler = prepare_sampler(built, bound, u; backend = _OS_BACKEND)
    value, g = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ expected rtol = 1e-12
    @test g ≈ gradient rtol = 1e-10 atol = 1e-12
    return built, post, sampler
end

@testset "observation domains follow Julia broadcasting" begin
    model = _os_model()
    cases = (
        (0.5, [0.2, -0.1, 0.4]),
        ([0.5], [0.2, -0.1, 0.4]),
        (fill(0.5, 3), [0.2, -0.1, 0.4]),
        ([0.5], reshape([0.2, -0.1], 1, 2)),
        (0.5, reshape([0.2, -0.1], 1, 2)),
        (reshape([0.5, -0.3], 2, 1), reshape(collect(0.1:0.1:0.6), 2, 3)),
        (reshape([0.5, -0.3], 1, 2, 1), reshape(collect(0.1:0.1:0.8), 2, 2, 2)),
        ([0.5, -0.3], [0.2]),
        (fill(0.5), reshape([0.2, -0.1], 1, 2)),
        ([0.5], Float64[]),
        (ones(1, 1), zeros(0, 2)),
        (ones(1, 2, 1), zeros(0, 2, 2)),
    )
    for (x, y) in cases
        before = deepcopy((x, y))
        bound = model(; x) | (; y)
        @test bound.n_obs == length(y .- (_OS_U[1] .+ _OS_U[2] .* x))
        @test size(bound.columns[:y]) == size(y)
        _os_check(bound, _os_oracle(x, y, _OS_U)...)
        @test (x, y) == before
    end
    # Refused: Julia broadcasting cannot combine non-singleton dimensions
    # of lengths 2 and 3 (language principle P3).
    @test_throws ContractValidationError (model(; x = [0.5, -0.3]) |
        (; y = [0.2, -0.1, 0.4]))
    @test_throws ContractValidationError (model(; x = zeros(2, 3)) |
        (; y = zeros(2, 4)))
    indexed = @rkppl begin
        a ~ Normal(0, 1)
        @plate for i in eachindex(y)
            y[i] ~ Normal(a + x[i], 0.7)
        end
    end
    # Evaluation preserves ordinary Julia bounds checks for indexed reads.
    invalid = indexed(; x = [0.5]) | (; y = zeros(3))
    @test_throws BoundsError begin
        built = build_kernel(invalid)
        post = prepare_query(built, invalid, :sampler)
        Base.invokelatest(post, [0.3])
    end
    # The authored loop linearly indexes the matrix. Broadcasting these
    # differently shaped operands would instead create four observations.
    x = reshape([0.5, -0.3], 1, 2)
    y = [0.2, -0.1]
    mu = 0.3 .+ vec(x)
    linear = indexed(; x) | (; y)
    @test linear.n_obs == 2
    _os_check(linear, logpdf(Normal(), 0.3) +
        sum(logpdf.(Normal.(mu, 0.7), y)),
        [sum(y .- mu) / 0.7^2 - 0.3], [0.3])
    empty_loop = indexed(; x = ones(3)) | (; y = Float64[])
    _os_check(empty_loop, logpdf(Normal(), 0.3), [-0.3], [0.3])

    bare = @rkppl begin
        a ~ Normal(0, 1)
        y .~ Normal.(a, 0.7)
    end
    for y in (reshape([0.2, -0.1], 1, 2), fill(0.2, 2, 2, 2))
        bound = bare() | (; y)
        _os_check(bound, logpdf(Normal(), 0.3) +
            sum(logpdf.(Normal(0.3, 0.7), y)),
            [sum(y .- 0.3) / 0.7^2 - 0.3], [0.3])
    end

    offsets = @rkppl begin
        a ~ Normal(0, 1)
        z[1:2] .~ Normal.(0, 1)
        shift = z[2]
        mu = a .+ shift .+ x
        y .~ Normal.(mu, 0.7)
    end
    x, y = ones(1, 2, 1), zeros(2, 2, 2)
    bound = offsets(; x) | (; y)
    u = [0.3, 0.1, -0.2]
    mu = u[1] .+ u[3] .+ x
    residual = y .- mu
    _os_check(bound, sum(logpdf.(Normal(), u)) +
        sum(logpdf.(Normal.(mu, 0.7), y)),
        [sum(residual) / 0.7^2 - u[1], -u[2],
            sum(residual) / 0.7^2 - u[3]], u)
end

@testset "singleton operands keep response domains independent" begin
    model = @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y1 .~ Normal.(mu, 0.7)
        y2 .~ Normal.(mu, 0.7)
    end
    for (x, y1, y2) in (
        ([0.5], [0.2, -0.1, 0.4], [0.2, 0.3]),
        (ones(1, 1, 1), zeros(2, 2, 2), zeros(3, 2, 2)),
        (reshape([0.5, -0.3], 2, 1), zeros(2, 3), zeros(2, 4)),
        ([0.5], Float64[], zeros(3)),
    )
        original = deepcopy((x, y1, y2))
        bound = model(; x) | (; y1, y2)
        n1 = length(y1 .- (_OS_U[1] .+ _OS_U[2] .* x))
        n2 = length(y2 .- (_OS_U[1] .+ _OS_U[2] .* x))
        @test bound.n_obs == n1 + n2
        v1, g1 = _os_oracle(x, y1, _OS_U)
        v2, g2 = _os_oracle(x, y2, _OS_U)
        _os_check(bound, v1 + v2 - sum(logpdf.(Normal(), _OS_U)),
            g1 + g2 + _OS_U)
        @test (x, y1, y2) == original
    end
    # Refused: the shared vector's length 3 cannot broadcast with y2's
    # length 2 (P3). A singleton is the only vector that can serve both.
    @test_throws ContractValidationError (model(; x = ones(3)) |
        (; y1 = zeros(3), y2 = zeros(2)))
end

@testset "log-link responses stretch a singleton response" begin
    # Bernoulli-logit and Poisson-log fuse a full response into whole-vector
    # sums (`dot(y, eta)`); a number or one-entry response stretches over the
    # location's rows, as Julia broadcasting and the Normal response do.
    x = [-1.2, -0.4, 0.3, 0.9, 1.6]
    u = [0.1, 0.2]
    eta = u[1] .+ u[2] .* x
    families = (
        (@rkppl(begin
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            y .~ Bernoulli.(logistic.(mu))
        end), y -> logpdf.(Bernoulli.(1 ./ (1 .+ exp.(-eta))), y),
            y -> y .- 1 ./ (1 .+ exp.(-eta)),
            (true, 0, [true], [1], fill(false), [true, false, true, true, false])),
        (@rkppl(begin
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            y .~ Poisson.(exp.(mu))
        end), y -> logpdf.(Poisson.(exp.(eta)), y),
            y -> y .- exp.(eta),
            (3, [2], fill(0), [0, 2, 1, 4, 3])),
    )
    for (model, pointwise, score, ys) in families, y in ys
        before = deepcopy((x, y))
        bound = model(; x) | (; y)
        residual = score(y)
        expected = sum(pointwise(y)) + sum(logpdf.(Normal(), u))
        gradient = [sum(residual) - u[1], sum(residual .* x) - u[2]]
        _os_check(bound, expected, gradient, u)
        @test (x, y) == before
    end
end

@testset "elementwise family operands retain observation shapes" begin
    model = @rkppl begin
        p ~ Beta(2, 3)
        k .~ Binomial.(n, p)
    end
    k = reshape([0, 1, 2, 3], 2, 2)
    n = [5]
    bound = model(; n) | (; k)
    built = build_kernel(bound)
    u = [0.2]
    p = constrain(built.layout, u).p
    post = prepare_query(built, bound, :likelihood)
    @test Base.invokelatest(post, u) ≈ sum(logpdf.(Binomial.(n, p), k))
    bernoulli = @rkppl begin
        p ~ Beta(2, 3)
        y .~ Bernoulli.(p)
    end
    y = reshape([0, 1, 1, 0], 2, 2)
    b = bernoulli() | (; y)
    k = build_kernel(b)
    @test Base.invokelatest(prepare_query(k, b, :likelihood), u) ≈
        sum(logpdf.(Bernoulli(p), y))
end

function _os_design_fixture(kind, n, weighted_case; value_aware = false)
    x = [0.1 + 0.07i for i in 1:n]
    y = [0.2 - 0.03cos(i) for i in 1:n]
    w = [1 + mod(i, 3) for i in 1:n]
    data = kind === :supplied ? (; X = hcat(ones(n), x), y, w) : (; x, y, w)
    source = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
    end
    kind === :supplied || push!(source.args, :(X = hcat(ones(length(x)), x)))
    push!(source.args, :(beta = [a, b]))
    location = kind === :inline ? :(X * [a, b]) :
        kind === :broadcast_coefficients ? :(X * (beta .* 1.0)) : :(X * beta)
    push!(source.args, :(mu = $location))
    observation = weighted_case ? :(y .~ weighted.(Normal.(mu, 0.7), w)) :
        :(y .~ Normal.(mu, 0.7))
    push!(source.args, observation)
    plan = lower_rkppl(source, value_aware ? data : keys(data); conditioned = (:y,))
    return bind_data(plan, data), x, y, weighted_case ? w : ones(n)
end

function _os_design_oracle(x, y, w)
    mu = _OS_U[1] .+ _OS_U[2] .* x
    residual = w .* (y .- mu)
    value = sum(w .* logpdf.(Normal.(mu, 0.7), y)) +
        sum(logpdf.(Normal(), _OS_U))
    gradient = [sum(residual) / 0.7^2 - _OS_U[1],
        sum(residual .* x) / 0.7^2 - _OS_U[2]]
    return value, gradient
end

@testset "matrix-vector products count their consuming observation domain" begin
    for kind in (:prepared, :supplied, :inline, :broadcast_coefficients),
            weighted_case in (false, true), n in (0, 1, 6, 18)
        bound, x, y, w = _os_design_fixture(kind, n, weighted_case)
        value_bound, _, _, _ = _os_design_fixture(kind, n, weighted_case;
            value_aware = true)
        @test bound.n_obs == value_bound.n_obs == n
        @test size(bound.columns[:X]) == (n, 2)
        original = deepcopy(bound.columns)
        _os_check(bound, _os_design_oracle(x, y, w)...)
        @test bound.columns == original
    end
    # A scalar product retains the whole matrix result. Its columns really
    # do broadcast with y, so this domain has twice as many cells.
    for n in (0, 1, 6, 18)
        source = quote
            X = hcat(ones(length(x)), x)
            mu = X * 2.0
            y .~ Normal.(mu, 0.7)
        end
        data = (; x = collect(1.0:n), y = zeros(n))
        bound = bind_data(lower_rkppl(source, keys(data); conditioned = (:y,)), data)
        @test bound.n_obs == 2n
        built = build_kernel(bound)
        likelihood = prepare_query(built, bound, :likelihood)
        @test Base.invokelatest(likelihood, Float64[]) ≈
            sum(logpdf.(Normal.(2.0 .* hcat(ones(n), data.x), 0.7), data.y))
    end
end
