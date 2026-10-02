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
    # Refused: x[2] is out of bounds in the authored Julia loop (P3).
    @test_throws "explicit `@plate` indexing does not stretch" (indexed(;
        x = [0.5]) | (; y = zeros(3)))
    # Linear indexing of differently shaped arrays needs a distinct
    # lowering; silently broadcasting would change this loop's density.
    @test_throws "different operand axes is not built yet" (indexed(;
        x = zeros(1, 2)) | (; y = zeros(2)))

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
