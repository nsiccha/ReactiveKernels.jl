module PreparationIdentityTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Test

function model(kind)
    kind === :plate && return @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        @plate for i in eachindex(y)
            mu = a + sin(b * x[i])
            y[i] ~ Normal(mu, scale)
        end
    end
    @rkppl begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        mu = a .+ sin.(b .* x)
        y .~ Normal.(mu, scale)
    end
end

function oracle(u, x, y, scale)
    r = y .- (u[1] .+ sin.(u[2] .* x))
    value = -log(2pi) - log(2) - u[1]^2/2 - u[2]^2/8 +
        sum(-log(2pi)/2 - log(scale) .- r.^2 ./ (2scale^2))
    gradient = [-u[1] + sum(r)/scale^2,
        -u[2]/4 + sum(r .* cos.(u[2].*x) .* x)/scale^2]
    value, gradient
end

function check(q, x, y, scale)
    u = [0.3, -0.2]
    g = similar(u)
    value, expected = oracle(u, x, y, scale)
    actual, _ = sampler_value_and_gradient!(q, g, u)
    @test actual ≈ value
    @test q(u) ≈ value
    @test g ≈ expected
    @test Base.return_types(q.kernel, (Vector{Float64},)) == [Float64]
end

@testset "fresh programs reuse executable and ordinary Reverse types" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for kind in (:plate, :broadcast)
        first_built = first_query = nothing
        for (n, shift) in ((3, 0.0), (3, 0.2), (17, -0.4))
            x = collect(range(-0.8, 1.2; length=n))
            y = collect(range(0.1, 0.7; length=n)) .+ shift
            original = deepcopy((; x, y))
            plan = model(kind)(; x, scale=1.3) | (; y)
            built = build_kernel(plan)
            q = prepare_sampler(built, plan, [0.3, -0.2]; backend,
                native_forward_batch=false)
            check(q, x, y, 1.3)
            if first_built === nothing
                first_built, first_query = built, q
            else
                @test typeof(q.kernel.f) === typeof(first_query.kernel.f)
                @test typeof(q.kernel.ops) === typeof(first_query.kernel.ops)
                @test typeof(q.ad) === typeof(first_query.ad)
                rebound = prepare_sampler(first_built, plan, [0.3, -0.2]; backend,
                    native_forward_batch=false)
                check(rebound, x, y, 1.3)
            end
            @test (; x, y) == original
        end

        # A different expression is still a different program. A changed
        # bound scale, conversely, is the same program over different data.
        x, y = [-0.3, 0.8, 1.2], [0.4, 0.5, 0.7]
        changed_data = model(kind)(; x, scale=0.9) | (; y)
        check(prepare_sampler(first_built, changed_data, [0.3, -0.2]; backend,
            native_forward_batch=false), x, y, 0.9)
        different = @rkppl begin
            a ~ Normal(1, 1)
            b ~ Normal(0, 2)
            mu = a .+ sin.(b .* x)
            y .~ Normal.(mu, scale)
        end
        changed_program = different(; x, scale=1.3) | (; y)
        # Refused because using another program's graph would evaluate the
        # wrong prior; the changed program's own build remains supported.
        @test_throws "needs its own `build_kernel(plan)`" prepare_sampler(
            first_built, changed_program, [0.3, -0.2]; backend, native_forward_batch=false)
        own = build_kernel(changed_program)
        own_query = prepare_sampler(own, changed_program, [0.3, -0.2]; backend,
            native_forward_batch=false)
        expected_value, expected_gradient = oracle([0.3, -0.2], x, y, 1.3)
        gradient = zeros(2)
        actual_value, _ = sampler_value_and_gradient!(own_query, gradient, [0.3, -0.2])
        @test actual_value ≈ expected_value - 0.2
        @test gradient ≈ expected_gradient + [1.0, 0.0]
    end
end
end
