module BoundStatisticsQueryTests
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, DifferentiationInterface, MutatingFunctions, Test
const backend = AutoEnzyme(; mode = Enzyme.Reverse)

@testset "bound-statistics query types" begin
    @testset "PPL threshold and ownership" begin
        source = @rkppl begin
            z[1:z_length] .~ LogNormal.(mu, sd)
            y .~ Normal.(z[idx], noise)
        end
        for schedule in (:flat, :runs), reuse in (false, true)
            data(n, shift) = begin
                idx = schedule === :flat ? repeat(1:3, n) : repeat(1:3, inner = 8n)
                (; y = fill(0.4 + shift, length(idx)), idx, z_length = 3,
                   mu = 0.1 + shift, sd = 0.5 + shift, noise = 0.2 + shift)
            end
            A = data(1, 0.0)
            fit = source(; A.idx, A.z_length, A.mu, A.sd, A.noise) | (; A.y)
            built = build_kernel(fit)
            u = [0.1, -0.2, 0.3]
            bindings = (A, data(1, 0.2), data(2, 0.1), data(20, 0.1), A)
            queries = map(b -> prepare_sampler(built, bind_data(fit, b), u;
                backend, nonallocating = reuse), bindings)
            @test allequal(typeof.(queries))
            for _ in 1:2, (i, (b, q)) in enumerate(zip(bindings, queries))
                gradient = zeros(3)
                value, _ = sampler_value_and_gradient!(q, gradient, u)
                residual = b.y - exp.(u[b.idx])
                reference = sum(-((u .- b.mu) ./ b.sd).^2 ./ 2 .- log(b.sd) .- log(2pi)/2) +
                    sum(-residual.^2 ./ (2b.noise^2) .- log(b.noise) .- log(2pi)/2)
                refgradient = -(u .- b.mu) ./ b.sd^2
                for (j, idx) in enumerate(b.idx)
                    refgradient[idx] += residual[j] * exp(u[idx]) / b.noise^2
                end
                @test value ≈ reference
                @test q(u) ≈ reference
                @test gradient ≈ refgradient
            end

        end
    end
end
function gradient_bytes(q, u)
    g = similar(u)
    for _ in 1:4
        sampler_value_and_gradient!(q, g, u)
    end
    @allocated sampler_value_and_gradient!(q, g, u)
end

@testset "reused storage survives the runtime branch" begin
    source = @rkppl begin
        z[1:z_length] .~ LogNormal.(mu, sd)
        y .~ Normal.(z[idx], noise)
    end
    u = [0.1, -0.2, 0.3]
    for n in (1, 2, 20)
        idx = repeat(1:3, n)
        data = (; y = fill(0.4, length(idx)), idx, z_length = 3, mu = 0.1, sd = 0.5, noise = 0.2)
        fit = source(; data.idx, data.z_length, data.mu, data.sd, data.noise) | (; data.y)
        queries = map((false, true)) do statistics
            with_bound_statistics(statistics) do
                built = build_kernel(fit)
                prepare_sampler(built, bind_data(fit, data), u; backend, nonallocating = true)
            end
        end
        allocations = map(q -> gradient_bytes(q, u), queries)

        @test allocations[2] <= allocations[1]
    end
end


end # module
