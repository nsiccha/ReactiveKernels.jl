using Distributions
using ReactiveKernels
using ReactiveKernelsPPL
using Test

_bound_extent_nodes(ex) = ex isa Expr ?
    1 + sum(_bound_extent_nodes, ex.args; init = 0) : 0

@testset "declared arrays sized by a bound integer" begin
    ast = quote
        z[1:J] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ z[k]
        y .~ Normal.(mu, 1)
    end
    plan = lower_rkppl(ast, (:J, :k, :y); conditioned = (:y,))
    domain_plan = lower_rkppl(quote
        z[domain] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ z[k]
        y .~ Normal.(mu, 1)
    end, (:domain, :k, :y); conditioned = (:y,))
    sizes = Tuple{Int,Int}[]
    for J in (4, 40)
        k, y = [1, 3], [0.2, -0.3]
        bound = bind_data(plan, Dict{Symbol,Any}(:J => J, :k => k, :y => y))
        built = build_kernel(bound)
        @test bound.n_obs == length(y)
        @test built.layout.total == J + 1
        u = [0.3sin(i) for i in 1:built.layout.total]
        values = constrain(built.layout, u)
        want = sum(logpdf.(Normal(), values.z)) + logpdf(Normal(), values.a) +
            sum(logpdf.(Normal.(values.a .+ values.z[k], 1), y))
        @test _query(built.spec, bound, :posterior, u) ≈ want
        # A supplied 1:J domain is equivalent but adds an input to the model.
        domain = bind_data(domain_plan,
            Dict{Symbol,Any}(:domain => 1:J, :k => k, :y => y))
        domain_built = build_kernel(domain)
        @test domain_built.layout.total == built.layout.total
        @test _query(domain_built.spec, domain, :posterior, u) ≈ want
        push!(sizes, (_bound_extent_nodes(kernel_expr(bound, built.layout)),
            _bound_extent_nodes(code_expr(built.spec))))
        J == 4 && _check_gradient(built.spec, bound, u)
    end
    # The latent dimension changes the layout; it does not replicate work in
    # the RK kernel definition or planned program.
    @test length(unique(sizes)) == 1

    # Observed labels are not the declared support: 2 and 4 are unobserved.
    levels_plan = lower_rkppl(quote
        z[levels(k)] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        mu = a .+ z[k]
        y .~ Normal.(mu, 1)
    end, (:k, :y); conditioned = (:y,))
    levels_bound = bind_data(levels_plan,
        Dict{Symbol,Any}(:k => [1, 3], :y => [0.2, -0.3]))
    @test build_kernel(levels_bound).layout.total == 3
end

@testset "bound array extent expressions and validation" begin
    plan = lower_rkppl(quote
        K = J + 1
        z[1:K] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        y .~ Normal.(a, 1)
    end, (:J, :y); conditioned = (:y,))
    @test build_kernel(bind_data(plan,
        Dict{Symbol,Any}(:J => 3, :y => [0.2]))).layout.total == 5

    two_axes = lower_rkppl(quote
        B[1:(2 * J), 1:J] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        y .~ Normal.(a, 1)
    end, (:J, :y); conditioned = (:y,))
    @test build_kernel(bind_data(two_axes,
        Dict{Symbol,Any}(:J => 2, :y => [0.2]))).layout.total == 9

    level_expr = lower_rkppl(quote
        z[1:(length(levels(g)) + 1)] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        y .~ Normal.(a, 1)
    end, (:g, :y); conditioned = (:y,))
    @test build_kernel(bind_data(level_expr,
        Dict{Symbol,Any}(:g => [1, 2], :y => [0.2]))).layout.total == 4

    direct = lower_rkppl(quote
        z[1:J] .~ Normal.(0, 1)
        a ~ Normal(0, 1)
        y .~ Normal.(a, 1)
    end, (:J, :y); conditioned = (:y,))
    @test build_kernel(bind_data(direct,
        Dict{Symbol,Any}(:J => 0, :y => [0.2]))).layout.total == 1
    for bad in (-1, 1.5, true)
        # A declared extent is an ordinary nonnegative integer size.
        @test_throws ContractValidationError bind_data(direct,
            Dict{Symbol,Any}(:J => bad, :y => [0.2]))
    end
    @test_throws ContractValidationError bind_data(direct,
        Dict{Symbol,Any}(:y => [0.2]))

    live = lower_rkppl(quote
        a ~ Normal(0, 1)
        z[1:a] .~ Normal.(0, 1)
        y .~ Normal.(a, 1)
    end, (:y,); conditioned = (:y,))
    # A sampled value cannot set the structural coordinate pack at bind.
    @test_throws ContractValidationError bind_data(live,
        Dict{Symbol,Any}(:y => [0.2]))
end
