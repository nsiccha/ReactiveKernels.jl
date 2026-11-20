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


@testset "row scales in a bound Normal regression" begin
    scaled = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); sigma ~ Exponential(1.0)
        y .~ Normal.(a .+ b .* x, sigma .* w)
    end
    by_precision = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); sigma ~ Exponential(1.0)
        y .~ Normal.(a .+ b .* x, sigma ./ sqrt.(w))
    end
    by_product = @rkppl begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); sigma ~ Exponential(1.0)
        y .~ Normal.(a .+ b .* x .* z, sigma .* w)
    end
    u = [0.25, -0.45, -0.2]
    a, b, logscale = u
    sigma = exp(logscale)
    for model in (scaled, by_precision, by_product), n in (0, 48)
        x = [cos(i) for i in 1:n]
        y = [sin(i) for i in 1:n]
        w = [isodd(i) ? 2.0 : 0.5 for i in 1:n]
        z = [cos(2i) for i in 1:n]
        inputs = model === by_product ? (; x, z, w) : (; x, w)
        fit = model(; inputs...) | (; y)
        built = build_kernel(fit)
        k = prepare_query(built, fit, :sampler; exemplars = (u,))
        # A binding without rows has no usable fit and binds none.
        stats = filter(s -> s isa ReactiveKernels._LSWeightedStatistics, k.constants)
        @test length(stats) == (n > 0)
        @test all(s -> s.fit.usable, stats)
        ref = -(a^2 + b^2)/2 - log(2pi) - sigma + logscale
        grad = [-a, -b, 1 - sigma]
        for i in eachindex(y)
            sd = model === by_precision ? sigma / sqrt(w[i]) : sigma * w[i]
            feature = model === by_product ? x[i]*z[i] : x[i]
            residual = y[i] - a - b*feature
            ref += -0.5*log(2pi) - log(sd) - 0.5*(residual/sd)^2
            grad[1] += residual/sd^2
            grad[2] += feature*residual/sd^2
            grad[3] += (residual/sd)^2 - 1
        end
        q = prepare_sampler(built, fit, u; backend)
        actual = similar(u)
        value, _ = sampler_value_and_gradient!(q, actual, u)
        @test value ≈ ref
        @test actual ≈ grad
    end
    # A common location and scale give a design and a weight of one.
    iid = @rkppl begin
        m ~ Normal(0, 1); sigma ~ Exponential(1.0)
        y .~ Normal.(m, sigma)
    end
    # Enough rows that the statistics cost less than the vectorized loop.
    y = [sin(i) for i in 1:480]
    fit = iid() | (; y)
    built = build_kernel(fit)
    point = [a, logscale]
    k = prepare_query(built, fit, :sampler; exemplars = (point,))
    @test only(s.fit.usable for s in k.constants if s isa ReactiveKernels._LSWeightedStatistics)
    q = prepare_sampler(built, fit, point; backend)
    r = y .- a
    gradient = similar(point)
    sampler_value_and_gradient!(q, gradient, point)
    @test gradient ≈ [sum(r)/sigma^2 - a, sum(abs2, r)/sigma^2 - length(r) + 1 - sigma]
end

@testset "a binding runs its statistics only where they cost less" begin
    fits(k) = filter(s -> s isa ReactiveKernels._LSWeightedStatistics, k.constants)
    shape(k) = replace(string(code_expr(k)), r"#\d+" => "#")
    # Coefficients declared as numbers, and read from a declared vector.
    numbers = @rkppl begin
        a ~ Flat(); b ~ Flat(); sigma ~ restricted(Flat(), 0, Inf)
        y .~ Normal.(a .+ b .* x, sigma)
    end
    vector = @rkppl begin
        beta[1:2] .~ Flat()
        sigma ~ restricted(Flat(), 0, Inf)
        mu = beta[1] .+ beta[2] .* x
        y .~ Normal.(mu, sigma)
    end
    a, b, sigma = 0.3, -0.6, 1.2
    # (rows, distinct rows, statistics). 240 rows of two binary columns are
    # four distinct rows: the grouped loop costs less than any statistics.
    cases = ((240, 4, false), (60, 60, false), (2000, 2000, true), (2000, 500, true))
    for model in (numbers, vector), (n, distinct, statistics) in cases
        base = distinct == 4 ? [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] :
            [(cos(i), sin(3i)) for i in 1:distinct]
        rows = [base[mod1(7i, distinct)] for i in 1:n]
        x, y = first.(rows), last.(rows)
        fit = model(; x) | (; y)
        built = build_kernel(fit)
        named = model === numbers ? (; a, b, sigma) : (; beta = [a, b], sigma)
        u = collect(Float64, unconstrain(built.layout, named))
        # Where each coordinate sits in the packed point.
        at(named) = findfirst(!iszero, collect(Float64, unconstrain(built.layout, named)) .- u)
        ia = at(model === numbers ? (; a = a + 1, b, sigma) : (; beta = [a + 1, b], sigma))
        ib = at(model === numbers ? (; a, b = b + 1, sigma) : (; beta = [a, b + 1], sigma))
        is = at(model === numbers ? (; a, b, sigma = 2sigma) : (; beta = [a, b], sigma = 2sigma))
        k = prepare_query(built, fit, :sampler; exemplars = (u,))
        literal = with_bound_statistics(false) do
            prepare_query(build_kernel(fit), fit, :sampler; exemplars = (u,))
        end
        @test length(fits(k)) == statistics
        @test all(s -> s.fit.usable, fits(k))
        # Rows that repeat are grouped in either program: in the plan without
        # the rewrite, and in the authored arm beside the statistics.
        grouped(q) = [length(c) for c in q.constants if c isa Vector{Int}]
        @test grouped(literal) == (distinct < n ? [distinct] : Int[])
        @test grouped(k) == grouped(literal)
        # Where the rows cost less, the program is the one without the rewrite.
        statistics || @test shape(k) == shape(literal)
        residual = y .- a .- b .* x
        reference = sum(-log(2pi) / 2 .- log(sigma) .- (residual ./ sigma) .^ 2 ./ 2) + log(sigma)
        expected = zeros(3)
        expected[ia] = sum(residual) / sigma^2
        expected[ib] = sum(residual .* x) / sigma^2
        expected[is] = sum((residual ./ sigma) .^ 2 .- 1) + 1
        q = prepare_sampler(built, fit, u; backend)
        gradient = similar(u)
        value, _ = sampler_value_and_gradient!(q, gradient, u)
        @test value ≈ reference rtol = 1e-12
        @test q(u) ≈ reference rtol = 1e-12
        @test gradient ≈ expected rtol = 1e-10
        @test gradient_bytes(q, u) == 0
    end
end

end # module
