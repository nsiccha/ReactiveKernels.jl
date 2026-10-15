using DifferentiationInterface, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

@testset "program identity compares read-only array literals by value" begin
    same = ReactiveKernelsPPL._same_program
    identity = ReactiveKernelsPPL._program_identity
    literal(x) = Expr(:call, :identity, QuoteNode(x))
    mask = Bool[true, false, true]
    a, b = identity(literal(mask)), identity(literal(copy(mask)))
    @test same(a, b)
    @test a.args[2].value !== b.args[2].value
    @test same(a, identity(literal(Bool[true, false, true])))
    @test !same(a, identity(literal(Bool[true, true, false])))
    @test !same(a, identity(literal(reshape(copy(mask), 1, :))))
    @test !same(a, identity(literal(Int.(mask))))
    @test !same(literal(Any[1]), literal(Any[1.0]))
    @test same(literal([copy(mask)]), literal([copy(mask)]))
    @test !same(a, Expr(:call, :reverse, b.args[2]))
    @test !same(a, Expr(:call, :identity, b.args[2], 1))
    @test !same(:(x = 1), :(x = 1.0))
    @test !same(literal(0.0), literal(-0.0))
    @test same(literal(NaN), literal(NaN))
    @test !same(GlobalRef(Base, :identity), GlobalRef(Main, :identity))
    @test same(identity(Expr(:call, :identity, gensym(:arg))),
        identity(Expr(:call, :identity, gensym(:arg))))
    # Quoted names are data, not private bindings to alpha-rename.
    @test !same(identity(literal(gensym(:label))), identity(literal(gensym(:label))))
    @test mask == Bool[true, false, true]

    # An equal quoted mask must not hide the later, actual difference.
    def(mask, value) = Expr(:function, :(f(x)), Expr(:block,
        Expr(:(=), :present, QuoteNode(mask)), Expr(:(=), :result, value)))
    built, other = def(mask, 1.0), def(copy(mask), 2.0)
    @test startswith(ReactiveKernelsPPL._program_difference(built, other), "statement 2:")
end

@testset "missing-observation programs prepare across repeated lowerings" begin
    model = @rkppl begin
        a ~ Normal(0, 1)
        @plate for i in eachindex(y)
            m = a * sqrt(x[i])
            y[i] ~ Normal(m, 1.3)
        end
    end
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    normal_logpdf(y, mu, sigma) = -log(2pi) / 2 - log(sigma) - (y - mu)^2 / (2sigma^2)
    for n in (3, 9)
        x = collect(range(0.2, 1.4; length = n))
        x[2] = -1.0 # A missing cell must skip the invalid sqrt too.
        y = Union{Missing,Float64}[0.1i for i in 1:n]
        y[2] = missing
        original = deepcopy((; x, y))
        plan = model(; x) | (; y)
        built = build_kernel(plan)
        # Same plan, fresh equal binding, and changed values with the same mask.
        for other in (plan, model(; x = copy(x)) | (; y = copy(y)),
                model(; x = copy(x)) | (; y = [ismissing(v) ? missing : v + 0.2 for v in y]))
            data = other.columns[:y]
            u = [0.3]
            expected = [i == 2 ? 0.0 : normal_logpdf(data[i], u[1] * sqrt(x[i]), 1.3)
                for i in 1:n]
            likelihood_gradient = sum((data[i] - u[1] * sqrt(x[i])) * sqrt(x[i]) / 1.3^2
                for i in 1:n if i != 2)
            pointwise = prepare_query(built, other, :pointwise)
            @test Base.invokelatest(pointwise, u).y ≈ expected
            sampler = prepare_sampler(built, other, u; backend)
            g = similar(u)
            value, _ = sampler_value_and_gradient!(sampler, g, u)
            @test value ≈ normal_logpdf(u[1], 0.0, 1.0) + sum(expected)
            @test g[1] ≈ -u[1] + likelihood_gradient
            cell = prepare_cell_sampler(built, other, :y, u; backend)
            for i in (1, 2, n)
                value, _ = cell_value_and_gradient!(cell, g, u, i)
                @test value ≈ expected[i]
                @test g[1] ≈ (i == 2 ? 0.0 :
                    (data[i] - u[1] * sqrt(x[i])) * sqrt(x[i]) / 1.3^2)
            end
        end
        @test isequal((; x, y), original)

        changed_x, changed_y = copy(x), copy(y)
        changed_x[2], changed_y[2], changed_y[3] = 0.5, 0.2, missing
        changed = model(; x = changed_x) | (; y = changed_y)
        # Refused: this graph embeds the old selected-argument guard; ignoring
        # a changed mask would skip a present value and evaluate the wrong model.
        @test_throws "needs its own `build_kernel(plan)`" prepare_query(built, changed, :pointwise)
        @test_throws ContractValidationError prepare_cell_query(built, changed, :y)
        own = build_kernel(changed)
        own_values = Base.invokelatest(prepare_query(own, changed, :pointwise), [0.3]).y
        @test own_values ≈ [ismissing(changed_y[i]) ? 0.0 :
            normal_logpdf(changed_y[i], 0.3 * sqrt(changed_x[i]), 1.3) for i in 1:n]
    end
end
