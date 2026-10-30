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

    # A gensym-created module is an isolation namespace in quoted data and as
    # a value too. Stable modules, and which namespace a reference names, stay.
    a, b = Module(gensym(:Lowering)), Module(gensym(:Lowering))
    quoted(m, n = m) = Expr(:call, :+, QuoteNode(GlobalRef(m, :f)), GlobalRef(n, :g))
    @test same(identity(quoted(a)), identity(quoted(b)))
    @test !same(identity(quoted(a)), identity(quoted(a, b)))
    @test !same(identity(quoted(Base)), identity(quoted(Main)))
    rebuilt(m) = Expr(:call, GlobalRef(Core, :GlobalRef), m, QuoteNode(:f))
    @test same(identity(rebuilt(a)), identity(rebuilt(b)))
    @test !same(identity(rebuilt(Base)), identity(rebuilt(Main)))
    @test !same(identity(rebuilt(a)), identity(rebuilt(Main)))
    # A differing statement that holds a module is named, not copied.
    held(m) = Expr(:function, :(f(x)), Expr(:block, LineNumberNode(1),
        Expr(:(=), :ref, rebuilt(m))))
    built, other = held(Base), held(Main)
    @test occursin("statement 1:", ReactiveKernelsPPL._program_difference(built, other))
    @test built.args[2].args[1] isa LineNumberNode

    # An equal quoted mask must not hide the later, actual difference.
    def(mask, value) = Expr(:function, :(f(x)), Expr(:block,
        Expr(:(=), :present, QuoteNode(mask)), Expr(:(=), :result, value)))
    built, other = def(mask, 1.0), def(copy(mask), 2.0)
    @test startswith(ReactiveKernelsPPL._program_difference(built, other), "statement 2:")
end

@testset "level labels are the identity's quoted data, not compiled code" begin
    m = @rkppl begin
        a ~ Normal(0, 1)
        c[levels(g)] .~ Normal.(0, 1)
        y .~ Normal.(a .+ c[g], 1)
    end
    g = ["b", "a", "c", "a"]
    y = [0.1, -0.2, 0.3, 0.0]
    plan = m(; g) | (; y)
    # The program reads the axis's labels as a data-only value: no label is
    # a literal of the generated code (decision 1id2a4o: two 10,950-label
    # literals were 60 s and 45 s of the ArbeitsplatzAtlas prefix compile).
    program = string(kernel_expr(plan, assign_layout(plan)))
    # This location is an affine design: the factor contrast reads the labels.
    @test occursin("permutedims(_ppl_levels(g))", program)
    @test !occursin("\"a\"", program)
    # A gather outside a design reads them through its level codes.
    gathered = (@rkppl begin
        a ~ Normal(0, 1)
        c[levels(g)] .~ Normal.(0, 1)
        y .~ Normal.(a, exp.(c[g]))
    end)(; g) | (; y)
    program = string(kernel_expr(gathered, assign_layout(gathered)))
    @test occursin("_ppl_lvx_c_g_axis_1 = _declared_codes(g, _ppl_levels(g))", program)
    @test !occursin("\"a\"", program)
    built = build_kernel(plan)
    quoted = Any[]
    walk(x) = x isa QuoteNode ? push!(quoted, x.value) :
        x isa Expr ? foreach(walk, x.args) : nothing
    walk(built.program)
    @test ["a", "b", "c"] in quoted
    @test !occursin("_ppl_levels", string(built.program))
    # The same labels bound in another order are the same program; other
    # labels another (`test_rebinding_rows.jl` pins the refusal).
    same = ReactiveKernelsPPL._same_program
    @test same(built.program, build_kernel(m(; g = ["c", "b", "a", "b"]) | (; y)).program)
    @test !same(built.program, build_kernel(m(; g = ["b", "a", "d", "a"]) | (; y)).program)
    # A long label vector prints as its summary and ends in the difference.
    wide = build_kernel(m(; g = string.("g", 1:40)) | (; y = zeros(40))).program
    other = build_kernel(m(; g = string.("h", 1:40)) | (; y = zeros(40))).program
    difference = ReactiveKernelsPPL._program_difference(wide, other)
    @test occursin("<40-element Vector{String}: g1, g10, g11", difference)
    @test occursin("(32 more), g8, g9>", difference)
    @test !occursin("g20", difference)
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
