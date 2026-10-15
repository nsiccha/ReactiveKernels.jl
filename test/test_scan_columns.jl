using ReactiveKernels, DifferentiationInterface, Enzyme, Test

@kernel _coupled_lags(x::Vector{Float64}, r::Float64) = begin
    @scan begin
        a[1] = 1
        b[1] = 2
        for t in 2:length(x)
            shared = sin(x[t] * r)
            a[t] = r * b[t - 1] + shared
            b[t] = 0.3 * a[t - 1] + a[t] + shared
        end
    end
    total::Float64 = sum(a) + 2sum(b)
    return (a, b)
end

@kernel _coupled_windows(x::Vector{Float64}, r::Float64) = begin
    @scan begin
        a[1] = 1.0
        a[2] = 0.5
        b[1] = -0.25
        b[2] = a[1]
        for t in 3:length(x)
            shared = sin(x[t] * r)
            a[t] = r * sum(b[s] for s in 1:(t - 1); init = 0.0) / t + shared
            b[t] = 0.3 * sum(a[(t - 2):(t - 1)]) + a[t]
        end
    end
    total::Float64 = sum(a) + 2sum(b)
    return (a, b)
end

@kernel _independent_columns(x, r) = begin
    @scan begin
        a[1] = 1.0
        for t in 2:length(x)
            a[t] = r * a[t - 1] + x[t]
        end
    end
    @scan begin
        b[1] = -0.25
        for t in 2:length(x)
            b[t] = r * b[t - 1] - x[t]
        end
    end
    return (a, b)
end

function _coupled_reference(x, r, history)
    a, b = history ? ([1.0, 0.5], [-0.25, 1.0]) : ([1.0], [2.0])
    for t in (length(a) + 1):length(x)
        shared = sin(x[t] * r)
        next_a = history ? r * sum(b) / t + shared : r * b[end] + shared
        next_b = history ? 0.3 * sum(a[(end - 1):end]) + next_a :
            0.3 * a[end] + next_a + shared
        push!(a, next_a)
        push!(b, next_b)
    end
    return a, b
end

_count_scan_loops(ex) = ex isa Expr ? (ex.head === :for ? 1 : 0) +
    sum(_count_scan_loops, ex.args; init = 0) : 0
_is_columns_scan(r) = r.op isa ReactiveKernels._AuthoredScanOp
_scan_sin_calls(ex) = ex isa Expr ?
    ((ex.head === :call && (ex.args[1] === :sin ||
        ex.args[1] isa GlobalRef && ex.args[1].name === :sin)) ? 1 : 0) +
    sum(_scan_sin_calls, ex.args; init = 0) : 0

@testset "one recurrence for multiple scan columns" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for (spec, history, seed_count) in ((_coupled_lags, false, 1), (_coupled_windows, true, 2))
        kernel = prepare(spec)
        @test count(e -> e.kind === :scan, recipe_inventory(spec)) == 1
        @test count(_is_columns_scan, kernel.plan.recipes) == 1
        single = prepare(spec; want = :a)
        @test count(_is_columns_scan, single.plan.recipes) == 1
        structures = Int[]
        for n in (seed_count, seed_count + 1, 7, 19)
            x = [cos(t) for t in 1:n]
            before = copy(x)
            a, b = kernel(x, 0.4)
            expected = _coupled_reference(x, 0.4, history)
            @test a ≈ expected[1]
            @test b ≈ expected[2]
            @test single(x, 0.4) ≈ a
            @test x == before
            @test !Base.mightalias(a, b)
            @test !Base.mightalias(a, x)
            total = prepare(spec; want = :total, bound = (; x))
            push!(structures, _count_scan_loops(code_expr(total)))
            ad = prepare_ad(total, backend, 0.4; active = :r)
            value, grad = ad_value_and_gradient(ad, 0.4)
            f(r) = let (aa, bb) = _coupled_reference(x, r, history)
                sum(aa) + 2sum(bb)
            end
            @test value ≈ f(0.4)
            @test grad ≈ (f(0.400001) - f(0.399999)) / 2e-6 atol = 1e-8 rtol = 1e-6
        end
        # Bound lengths alter the domain, not copies of the recurrence body.
        @test allequal(structures)
        # The retained step graph contains the shared nonlinear operation once;
        # one scan recipe supplies all requested trajectory projections.
        scan_recipe = only(filter(_is_columns_scan, kernel.plan.recipes))
        @test length(scan_body(scan_recipe).want) == 2
        @test sum(r -> _scan_sin_calls(r.source), scan_body(scan_recipe).recipes) == 1
    end
    @test count(e -> e.kind === :scan, recipe_inventory(_independent_columns)) == 2
    x = [0.1, 0.4, -0.2]
    a, b = prepare(_independent_columns)(x, 0.5)
    @test a ≈ [1.0, 0.9, 0.25]
    @test b ≈ [-0.25, -0.525, -0.0625]
end

@kernel _direct_history_columns(x::Vector{Float64}, r::Float64) = begin
    columns = scan(eachindex(x); init = 0.0,
            history = (a = [1.0, 2.0], b = 0.0)) do carry, j, earlier
        a = r * earlier.a[j + 1] + x[j]
        b = (j == 1 ? 0.0 : earlier.b[j - 1]) + a
        (carry, (a = a, b = b))
    end
    total::Float64 = sum(columns.a) + sum(columns.b)
    return columns
end

@testset "named history columns retain independent seed offsets" begin
    for n in (0, 1, 6)
        x = [sin(t) for t in 1:n]
        expected(r) = let a = [1.0, 2.0], b = Float64[]
            for t in eachindex(x)
                push!(a, r * a[end] + x[t])
                push!(b, (isempty(b) ? 0.0 : b[end]) + a[end])
            end
            (; a, b)
        end
        columns = prepare(_direct_history_columns)(x, 0.4)
        ref = expected(0.4)
        @test columns.a ≈ ref.a
        @test columns.b ≈ ref.b
        total = prepare(_direct_history_columns; want = :total, bound = (; x))
        ad = prepare_ad(total, AutoEnzyme(; mode = Enzyme.Reverse), 0.4; active = :r)
        value, grad = ad_value_and_gradient(ad, 0.4)
        f(r) = sum(expected(r).a) + sum(expected(r).b)
        @test value ≈ f(0.4)
        @test grad ≈ (f(0.400001) - f(0.399999)) / 2e-6 atol = 1e-8 rtol = 1e-6
    end
end
