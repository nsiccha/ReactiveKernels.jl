using ReactiveKernels, DifferentiationInterface, Enzyme, Test

module NativeTypeQueryFixtures
using ReactiveKernels

# Each structural stage reads both preceding values. The mathematical graph
# has O(depth) recipes, but recursively substituted result-type expressions
# used to contain O(2^depth) calls. Depth is authored structure, never data.
function diamonds(depth, left, right)
    statements = Expr[]
    for i in 1:depth
        a, b = Symbol(:left_, i), Symbol(:right_, i)
        push!(statements, :($a = ($left + $right) / 2))
        push!(statements, :($b = ($left - $right) / 4))
        left, right = a, b
    end
    statements, left, right
end

function nested(depth)
    statements, left, right = diamonds(depth, :left_0, :right_0)
    source = quote
        @kernel function diamond_plates(groups, gain)
            rows = plate(groups) do xs
                cells = plate(xs) do x
                    left_0 = x * gain
                    right_0 = x / gain
                    $(statements...)
                    $left + $right
                end
                sum(cells)
            end
            return sum(rows)
        end
    end
    Core.eval(@__MODULE__, source)
end

function scanned(depth)
    statements, left, right = diamonds(depth, :left_0, :right_0)
    source = quote
        @kernel function diamond_scan(xs, gain)
            values = scan(xs; init=zero(gain)) do previous, x
                left_0 = previous + x * gain
                right_0 = previous - x / gain
                $(statements...)
                next = $left + $right
                (next, next)
            end
            return values
        end
    end
    Core.eval(@__MODULE__, source)
end

function reference(x, gain, depth; previous=zero(gain))
    left, right = previous + x * gain, previous - x / gain
    for _ in 1:depth
        left, right = (left + right) / 2, (left - right) / 4
    end
    left + right
end

function plate_reference(x, gain, depth)
    left, right = x * gain, x / gain
    for _ in 1:depth
        left, right = (left + right) / 2, (left - right) / 4
    end
    left + right
end

function plate_derivative(x, gain, depth)
    left, right = x, -x / gain^2
    for _ in 1:depth
        left, right = (left + right) / 2, (left - right) / 4
    end
    left + right
end

count_head(ex, head) = ex isa Expr ? Int(ex.head === head) +
    sum(x -> count_head(x, head), ex.args; init=0) : 0
count_query(ex) = ex isa Expr ?
    Int(ex.head === :call && ex.args[1] == GlobalRef(Base, :promote_op)) +
    sum(count_query, ex.args; init=0) : 0
end

@testset "Native type queries preserve a shared graph" begin
    F = NativeTypeQueryFixtures
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    counts, sizes = Int[], Int[]
    for depth in (4, 8, 16)
        spec = F.nested(depth)
        reader = prepare(spec)
        @test typeof(reader.f) === typeof(prepare(spec).f)
        ast = deepcopy(code_expr(reader))
        push!(counts, F.count_query(ast))
        push!(sizes, sizeof(sprint(Base.show_unquoted, ast)))
        @test F.count_head(ast, :for) == 2
        for n in (0, 3, 17), gain in (0.7, 1.3)
            groups = [sin.(1.0:n), Float64[], cos.(1.0:(n ÷ 2))]
            original = deepcopy(groups)
            expected = sum(xs -> sum(x -> F.plate_reference(x, gain, depth), xs;
                                     init=0.0), groups; init=0.0)
            derivative = sum(xs -> sum(x -> F.plate_derivative(x, gain, depth), xs;
                                       init=0.0), groups; init=0.0)
            bound = prepare(spec; bound=(; groups))
            @test F.count_head(code_expr(bound), :for) == 2
            @test reader(groups, gain) ≈ expected
            @test bound(gain) ≈ expected
            for (k, args) in ((reader, (groups, gain)), (bound, (gain,)))
                ad = prepare_ad(k, backend, args...; active=:gain)
                value, gradient = ad_value_and_gradient(ad, args...)
                @test value ≈ expected
                @test gradient ≈ derivative
            end
            @test isequal(groups, original)
            @test code_expr(reader) == ast
        end
    end
    # Doubling authored work may grow the emitted program linearly. This
    # distinguishes DAG preservation from the former exponential expansion.
    @test counts[2] < 3counts[1]
    @test counts[3] < 3counts[2]
    @test sizes[2] < 3sizes[1]
    @test sizes[3] < 3sizes[2]
end

# The inferred output type allocates the scan's one buffer before the
# emptiness branch (see `_lower_authored_scan_native!`), so its shared
# queries are bound once there and neither arm repeats them.
@testset "Scan output type queries precede the emptiness branch" begin
    F = NativeTypeQueryFixtures
    for depth in (4, 8, 16)
        reader = prepare(F.scanned(depth))
        ast = code_expr(reader)
        @test F.count_head(ast, :for) == 1
        statements = ast.args[2].args
        position = only(i for (i, ex) in enumerate(statements) if ex isa Expr &&
            ex.head === :if && ex.args[1] isa Expr &&
            ex.args[1].head === :call && ex.args[1].args[1] == GlobalRef(Base, :isempty))
        guard = statements[position]
        @test F.count_query(guard.args[2]) == 0
        @test F.count_query(guard.args[3]) == 0
        @test F.count_query(Expr(:block, statements[1:(position - 1)]...)) > 0
        @test F.count_query(ast) ==
            F.count_query(Expr(:block, statements[1:(position - 1)]...))
        for T in (Float32, Float64), n in (0, 3, 17)
            xs, gain = T.(sin.(1.0:n)), T(1.3)
            original = copy(xs)
            expected, previous = T[], zero(T)
            for x in xs
                previous = F.reference(x, gain, depth; previous)
                push!(expected, previous)
            end
            actual = reader(xs, gain)
            @test actual ≈ expected
            @test eltype(actual) == T
            @test xs == original
        end
    end
end

module TypeOperationFixtures
using ReactiveKernels
# A type called directly on graph ports is the recipe's operation itself.
const Pair2 = ComplexF64

@kernel pair_product(a::Float64, b::Float64) = begin
    pair = Pair2(a, b)
    product = real(pair) * imag(pair)
    return product
end

# The child's scan step outputs a bare constructor call; its result feeds a
# plate cell whose values a second plate reads beside fixed data vectors.
@kernel trace_reads(xs, ws, idx, rate) = begin
    path = scan(xs, ws; init = 0.0) do previous, x, w
        level = previous * exp(-rate) + w * x
        area = previous + level
        (level, Pair2(level, area))
    end
    selected = path[idx]
    return vcat(real.(selected), imag.(selected))
end

@kernel grouped_trace(xs, ws, idx, first_idx, second_idx, times, obs, sd, theta) = begin
    rates = exp.(theta[1] .+ 0.1 .* collect(1:length(xs)))
    slopes = theta[2] .+ 0.05 .* collect(1:length(xs))
    cells = plate(1:length(xs), rates, slopes) do s, rate, slope
        reads = trace_reads(xs[s], ws[s], idx[s], rate)
        level = reads[first_idx[s]]
        scaled = reads[second_idx[s]] ./ 3.0
        slope * times[s] + -slope * scaled .+ level
    end
    location = convert(Vector{Float64}, reduce(vcat, cells; init = Float64[]))
    terms = plate(location, obs, sd) do m, y, s
        -0.5 * abs2((y - m) / (s * exp(theta[3])))
    end
    return sum(terms)
end

function grouped_reference(xs, ws, idx, first_idx, second_idx, times, obs, sd, theta)
    location = Float64[]
    for s in eachindex(xs)
        rate, slope = exp(theta[1] + 0.1 * s), theta[2] + 0.05 * s
        previous, levels, areas = 0.0, Float64[], Float64[]
        for (x, w) in zip(xs[s], ws[s])
            level = previous * exp(-rate) + w * x
            push!(levels, level)
            push!(areas, previous + level)
            previous = level
        end
        reads = vcat(levels[idx[s]], areas[idx[s]])
        append!(location, slope .* times[s] .- slope .* (reads[second_idx[s]] ./ 3.0) .+
                          reads[first_idx[s]])
    end
    sum(-0.5 * abs2((y - m) / (σ * exp(theta[3]))) for (m, y, σ) in zip(location, obs, sd))
end
end

# A type in a heterogeneous operation table is only a `DataType`, so calls
# through it and every result type derived from it were uninferred. In this
# grouped graph that left the plate result-type queries unfolded and the body
# in dynamic dispatch, where native Reverse raised EnzymeRuntimeActivityError
# on a boxed constant-data broadcast operand.
@testset "Type-constructor operations keep native result types" begin
    F = TypeOperationFixtures
    reader = prepare(F.pair_product)
    @test reader(1.5, 2.0) == 3.0
    @test Base.return_types(reader, (Float64, Float64)) == Any[Float64]
    @test occursin("Complex(a, b)", sprint(show, MIME"text/plain"(), kernel_graph(F.pair_product)))

    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    grouped = prepare(F.grouped_trace)
    cases = (
        (xs = [[1.0, 0.5, 2.0], [0.3, 1.2, 0.7, 0.9]],
         ws = [[1.0, 1.0, 0.5], [0.2, 1.0, 1.0, 0.4]],
         idx = [[2, 1, 3], [4, 3, 2, 1]], first_idx = [[1, 2], [2, 4]],
         second_idx = [[4, 6], [5, 7]], times = [[0.1, 0.4], [0.2, 0.9]],
         obs = [1.0, 0.7, 0.4, 1.3], sd = [0.5, 0.6, 0.7, 0.8]),
        (xs = [[0.4, 1.1], [2.0], [0.3, 0.8, 1.5]],
         ws = [[1.0, 0.7], [0.9], [0.4, 1.0, 0.6]],
         idx = [[1, 2], [1], [3, 1, 2]], first_idx = [[2], [1], [1, 3]],
         second_idx = [[3], [2], [4, 6]], times = [[0.5], [0.3], [0.1, 0.7]],
         obs = [0.2, 0.9, 1.1, 0.6], sd = [0.4, 0.5, 0.9, 0.3]),
    )
    for case in cases, theta in ([0.3, 0.8, -0.2], [-0.4, 1.3, 0.1])
        args = (values(case)[1:6]..., case.obs, case.sd, theta)
        original = deepcopy(args)
        @test Base.return_types(grouped, map(typeof, args)) == Any[Float64]
        expected = F.grouped_reference(args...)
        @test grouped(args...) ≈ expected rtol = 1e-12
        ad = prepare_ad(grouped, backend, args...; active = :theta)
        value, gradient = ad_value_and_gradient(ad, args...)
        @test value ≈ expected rtol = 1e-12
        central = map(eachindex(theta)) do i
            h = 1e-6
            up, down = copy(theta), copy(theta)
            up[i] += h
            down[i] -= h
            (F.grouped_reference(args[1:8]..., up) -
             F.grouped_reference(args[1:8]..., down)) / 2h
        end
        @test gradient ≈ central rtol = 1e-6
        @test isequal(args, original)
    end
end
