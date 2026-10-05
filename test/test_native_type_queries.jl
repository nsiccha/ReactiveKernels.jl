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
            rows = plate(groups, Ref(gain)) do xs, scale
                cells = plate(xs, Ref(scale)) do x, s
                    left_0 = x * s
                    right_0 = x / s
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
            values = scan(xs, Ref(gain); init=zero(gain)) do previous, x, s
                left_0 = previous + x * s
                right_0 = previous - x / s
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

@testset "Empty scan type queries stay in the empty arm" begin
    F = NativeTypeQueryFixtures
    for depth in (4, 8, 16)
        reader = prepare(F.scanned(depth))
        ast = code_expr(reader)
        @test F.count_head(ast, :for) == 1
        guard = only(ex for ex in ast.args[2].args if ex isa Expr &&
            ex.head === :if && ex.args[1] isa Expr &&
            ex.args[1].head === :call && ex.args[1].args[1] == GlobalRef(Base, :isempty))
        @test F.count_query(guard.args[2]) > 0
        @test F.count_query(guard.args[3]) == 0
        @test F.count_query(ast) == F.count_query(guard.args[2])
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
