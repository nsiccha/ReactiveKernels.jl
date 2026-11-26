module BatchedDestinationTests

using ReactiveKernels
using Test

@kernel scaled(p, x) = begin
    left = p * x
    right = x * p
    total = sum(left)
end
@kernel scaled_with_keyword(p; data) = begin
    result = p * data
end
@kernel recorded(p, x) = begin
    line = x * p
    result = (; line, stats=(sum(line), maximum(line)))
end
@kernel pass(p::Vector{Float64}) = begin
    p
end
@kernel observed_column(p) = begin
    points = plate(p) do item
        item
    end
    from_column = points isa SubArray
end
@kernel shared_rows(p, cells) = begin
    landscape = stack(cells; dims=1)
    grid = landscape .+ 1.0
    peak = maximum(grid)
    total = p * peak
end
@kernel stacked_columns(p, cells) = begin
    landscape = stack(cells; dims=2)
    total = sum(p * landscape)
end

function allocated_pair(reader, full, tail, x)
    reader(full, x)
    reader(tail, x)
    @allocated begin
        reader(full, x)
        reader(tail, x)
    end
end

@testset "shared stack and broadcast refresh changed design inputs" begin
    cells = [collect(1.:1024) for _ in 1:48]
    p = [1., 2., -1.]
    for graph in (shared_rows, stacked_columns)
        k = vectorize(graph; batched=:p, reuse=true)
        scalar = prepare(graph)
        ast = code_expr(k)
        @test occursin("_lane_stack", string(ast))
        k(p, cells)
        bytes = @allocated k(p, cells)
        @test bytes < length(cells) * sizeof(first(cells))
        cells[2][1] += 2000.
        @test k(p, cells) == map(v -> scalar(v, cells), p)
        @test_throws DimensionMismatch k(p, [cells[1], cells[2][2:end]])
        @test k(p, cells[2:5]) == map(v -> scalar(v, cells[2:5]), p)
        @test code_expr(k) === ast
    end
end

@testset "two dense batch widths and scalar products" begin
    x = collect(1.0:4096)
    full, tail = collect(1.0:32), collect(-13.0:-1)
    for schedule in (nothing, NativeScheduling(workers=1, chunk_size=8))
        k = vectorize(scaled; batched=:p, want=(:left, :right), reuse=true, schedule)
        a = k(full, x)
        b = k(tail, x)
        @test all(y -> y isa Matrix{Float64}, (a..., b...))
        @test all(y -> y == x .* transpose(full), a)
        @test all(y -> y == x .* transpose(tail), b)
        @test all(map(===, a, k(-full, x)))
        @test all(map(===, b, k(-tail, x)))
        @test all(y -> y == x .* transpose(-tail), b)
        @test copy(k)(full, x)[1] !== a[1]
        # One dense stack per leaf and width would already exceed this bound.
        @test allocated_pair(k, full, tail, x) < sizeof(first(b))
        first_three = k(full[1:3], x)
        @test all(map(===, b, k(tail, x)))
        @test all(map(!==, a, k(full, x)))
        @test first_three[1] == x .* transpose(full[1:3])
        @test size(k(tail, x[1:7])[1]) == (7, length(tail))
        @test k(Float32.(tail), Float32.(x))[1] isa Matrix{Float32}
    end

    # Feeding the inactive width back into the reader must detach it too.
    k = vectorize(pass; batched=:p, want=:p, reuse=true)
    full_input = reshape(collect(1.0:12), 3, 4)
    old = k(full_input)
    k(full_input[:, 1:2])
    expected = copy(old)
    @test k(old) == expected
    @test k(-full_input) !== old
    @test old == expected
end

@testset "batched writers use caller rows and validate overlap" begin
    p, x = [1., -2., 3.], [2., 0., -1., 4.]
    for schedule in (nothing, NativeScheduling(workers=2, chunk_size=2)), reuse in (false, true)
        reader = vectorize(prepare(scaled; want=(:left,:right,:total)); batched=:p, schedule, reuse)
        write! = prepare_into(reader)
        bank = fill(-100., 2length(x) + 1, length(p) + 2)
        columns = 2:length(p)+1
        dest = (view(bank, 1:length(x), columns), view(bank, length(x)+1:2length(x), columns),
                view(bank, size(bank, 1), columns))
        for scale in (1, -3)
            expected = reader(scale .* p, x)
            @test write!(dest, scale .* p, x) === dest
            @test all(map(isequal, dest, expected))
        end
        @test all(==(-100.), bank[:, [1,end]])
        @test_throws ArgumentError write!((dest[1], dest[1], dest[3]), p, x)
        @test_throws ArgumentError write!((dest[1], dest[2], p), p, x)
        @test_throws ArgumentError write!((dest[1],), p, x)
        @test_throws DimensionMismatch write!((dest[1], zeros(2,3), dest[3]), p, x)
        @test write!(dest, p, x) === dest

        kw = prepare_into(vectorize(scaled_with_keyword; batched=:p, schedule, reuse))
        float32 = zeros(Float32, length(x), length(p))
        @test kw(float32, p; data=x) === float32
        @test float32 == x .* transpose(p)

        record_writer = prepare_into(vectorize(recorded; batched=:p, schedule, reuse))
        record = (; line=similar(dest[1]), stats=(similar(p), similar(p)))
        @test record_writer(record, p, x) === record
        @test record.line == x .* transpose(p)
        @test record.stats[1] == vec(sum(record.line; dims=1))
        @test record.stats[2] == vec(maximum(record.line; dims=1))

        observed = vectorize(prepare(observed_column; want=(:points, :from_column)); batched=:p, schedule, reuse)
        X = reshape(collect(1.0:12), 4, 3)
        obs_dest = (dest[1], falses(3))
        @test prepare_into(observed)(obs_dest, X) === obs_dest
        @test obs_dest == observed(X)
    end
    # Dense batched HAVE -> WANT copy, including the declared empty shape.
    pw = prepare_into(vectorize(pass; batched=:p, want=:p, reuse=true))
    X = reshape(collect(1.0:12), 4, 3)
    dst = similar(X)
    @test pw(dst, X) === dst && dst == X
    @test_throws ArgumentError pw(view(X,:,:), X)
    @test isempty(pw(zeros(0,0), zeros(4,0)))
    @test_throws DimensionMismatch pw(zeros(4,0), zeros(4,0))
    bound = prepare(scaled; have=(:p,:x), want=:left, bound=(; x))
    @test_throws ArgumentError prepare_into(vectorize(bound; batched=:p))(reshape(x,:,1), [1.])
end

@testset "a warmed view writer avoids a second full stack" begin
    x = collect(1.:4096)
    p = collect(1.:32)
    bank = zeros(length(x) + 1, length(p))
    dst = view(bank, 2:size(bank, 1), :)
    for schedule in (nothing, NativeScheduling(workers=1, chunk_size=8))
        reader = vectorize(scaled; batched=:p, want=:left, schedule)
        writer = prepare_into(reader)
        writer(dst, p, x)
        bytes = @allocated writer(dst, p, x)
        @test bytes < sizeof(dst) ÷ 2
        @test dst == x .* transpose(p)
        @test all(iszero, bank[1,:])
    end
end

end
