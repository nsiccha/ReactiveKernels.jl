using Test
using ReactiveKernels
using LinearAlgebra
using InteractiveUtils: code_typed

struct PositionRankArray{T,A,N} <: AbstractArray{T,N}
    storage::A
end

@testset "untyped numeric position boundary" begin
    @kernel untyped_positions(position, shared) = begin
        invariant = sum(abs2, shared)
        result = dot(position, position) + invariant
        return result
    end
    positions = reshape(collect(1.0:15.0), 3, 5)
    shared = [2.0, 3.0]
    scalar = prepare(untyped_positions)
    expected = [scalar(positions[:, i], shared) for i in axes(positions, 2)]
    for constructor in (vectorize, prepare_batched)
        batch = constructor(untyped_positions; batched=:position)
        @test batch(positions, shared) == expected
        @test batch(positions[:, 5:-1:1], shared) == reverse(expected)
        @test batch(positions[:, 1:1], shared) == expected[1:1]
        @test batched_ports(batch) == (:position,)
        @test batch isa ReactiveKernels.GraphReplicatedKernel
    end
    @test vectorize(prepare(untyped_positions; have=(:position, :shared));
                    batched=:position)(positions, shared) == expected
end

@testset "record positions and owned record outputs" begin
    @kernel record_positions(position, times) = begin
        invariant = sum(times)
        result = (; trajectory=position.scale .* times .+ position.offset,
                    total=position.scale * invariant,
                    flags=(position.scale > 0,))
        return result
    end
    positions = (; scale=[1.0, -2.0, 3.0], offset=[0.5, 1.0, -0.5])
    records = [(; scale=positions.scale[i], offset=positions.offset[i]) for i in 1:3]
    times = [0.0, 1.0, 2.0, 4.0]
    scalar = prepare(record_positions)
    expected = [scalar(position, times) for position in records]
    for constructor in (vectorize, prepare_batched, replica)
        batch = constructor(record_positions; batched=:position)
        actual = batch(positions, times)
        @test actual.trajectory == reduce(hcat, getproperty.(expected, :trajectory))
        @test actual.total == getproperty.(expected, :total)
        @test actual.flags[1] == [true, false, true]
        @test batch(records, times) == actual
        snapshot = deepcopy(actual)
        batch((; scale=[2.0], offset=[0.0]), times)
        @test actual == snapshot
        @test positions == (; scale=[1.0, -2.0, 3.0], offset=[0.5, 1.0, -0.5])
        @test times == [0.0, 1.0, 2.0, 4.0]
        @test_throws DimensionMismatch batch((; scale=[1.0], offset=[1.0, 2.0]), times)
    end
    @kernel typed_record(position::NamedTuple{(:scale,),Tuple{Float64}}) = begin
        result::Float64 = position.scale^2
    end
    @test vectorize(typed_record; batched=:position)((; scale=[2.0, 3.0])) == [4.0, 9.0]
end

@testset "same-graph fixed-position build and changing read cut" begin
    @kernel staged_trajectory(position, schedule, amount) = begin
        grid = schedule.times .^ 2
        units = position.scale .* grid
        trajectory = units .* amount
        result = (; trajectory, total=sum(trajectory))
        return result
    end
    positions = (; scale=[1.0, 2.0, 4.0])
    schedule = (; times=[0.0, 0.5, 1.0, 2.0])
    build = vectorize(prepare(staged_trajectory;
        have=(:position, :schedule), want=:units, bound=(; schedule)); batched=:position)
    units = build(positions)
    read = vectorize(prepare(staged_trajectory;
        have=(:units, :amount), want=:result); batched=:units)
    direct = vectorize(staged_trajectory; batched=:position, want=:result)
    for amount in (0.0, 0.5, -2.0, 3.0)
        @test read(units, amount) == direct(positions, schedule, amount)
    end
    # The read plan contains no unit solve or grid construction. The direct
    # plan is the positive control for both checks, and numeric results match.
    read_nodes = [value.name for recipe in plan(read).recipes for value in recipe.outputs]
    direct_nodes = [value.name for recipe in plan(direct).recipes for value in recipe.outputs]
    @test :units ∉ read_nodes && :units ∈ direct_nodes
    @test :grid ∉ read_nodes && :grid ∈ direct_nodes
    @test batched_ports(build) == (:position,)
    @test batched_ports(read) == (:units,)
end

@testset "composite shared work stays above the position loop" begin
    @kernel shared_plate(position, data) = begin
        transformed = plate(data) do x
            x^2
        end
        result = position * sum(transformed)
    end
    batch = vectorize(shared_plate; batched=:position)
    @test batch isa ReactiveKernels.GraphReplicatedKernel
    @test batch([2.0, 3.0], [1.0, 2.0, 3.0]) == [28.0, 42.0]
    expression = code_expr(batch)
    # The position loop is the last top-level loop; a hoisted plate emits its
    # own fused loop above it.
    loop_index = findlast(node -> node isa Expr && node.head === :for,
                          expression.args[2].args)
    before_loop = Expr(:block, expression.args[2].args[1:loop_index-1]...)
    loop = expression.args[2].args[loop_index]
    @test occursin("transformed =", string(before_loop))
    @test !occursin("transformed =", string(loop))
    @test batch([2.0], [1.0, 2.0, 3.0]) == [28.0]
    @test batch(fill(2.0, 23), [1.0, 2.0, 3.0]) == fill(28.0, 23)
    @test code_expr(batch) === expression
end

_position_plate_bytes(kernel, args...) =
    minimum(@allocated(kernel(args...)) for _ in 1:3)

@testset "per-position plates lower as in prepare" begin
    # A plate whose operands vary with the position runs once per position. It
    # emits the scalar kernel's fused native loop, not the plate operation's
    # per-cell fallback (snag inline-natural-s-126a07a6: 3.0 MB against
    # 0.33 MB per position and read on the ShinyRK superposition plate).
    @kernel per_position_plate(position::Vector{Float64}, shifts::Vector{Int},
                               n::Int) = begin
        response::Vector{Float64} = exp.(-position[1] .* (0:(n - 1)))
        weights::Vector{Float64} = position[2] .* (1:length(shifts))
        observations = 1:n
        concentration::Vector{Float64} = plate(observations) do t
            sum((weights[j] * get(response, t - shifts[j], 0.0) for j in eachindex(shifts)); init = 0.0)
        end
        return concentration
    end
    positions = [0.1 0.2 0.3 0.4; 1.0 2.0 3.0 4.0]
    shifts, n = [0, 5, 9], 2000
    scalar = prepare(per_position_plate)
    expected = reduce(hcat, [scalar(positions[:, i], shifts, n)
                             for i in axes(positions, 2)])
    scalar_bytes = (scalar(positions[:, 1], shifts, n);
                    _position_plate_bytes(scalar, positions[:, 1], shifts, n))
    for reuse in (false, true)
        batch = vectorize(per_position_plate; batched = :position, reuse)
        @test batch(positions, shifts, n) == expected
        bytes = _position_plate_bytes(batch, positions, shifts, n)
        # Per position: the scalar kernel's own arrays plus the stacked output.
        @test bytes <= 2 * size(positions, 2) * scalar_bytes
    end
end

@testset "rank checks and HAVE passthrough" begin
    @kernel typed_positions(position::Vector{Float64}) = begin
        result::Float64 = sum(position)
    end
    batch = vectorize(typed_positions; batched=:position)
    @test_throws DimensionMismatch batch([1.0, 2.0])
    @test batch(zeros(2, 0)) == Float64[]
    # A declaration fixing only the rank is checked the same way; it used to
    # accept any stack and run the scalar graph on matrix slices (snag
    # rk-declared-rank-317aa725). One leaving the rank open stays unchecked.
    @kernel rank_positions(position::AbstractVector) = begin
        result::Float64 = sum(position)
    end
    @kernel real_positions(position::AbstractVector{<:Real}) = begin
        result::Float64 = sum(position)
    end
    for graph in (rank_positions, real_positions)
        rank_batch = vectorize(graph; batched=:position)
        @test rank_batch([1.0 2.0; 3.0 4.0]) == [4.0, 6.0]
        @test rank_batch([1 2; 3 4]) == [4.0, 6.0]
        @test_throws DimensionMismatch rank_batch([1.0, 2.0])
        @test_throws DimensionMismatch rank_batch(ones(2, 2, 2))
    end
    @kernel rankless_positions(position::AbstractArray{Float64}) = begin
        result::Float64 = sum(position)
    end
    @test vectorize(rankless_positions; batched=:position)(ones(2, 2, 2)) ==
          [4.0, 4.0]
    identity_batch = vectorize(typed_positions;
        have=:position, want=:position, batched=:position)
    positions = reshape(collect(1.0:6.0), 2, 3)
    output = identity_batch(positions)
    @test output == positions
    @test output !== positions
end

@testset "shared scans and embedded scalar operations" begin
    @kernel shared_scan(position, data) = begin
        cumulative = scan(data; init=0.0) do carry, item
            next = carry + item
            (next, next)
        end
        result = position * sum(cumulative)
    end
    @kernel scan_endpoint(data) = begin
        cumulative = scan(data; init=0.0) do carry, item
            next = carry + item
            (next, next)
        end
        total = sum(cumulative)
    end
    endpoint = prepare(scan_endpoint)
    @kernel embedded_positions(position, data) = begin
        invariant = endpoint(data)
        result = position * invariant
    end
    for graph in (shared_scan, embedded_positions)
        batch = vectorize(graph; batched=:position)
        scalar = prepare(graph)
        for count in (1, 23)
            positions = collect(1.0:count)
            data = [1.0, 2.0, 3.0]
            @test batch(positions, data) == [scalar(x, data) for x in positions]
        end
        @test batch isa ReactiveKernels.GraphReplicatedKernel
    end
end

@testset "runtime output layouts and numeric array subtypes" begin
    @kernel subtype_position(position::PositionRankArray{Float64,Vector{Float64},1}) = begin
        result::Float64 = sum(position)
    end
    batch = vectorize(subtype_position; batched=:position)
    @test batch([1.0 2.0; 3.0 4.0]) == [4.0, 6.0]
    @test_throws DimensionMismatch batch([1.0, 2.0])

    @kernel mixed_output(position) = begin
        result = position > 0 ? position : round(Int, -position)
    end
    @test_throws ArgumentError vectorize(mixed_output; batched=:position)([1.0, -2.0])
    @kernel mixed_tuple(position) = begin
        result = position > 0 ? (position,) : (position, position)
    end
    @test_throws ArgumentError vectorize(mixed_tuple; batched=:position)([1.0, -2.0])
    @kernel broad_record(position) = begin
        result::NamedTuple = (; value=position^2)
    end
    @test replica(broad_record; batched=:position)([2.0, 3.0]).value == [4.0, 9.0]
end

# Structural record depth must not erase scalar types inside the retained
# position/scan loops. The recurrence has an independent, exact prefix oracle.
@kernel nested_position_scan(position, xs) = begin
    trajectory = scan(xs, Ref(position.group.response);
                      init=position.group.response.seed, include_init=true) do previous, x, p
        next = previous + p.rate * x
        (next, next)
    end
    return trajectory
end
_nested_position_bytes(kernel, args...) =
    (kernel(args...); minimum(@allocated(kernel(args...)) for _ in 1:5))
_nested_position_lane(position, xs) = Vector{Float64}(undef, length(xs) + 1)
_nested_position_output(position, xs) =
    Matrix{Float64}(undef, length(xs) + 1, length(position.group.response.seed))
_nested_position_copy(kernel, position, xs) = copy(view(kernel(position, xs), :, 1))

@testset "nested record positions retain concrete loop arithmetic" begin
    position = (; group=(; response=(; seed=[1.0, 2.0, 3.0], rate=[0.5, -0.5, 2.0]),
                            unused=[4, 5, 6]), unused=[7.0, 8.0, 9.0])
    before = deepcopy(position)
    short, long = ones(8), ones(8192)
    expected(xs) = hcat([position.group.response.seed[i] .+
                        position.group.response.rate[i] .* (0:length(xs))
                        for i in 1:3]...)
    # Match the two intended allocations: the first scan lane and the owned
    # output stack. @allocated counts usable allocator blocks, which can grow
    # by more than sizeof's payload on macOS. Keep the fixed allowance small;
    # an additional length-dependent allocation must still fail the guard.
    lane_growth = _nested_position_bytes(_nested_position_lane, position, long) -
                  _nested_position_bytes(_nested_position_lane, position, short)
    output_growth = _nested_position_bytes(_nested_position_output, position, long) -
                    _nested_position_bytes(_nested_position_output, position, short)
    accounting_slack = 4096
    for reuse in (false, true)
        batch = vectorize(nested_position_scan; batched=:position, reuse)
        @test batch(position, short) == expected(short)
        @test batch(position, long) == expected(long)
        # Inspect the generated entry rather than only its wrapper.
        argtypes = reuse ?
            Tuple{typeof(batch.native), typeof(batch.ops), typeof(batch.caches),
                  typeof(position), typeof(long)} :
            Tuple{typeof(batch.native), typeof(batch.ops), typeof(position), typeof(long)}
        @test last(only(code_typed(ReactiveKernels.RuntimeGeneratedFunctions.generated_callfunc,
                                   argtypes))) === Matrix{Float64}
        short_bytes = _nested_position_bytes(batch, position, short)
        long_bytes = _nested_position_bytes(batch, position, long)
        storage_growth = reuse ? 0 : output_growth + lane_growth
        @test long_bytes - short_bytes <= storage_growth + accounting_slack
        # Negative control: even one extra lane copy from the same reader
        # must violate the guard in both ownership modes.
        short_copy = _nested_position_bytes(_nested_position_copy, batch, position, short)
        long_copy = _nested_position_bytes(_nested_position_copy, batch, position, long)
        @test long_copy - short_copy > storage_growth + accounting_slack
        println("NESTED_SCAN_ALLOC reuse=", reuse, " short=", short_bytes,
                " long=", long_bytes, " lane_growth=", lane_growth,
                " output_growth=", output_growth, " copy_growth=", long_copy - short_copy,
                " accounting_slack=", accounting_slack)
        @test batch(position, Float64[]) == reshape(position.group.response.seed, 1, :)
        @test position == before
        bad = (; group=(; response=(; seed=[1.0], rate=[0.5, -0.5, 2.0]),
                          unused=[4, 5, 6]), unused=position.unused)
        @test_throws DimensionMismatch batch(bad, short)
    end
end

@testset "nested tuple projection preserves layouts and dynamic fields" begin
    @kernel nested_record_sum(position) = begin
        result = sum(position.payload[1].values) + position.payload[2]
    end
    values = reshape(collect(1.0:12.0), 4, 3)
    tree = (; payload=((; values), [1, 2, 3]))
    expected = vec(sum(values; dims=1)) .+ [1, 2, 3]
    broad = NamedTuple{(:payload,),Tuple{Any}}((tree.payload,))
    for reuse in (false, true)
        batch = vectorize(nested_record_sum; batched=:position, reuse)
        @test batch(tree) == expected
        @test batch(broad) == expected
        retained = copy(batch(tree))
        @test batch((; payload=((; values=values[:, 1:1]), [1]))) == expected[1:1]
        @test retained == expected
    end
end
