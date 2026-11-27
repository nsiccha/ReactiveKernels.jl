module PositionRecordReverseTests

using Test
using ReactiveKernels
using Enzyme

@kernel record_child(x, offset) = begin
    total = sum(abs2, x) + offset
    record = (; tail = x[2:end], total = total)
    return record
end
@kernel record_parent(x, offset) = begin
    record = record_child(x, offset)
    return record
end

record_loss(x, reader, args) = sum(reader(x, args...).total)

@kernel function paired_plates(p::Vector{Float64}, t::Vector{Int})
    left = plate(t) do x
        exp(-p[1] * x) * (1 + p[2] * x)
    end
    right = plate(t) do x
        sin(p[1] + x) * p[2]
    end
    return left, right
end

function paired_loss(p, reader, t)
    left, right = reader(p, t)
    sum(left) + sum(right)
end

@testset "ordinary Reverse of position plates with inferred output types" begin
    reader = vectorize(prepare(paired_plates); batched=:p)
    for t in ([2, 4, 7, 11], [1], Int[]), width in (1, 7)
        p = reshape(collect(range(0.1, 0.9; length=2 * width)), 2, width)
        dp = zero(p)
        expected = zero(p)
        for i in axes(p, 2), x in t
            a, b = p[1, i], p[2, i]
            decay = exp(-a * x)
            expected[1, i] += -x * decay * (1 + b * x) + cos(a + x) * b
            expected[2, i] += x * decay + sin(a + x)
        end
        Enzyme.autodiff(Enzyme.Reverse, paired_loss, Enzyme.Active,
                        Enzyme.Duplicated(p, dp), Enzyme.Const(reader), Enzyme.Const(t))
        @test dp ≈ expected
    end
end

@kernel function shared_array_work(p::Vector{Float64}, t::Vector{Float64}, indices::Vector{Int})
    wave = plate(t) do x
        sin(x)
    end
    energy = plate(t) do x
        x * x
    end
    combined = p[1] .* wave .+ p[2] .* energy
    selected = combined[indices]
    return selected
end
shared_array_loss(p, t, indices, reader) = sum(reader(p, t, indices))

@testset "ordinary Reverse through shared plates and reusable array work" begin
    reader = vectorize(prepare(shared_array_work); batched=:p)
    for count in (0, 1, 7, 73), width in (1, 3)
        t = [0.1 * i for i in 1:count]
        indices = count == 0 ? Int[] : [count, 1, count]
        p = reshape(collect(range(0.1, 0.9; length=2width)), 2, width)
        expected = [p[1,j] * sin(t[i]) + p[2,j] * t[i]^2 for i in indices, j in axes(p,2)]
        @test reader(p,t,indices) ≈ expected
        dp, dt = zero(p), zero(t)
        expected_p, expected_t = zero(p), zero(t)
        for j in axes(p,2), i in indices
            expected_p[1,j] += sin(t[i])
            expected_p[2,j] += t[i]^2
            expected_t[i] += p[1,j] * cos(t[i]) + 2p[2,j] * t[i]
        end
        Enzyme.autodiff(Enzyme.Reverse, shared_array_loss, Enzyme.Active,
            Enzyme.Duplicated(p,dp), Enzyme.Duplicated(t,dt),
            Enzyme.Const(indices), Enzyme.Const(reader))
        @test dp ≈ expected_p
        @test dt ≈ expected_t
    end
end

@testset "ordinary Reverse of composed position records" begin
    # The empty/nonempty tail reaches ownership checks and heterogeneous
    # record storage even though the loss reads only total. Enzyme dogfood
    # 0d42dc42 repairs the ownership walk; b916d65b repairs tuple foreach.
    # Keep ordinary Reverse: no runtime activity or replacement AD rule.
    for bound_offset in (false, true), offset in (0.5, -1.25)
        bound = bound_offset ? (; offset) : NamedTuple()
        scalar = prepare(record_parent; bound)
        reader = vectorize(scalar; batched = :x)
        args = bound_offset ? () : (offset,)
        expression = code_expr(reader)
        for width in (1, 4), count in (1, 7), shift in (0.0, -2.0)
            x = reshape(collect(1.0:(width * count)), width, count) .+ shift
            original = copy(x)
            expected = (; tail = x[2:end, :],
                        total = vec(sum(abs2, x; dims = 1)) .+ offset)
            retained = reader(x, args...)
            @test retained == expected

            dx = zero(x)
            Enzyme.autodiff(Enzyme.Reverse, record_loss, Enzyme.Active,
                Enzyme.Duplicated(x, dx), Enzyme.Const(reader), Enzyme.Const(args))
            @test dx == 2 .* original
            @test x == original
            @test retained == expected

            column = copy(x[:, 1])
            dcolumn = zero(column)
            Enzyme.autodiff(Enzyme.Reverse, record_loss, Enzyme.Active,
                Enzyme.Duplicated(column, dcolumn), Enzyme.Const(scalar), Enzyme.Const(args))
            @test dcolumn == 2 .* original[:, 1]
            @test column == original[:, 1]

            # Later evaluations and changes to the input must preserve an
            # owning reader's published record, including the sliced tail.
            x .+= 1.0
            next_result = reader(x, args...)
            @test next_result.total == vec(sum(abs2, x; dims = 1)) .+ offset
            @test retained == expected
            @test code_expr(reader) === expression
        end
    end
end

end
