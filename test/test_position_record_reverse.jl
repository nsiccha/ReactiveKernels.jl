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
