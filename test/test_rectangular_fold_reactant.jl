module RectangularFoldReactantTests
using ReactiveKernels, Reactant, Test
import Enzyme
const RK = ReactiveKernels

function segmented_sum(q, reset, x)
    step = function(carry, row, q)
        total = ifelse(row[1], zero(carry.total), carry.total) +
            row[2] * RK._tensorized_getindex(q, 1)
        out = RK._tensorized_setindex(carry.out, total, carry.i)
        (total=total, out=out, i=carry.i + 1)
    end
    carry = (total=0.0, out=zeros(length(x)), i=1)
    RK._rectangular_fold(step, carry, (reset, x), (q,), q).out
end

@testset "rectangular fold retains bound data and fixed carry buffers" begin
    q = [2.0]
    for n in (4, 40)
        reset = [isodd(i) for i in 1:n]
        x = collect(1.0:n)
        f = q -> segmented_sum(q, reset, x)
        expected = f(q)
        rq = Reactant.to_rarray(q)
        hlo = repr(Reactant.@code_hlo optimize=false f(rq))
        @test count("stablehlo.while", hlo) == 1
        compiled = Reactant.@compile f(rq)
        @test Array(compiled(rq)) == expected
        @test Array(rq) == q
        loss = q -> sum(f(q))
        gradient = q -> only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(loss), q))
        native_grad = gradient(q)
        compiled_grad = Reactant.@compile gradient(rq)
        @test Array(compiled_grad(rq)) ≈ native_grad
    end
    @test isempty(segmented_sum(q, Bool[], Float64[]))
    @test_throws DimensionMismatch segmented_sum(q, [true], [1.0, 2.0])
    empty_fold(q) = RK._rectangular_fold((c, row) -> c + row[1],
        sum(q), (Float64[],), (), q)
    rq = Reactant.to_rarray(q)
    compiled_empty = Reactant.@compile empty_fold(rq)
    @test Float64(compiled_empty(rq)) == sum(q)
end

@testset "recurrence branches are lazy" begin
    positive(x) = sqrt(x)
    negative(x) = sqrt(-x)
    f(q) = RK._recurrence_branch(sum(q) > 0, positive, negative, (sum(q),))
    rq = Reactant.to_rarray([4.0])
    compiled = Reactant.@compile f(rq)
    @test Float64(compiled(rq)) == 2.0
    @test Float64(compiled(Reactant.to_rarray([-9.0]))) == 3.0
    grad(q) = only(Enzyme.gradient(Enzyme.Reverse, f, q))
    compiled_grad = Reactant.@compile grad(rq)
    @test Array(compiled_grad(rq)) == [0.25]
    @test Array(compiled_grad(Reactant.to_rarray([-9.0]))) ≈ [-1/6]
end
end
