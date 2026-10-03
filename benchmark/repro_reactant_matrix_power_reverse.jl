# Backend-only matrix binary power: retained integer loop and lazy multiply.
# Reactant 0.2.290 compiles the primal but ordinary default reverse fails with
# "WhileOp does not have known iteration count for cache removal". No RK
# lowering, derivative rule, unrolling, or optimizer override is involved.
using Reactant, Enzyme, Test, LinearAlgebra
Reactant.set_default_backend("cpu")

function power_base(q)
    B = reshape(vcat(q[1:1], zeros(3), [0.2], q[2:2], zeros(2),
        [0.0, 0.3, 0.8, 0.0], [0.0, 0.0, 0.1, 1.0]), 4, 4)
    q isa Reactant.TracedRArray ? Reactant.promote_to(Reactant.TracedRArray, B) : B
end

function conditional_product(R, B, e)
    Reactant.@trace if e & 1 == 1
        result = R * B
    else
        result = R
    end
    result
end

function power_loss(q, n)
    B = power_base(q)
    # Dense traced carries isolate the loop without StaticArrays staging.
    R = zero(B) + Matrix{Float64}(I, 4, 4)
    e = n
    Reactant.@trace while e > 0
        R = conditional_product(R, B, e)
        B = B * B
        e = div(e, 2)
    end
    sum(R)
end
gradient(q, n) = only(Enzyme.gradient(Enzyme.Reverse,
    Enzyme.Const(x -> power_loss(x, n)), q))

@testset "backend-only retained matrix integer power" begin
    q = [0.7, 0.9]
    rq = Reactant.to_rarray(q)
    rn = Reactant.to_rarray(Int64(3); track_numbers=true)
    compiled = Reactant.@compile power_loss(rq, rn)
    for n in (0, 1, 3, 9, 17)
        reference = sum(power_base(q)^n)
        @test power_loss(q, n) ≈ reference
        @test Float64(compiled(rq, Reactant.to_rarray(Int64(n); track_numbers=true))) ≈ reference
    end
    hlo = repr(Reactant.@code_hlo optimize=true power_loss(rq, rn))
    @test count("stablehlo.while", hlo) == 1
    @test count("stablehlo.if", hlo) == 1
    h = 1e-6
    finite_gradient = [(e = zeros(2); e[i] = h;
        (power_loss(q + e, 3) - power_loss(q - e, 3)) / (2h)) for i in 1:2]
    @test gradient(q, 3) ≈ finite_gradient rtol=1e-8
    result = try
        reverse = Reactant.@compile gradient(rq, rn)
        Array(reverse(rq, rn))
    catch err
        @test occursin("WhileOp does not have known iteration count for cache removal",
            sprint(showerror, err))
        nothing
    end
    @test_broken result !== nothing && isapprox(result, finite_gradient; rtol=1e-8)
end
