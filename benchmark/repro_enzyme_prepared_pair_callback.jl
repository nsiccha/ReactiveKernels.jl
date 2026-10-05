# Generic RK reproducer: an opaque prepared pair callback with bound locations.
# Julia 1.10.12 / Enzyme 0.13.210 rejects its readonly storage during reverse;
# the assigned graph control passes without activity annotations.
using ReactiveKernels, Test
import Enzyme
using DifferentiationInterface: AutoEnzyme
@kernel coordinates(x) = begin
    locations = Float64.(x)
    left = locations
    right = reshape(locations, 1, :)
    return left, right
end
@kernel pair_grid(x, scale) = begin
    left, right = coordinates(x)
    matrix = plate(left, right, Ref(scale)) do a, b, s
        (a + 2b) * s
    end
    return matrix
end
const callback = prepare(pair_grid)
ordinary_grid(x, scale) = callback(x, scale)
@kernel direct_loss(x, q) = begin
    matrix = pair_grid(x, q[1])
    result = sum(matrix)
    return result
end
@kernel opaque_loss(x, q) = begin
    matrix = ordinary_grid(x, q[1])
    result = sum(matrix)
    return result
end
x=[-0.4,0.2,0.8]; q=[0.7]
backend=AutoEnzyme(;mode=Enzyme.Reverse)
for (label,spec) in ((:direct,direct_loss),(:opaque,opaque_loss))
    kernel=prepare(spec;bound=(;x))
    ad=prepare_ad(kernel,backend,q;active=:q)
    println(label, " primal=",kernel(q));flush(stdout)
    try
        v,g=ad_value_and_gradient(ad,q)
        @test v ≈ 3length(x)*sum(x)*q[1]
        @test g ≈ [3length(x)*sum(x)]
        println(label," reverse=",g)
    catch err
        println(label," error type=",typeof(err))
        println(first(split(sprint(showerror,err),'\n')))
        nameof(typeof(err)) === :EnzymeRuntimeActivityError || rethrow()
        @test_broken false  # exact error above; transparent graph control succeeds
    end
    flush(stdout)
end
