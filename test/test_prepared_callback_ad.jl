module PreparedCallbackADTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

@kernel pair_terms(x, scale, rate) = begin
    right = reshape(x, 1, :)
    terms = plate(x, right) do a, b
        delta = a - b
        scale * exp(-rate * delta * delta)
    end
    return terms
end
const PAIR_TERMS = prepare(pair_terms)
ordinary_pair_terms(x, scale, rate) = PAIR_TERMS(x, scale, rate)

@kernel callback_sum(q, x) = begin
    scale = exp(q[1])
    rate = exp(q[2])
    terms = ordinary_pair_terms(x, scale, rate)
    total = sum(terms)
    return total
end

# Independent scalar algebra, with no RK call or generated body.
function reference(q, x)
    scale, rate = exp(q[1]), exp(q[2])
    total = zero(scale)
    dq, dx = zero(q), zero(x)
    for i in eachindex(x), j in eachindex(x)
        delta = x[i] - x[j]
        term = scale * exp(-rate * delta^2)
        total += term
        dq[1] += term
        dq[2] -= rate * delta^2 * term
        dx[i] -= 2 * rate * delta * term
        dx[j] += 2 * rate * delta * term
    end
    total, dq, dx
end

@testset "ordinary callbacks invoke prepared kernels under native Reverse" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const)
    kernel = prepare(callback_sum)
    original_ast = deepcopy(code_expr(kernel))
    for T in (Float32, Float64), n in (1, 4, 9)
        x = collect(range(T(-0.7); step=T(0.2), length=n))
        saved_x = copy(x)
        initial = T[0.2, -0.3]
        ad = prepare_ad(kernel, backend, initial, x; active=:q)
        joint = prepare_ad(kernel, backend, initial, x; active=(:x, :q))
        pullback = prepare_ad_pullback(kernel, backend, one(T), initial, x;
                                      active=:q)
        for q in (initial, T[-0.4, 0.1])
            saved_q = copy(q)
            expected, dq, dx = reference(q, x)
            @test kernel(q, x) ≈ expected
            gradient = similar(q)
            value, returned = ad_value_and_gradient!(ad, gradient, q, x)
            @test value ≈ expected
            @test returned === gradient
            @test gradient ≈ dq
            joint_value, (gx, gq) = ad_value_and_gradient(joint, q, x)
            @test joint_value ≈ expected
            @test gx ≈ dx rtol=sqrt(eps(T)) atol=20eps(T)
            @test gq ≈ dq
            @test ad_pullback(pullback, T(1.7), q, x) ≈ T(1.7) .* dq
            @test q == saved_q
            @test x == saved_x
        end
    end
    @test code_expr(kernel) == original_ast
end
end
