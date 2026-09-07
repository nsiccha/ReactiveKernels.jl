using ReactiveKernelsPPLExamples.Rate3Example
using LogExpFunctions: logistic, log1pexp
@testset "PPL graph — Rate_3 (posteriordb)" begin
    a = evaluate_rate_3_source(); @test a.source == strip(RATE_3_SOURCE, '\n')
    @test a.output == Base.invokelatest(a.kernel, Tuple(a.inputs)...)
    q = [0.2]; t = logistic(q[1]); bl(n,k) = log(float(Base.binomial(n,k))) + k*log(t) + (n-k)*log1p(-t)
    ref = bl(RATE3_N1,RATE3_K1) + bl(RATE3_N2,RATE3_K2) + (-log1pexp(-q[1])-log1pexp(q[1]))
    _, _, _, posterior = prepare(a.model; have=(:unconstrained,:n1,:n2,:k1,:k2), want=(:parameters,:prior,:likelihood,:posterior))(q, RATE3_N1, RATE3_N2, RATE3_K1, RATE3_K2)
    @test posterior ≈ ref
end
