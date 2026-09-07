using ReactiveKernelsPPLExamples.Rate5Example
using LogExpFunctions: logistic, log1pexp
@testset "PPL graph — Rate_5 (posteriordb)" begin
    a = evaluate_rate_5_source(); @test a.source == strip(RATE_5_SOURCE, '\n')
    @test a.output == Base.invokelatest(a.kernel, Tuple(a.inputs)...)
    q = [0.0]; t = logistic(q[1]); bl(n,k) = log(float(Base.binomial(n,k))) + k*log(t) + (n-k)*log1p(-t)
    ref = bl(RATE5_N1,RATE5_K1) + bl(RATE5_N2,RATE5_K2) + (-log1pexp(-q[1])-log1pexp(q[1]))
    _, _, _, posterior = prepare(a.model; have=(:unconstrained,:n1,:n2,:k1,:k2), want=(:parameters,:prior,:likelihood,:posterior))(q, RATE5_N1, RATE5_N2, RATE5_K1, RATE5_K2)
    @test posterior ≈ ref
end
