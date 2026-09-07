using ReactiveKernelsPPLExamples.Rate2Example
using ReactiveKernelsDistributionKernels.DistributionKernelSources: beta, binomial
using LogExpFunctions: logistic, log1pexp
@testset "PPL graph — Rate_2 (posteriordb)" begin
    a = evaluate_rate_2_source(); @test a.source == strip(RATE_2_SOURCE, '\n')
    @test a.output == Base.invokelatest(a.kernel, Tuple(a.inputs)...)
    q = [0.1, -0.1]; t1 = logistic(q[1]); t2 = logistic(q[2])
    bl(n,k,t) = log(float(Base.binomial(n,k))) + k*log(t) + (n-k)*log1p(-t)
    jac = (-log1pexp(-q[1])-log1pexp(q[1])) + (-log1pexp(-q[2])-log1pexp(q[2]))
    ref = bl(RATE2_N1,RATE2_K1,t1) + bl(RATE2_N2,RATE2_K2,t2) + jac
    _, _, likelihood, posterior = prepare(a.model; have=(:unconstrained,:n1,:n2,:k1,:k2), want=(:parameters,:prior,:likelihood,:posterior))(q, RATE2_N1, RATE2_N2, RATE2_K1, RATE2_K2)
    @test posterior ≈ ref
end
