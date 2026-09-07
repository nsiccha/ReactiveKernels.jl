using ReactiveKernelsPPLExamples.Rate4Example
using LogExpFunctions: logistic, log1pexp
@testset "PPL graph — Rate_4 (posteriordb)" begin
    a = evaluate_rate_4_source(); @test a.source == strip(RATE_4_SOURCE, '\n')
    @test a.output == Base.invokelatest(a.kernel, Tuple(a.inputs)...)
    q = [0.1, 0.0]; t = logistic(q[1])
    ref = (log(float(Base.binomial(RATE4_N,RATE4_K))) + RATE4_K*log(t) + (RATE4_N-RATE4_K)*log1p(-t)) +
          (-log1pexp(-q[1])-log1pexp(q[1])) + (-log1pexp(-q[2])-log1pexp(q[2]))
    _, _, _, posterior = prepare(a.model; have=(:unconstrained,:n,:k), want=(:parameters,:prior,:likelihood,:posterior))(q, RATE4_N, RATE4_K)
    @test posterior ≈ ref
end
