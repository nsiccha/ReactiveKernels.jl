module PKSubjectPlateADTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Test
using ..PKSubjectPlateTests: fixture, columns, oracle, finite_gradient, auc_objective
const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)
@testset "PK subject plate: native reverse" begin
    s = fixture(3)
    cols = columns(s)
    q = log.([10.0, 0.1, 0.2, 0.3, 0.5])
    f = [0.02cos(j) for j in eachindex(s.op_type)]
    names = (:op_type, :op_dt, :op_amount, :op_interval, :op_count, :op_read_idx)
    data = merge((ends=s.op_ends,), NamedTuple{names}(cols))
    value = prepare(auc_objective; bound=merge(data, (log_F=f,)))
    ad = prepare_ad(value, BACKEND, q; active=:q)
    reference(q) = sum(oracle(s.op_ends, cols, f, repeat(q, 1, 3); auc=true))
    @test ad_gradient(ad, q) ≈ finite_gradient(reference, q) rtol=2e-6 atol=2e-7
    event_value = prepare(auc_objective; bound=merge(data, (q=q,)))
    event_ad = prepare_ad(event_value, BACKEND, f; active=:log_F)
    g = ad_gradient(event_ad, f)
    event_reference(f) = sum(oracle(s.op_ends, cols, f, repeat(q, 1, 3); auc=true))
    @test g ≈ finite_gradient(event_reference, f) rtol=2e-6 atol=2e-7
    @test all(iszero, g[s.op_type .== LINEAR_EVENT_READ])
end
end
