# Focused synthetic PK subject-plate reproducer. The historical upstream
# while/nested-if reverse defects are fixed in Reactant 0.2.289+. This ordinary
# graph currently fails compilation at fixed-size system-matrix batching,
# before the retained event scan. No private diagnostic toggle is needed.
using ReactiveKernels, ReactiveKernelsPPL, Reactant, Enzyme, DifferentiationInterface
const PK_READS = ReactiveKernelsPPL._pk_auc_spec
@kernel pk_objective(q::Vector{Float64}, ends, op_type, op_dt, op_amount,
        op_interval, op_count, op_read_idx, log_F) = begin
    log_Vc = q[1]
    log_k10 = q[2]
    log_k12 = q[3]
    log_k21 = q[4]
    log_ka = q[5]
    reads = PK_READS(ends, op_type, op_dt, op_amount, op_interval, op_count,
        op_read_idx, log_F, log_Vc, log_k10, log_k12, log_k21, log_ka)
    objective = sum(reads)
    return objective
end
s = build_linear_pk_schedule([1, 1, 2, 2], [96., 120., 0., 5.],
    [1, 1, 1, 1, 2], [0., 24., 48., 72., 0.], [100., 100., 100., 100., 50.])
names = (:op_type, :op_dt, :op_amount, :op_interval, :op_count, :op_read_idx)
cols = NamedTuple{names}(Tuple(getproperty(s, n) for n in names))
k = prepare(pk_objective; bound=merge((ends=s.op_ends,
    log_F=zeros(length(s.op_type))), cols))
q = log.([10., .1, .2, .3, .5])
ad = prepare_ad(k, AutoEnzyme(; mode=Enzyme.Reverse), q; active=:q)
value, gradient = ReactiveKernels.ad_value_and_gradient!(ad, similar(q), q)
println("native value: ", value, ", gradient: ", gradient)
flush(stdout)
rq = Reactant.to_rarray(q)
compiled = compile_ad_value_and_gradient(ad, rq)
rvalue, rgradient = compiled(rq)
@assert Float64(rvalue) ≈ value
@assert Array(rgradient) ≈ gradient
