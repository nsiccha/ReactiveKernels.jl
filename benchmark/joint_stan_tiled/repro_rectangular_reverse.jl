# Focused reproducer for the Reactant 0.2.285 / EnzymeMLIR reverse failure:
# "had set op which was not a direct descendant". Two subjects, six operations,
# one repeated-dose segment; no PPL generator, likelihood, or joint model.
# With that fixed (EnzymeAD/Enzyme-JAX#3240) it compiles, but the gradient is
# wrong on subject 1's rate parameters until the nested-if adjoint fix
# (repro_nested_if_reverse.jl) is in as well.
# Run in an environment with this checkout, ReactiveKernelsPPL, Reactant, Enzyme.
using ReactiveKernels, ReactiveKernelsPPL, Reactant, Enzyme
const PPL = ReactiveKernelsPPL
PPL._rectangular_pk_enabled[] = true
const schedule = build_linear_pk_schedule([1, 1, 2, 2], [96., 120., 0., 5.],
    [1, 1, 1, 1, 2], [0., 24., 48., 72., 0.], [100., 100., 100., 100., 50.])
const columns = (schedule.op_type, schedule.op_dt, schedule.op_amount,
    schedule.op_interval, schedule.op_count, schedule.op_read_idx)

function pk(q)
    args = ntuple(i -> PPL.SubjectScalar(
        ReactiveKernels._tensorized_getindex(q, [2i-1, 2i])), 5)
    sum(PPL.linear_pk_read_locs_auc_over_subjects(schedule.op_ends, columns...,
        PPL.SubjectSlice(zeros(length(schedule.op_type))), args...))
end

q = repeat(log.([10., .1, .2, .3, .5]); inner=2)
println("native value: ", pk(q))
flush(stdout)
gradient(q) = only(Enzyme.gradient(Enzyme.Reverse, pk, q))
expected_gradient = gradient(q)
println("native gradient: ", expected_gradient)
flush(stdout)
compiled = Reactant.@compile sync=true gradient(Reactant.to_rarray(q))
got = Array(compiled(Reactant.to_rarray(q)))
@assert isapprox(got, expected_gradient; rtol=1e-8) "compiled gradient $got, native $expected_gradient"
println("compiled gradient matches native: ", got)
