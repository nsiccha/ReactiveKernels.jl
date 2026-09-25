module PKRectangularTests
using ReactiveKernels, ReactiveKernelsPPL, Test
const RKP = ReactiveKernelsPPL

# Native-only: the Reactant compiles of the rectangular path need
# Reactant-side control-flow support for the StaticArrays-`exp` branches
# (deferred per user direction) — the HLO/compile assertions are removed
# until that lands. The native fold still proves the restructure.
@testset "rectangular PK recurrence (native fold)" begin
    sched = build_linear_pk_schedule(
        [1, 1, 2, 2], [96.0, 120.0, 0.0, 5.0],
        [1, 1, 1, 1, 2], [0.0, 24.0, 48.0, 72.0, 0.0],
        [100.0, 100.0, 100.0, 100.0, 50.0])
    cols = (sched.op_type, sched.op_dt, sched.op_amount,
        sched.op_interval, sched.op_count, sched.op_read_idx)
    lp = log.([10.0, 0.1, 0.2, 0.3, 0.5])
    function f(lp)
        cellargs = (RKP.SubjectSlice(zeros(length(sched.op_type))),
            ntuple(i -> ReactiveKernels._tensorized_getindex(lp, i), 5)...)
        RKP.linear_pk_read_locs_auc_over_subjects(sched.op_ends, cols..., cellargs...)
    end
    expected = f(lp)
    args = (RKP.SubjectSlice(zeros(length(sched.op_type))), lp...)
    @test RKP._pk_rectangular(linear_pk_read_locs_auc, sched.op_ends, cols, args, lp) ≈ expected
    # Same-time read and dose: neither propagation nor a repeated-dose
    # affine map fires; both hoisted tables are `nothing` and the lazy
    # branches must still execute without indexing a missing table.
    zero_sched = build_linear_pk_schedule([1], [0.0], [1], [0.0], [100.0])
    @test all(iszero, zero_sched.op_dt)
    zero_cols = (zero_sched.op_type, zero_sched.op_dt, zero_sched.op_amount,
        zero_sched.op_interval, zero_sched.op_count, zero_sched.op_read_idx)
    zero_dt(lp) = RKP.linear_pk_read_locs_auc_over_subjects(
        zero_sched.op_ends, zero_cols...,
        RKP.SubjectSlice(zeros(length(zero_sched.op_type))),
        ntuple(i -> ReactiveKernels._tensorized_getindex(lp, i), 5)...)
    zargs = (RKP.SubjectSlice(zeros(length(zero_sched.op_type))), lp...)
    @test RKP._pk_rectangular(linear_pk_read_locs_auc, zero_sched.op_ends,
        zero_cols, zargs, lp) ≈ zero_dt(lp)
end
end
