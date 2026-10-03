module PKRectangularTests
using ReactiveKernels, ReactiveKernelsPPL, Test, LinearAlgebra, StaticArrays
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

@testset "PK integer powers and repeated-dose entry points" begin
    A = linear_pk_system_3(log.([1.0, 10.0, 2.0, 20.0, 0.5])...)
    state = SVector(0.4, 0.2, 0.1)
    amount, interval = 2.3, 0.7
    B = RKP._pk_dose_affine(A, amount, interval)
    for count in (Int8(1), Int32(3), Int64(9), UInt32(17))
        # Independent dense propagation visits each individual dose.
        expected = Vector(state)
        transition = exp(Matrix(A) * interval)
        for j in 1:count
            j > 1 && (expected = transition * expected)
            expected[1] += amount
        end
        @test linear_pk_add_regular_doses_3(A, state, amount, interval, count) ≈ expected
        @test ReactiveKernels.traced(linear_pk_add_regular_doses_3,
            A, state, amount, interval, count) ≈ expected
        @test RKP._pk_smat_pow4(B, count) ≈ Matrix(B)^count
        @test ReactiveKernels.traced(RKP._pk_smat_pow4, B, count) ≈ Matrix(B)^count
    end
    @test RKP._pk_smat_pow4(B, 0) == SMatrix{4,4}(I)
    @test ReactiveKernels.traced(RKP._pk_smat_pow4, B, 0) == SMatrix{4,4}(I)
    # The inactive affine/exponential arm must not read invalid values.
    invalid_A = SMatrix{3,3}(fill(NaN, 3, 3))
    expected = linear_pk_add_dose_3(state, amount)
    @test linear_pk_add_regular_doses_3(invalid_A, state, amount, NaN, 1) == expected
    @test ReactiveKernels.traced(linear_pk_add_regular_doses_3,
        invalid_A, state, amount, NaN, 1) == expected
    # Refused: these helpers' existing count/exponent contracts require an
    # integer scalar; accepting a Float64 would change native admission.
    @test_throws MethodError linear_pk_add_regular_doses_3(A, state, amount, interval, 3.0)
    @test_throws MethodError RKP._pk_smat_pow4(B, 3.0)
    @test_throws ArgumentError ReactiveKernels.traced(linear_pk_add_regular_doses_3,
        A, state, amount, interval, 3.0)
    @test_throws ArgumentError ReactiveKernels.traced(RKP._pk_smat_pow4, B, 3.0)
end
end
