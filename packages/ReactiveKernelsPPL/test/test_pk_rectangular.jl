module PKRectangularTests
using ReactiveKernels, ReactiveKernelsPPL, Reactant, Test
const RKP = ReactiveKernelsPPL

@testset "rectangular PK recurrence" begin
    previous_mode = RKP._rectangular_pk_enabled[]
    try
    RKP._rectangular_pk_enabled[] = true
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
    rlp = Reactant.to_rarray(lp)
    hlo = repr(Reactant.@code_hlo optimize=false f(rlp))
    @test count("stablehlo.while", hlo) == 2
    println("PK_RECT_HLO bytes=", sizeof(hlo), " whiles=", count("stablehlo.while", hlo))
    compiled = Reactant.@compile f(rlp)
    @test Array(compiled(rlp)) ≈ expected rtol=1e-9
    finally
        RKP._rectangular_pk_enabled[] = previous_mode
    end
end
end
