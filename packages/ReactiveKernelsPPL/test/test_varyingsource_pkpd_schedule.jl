using ReactiveKernelsPPL, Test

function _vs_pkpd_schedule_fixture()
    return (; subject=[1,1,1,1,1,1,2,3,3,3],
        time=[12.,0.,14.,6.,2.,14.,0.,0.,4.,9.],
        assay=[1,2,2,1,3,3,2,1,3,3],
        dose_subject=[1,3,1,1,3], dose_time=[0.,0.,4.,8.,0.],
        dose_amount=[10000.,15000.,20000.,40000.,25000.],
        treatment=[101,302,101,203,404],
        discretization=[0.,.5,1.,2.,4.,8.,12.,20.])
end

_vs_build_full(d) = build_varyingsource_pkpd_schedule(d.subject, d.time,
    d.assay, d.dose_subject, d.dose_time, d.dose_amount, d.treatment, d.discretization)

@testset "varyingsource raw full-grid contract and source boundaries" begin
    d = _vs_pkpd_schedule_fixture()
    s = _vs_build_full(d)
    # These boundary products were checked against the actual original
    # _vs_subject_design at Bruno 896137dd, including the equal-time doses.
    @test s.reference_ends == [18,18,25]
    @test s.obs_ends == [6,7,10]
    @test s.n_reads == [6,1,3]
    @test s.n_reads_total == 10
    @test s.dose_index == [1,3,4,2,5]
    @test s.treatment_map == [1,1,2,1,2]
    @test s.dose_ends == [3,3,5]
    @test s.concentration_ends == [54,54,68]
    @test s.pk_ends == [2,2,3]
    @test s.pk_idxs == [17,10,1]
    @test s.pd2_idxs == [1,17,1]
    @test s.pd3_idxs == [5,17,7,9]
    @test s.pd2_step_ends == [16,16,16]
    @test s.pd3_step_ends == [16,16,24]
    @test s.pd3_dts[17:24] == [0.,0.,.5,.5,1.,2.,4.,1.]
    @test s.placebo_ends == [32,32,40]
    @test s.placebo_time[33:40] == [0.,0.,.25,.75,1.5,3.,6.,8.5]
    @test s.assay[s.obs_map] == d.assay
    # Interleave subjects while retaining each PD stream's chronology.
    order = [8,1,2,9,3,7,4,5,10,6]
    shuffled = _vs_build_full(merge(d, (; subject=d.subject[order],
        time=d.time[order], assay=d.assay[order])))
    @test shuffled.assay[shuffled.obs_map] == d.assay[order]
    for field in (:reference_ends, :dose_index, :unique_dts, :concentration_idxs,
            :pd2_idxs, :pd2_dts, :pd3_idxs, :pd3_dts, :placebo_time)
        @test getproperty(shuffled,field) == getproperty(s,field)
    end
    # Initial-state-only PD requires no fake PK reference or placebo row.
    initial = build_varyingsource_pkpd_schedule([1],[0.],[2],Int[],Float64[],
        Float64[],Int[],Float64[])
    @test initial.reference_ends == [0]
    @test initial.unique_dts == [0.]
    @test initial.pd2_idxs == [1]
    @test isempty(initial.placebo_time)
    @test initial.obs_map == [1]
    @test_throws "strictly increase" _vs_build_full(merge(d,(; time=[12.,0.,0.,d.time[4:end]...])))
    @test_throws "assay codes" _vs_build_full(merge(d,(; assay=[4,d.assay[2:end]...])))
    @test_throws "nondecreasing" _vs_build_full(merge(d,(; dose_time=[4.,0.,0.,8.,0.])))
    @test_throws "discretization" _vs_build_full(merge(d,(; discretization=[1.,0.])))
    @test_throws "contiguous" _vs_build_full(merge(d,(; subject=replace(d.subject,2=>4))))
    @test_throws "observed subjects" _vs_build_full(merge(d,(; dose_subject=[4,d.dose_subject[2:end]...])))
    # The source helper stalls when its exhausted fallback lag is behind the
    # next PD measurement. Reject that corner instead of entering its loop.
    @test_throws "would not advance" build_varyingsource_pkpd_schedule([1],[2.],[2],
        [1],[-10.],[1.],[1],Float64[])
end
