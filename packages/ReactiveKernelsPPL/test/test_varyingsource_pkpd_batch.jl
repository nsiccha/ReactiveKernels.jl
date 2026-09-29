using DifferentiationInterface: AutoEnzyme, Constant, gradient
import Enzyme
using InteractiveUtils: code_llvm
using ReactiveKernelsPPL, Test

function _vs_full_batch_point()
    q = _vs_pkpd_point()
    return vcat(q[1:13], [0.,.05,.15,.2,-.1], [0.,-.02,-.1,.05,.1],
        [-.1,-.2,.05,0.,-.15], q[23:end])
end

function _vs_full_batch(q, s)
    n_subjects = length(s.obs_ends)
    logs = ntuple(i -> ReactiveKernelsPPL.SubjectScalar(q[i] .+
        0.01i .* collect(0:(n_subjects-1))), Val(13))
    return varyingsource_pkpd_read_locs_over_subjects(s, q[14:18], q[19:23],
        q[24:28], q[34:37], q[29], q[30], exp(q[31]), exp(q[32]), exp(q[33]),
        q[42:44], exp(q[38]), exp(q[39]), q[45:47], exp(q[40]), exp(q[41]),
        0., 24., logs...)
end

_vs_full_batch_objective(q,s) = dot(sin.(collect(1:s.obs_ends[end])),_vs_full_batch(q,s))

function _vs_full_batch_oracle(q,s;subject_logs=nothing)
    out = zeros(s.obs_ends[end])
    block = ReactiveKernelsPPL._vs_block
    for subject in eachindex(s.obs_ends)
        dr, rr = block(s.dose_ends,subject), block(s.reference_ends,subject)
        rows = s.dose_index[dr]
        mods = [vcat(q[start .+ rows],zeros(3-length(rows))) for start in (13,18,23)]
        logs = subject_logs === nothing ? [q[i]+0.01i*(subject-1) for i in 1:13] :
            subject_logs[:,subject]
        localq = vcat(logs,mods...,q[29:end])
        b2, b3 = block(s.pd2_step_ends,subject), block(s.pd3_step_ends,subject)
        d = (; pk=(; n_times=length(rr),dose=s.dose_amount[dr],
            treatment=s.treatment_map[dr],lags=s.unique_dts[block(s.lag_ends,subject)],
            indices=s.concentration_idxs[block(s.concentration_ends,subject)],
            dosing_indices=s.dosing_time_idxs[dr]),
            assay=s.assay[block(s.obs_ends,subject)], pk_idxs=s.pk_idxs[block(s.pk_ends,subject)],
            pd2_idxs=s.pd2_idxs[block(s.pd2_write_ends,subject)],
            pd3_idxs=s.pd3_idxs[block(s.pd3_write_ends,subject)],
            dts=Float64[],centers=Int[],pd2_dts=s.pd2_dts[b2],pd3_dts=s.pd3_dts[b3],
            pd2_centers=s.pd2_center_idxs[b2],pd3_centers=s.pd3_center_idxs[b3],
            placebo_times=s.placebo_time[block(s.placebo_ends,subject)])
        out[block(s.obs_ends,subject)] = _vs_pkpd_oracle(localq,d)
    end
    return out
end

function _vs_initial_batch(q,s)
    absent = ReactiveKernelsPPL.SubjectScalar(Float64[])
    return varyingsource_pkpd_read_locs_over_subjects(s,
        absent,absent,absent,Float64[],absent,absent,absent,absent,absent,
        Float64[],absent,absent,Float64[],absent,absent,absent,absent,
        absent,absent,absent,absent,q[1],absent,absent,absent,
        absent,absent,absent,absent,absent)
end

@testset "varyingsource full subject batch: independent locations and reverse" begin
    d = _vs_pkpd_schedule_fixture()
    s, q = _vs_build_full(d), _vs_full_batch_point()
    want = _vs_full_batch_oracle(q,s)
    @test _vs_full_batch(q,s) ≈ want rtol=5e-9 atol=2e-8
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    @test gradient(_vs_full_batch_objective,backend,q,Constant(s)) ≈
        _transit_fd_gradient(p -> dot(sin.(collect(1:10)),_vs_full_batch_oracle(p,s)),q) rtol=5e-6 atol=2e-6
    @test _vs_full_batch(q,s)[7] == exp(q[5]+0.05)
    order = [8,1,2,9,3,7,4,5,10,6]
    shuffled = _vs_build_full(merge(d,(; subject=d.subject[order],time=d.time[order],assay=d.assay[order])))
    @test _vs_full_batch(q,shuffled)[shuffled.obs_map] ≈ want[s.obs_map][order]
    initial = build_varyingsource_pkpd_schedule([1],[0.],[2],Int[],Float64[],Float64[],Int[],Float64[])
    @test _vs_initial_batch([log(120.)],initial) == [exp(log(120.))]
    initial_objective(p,s) = sum(_vs_initial_batch(p,s))
    @test gradient(initial_objective,backend,[log(120.)],Constant(initial)) ≈ [120.]
end

@testset "full native cell keeps runtime subject and grid loops" begin
    d = _vs_pkpd_schedule_fixture()
    s, q = _vs_build_full(d), _vs_full_batch_point()
    large = _vs_build_full((; subject=reduce(vcat,[d.subject .+ 3i for i in 0:9]),
        time=repeat(d.time,10),assay=repeat(d.assay,10),
        dose_subject=reduce(vcat,[d.dose_subject .+ 3i for i in 0:9]),
        dose_time=repeat(d.dose_time,10),dose_amount=repeat(d.dose_amount,10),
        treatment=repeat(d.treatment,10),discretization=d.discretization))
    ir = String[]
    for schedule in (s,large)
        io = IOBuffer()
        code_llvm(io,_vs_full_batch,Tuple{typeof(q),typeof(schedule)};debuginfo=:none)
        push!(ir,String(take!(io)))
    end
    normalized = [replace(v,r"(?<=_)\d+(?=\"?\()"=>"JIT") for v in ir]
    @test normalized[1] == normalized[2]
    @test occursin(" phi i64 ",ir[1]) && occursin("br i1",ir[1])
end
