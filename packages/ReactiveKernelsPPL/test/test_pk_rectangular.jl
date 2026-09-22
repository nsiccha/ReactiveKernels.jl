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
    # Repeat the bound ragged schedule: subjects and operations both grow,
    # while the traced recurrence body must occur exactly once.
    nops = length(sched.op_type)
    tiled_ends = vcat((sched.op_ends .+ k * nops for k in 0:2)...)
    tiled_cols = map(c -> repeat(c, 3), cols)
    function tiled(lp)
        cellargs = (RKP.SubjectSlice(zeros(3nops)),
            ntuple(i -> ReactiveKernels._tensorized_getindex(lp, i), 5)...)
        RKP.linear_pk_read_locs_auc_over_subjects(tiled_ends, tiled_cols..., cellargs...)
    end
    tiled_hlo = repr(Reactant.@code_hlo optimize=false tiled(rlp))
    @test count("stablehlo.while", tiled_hlo) == 2
    opcount(hlo) = length(collect(eachmatch(r"stablehlo\.\w+", hlo)))
    @test opcount(tiled_hlo) == opcount(hlo)
    println("PK_RETAINED subjects=", (length(sched.op_ends), length(tiled_ends)),
        " ops=", (opcount(hlo), opcount(tiled_hlo)))
    tiled_compiled = Reactant.@compile tiled(rlp)
    @test Array(tiled_compiled(rlp)) ≈ repeat(expected, 3) rtol=1e-9
    finally
        RKP._rectangular_pk_enabled[] = previous_mode
    end
end

@testset "standalone dose power retains its bit loop" begin
    counts = Int[]
    for n in (3, 31)
        function f(q)
            a = ReactiveKernels._tensorized_getindex(q, 1)
            A = (a, 0.0, 0.0, 0.0, 0.0, a, 0.0, 0.0,
                0.0, 0.0, a, 0.0, 1.0, 0.0, 0.0, 1.0)
            RKP._pk_matpow4(A, n)[13]
        end
        q = [0.9]
        rq = Reactant.to_rarray(q)
        hlo = repr(Reactant.@code_hlo optimize=false f(rq))
        @test count("stablehlo.while", hlo) == 1
        push!(counts, length(collect(eachmatch(r"stablehlo\.\w+", hlo))))
        println("PK_POWER_HLO exponent=", n, " ops=", last(counts))
        compiled = Reactant.@compile f(rq)
        @test Float64(compiled(rq)) ≈ sum(0.9^j for j in 0:n-1) rtol=1e-12
        @test Float64(compiled(rq)) ≈ f(q) rtol=1e-12
    end
    @test all(==(first(counts)), counts)
end
end
