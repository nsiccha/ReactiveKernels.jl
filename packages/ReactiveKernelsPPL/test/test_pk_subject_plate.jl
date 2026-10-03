module PKSubjectPlateTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using Test, LinearAlgebra, StaticArrays
const PPL = ReactiveKernelsPPL
const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)
const AUC_CELL = PPL._pk_auc_spec
@kernel auc_objective(q, ends, op_type, op_dt, op_amount, op_interval,
        op_count, op_read_idx, log_F) = begin
    vc = q[1]
    k10 = q[2]
    k12 = q[3]
    k21 = q[4]
    ka = q[5]
    reads = AUC_CELL(ends, op_type, op_dt, op_amount, op_interval,
        op_count, op_read_idx, log_F, vc, k10, k12, k21, ka)
    objective = sum(reads)
    return objective
end

# Independent dense-matrix oracle, with every repeated dose visited explicitly.
function oracle(ends, cols, log_f, parameters; auc=false)
    result = Float64[]
    lo = 1
    for (s, hi) in enumerate(ends)
        vc, k10, k12, k21, ka = parameters[:, s]
        A = Matrix(linear_pk_system_3(vc+k10, vc, vc+k12, vc+k12-k21, ka))
        state = zeros(3)
        given = 0.0
        concentration, exposure = Float64[], Float64[]
        for j in lo:hi
            cols[2][j] > 0 && (state = exp(A*cols[2][j])*state)
            if cols[1][j] == LINEAR_EVENT_READ
                push!(concentration, state[2]/exp(vc))
                push!(exposure, (given-sum(state))/exp(vc+k10))
            else
                effective = cols[3][j]*exp(log_f[j])
                count = cols[1][j] == LINEAR_EVENT_DOSE ? 1 : cols[5][j]
                for d in 1:count
                    d > 1 && (state = exp(A*cols[4][j])*state)
                    state[1] += effective
                    given += effective
                end
            end
        end
        append!(result, concentration)
        auc && append!(result, exposure)
        lo = hi+1
    end
    result
end
columns(s) = (s.op_type, s.op_dt, s.op_amount, s.op_interval, s.op_count, s.op_read_idx)
function fixture(G)
    subj = [s for s in 1:G for _ in 1:s+1]
    time = [t for s in 1:G for t in range(0.0, 16.0; length=s+1)]
    ds = repeat(collect(1:G); inner=5)
    dt = repeat([0.0, 2.0, 4.0, 6.0, 8.0], G)
    da = [2.0+0.3s for s in ds]
    build_linear_pk_schedule(subj, time, ds, dt, da)
end
finite_gradient(f, x) = [(f(x + [i==j ? 1e-5 : 0.0 for i in eachindex(x)]) -
    f(x - [i==j ? 1e-5 : 0.0 for i in eachindex(x)]))/2e-5 for j in eachindex(x)]

@testset "PK subject plate: ragged reads, AUC and independent subjects" begin
    for G in (1, 3, 8)
        s = fixture(G)
        cols = columns(s)
        q = log.([10.0, 0.1, 0.2, 0.3, 0.5])
        parameters = repeat(q, 1, G)
        parameters[1, :] .+= 0.4 .* (0:G-1) ./ max(G-1, 1)
        f = [0.04sin(j) for j in eachindex(s.op_type)]
        f[s.op_type .== LINEAR_EVENT_READ] .= NaN  # the READ arm never reads F
        saved = deepcopy((s, f, parameters))
        args = (PPL.SubjectSlice(f), (PPL.SubjectScalar(parameters[i, :]) for i in 1:5)...)
        for (cell, auc) in ((linear_pk_read_locs, false), (linear_pk_read_locs_auc, true))
            value = PPL._cell_over_subjects(cell, s.op_ends, cols, args)
            @test value ≈ oracle(s.op_ends, cols, f, parameters; auc) rtol=2e-11 atol=1e-12
            @test length(value) == sum(s.n_reads) * (auc ? 2 : 1)
        end
        @test isequal((s, f, parameters), saved)
    end
end

@testset "PK subject plate: empty cells and read-before-dose" begin
    emptycols = (Int[], Float64[], Float64[], Float64[], Int[], Int[])
    q = log.([10.0, 0.1, 0.2, 0.3, 0.5])
    for ends in (Int[], [0], [0, 0])
        @test isempty(PPL.linear_pk_read_locs_over_subjects(ends, emptycols..., q...))
        @test isempty(PPL.linear_pk_read_locs_auc_over_subjects(ends, emptycols...,
            PPL.SubjectSlice(Float64[]), q...))
    end
    s = build_linear_pk_schedule([1], [0.0], [1], [0.0], [2.0])
    @test PPL.linear_pk_read_locs_auc_over_subjects(s.op_ends, columns(s)...,
        PPL.SubjectSlice([NaN, 0.0]), q...) == [0.0, 0.0]
    # A subject with no reads still executes its doses; packing drops its empty block.
    cols = ([LINEAR_EVENT_DOSE, LINEAR_EVENT_READ], [0.0, 0.0], [2.0, 0.0],
        [0.0, 0.0], [1, 0], [0, 1])
    @test PPL.linear_pk_read_locs_auc_over_subjects([1, 2], cols...,
        PPL.SubjectSlice([0.0, NaN]), q...) == [0.0, 0.0]
end
end
