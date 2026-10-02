using DifferentiationInterface: AutoEnzyme
using Distributions: Normal, Exponential, logpdf
import Enzyme
using InteractiveUtils: code_llvm
using ReactiveKernels
using ReactiveKernelsPPL
using Test

const _VS_EMIT_DATA = Set([:subj, :time, :dsubj, :dtime, :damt, :treatment,
    :dose_x, :gp_weights, :age_s, :dv])

function _vs_emit_ast()
    return quote
        sigma ~ Exponential(1.0)
        b_vc ~ Normal(0.0, 1.0)
        b_age ~ Normal(0.0, 1.0)
        b_k10 ~ Normal(0.0, 1.0)
        b_k12 ~ Normal(0.0, 1.0)
        b_k21 ~ Normal(0.0, 1.0)
        b_rate ~ Normal(0.0, 1.0)
        b_mode ~ Normal(0.0, 1.0)
        d_rate ~ Normal(0.0, 1.0)
        d_mode ~ Normal(0.0, 1.0)
        d_f ~ Normal(0.0, 1.0)
        dose_slope ~ Normal(0.0, 1.0)
        conc_slope ~ Normal(0.0, 1.0)
        log_Vc = b_vc .+ b_age .* age_s
        log_k10 = b_k10
        log_k12 = b_k12
        log_k21 = b_k21
        log_rate = b_rate
        log_mode = b_mode
        vs = varyingsource_pk_schedule(obs = (:subj, :time),
            dose = (:dsubj, :dtime, :damt, :treatment))
        @plate conc for s in 1:kernel_nsub_conc
            rate_mod = d_rate .* dose_x
            mode_mod = d_mode .* dose_x
            f_mod = d_f .* dose_x
            reads = varyingsource_pk_read_locs(vs, rate_mod, mode_mod, f_mod,
                gp_weights, dose_slope, conc_slope, log_Vc, log_k10, log_k12,
                log_k21, log_rate, log_mode)
            mu = reads[vs.obs_map]
            dv .~ Normal.(mu, sigma)
            mu
        end
    end
end

function _vs_emit_columns()
    # Ragged, interleaved axes; subject 2 has no doses. Subject 1's third
    # dose first introduces the second treatment; equal-time doses stay split.
    return Dict{Symbol,AbstractVector}(
        :subj => [3, 1, 2, 1, 3, 1, 2, 1],
        :time => [20., 1., 7., 12., 0., 24., 0., 12.],
        :dsubj => [1, 3, 1, 1, 3], :dtime => [0., 0., 4., 8., 0.],
        :damt => [10000., 15000., 20000., 40000., 25000.],
        :treatment => [10, 20, 10, 30, 30], :dose_x => [0., 0.4, 1., 2., -0.3],
        :gp_weights => vec([0.02 -0.03; -0.01 0.04]),
        :age_s => [0.2, -0.3, 0.5], :dv => [2., 3., 0., 10., 0., 4., 0., 10.])
end

function _vs_emit_point(names)
    values = Dict(:sigma => log(2.0), :d_rate => 0.15, :d_mode => -0.1,
        :d_f => -0.2, :dose_slope => 0.1, :conc_slope => -0.15,
        :b_vc => log(100.0), :b_age => 0.1, :b_k10 => log(0.08),
        :b_k12 => log(0.15), :b_k21 => log(0.05), :b_rate => log(0.2),
        :b_mode => 0.0)
    for (i, w) in enumerate(vec([0.02 -0.03; -0.01 0.04]))
        values[Symbol("gp_weights.", i)] = w
    end
    return [values[n] for n in names]
end

function _vs_emit_oracle(u, names, cols)
    p = Dict(zip(names, u))
    weights = haskey(p, Symbol("gp_weights.1")) ?
        [p[Symbol("gp_weights.", i)] for i in 1:4] : cols[:gp_weights]
    mu = zeros(length(cols[:subj]))
    for s in eachindex(cols[:age_s])
        orows = findall(==(s), cols[:subj])
        drows = findall(==(s), cols[:dsubj])
        refs = sort(unique(vcat(cols[:time][orows], cols[:dtime][drows])))
        keys = unique(cols[:treatment][drows])
        treatments = [findfirst(==(cols[:treatment][i]), keys) for i in drows]
        d = _vs_pk_test_data(refs, cols[:dtime][drows], cols[:damt][drows], treatments)
        q = vcat([p[:b_vc] + p[:b_age] * cols[:age_s][s]],
            [p[x] for x in (:b_k10, :b_k12, :b_k21, :b_rate, :b_mode)],
            [p[:dose_slope], p[:conc_slope]],
            p[:d_rate] .* cols[:dose_x][drows], p[:d_mode] .* cols[:dose_x][drows],
            p[:d_f] .* cols[:dose_x][drows], weights, [0.0, 0.0])
        concentration = _vs_pk_oracle(q, d)
        for i in orows
            mu[i] = concentration[searchsortedfirst(refs, cols[:time][i])]
        end
    end
    sigma = exp(p[:sigma])
    likelihood = sum(logpdf.(Normal.(mu, sigma), cols[:dv]))
    prior = logpdf(Exponential(1.0), sigma) +
        sum(logpdf(Normal(0.0, 1.0), v) for (k, v) in p if k !== :sigma)
    return likelihood + prior + p[:sigma], mu
end

function _vs_emit_vector_plan(plan; size = 4)
    kp = only(plan.kernel_plates)
    slices = filter(x -> x[2] !== :gp_weights, kp.slices)
    plate = KernelPlate(kp.result, kp.subjects, kp.timepoints, slices,
        kp.assignments, kp.obs, kp.collected, kp.label, kp.lp_args, kp.schedules)
    return StructuralPlan(plan.responses, plan.predictors, plan.population_priors,
        plan.parameters, plan.assignments, plan.columns, plan.n_obs;
        vector_parameters = [VectorParameter(:gp_weights, :vector_normal,
            (arg1 = 0.0, arg2 = 1.0), size)], kernel_plates = [plate])
end

@testset "varyingsource bound schedule and fail-closed contract" begin
    cols = _vs_emit_columns()
    build(c) = build_varyingsource_pk_schedule(c[:subj], c[:time],
        c[:dsubj], c[:dtime], c[:damt], c[:treatment])
    sched = build(cols)
    @test sched.n_subjects == 3
    @test sched.dose_index == [1, 3, 4, 2, 5]
    @test sched.treatment_map == [1, 1, 2, 1, 2]
    @test sched.dose_ends == [3, 3, 5]
    @test sched.reference_ends == [6, 8, 10]
    @test sched.obs_map == [10, 2, 8, 5, 9, 6, 7, 5]
    @test sched.concentration_ends == [18, 18, 22]
    @test length(sched.dose_amount) == 5
    @test_throws "positive integers" build(merge(cols, Dict(:subj => [1.5; cols[:subj][2:end]])))
    @test_throws "nondecreasing" build(merge(cols, Dict(:dtime => [8., 0., 4., 0., 0.])))
    @test_throws "positive" build(merge(cols, Dict(:damt => zeros(5))))
    unbound = lower_rkppl(_vs_emit_ast(), _VS_EMIT_DATA)
    @test only(only(unbound.kernel_plates).schedules) isa VaryingSourcePKScheduleSpec
    bind(c) = bind_data(unbound, c; dims = Dict(:kernel_nsub_conc => 3))
    bound = bind(cols)
    @test validate_data(bound) === nothing
    @test_throws "dose axis" bind(merge(cols, Dict(:dose_x => [1.])))
    @test_throws "square coefficient" bind(merge(cols, Dict(:gp_weights => ones(3))))
    @test_throws "is reserved for schedule" bind(merge(cols, Dict(:vs_dose_ends => [1, 2, 3])))
    bound.columns[:vs_dose_index][1] = 2
    @test_throws "not the schedule build" validate_data(bound)
end

@testset "varyingsource emitted density and native reverse" begin
    cols = _vs_emit_columns()
    bound = bind_data(lower_rkppl(_vs_emit_ast(), _VS_EMIT_DATA), cols;
        dims = Dict(:kernel_nsub_conc => 3))
    built = build_kernel(bound)
    names = coordinate_names(built.layout)
    u = _vs_emit_point(names)
    want, mu = _vs_emit_oracle(u, names, cols)
    sampler = prepare_query(built, bound, :sampler)
    @test sampler(u) ≈ want rtol = 2e-9 atol = 2e-7
    @test mu[[3, 5, 7]] == zeros(3)
    @test mu[4] == mu[8]
    q = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
    g = zeros(length(u))
    value, _ = sampler_value_and_gradient!(q, g, u)
    @test value ≈ want rtol = 2e-9 atol = 2e-7
    fd = _transit_fd_gradient(w -> _vs_emit_oracle(w, names, cols)[1], u)
    @test g ≈ fd rtol = 5e-6 atol = 2e-5
    @test all(isfinite, g)
end

@testset "varyingsource emitter with active GP coefficients" begin
    cols = _vs_emit_columns()
    plan = _vs_emit_vector_plan(lower_rkppl(_vs_emit_ast(), _VS_EMIT_DATA))
    @test validate_structure(plan) === nothing
    @test_throws "concrete square" validate_structure(_vs_emit_vector_plan(plan; size = 3))
    bound = bind_data(plan, filter(p -> p.first !== :gp_weights, cols);
        dims = Dict(:kernel_nsub_conc => 3))
    built = build_kernel(bound)
    names = coordinate_names(built.layout)
    u = _vs_emit_point(names)
    want, _ = _vs_emit_oracle(u, names, cols)
    q = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
    g = zeros(length(u))
    value, _ = sampler_value_and_gradient!(q, g, u)
    @test value ≈ want rtol = 2e-9 atol = 2e-7
    fd = _transit_fd_gradient(w -> _vs_emit_oracle(w, names, cols)[1], u)
    @test g ≈ fd rtol = 5e-6 atol = 2e-5
    @test all(isfinite, g)
end

_vs_ast_calls(ex, fn) = ex isa Expr ?
    Int(ex.head === :call && !isempty(ex.args) && ex.args[1] === fn) +
        sum(_vs_ast_calls(a, fn) for a in ex.args; init = 0) : 0
_vs_ast_size(ex) = ex isa Expr ? 1 + sum(_vs_ast_size, ex.args; init = 0) : 1

function _vs_replicate_columns(cols, replicas)
    out = Dict{Symbol,AbstractVector}()
    for (key, value) in cols
        if key === :gp_weights
            out[key] = copy(value)
        elseif key === :subj || key === :dsubj
            out[key] = reduce(vcat, [value .+ 3r for r in 0:(replicas - 1)])
        else
            out[key] = repeat(value, replicas)
        end
    end
    return out
end

function _vs_schedule_args(sched)
    return (sched.reference_ends, sched.dose_ends, sched.lag_ends,
        sched.concentration_ends, sched.dose_amount, sched.dose_index,
        sched.treatment_map, sched.unique_dts, sched.concentration_idxs,
        sched.dosing_time_idxs)
end

function _vs_empty_batched(q, sched)
    return sum(varyingsource_pk_read_locs_over_subjects(sched.reference_ends,
        sched.dose_ends, sched.lag_ends, sched.concentration_ends, sched.dose_amount,
        sched.dose_index, sched.treatment_map, sched.unique_dts,
        sched.concentration_idxs, sched.dosing_time_idxs, q[1], q[2], q[3],
        q[1:4], q[5], q[6], q[1], q[2], q[3], q[4], q[5], q[6]))
end

@testset "varyingsource runtime structure and lazy dose-free batch" begin
    plan = lower_rkppl(_vs_emit_ast(), _VS_EMIT_DATA)
    expressions = Expr[]
    llvm = String[]
    for replicas in (1, 10)
        cols = _vs_replicate_columns(_vs_emit_columns(), replicas)
        bound = bind_data(plan, cols; dims = Dict(:kernel_nsub_conc => 3replicas))
        push!(expressions, kernel_expr(bound, assign_layout(bound)))
        sched = build_varyingsource_pk_schedule(cols[:subj], cols[:time],
            cols[:dsubj], cols[:dtime], cols[:damt], cols[:treatment])
        args = (_vs_schedule_args(sched)..., cols[:dose_x], cols[:dose_x],
            cols[:dose_x], cols[:gp_weights], 0.1, -0.15, 4.6, -2.5, -1.9, -3., -1.6, 0.)
        io = IOBuffer()
        code_llvm(io, varyingsource_pk_read_locs_over_subjects,
            Tuple{map(typeof, args)...}; debuginfo = :none)
        push!(llvm, String(take!(io)))
    end
    @test all(==(1), [_vs_ast_calls(e, :varyingsource_pk_read_locs_over_subjects)
        for e in expressions])
    @test _vs_ast_size(expressions[1]) == _vs_ast_size(expressions[2])
    # code_llvm emits fresh numeric ids for function symbols on every call.
    # Strip only those ids; all native instructions and blocks must match.
    normalized = [replace(ir, r"(?<=_)\d+(?=\"?\()" => "JIT") for ir in llvm]
    same_llvm = normalized[1] == normalized[2]
    @test same_llvm
    @test occursin(" phi i64 ", llvm[1]) && occursin("br i1", llvm[1])
    empty = build_varyingsource_pk_schedule([1, 2, 2], [1., 0., 5.],
        Int[], Float64[], Float64[], Int[])
    q = fill(NaN, 6)
    @test _vs_empty_batched(q, empty) == 0.0
    @test gradient(_vs_empty_batched, AutoEnzyme(; mode = Enzyme.Reverse), q,
        Constant(empty)) == zeros(6)
    scalar = ReactiveKernelsPPL.SubjectScalar(Float64[])
    @test varyingsource_pk_read_locs_over_subjects(_vs_schedule_args(empty)...,
        scalar, scalar, scalar, Float64[], scalar, scalar,
        scalar, scalar, scalar, scalar, scalar, scalar) == zeros(3)
end
