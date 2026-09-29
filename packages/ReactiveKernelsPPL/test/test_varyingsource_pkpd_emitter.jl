using Distributions: Normal, Exponential, logpdf
using ReactiveKernels, ReactiveKernelsPPL, Test

const _VS_FULL_LOG_NAMES = (:log_Vc,:log_k10,:log_k12,:log_k21,
    :log_baseline_pbmc,:log_kout,:log_theta1_pbmc,:log_theta2_pbmc,
    :log_baseline_csf,:log_theta1_csf,:log_theta2_csf,:log_absorption_rate,:log_absorption_mode)
const _VS_FULL_POSITIVE = (:sigma,:rho_d,:rho_c,:eff_sd,:rho_p,:sd_p,:rho_csf,:sd_csf)

function _vs_full_emit_ast()
    body = quote
        sigma ~ Exponential(1.)
        rho_d ~ Exponential(1.)
        rho_c ~ Exponential(1.)
        eff_sd ~ Exponential(1.)
        rho_p ~ Exponential(1.)
        sd_p ~ Exponential(1.)
        rho_csf ~ Exponential(1.)
        sd_csf ~ Exponential(1.)
        d_rate ~ Normal(0.,1.)
        d_mode ~ Normal(0.,1.)
        d_f ~ Normal(0.,1.)
        dose_slope ~ Normal(0.,1.)
        conc_slope ~ Normal(0.,1.)
        b_age ~ Normal(0.,1.)
    end
    for lp in _VS_FULL_LOG_NAMES
        coefficient = Symbol(:b_,lp)
        push!(body.args,:($coefficient ~ Normal(0.,1.)))
        push!(body.args,lp === :log_Vc ? :($lp = $coefficient .+ b_age .* age_s) :
            :($lp = $coefficient))
    end
    append!(body.args,(quote
        vs = varyingsource_pkpd_schedule(obs=(:subj,:time,:assay),
            dose=(:dsubj,:dtime,:damt,:treatment),discretization=:disc)
        @plate loc for s in 1:kernel_nsub_loc
            rate_mod = d_rate .* dose_x
            mode_mod = d_mode .* dose_x
            f_mod = d_f .* dose_x
            reads = varyingsource_pkpd_read_locs(vs,rate_mod,mode_mod,f_mod,
                gp_w,dose_slope,conc_slope,rho_d,rho_c,eff_sd,
                p_w,rho_p,sd_p,c_w,rho_csf,sd_csf,0.,24.,
                log_Vc,log_k10,log_k12,log_k21,log_baseline_pbmc,log_kout,
                log_theta1_pbmc,log_theta2_pbmc,log_baseline_csf,
                log_theta1_csf,log_theta2_csf,log_absorption_rate,log_absorption_mode)
            mu = reads[vs.obs_map]
            dv .~ Normal.(mu,sigma)
            mu
        end
    end).args)
    return body
end

function _vs_full_emit_columns()
    d,q = _vs_pkpd_schedule_fixture(),_vs_pkpd_point()
    return Dict{Symbol,AbstractVector}(:subj=>d.subject,:time=>d.time,:assay=>d.assay,
        :dsubj=>d.dose_subject,:dtime=>d.dose_time,:damt=>d.dose_amount,
        :treatment=>d.treatment,:disc=>d.discretization,:dose_x=>[0.,.2,.7,1.,-.2],
        :gp_w=>q[28:31],:p_w=>q[36:38],:c_w=>q[39:41],:age_s=>[0.,.1,.2],
        :dv=>[8.,120.,85.,6.,80.,55.,130.,0.,75.,65.])
end

function _vs_full_emit_vector_plan(plan)
    kp = only(plan.kernel_plates)
    vectors = (:gp_w,:p_w,:c_w)
    plate = KernelPlate(kp.result,kp.subjects,kp.timepoints,
        filter(x -> !(x[2] in vectors),kp.slices),kp.assignments,kp.obs,
        kp.collected,kp.label,kp.lp_args,kp.schedules)
    params = [VectorParameter(n,:vector_normal,(arg1=0.,arg2=1.),size)
        for (n,size) in zip(vectors,(4,3,3))]
    return StructuralPlan(plan.responses,plan.predictors,plan.population_priors,
        plan.parameters,plan.assignments,plan.columns,plan.n_obs;
        vector_parameters=params,kernel_plates=[plate])
end

function _vs_full_emit_point(names)
    q = _vs_pkpd_point()
    p = Dict{Symbol,Float64}(:sigma=>log(2.),:rho_d=>q[25],:rho_c=>q[26],:eff_sd=>q[27],
        :rho_p=>q[32],:sd_p=>q[33],:rho_csf=>q[34],:sd_csf=>q[35],
        :d_rate=>.15,:d_mode=>-.1,:d_f=>-.2,:dose_slope=>.1,:conc_slope=>-.15,
        Symbol("log_Vc.age_s")=>.1)
    for (i,n) in enumerate(_VS_FULL_LOG_NAMES)
        p[Symbol(n,".Intercept")] = q[i]
    end
    for (n,weights) in ((:gp_w,q[28:31]),(:p_w,q[36:38]),(:c_w,q[39:41]))
        for (i,w) in enumerate(weights)
            p[Symbol(n,".",i)] = w
        end
    end
    return [p[n] for n in names]
end

function _vs_full_emit_oracle(u,names,cols)
    p = Dict(zip(names,u))
    vector(n,k) = haskey(p,Symbol(n,".1")) ? [p[Symbol(n,".",i)] for i in 1:k] : cols[n]
    logs = [p[Symbol(n,".Intercept")] for n in _VS_FULL_LOG_NAMES, s in 1:3]
    logs[1,:] .+= p[Symbol("log_Vc.age_s")] .* cols[:age_s]
    q = vcat(logs[:,1],p[:d_rate].*cols[:dose_x],p[:d_mode].*cols[:dose_x],
        p[:d_f].*cols[:dose_x],[p[:dose_slope],p[:conc_slope],p[:rho_d],p[:rho_c],p[:eff_sd]],
        vector(:gp_w,4),[p[:rho_p],p[:sd_p],p[:rho_csf],p[:sd_csf]],vector(:p_w,3),vector(:c_w,3))
    s = build_varyingsource_pkpd_schedule(cols[:subj],cols[:time],cols[:assay],
        cols[:dsubj],cols[:dtime],cols[:damt],cols[:treatment],cols[:disc])
    mu = _vs_full_batch_oracle(q,s;subject_logs=logs)[s.obs_map]
    prior = sum(n in _VS_FULL_POSITIVE ? logpdf(Exponential(1.),exp(v))+v :
        logpdf(Normal(0.,1.),v) for (n,v) in p)
    return prior+sum(logpdf.(Normal.(mu,exp(p[:sigma])),cols[:dv]))
end

@testset "full varying-source typed schedule and emitted density" begin
    cols = _vs_full_emit_columns()
    plan = lower_rkppl(_vs_full_emit_ast(),Set(keys(cols)))
    @test only(only(plan.kernel_plates).schedules) isa VaryingSourcePKPDScheduleSpec
    for active in (false,true)
        unbound = active ? _vs_full_emit_vector_plan(plan) : plan
        @test validate_structure(unbound) === nothing
        data = active ? filter(p -> !(p.first in (:gp_w,:p_w,:c_w)),cols) : cols
        bound = bind_data(unbound,data;dims=Dict(:kernel_nsub_loc=>3))
        @test validate_data(bound) === nothing
        built = build_kernel(bound)
        names = coordinate_names(built.layout)
        u = _vs_full_emit_point(names)
        query = prepare_query(built,bound,:sampler)
        @test query(u) ≈ _vs_full_emit_oracle(u,names,cols) rtol=2e-9 atol=3e-6
        @test _vs_ast_calls(kernel_expr(bound,built.layout),
            :varyingsource_pkpd_read_locs_over_subjects) == 1
        sampler = prepare_sampler(built,bound,u;backend=AutoEnzyme(;mode=Enzyme.Reverse))
        g = zeros(length(u))
        value,_ = sampler_value_and_gradient!(sampler,g,u)
        @test value ≈ query(u) rtol=2e-13
        @test g ≈ _transit_fd_gradient(p -> _vs_full_emit_oracle(p,names,cols),u) rtol=6e-6 atol=3e-5
        @test all(isfinite,g)
        bound.columns[:vs_pd3_center_idxs][1] = 99
        @test_throws "not the schedule build" validate_data(bound)
    end
    @test_throws "basis coefficient" bind_data(plan,merge(cols,Dict(:p_w=>[1.]));dims=Dict(:kernel_nsub_loc=>3))
    @test_throws "square coefficient" bind_data(plan,merge(cols,Dict(:gp_w=>ones(3)));dims=Dict(:kernel_nsub_loc=>3))
end

@testset "full grouped expression retains one batched call after data rebinding" begin
    cols = _vs_full_emit_columns()
    plan = lower_rkppl(_vs_full_emit_ast(),Set(keys(cols)))
    expressions = Expr[]
    for replicas in (1,10)
        data = Dict{Symbol,AbstractVector}()
        for (key,value) in cols
            data[key] = key in (:gp_w,:p_w,:c_w,:disc) ? copy(value) :
                key in (:subj,:dsubj) ? reduce(vcat,[value .+ 3i for i in 0:(replicas-1)]) :
                repeat(value,replicas)
        end
        bound = bind_data(plan,data;dims=Dict(:kernel_nsub_loc=>3replicas))
        push!(expressions,kernel_expr(bound,assign_layout(bound)))
    end
    @test all(e -> _vs_ast_calls(e,:varyingsource_pkpd_read_locs_over_subjects)==1,expressions)
    @test _vs_ast_size(expressions[1]) == _vs_ast_size(expressions[2])
end
