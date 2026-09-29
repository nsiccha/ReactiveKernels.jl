using ReactiveKernels, ReactiveKernelsPPL

const PKPD_LOGS = (:log_Vc, :log_k10, :log_k12, :log_k21,
    :log_baseline_pbmc, :log_kout, :log_theta1_pbmc, :log_theta2_pbmc,
    :log_baseline_csf, :log_theta1_csf, :log_theta2_csf,
    :log_absorption_rate, :log_absorption_mode)
const PKPD_LOC0 = vcat(log.([10., .01, sqrt(.01*50), sqrt(.001*5)]) .+
    [0., 3.2, 0., 0.], [log(130.)-1, -4., -4., 2., log(80.)-1, -4., 2.],
    [log(1/8), 0.])
const PKPD_SCALE0 = [.8, .8, 2., 2., .8, .8, .8, .8, .8, .8, .8, .8, .8]
const PKPD_DOSE_LPS = (:rate_mod, :mode_mod, :f_mod)
const PKPD_VESSELS = (:bottle, :bottle_20, :tablet, :tablet_60)
pkpd_rho_lower(k) = (6/pi)*sqrt(log(100)/(k*k-1))

"The original model's priors and full centered hierarchy on public synthetic data."
function pkpd_plan(; gp_basis=4, placebo_basis=10)
    gp_lo, p_lo = pkpd_rho_lower(gp_basis), pkpd_rho_lower(placebo_basis)
    gp_lo < 2 && p_lo < 2 || error("original GP priors need a nonempty interval")
    body = quote
        rho_d ~ Uniform($gp_lo, 2.)
        rho_c ~ Uniform($gp_lo, 2.)
        eff_sd ~ Normal(0., 1.; lower=0)
        dose_slope ~ Normal(0., 1.)
        conc_slope ~ Normal(0., 1.)
        rho_p ~ Uniform($p_lo, 2.)
        sd_p ~ LogNormal(0., 1.)
        rho_csf ~ Uniform($p_lo, 2.)
        sd_csf ~ LogNormal(0., 1.)
        # Julia uses scale; the original Stan observation-scale rate is 4.
        s_add1 ~ Exponential(.25)
        s_add2 ~ Exponential(.25)
        s_add3 ~ Exponential(.25)
        s_prop1 ~ Exponential(.25)
        s_prop2 ~ Exponential(.25)
        s_prop3 ~ Exponential(.25)
    end
    margins = Expr(:vect, fill(1, 13)...)
    push!(body.args, :(draws ~ varying_draws(sid, $margins;
        centered=true, eta=2., sd=Exponential($(1/1.5)))))
    for (i, lp) in enumerate(PKPD_LOGS)
        a, r = Symbol(:a_,lp), Symbol(:r_,lp)
        push!(body.args, :($a ~ Normal($(PKPD_LOC0[i]), $(PKPD_SCALE0[i]))))
        push!(body.args, :($r ~ varying_slice(draws, $i:$i)))
        terms = :($a .+ $r)
        if i <= 11
            d = Symbol(:d_,lp)
            push!(body.args, :($d ~ Normal(0.,1.)))
            terms = :($terms .+ $d .* diseased)
        end
        if i <= 2
            for column in (:male,:age_std,:weight_std)
                b = Symbol(:b_,lp,:_,column)
                push!(body.args, :($b ~ Normal(0.,.1)))
                terms = :($terms .+ $b .* $column)
            end
        end
        push!(body.args, :($lp = $terms))
    end
    for lp in PKPD_DOSE_LPS
        inc, beta = Symbol(:inc_,lp), Symbol(:b_,lp,:_diet)
        push!(body.args, :($inc ~ Dirichlet([1.,1.,1.])))
        push!(body.args, :($beta ~ Normal(0.,.5)))
        terms = :($beta .* mo(diet,$inc))
        for column in PKPD_VESSELS
            b = Symbol(:b_,lp,:_,column)
            push!(body.args, :($b ~ Normal(0.,.5)))
            terms = :($terms .+ $b .* $column)
        end
        push!(body.args, :($lp = $terms))
    end
    append!(body.args, (quote
        vs = varyingsource_pkpd_schedule(obs=(:subj,:time,:assay),
            dose=(:dsubj,:dtime,:damt,:key),discretization=:disc)
        @plate loc for s in 1:kernel_nsub_loc
            reads = varyingsource_pkpd_read_locs(vs,rate_mod,mode_mod,f_mod,
                gp_w,dose_slope,conc_slope,rho_d,rho_c,eff_sd,
                p_w,rho_p,sd_p,c_w,rho_csf,sd_csf,0.,24.,
                log_Vc,log_k10,log_k12,log_k21,log_baseline_pbmc,log_kout,
                log_theta1_pbmc,log_theta2_pbmc,log_baseline_csf,
                log_theta1_csf,log_theta2_csf,log_absorption_rate,log_absorption_mode)
            mu = reads[vs.obs_map]
            s_add = [s_add1,s_add2,s_add3][assay]
            s_prop = [s_prop1,s_prop2,s_prop3][assay]
            dv .~ CensoredAddpropnormal.(mu,s_add,s_prop,lloq)
            mu
        end
    end).args)
    columns = (:sid,:diseased,:male,:age_std,:weight_std,:diet,PKPD_VESSELS...,
        :subj,:time,:assay,:dsubj,:dtime,:damt,:key,:disc,:dv,:lloq,
        :gp_w,:p_w,:c_w)
    plan = lower_rkppl(body, columns)
    kp = only(plan.kernel_plates)
    innovations = (:gp_w,:p_w,:c_w)
    plate = KernelPlate(kp.result,kp.subjects,kp.timepoints,
        filter(x -> !(x[2] in innovations),kp.slices),kp.assignments,kp.obs,
        kp.collected,kp.label,kp.lp_args,kp.schedules)
    vectors = vcat(plan.vector_parameters,
        [VectorParameter(n,:vector_normal,(arg1=0.,arg2=1.),size)
            for (n,size) in zip(innovations,(gp_basis^2,placebo_basis,placebo_basis))])
    return StructuralPlan(plan.responses,plan.predictors,plan.population_priors,
        plan.parameters,plan.assignments,plan.columns,plan.n_obs;
        vector_parameters=vectors,kernel_plates=[plate],
        varying_draws=plan.varying_draws,varying_slices=plan.varying_slices,
        derived=plan.derived,levelmaps=plan.levelmaps)
end

function pkpd_point(layout; case=1)
    point = zeros(layout.total)
    for e in layout.entries
        r = e.offset:(e.offset+e.size-1)
        if e.kind === :coefficient
            lpidx = findfirst(==(e.predictor), PKPD_LOGS)
            for (j,label) in enumerate(e.labels)
                point[r[j]] = label === :Intercept && lpidx !== nothing ?
                    PKPD_LOC0[lpidx] + .04(case-1)*sin(lpidx) +
                        (case==3 && lpidx==13 ? log(56.) : 0.) :
                    .04sin(j+case)
            end
        elseif e.name === :tau_sid
            point[r] .= log.(.2 .+ .01collect(1:e.size))
        elseif e.name === :b_flat_sid
            point[r] .= .02sin.(collect(1:e.size) .+ case)
        elseif e.kind === :varying_corr
            point[r] .= .015sin.(collect(1:e.size) .+ case)
        elseif e.name in (:s_add1,:s_add2,:s_add3)
            point[r] .= log(e.name === :s_add1 ? .5 : e.name === :s_add2 ? 5. : 3.)
        elseif e.name in (:s_prop1,:s_prop2,:s_prop3)
            point[r] .= log(.12)
        elseif e.name in (:eff_sd,:sd_p,:sd_csf)
            point[r] .= log(.15)
        elseif e.name in (:dose_slope,:conc_slope)
            point[r] .= .1sin(case)
        elseif e.name in (:gp_w,:p_w,:c_w)
            point[r] .= .06sin.(collect(1:e.size) .+ case)
        elseif e.transform === :simplex
            point[r] .= .2sin.(collect(1:e.size) .+ case)
        end
    end
    return point
end

"Gamma shapes at the source columns selected by the native schedule."
function pkpd_gamma_shapes(layout,u,columns)
    nt = constrain(layout,u)
    coordinates = Dict(zip(coordinate_names(layout),u))
    coef(lp,label) = coordinates[Symbol(lp,".",label)]
    contrast = Dict(lp=>cumsum(vcat(0.,getproperty(nt,Symbol(:inc_,lp))))
        for lp in PKPD_DOSE_LPS)
    modifier(lp,row) = coef(lp,:diet)*contrast[lp][columns[:diet][row]]+
        sum(coef(lp,column)*columns[column][row] for column in PKPD_VESSELS)
    schedule = build_varyingsource_pkpd_schedule(columns[:subj],columns[:time],columns[:assay],
        columns[:dsubj],columns[:dtime],columns[:damt],columns[:key],columns[:disc])
    shapes = Float64[]
    first = 1
    for (subject,last) in enumerate(schedule.dose_ends)
        if first <= last
            treatments = maximum(view(schedule.treatment_map,first:last))
            for source in first:(first+treatments-1)
                row = schedule.dose_index[source]
                lograte = coef(:log_absorption_rate,:Intercept)+nt.b_sid[subject,12]+
                    modifier(:rate_mod,row)
                logmode = coef(:log_absorption_mode,:Intercept)+nt.b_sid[subject,13]+
                    modifier(:mode_mod,row)
                push!(shapes,1+exp(lograte+logmode))
            end
        end
        first = last+1
    end
    return shapes
end

function pkpd_columns(replicas=1)
    subject = [1,1,1,1,1,1,2,3,3,3]
    time = [12.,0.,14.,6.,2.,14.,0.,0.,4.,9.]
    assay = [1,2,2,1,3,3,2,1,3,3]
    dose_subject = [1,3,1,1,3,1,3]
    dose_time = [0.,0.,4.,8.,0.,10.,6.]
    vessel, diet = [1,3,1,2,4,5,5], [1,2,1,3,4,2,3]
    order = sortperm(dose_subject)
    dose_subject,dose_time,vessel,diet = dose_subject[order],dose_time[order],vessel[order],diet[order]
    raw = Dict{Symbol,AbstractVector}(:subj=>subject,:time=>time,:assay=>assay,
        :dsubj=>dose_subject,:dtime=>dose_time,
        :damt=>[10000.,15000.,20000.,40000.,25000.,50000.,200000.][order],
        :key=>100 .* vessel .+ diet,:vessel=>vessel,:diet=>diet,
        :disc=>[0.,.5,1.,2.,4.,8.,12.,20.],:sid=>[1,2,3],
        :male=>[0.,1.,1.],:age_std=>[-1.,0.,1.],:weight_std=>[-1.,1.,0.],
        :diseased=>[0.,1.,0.],:dv=>[8.,120.,85.,6.,80.,55.,130.,.5,75.,65.],
        :lloq=>[.5,1.,1.,.5,1.,1.,1.,.5,1.,1.],
        :bottle=>Float64.(vessel .∈ Ref([2,3])),:bottle_20=>Float64.(vessel .== 3),
        :tablet=>Float64.(vessel .∈ Ref([4,5])),:tablet_60=>Float64.(vessel .== 5))
    out = Dict{Symbol,AbstractVector}()
    for (key,value) in raw
        out[key] = key === :disc ? copy(value) : key in (:subj,:dsubj,:sid) ?
            reduce(vcat,[value .+ 3i for i in 0:replicas-1]) : repeat(value,replicas)
    end
    return out
end

pkpd_bound_columns(columns) = filter(p -> p.first !== :vessel,columns)
