# A public synthetic PK slice with interleaved/ragged observations, one
# dose-free subject, equal-time doses, duplicate reads, and GP feedback.
using ReactiveKernelsPPL

const PK_PORTS = [:sigma, Symbol("log_Vc.Intercept"), Symbol("log_Vc.age_s"),
    Symbol("log_k10.Intercept"), Symbol("log_k12.Intercept"),
    Symbol("log_k21.Intercept"), Symbol("log_rate.Intercept"),
    Symbol("log_mode.Intercept"), :d_rate, :d_mode, :d_f, :dose_slope,
    :conc_slope, [Symbol("gp_weights.", i) for i in 1:4]...]

function pk_columns(replicas)
    base = Dict{Symbol,AbstractVector}(
        :subj => [3, 1, 2, 1, 3, 1, 2, 1],
        :time => [20., 1., 7., 12., 0., 24., 0., 12.],
        :dsubj => [1, 3, 1, 1, 3], :dtime => [0., 0., 4., 8., 0.],
        :damt => [10000., 15000., 20000., 40000., 25000.],
        :treatment => [10, 20, 10, 30, 30], :dose_x => [0., 0.4, 1., 2., -0.3],
        :age_s => [0.2, -0.3, 0.5], :dv => [2., 3., 0., 10., 0., 4., 0., 10.])
    out = Dict{Symbol,AbstractVector}()
    for (key, value) in base
        out[key] = key in (:subj, :dsubj) ?
            reduce(vcat, [value .+ 3r for r in 0:(replicas - 1)]) :
            repeat(value, replicas)
    end
    return out
end

function pk_plan()
    ast = quote
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
    plan = lower_rkppl(ast, Set([:subj, :time, :dsubj, :dtime, :damt,
        :treatment, :dose_x, :age_s, :dv, :gp_weights]))
    kp = only(plan.kernel_plates)
    plate = KernelPlate(kp.result, kp.subjects, kp.timepoints,
        filter(x -> x[2] !== :gp_weights, kp.slices), kp.assignments,
        kp.obs, kp.collected, kp.label, kp.lp_args, kp.schedules)
    return StructuralPlan(plan.responses, plan.predictors, plan.population_priors,
        plan.parameters, plan.assignments, plan.columns, plan.n_obs;
        vector_parameters = [VectorParameter(:gp_weights, :vector_normal,
            (arg1 = 0.0, arg2 = 1.0), 4)], kernel_plates = [plate])
end

function pk_point(names, label)
    values = vcat([log(2.0), log(100.0), 0.1, log(0.08), log(0.15),
        log(0.05), log(0.2), 0.0, 0.15, -0.1, -0.2, 0.1, -0.15],
        vec([0.02 -0.03; -0.01 0.04]))
    if label == "P2"
        values[4:8] .= [log(0.5), log(0.3), log(0.4), log(0.05), log(1.0)]
        values[12:13] .= [-0.08, 0.12]
    elseif label == "P3-shape8"
        values[7:8] .= [log(0.3), log(7 / 0.3)]
    elseif label != "P1"
        error("unknown PK point $label")
    end
    mapping = Dict(zip(PK_PORTS, values))
    Set(names) == Set(PK_PORTS) || error("RK coordinate contract changed")
    return [mapping[name] for name in names]
end
