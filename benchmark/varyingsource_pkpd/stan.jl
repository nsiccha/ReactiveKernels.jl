using Libdl

struct StanPKPD
    library::Ptr{Cvoid}
    model::Ptr{Cvoid}
    names::Vector{String}
    n_unc::Int
end

function pkpd_stan_error(model, error_ptr)
    message = unsafe_string(error_ptr)
    ccall(Libdl.dlsym(model.library,:bs_free_error_msg),Cvoid,(Ptr{Cchar},),error_ptr)
    error(message)
end

function StanPKPD(library,json)
    lib = Libdl.dlopen(library)
    err = Ref{Ptr{Cchar}}(C_NULL)
    ptr = ccall(Libdl.dlsym(lib,:bs_model_construct),Ptr{Cvoid},
        (Cstring,Cuint,Ref{Ptr{Cchar}}),json,7,err)
    ptr == C_NULL && pkpd_stan_error((library=lib,),err[])
    names = split(unsafe_string(ccall(Libdl.dlsym(lib,:bs_param_names),Cstring,
        (Ptr{Cvoid},Cuchar,Cuchar),ptr,0,0)),',')
    n = ccall(Libdl.dlsym(lib,:bs_param_unc_num),Cint,(Ptr{Cvoid},),ptr)
    return StanPKPD(lib,ptr,names,n)
end

pkpd_json(x::Real) = isfinite(x) ? string(x) : error("nonfinite fixture data")
pkpd_json(x::AbstractVector) = "["*join(pkpd_json.(x),',')*"]"
pkpd_json(x::AbstractDict) = "{"*join(["\"$k\":"*pkpd_json(x[k])
    for k in sort!(collect(keys(x)))],',')*"}"

function pkpd_stan_data(cols;gp_basis=4,placebo_basis=10)
    n = length(cols[:sid])
    fields = Dict("subject"=>cols[:subj],"assay"=>cols[:assay],"ts"=>cols[:time],
        "lloq"=>cols[:lloq],"dosing_subject"=>cols[:dsubj],
        "dosing_times"=>cols[:dtime],"treatment"=>cols[:vessel],
        "dosing_diet"=>cols[:diet],"doses"=>cols[:damt],
        "discretization_times"=>cols[:disc],"male"=>cols[:male],
        "age_yr"=>35 .+ 10 .* cols[:age_std],
        "weight_kg"=>70 .+ 10 .* cols[:weight_std],"bmi_kgm2"=>fill(22.,n),
        "scale0"=>PKPD_SCALE0,"loc0"=>PKPD_LOC0,
        "subject_is_diseased"=>cols[:diseased],
        "effectiveness_parameters_scale"=>[1.,1.,1.],"obs"=>cols[:dv])
    data = Dict{String,Any}()
    for (key,value) in fields
        data[key],data[key*"_n"] = value,length(value)
    end
    merge!(data,Dict("age_yr_loc"=>35.,"age_yr_scale"=>10.,
        "weight_kg_loc"=>70.,"weight_kg_scale"=>10.,"pop_scale_rate"=>1.5,
        "eta"=>2.,"covariate_effects_scale"=>.1,
        "effectiveness_parameters_n_functions"=>gp_basis,
        "placebo_lo_time"=>0.,"placebo_hi_time"=>24.,
        "placebo_n_functions"=>placebo_basis,"obs_scale_rate"=>4.))
    return pkpd_json(data)
end

function pkpd_stan_constrained(layout,u,cols,stan_names)
    nt = constrain(layout,u)
    coordinates = Dict(zip(coordinate_names(layout),u))
    coef(lp,label) = coordinates[Symbol(lp,".",label)]
    out = Dict{String,Float64}()
    function put(name,value)
        haskey(out,name) && error("duplicate Stan parameter $name")
        out[name] = value
    end
    k = length(PKPD_LOGS)
    for (i,lp) in enumerate(PKPD_LOGS)
        center = coef(lp,:Intercept)-PKPD_LOC0[i]
        put("unit_params_x_loc.$i",center/PKPD_SCALE0[i])
        put("unit_params_x_scale.$i",nt.tau_sid[i])
        i <= 11 && put("unit_params_disease_effect.$i",coef(lp,:diseased))
        for s in eachindex(cols[:sid])
            put("unit_params_x_x.$i.$s",nt.b_sid[s,i]+center)
        end
    end
    for i in 2:k, j in 1:i-1
        q = (i-1)*(i-2)÷2+j
        put("unit_params_x_L_cor_xi.$q",coordinates[Symbol("L_sid.$q")]*sqrt(k-j))
    end
    for (i,lp) in enumerate(PKPD_LOGS[1:2]),
            (j,column) in enumerate((:male,:age_std,:weight_std))
        put("unit_params_covariate_effects.$i.$j",coef(lp,column)/.1)
    end
    for (lp,prefix) in zip(PKPD_DOSE_LPS,("source_parameters_m1",
            "source_parameters_m2","log_effective_bioavailability"))
        for (label,suffix) in zip((PKPD_VESSELS...,:diet),
                ("beta_bottle","beta_bottle_20","beta_tablet","beta_tablet_20","beta_diet"))
            put(prefix*"_"*suffix,coef(lp,label))
        end
        increments = getproperty(nt,Symbol(:inc_,lp))
        for i in eachindex(increments)
            put(prefix*"_simo_diet.$i",increments[i])
        end
    end
    for (param,suffix) in ((:dose_slope,"dose_slope"),(:conc_slope,"conc_slope"),
            (:eff_sd,"eff_scale"),(:rho_d,"dose_scale"),(:rho_c,"conc_scale"))
        put("effectiveness_parameters_"*suffix,getproperty(nt,param))
    end
    for i in eachindex(nt.gp_w)
        put("effectiveness_parameters_unit_weights.$i",nt.gp_w[i])
    end
    for (prefix,rho,sd,w) in (("placebo23_log_y",:rho_p,:sd_p,:p_w),
            ("placebo3_log_y",:rho_csf,:sd_csf,:c_w))
        put(prefix*"_x_scale",getproperty(nt,rho))
        put(prefix*"_y_scale",getproperty(nt,sd))
        for (i,value) in enumerate(getproperty(nt,w))
            put(prefix*"_unit_weights.$i",value)
        end
    end
    for i in 1:3
        put("obs_scale.$i.1",getproperty(nt,Symbol(:s_add,i)))
        put("obs_scale.$i.2",getproperty(nt,Symbol(:s_prop,i)))
    end
    Set(keys(out)) == Set(stan_names) || error("Stan parameter coverage differs: "*
        "missing=$(setdiff(stan_names,collect(keys(out)))) extra=$(setdiff(collect(keys(out)),stan_names))")
    return [out[name] for name in stan_names]
end

function pkpd_stan_unconstrain(model,theta)
    u,err = zeros(model.n_unc),Ref{Ptr{Cchar}}(C_NULL)
    status = ccall(Libdl.dlsym(model.library,:bs_param_unconstrain),Cint,
        (Ptr{Cvoid},Ptr{Cdouble},Ptr{Cdouble},Ref{Ptr{Cchar}}),model.model,theta,u,err)
    status == 0 || pkpd_stan_error(model,err[])
    return u
end

function pkpd_stan_gradient(model,u;jacobian=true)
    g,value,err = zeros(model.n_unc),Ref(0.),Ref{Ptr{Cchar}}(C_NULL)
    pkpd_stan_gradient!(value,g,model,u,err;jacobian)
    return value[],g
end

function pkpd_stan_gradient!(value,g,model,u,err=Ref{Ptr{Cchar}}(C_NULL);jacobian=true)
    status = ccall(Libdl.dlsym(model.library,:bs_log_density_gradient),Cint,
        (Ptr{Cvoid},Cuchar,Cuchar,Ptr{Cdouble},Ref{Cdouble},Ptr{Cdouble},Ref{Ptr{Cchar}}),
        model.model,0,jacobian,u,value,g,err)
    status == 0 || pkpd_stan_error(model,err[])
    return value[],g
end

function pkpd_stan_density(model,u;jacobian=true)
    value,err = Ref(0.),Ref{Ptr{Cchar}}(C_NULL)
    status = ccall(Libdl.dlsym(model.library,:bs_log_density),Cint,
        (Ptr{Cvoid},Cuchar,Cuchar,Ptr{Cdouble},Ref{Cdouble},Ref{Ptr{Cchar}}),
        model.model,0,jacobian,u,value,err)
    status == 0 || pkpd_stan_error(model,err[])
    return value[]
end

function pkpd_stan_parameters(model,u;include_gq=false)
    n = ccall(Libdl.dlsym(model.library,:bs_param_num),Cint,
        (Ptr{Cvoid},Cuchar,Cuchar),model.model,1,include_gq)
    names = split(unsafe_string(ccall(Libdl.dlsym(model.library,:bs_param_names),Cstring,
        (Ptr{Cvoid},Cuchar,Cuchar),model.model,1,include_gq)),',')
    values,err = zeros(n),Ref{Ptr{Cchar}}(C_NULL)
    rng = include_gq ? ccall(Libdl.dlsym(model.library,:bs_rng_construct),Ptr{Cvoid},
        (Cuint,Ref{Ptr{Cchar}}),7,err) : C_NULL
    include_gq && rng == C_NULL && pkpd_stan_error(model,err[])
    status = ccall(Libdl.dlsym(model.library,:bs_param_constrain),Cint,
        (Ptr{Cvoid},Cuchar,Cuchar,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cvoid},Ref{Ptr{Cchar}}),
        model.model,1,include_gq,u,values,rng,err)
    include_gq && ccall(Libdl.dlsym(model.library,:bs_rng_destruct),Cvoid,(Ptr{Cvoid},),rng)
    status == 0 || pkpd_stan_error(model,err[])
    return Dict(zip(names,values))
end

function close(model::StanPKPD)
    ccall(Libdl.dlsym(model.library,:bs_model_destruct),Cvoid,(Ptr{Cvoid},),model.model)
    Libdl.dlclose(model.library)
end

struct PKPDCoordinateMap
    affine::Matrix{Float64}
    bias::Vector{Float64}
    simplex_inputs::Vector{UnitRange{Int}}
    simplex_outputs::Vector{UnitRange{Int}}
end

"Map RK unconstrained coordinates into the original Stan program's layout."
function pkpd_coordinate_map(layout,model)
    indices = Dict(n=>i for (i,n) in enumerate(coordinate_names(layout)))
    index(lp,label) = indices[Symbol(lp,".",label)]
    k = length(PKPD_LOGS)
    a,c = zeros(model.n_unc,layout.total),zeros(model.n_unc)
    simplex_inputs,simplex_outputs = UnitRange{Int}[],UnitRange{Int}[]
    simplexes = Dict(prefix*"_simo_diet"=>Symbol(:inc_,lp)
        for (lp,prefix) in zip(PKPD_DOSE_LPS,("source_parameters_m1",
            "source_parameters_m2","log_effective_bioavailability")))
    scalar_map = Dict("effectiveness_parameters_dose_slope"=>:dose_slope,
        "effectiveness_parameters_conc_slope"=>:conc_slope,
        "effectiveness_parameters_eff_scale"=>:eff_sd,
        "effectiveness_parameters_dose_scale"=>:rho_d,
        "effectiveness_parameters_conc_scale"=>:rho_c,
        "placebo23_log_y_x_scale"=>:rho_p,"placebo23_log_y_y_scale"=>:sd_p,
        "placebo3_log_y_x_scale"=>:rho_csf,"placebo3_log_y_y_scale"=>:sd_csf)
    vector_map = Dict("effectiveness_parameters_unit_weights"=>:gp_w,
        "placebo23_log_y_unit_weights"=>:p_w,"placebo3_log_y_unit_weights"=>:c_w)
    xi_columns = [j for i in 2:k for j in 1:i-1]
    row = 0
    for name in model.names
        parts = split(name,'.')
        prefix = parts[1]
        if haskey(simplexes,prefix)
            q = parse(Int,parts[2])
            q == 3 && continue
            if q == 1
                e = only(e for e in layout.entries if e.name === simplexes[prefix])
                e.size == 2 && e.transform === :simplex || error("unexpected simplex layout")
                push!(simplex_inputs,e.offset:e.offset+1)
                push!(simplex_outputs,row+1:row+2)
            end
            row += 1
            continue
        end
        row += 1
        if prefix == "unit_params_disease_effect"
            i = parse(Int,parts[2]); a[row,index(PKPD_LOGS[i],:diseased)] = 1
        elseif prefix == "unit_params_x_loc"
            i = parse(Int,parts[2]); a[row,index(PKPD_LOGS[i],:Intercept)] = 1/PKPD_SCALE0[i]
            c[row] = -PKPD_LOC0[i]/PKPD_SCALE0[i]
        elseif prefix == "unit_params_x_scale"
            i = parse(Int,parts[2]); a[row,indices[Symbol("tau_sid.$i")]] = 1
        elseif prefix == "unit_params_x_L_cor_xi"
            i = parse(Int,parts[2]); a[row,indices[Symbol("L_sid.$i")]] = sqrt(k-xi_columns[i])
        elseif prefix == "unit_params_x_x"
            i,s = parse.(Int,parts[2:3])
            a[row,indices[Symbol("b_flat_sid.$((s-1)*k+i)")]] = 1
            a[row,index(PKPD_LOGS[i],:Intercept)] = 1
            c[row] = -PKPD_LOC0[i]
        elseif prefix == "unit_params_covariate_effects"
            i,j = parse.(Int,parts[2:3]); column = (:male,:age_std,:weight_std)[j]
            a[row,index(PKPD_LOGS[i],column)] = 10
        elseif haskey(scalar_map,prefix)
            a[row,indices[scalar_map[prefix]]] = 1
        elseif haskey(vector_map,prefix)
            a[row,indices[Symbol(vector_map[prefix],".",parts[2])]] = 1
        elseif prefix == "obs_scale"
            i,j = parse.(Int,parts[2:3]); parameter = Symbol(j == 1 ? :s_add : :s_prop,i)
            a[row,indices[parameter]] = 1
        else
            found = false
            for (lp,p) in zip(PKPD_DOSE_LPS,("source_parameters_m1",
                    "source_parameters_m2","log_effective_bioavailability")),
                    (label,suffix) in zip((PKPD_VESSELS...,:diet),
                        ("beta_bottle","beta_bottle_20","beta_tablet","beta_tablet_20","beta_diet"))
                if prefix == p*"_"*suffix
                    a[row,index(lp,label)] = 1
                    found = true
                end
            end
            found || error("unmapped Stan parameter $name")
        end
    end
    row == model.n_unc || error("Stan unconstrained width does not match constrained-name walk")
    return PKPDCoordinateMap(a,c,simplex_inputs,simplex_outputs)
end

function pkpd_stan_coordinates(u,map::PKPDCoordinateMap)
    out = map.affine*u+map.bias
    for (input,output) in zip(map.simplex_inputs,map.simplex_outputs)
        # RK's centered stick breaking, followed by Stan 2.39's Helmert ILR.
        v1 = inv(1+exp(-(u[input[1]]+log(2))))
        v2 = inv(1+exp(-u[input[2]]))
        logp1,logp2,logp3 = log(v1),log1p(-v1)+log(v2),log1p(-v1)+log1p(-v2)
        out[output[1]] = (logp1-logp2)/sqrt(2)
        out[output[2]] = (logp1+logp2-2logp3)/sqrt(6)
    end
    return out
end

pkpd_map_pullback_objective(u,map,g) = sum(pkpd_stan_coordinates(u,map).*g)
