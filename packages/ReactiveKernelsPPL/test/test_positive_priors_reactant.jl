# _pp_backend_check checks native Enzyme against finite differences,
# Reactant primal/reverse against native, and returns the backend operation
# multiset. It is shared with test_parameter_priors_reactant.jl.
@testset "positive prior slots retain native and compiled values, gradients and structure" begin
    for slot in (:sampled, :varying, :spline, :hsgp)
        structures = Dict{String,Int}[]
        for n in (5,9)
            x = collect(range(-1.,1.;length=n))
            data = Dict(:y=>fill(0.2,n))
            expr, q = if slot === :sampled
                (quote s ~ HalfCauchy(2); y .~ Normal.(s,1) end, (s=1.2,))
            elseif slot === :varying
                data[:g] = collect(1:n)
                (quote
                    a ~ Normal(0,1)
                    r ~ varying_effect(g,[1];sd=HalfCauchy(2))
                    mu = a .+ r
                    y .~ Normal.(mu,1)
                end, (a=0.1,L_g=ones(1,1),tau_g=[1.2],z_flat_g=fill(0.1,n)))
            elseif slot === :spline
                data[:x] = x
                (quote
                    a ~ Normal(0,1)
                    spline_basis(:s_x,x;k=4,sd=truncated(Normal(0.7,2),0.3,4))
                    mu = a .+ spline(:s_x)
                    y .~ Normal.(mu,1)
                end,(a=0.1,b_s_x_fixed=[0.1],b_s_x_raw=[0.1,0.1],sd_s_x=[1.2]))
            else
                data[:x] = x
                (quote
                    a ~ Normal(0,1)
                    hsgp_basis(:h_x,x;k=4,length_scale=truncated(Normal(0.7,2),0.3,4),sd=HalfNormal(2))
                    mu = a .+ hsgp(:h_x)
                    y .~ Normal.(mu,1)
                end,(a=0.1,rho_h_x=1.2,sigma_h_x=0.8,beta_raw_h_x=fill(0.1,4)))
            end
            println("POSITIVE_BACKEND_BEGIN ",slot," n=",n); flush(stdout)
            before = deepcopy(data)
            push!(structures,_pp_backend_check(expr,data,q))
            @test data == before
        end
        @test structures[1] == structures[2]
    end
end
