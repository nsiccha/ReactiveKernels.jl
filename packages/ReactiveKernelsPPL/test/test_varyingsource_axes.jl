using DifferentiationInterface: AutoEnzyme
using Distributions: Normal, logcdf, logpdf
using ReactiveKernels, ReactiveKernelsPPL, Test
import Enzyme

function _vs_dose_predictor_ast(; mixed=false)
    body = _vs_emit_ast()
    # Move a dose modifier from a cell-local calculation to a model LP.
    pushfirst!(body.args, :(inc ~ Dirichlet([1.,1.,1.])))
    pushfirst!(body.args, :(b_diet ~ Normal(0.,1.)))
    i = findfirst(ex -> ex isa Expr && ex.head === :macrocall,body.args)
    insert!(body.args,i,:(dose_lp = b_diet .* mo(diet,inc)))
    function rewrite(ex)
        ex isa Expr || return ex
        ex.head === :call && ex.args[1] === :varyingsource_pk_read_locs && begin
            args = copy(ex.args)
            args[3] = :dose_lp
            mixed && (args[9] = :dose_lp)
            return Expr(ex.head,args...)
        end
        return Expr(ex.head,map(rewrite,ex.args)...)
    end
    return rewrite(body)
end

@testset "varying-source LP dose rows and mixed-axis rejection" begin
    cols = merge(_vs_emit_columns(),Dict(:diet=>[1,2,4,3,2]))
    plan = lower_rkppl(_vs_dose_predictor_ast(),Set(keys(cols)))
    bound = bind_data(plan,cols;dims=Dict(:kernel_nsub_conc=>3))
    @test ReactiveKernelsPPL._predictor_rows(bound,:dose_lp) == 5
    @test ReactiveKernelsPPL._predictor_rows(bound,:log_Vc) == 3
    built = build_kernel(bound)
    names = coordinate_names(built.layout)
    u = zeros(length(names))
    u[findfirst(==(Symbol("dose_lp.diet")),names)] = .7
    nt = constrain(built.layout,u)
    contrast = cumsum(vcat(0.,nt.inc))[cols[:diet]]
    q = Base.invokelatest(prepare,built.spec;have=ReactiveKernelsPPL._query_have(bound),
        want=:_ppl_lp_dose_lp,bound=ReactiveKernelsPPL._query_bound(bound))
    @test Base.invokelatest(q,u) ≈ .7 .* contrast rtol=2e-14
    @test_throws "dose rows" bind_data(plan,merge(cols,Dict(:diet=>[1,2,3]));
        dims=Dict(:kernel_nsub_conc=>3))
    @test_throws "both dose and subject" lower_rkppl(_vs_dose_predictor_ast(;mixed=true),Set(keys(cols)))
    expressions = Expr[]
    for n in (1,10)
        b = bind_data(plan,_vs_replicate_columns(cols,n);dims=Dict(:kernel_nsub_conc=>3n))
        push!(expressions,kernel_expr(b,assign_layout(b)))
    end
    @test _vs_ast_size(expressions[1]) == _vs_ast_size(expressions[2])
    @test all(e -> _vs_ast_calls(e,:varyingsource_pk_read_locs_over_subjects)==1,expressions)
end

function _vs_assay_gather_ast(source=:([a1,a2]))
    return quote
        a1 ~ Exponential(1.)
        a2 ~ Exponential(1.)
        p1 ~ Exponential(1.)
        p2 ~ Exponential(1.)
        a ~ Normal(0.,1.)
        mu_lp = a
        vs = varyingsource_pk_schedule(obs=(:subj,:time),
            dose=(:dsubj,:dtime,:damt,:treatment))
        @plate result for s in 1:kernel_nsub_result
            mu = mu_lp[subj]
            add = $source[assay]
            prop = [p1,p2][assay]
            dv .~ CensoredAddpropnormal.(mu,add,prop,lloq)
            mu
        end
    end
end

@testset "assay scale gather and raw censoring ordinary reverse" begin
    cols = Dict{Symbol,AbstractVector}(:subj=>[1,1,1,2,2],:time=>[0.,1.,2.,0.,1.],
        :dsubj=>Int[],:dtime=>Float64[],:damt=>Float64[],:treatment=>Int[],
        :location=>[.8,.9,1.2,.7,1.1],:assay=>[1,2,1,2,1],
        :dv=>[.3,.5,1.3,.2,1.4],:lloq=>fill(.5,5))
    plan = lower_rkppl(_vs_assay_gather_ast(),Set(keys(cols)))
    bound = bind_data(plan,cols;dims=Dict(:kernel_nsub_result=>2))
    built = build_kernel(bound)
    names = coordinate_names(built.layout)
    values = Dict(:a1=>log(.4),:a2=>log(.6),:p1=>log(.15),:p2=>log(.2),
        Symbol("mu_lp.Intercept")=>.9)
    u = [values[n] for n in names]
    function oracle(p)
        coords = Dict(zip(names,p))
        scales = Dict(n=>exp(coords[n]) for n in (:a1,:a2,:p1,:p2))
        prior = sum(-v for v in Base.values(scales))+
            sum(coords[n] for n in keys(scales))+logpdf(Normal(0.,1.),coords[Symbol("mu_lp.Intercept")])
        likelihood = 0.
        for i in eachindex(cols[:dv])
            j = cols[:assay][i]
            mu = coords[Symbol("mu_lp.Intercept")]
            sd = hypot(scales[Symbol(:a,j)],mu*scales[Symbol(:p,j)])
            likelihood += cols[:dv][i] <= cols[:lloq][i] ?
                logcdf(Normal(mu,sd),cols[:lloq][i]) : logpdf(Normal(mu,sd),cols[:dv][i])
        end
        return prior+likelihood
    end
    query = prepare_query(built,bound,:sampler)
    @test query(u) ≈ oracle(u) rtol=2e-13
    sampler = prepare_sampler(built,bound,u;backend=AutoEnzyme(;mode=Enzyme.Reverse))
    g = zeros(length(u))
    value,_ = sampler_value_and_gradient!(sampler,g,u)
    @test value ≈ oracle(u) rtol=2e-13
    @test g ≈ _transit_fd_gradient(oracle,u) rtol=2e-6 atol=2e-8
    for bad in ([0,2,1,2,1],[1,3,1,2,1],[1.,2.,1.,2.,1.])
        @test_throws ContractValidationError bind_data(plan,merge(cols,Dict(:assay=>bad));
            dims=Dict(:kernel_nsub_result=>2))
    end
    @test_throws "empty" lower_rkppl(_vs_assay_gather_ast(:([])),Set(keys(cols)))
    @test_throws "model scalar" lower_rkppl(_vs_assay_gather_ast(:([location,a1])),Set(keys(cols)))
end
