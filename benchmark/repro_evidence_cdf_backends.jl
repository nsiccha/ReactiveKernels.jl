# Run with an environment containing same-worktree RK, DistributionKernels,
# ReactiveKernelsPPL, Reactant and Enzyme. One backend case per process:
# julia --project=<env> benchmark/repro_evidence_cdf_backends.jl gamma primal
# The wrapper lowers normally; this isolates a numerical backend failure.
using ReactiveKernels, ReactiveKernelsPPL, Reactant, Enzyme, DifferentiationInterface
family=Symbol(get(ARGS,1,"gamma"));mode=Symbol(get(ARGS,2,"primal"))
optimize=length(ARGS)>=3 ? Symbol(ARGS[3]) : nothing
ctor = family===:gamma ? :(Gamma.(2.,exp.(eta)./2)) :
    family===:beta ? :(Beta.(logistic.(eta) .* 4.,(1 .- logistic.(eta)) .* 4.)) :
    family===:poisson ? :(Poisson.(exp.(eta))) :
    family===:nb2 ? :(NegativeBinomial2.(exp.(eta),2.)) :
    family===:betabinomial ? :(BetaBinomial2.(5,logistic.(eta),4.)) :
    family===:vonmises ? :(VonMises.(eta,2.)) :
    family===:inversegaussian ? :(InverseGaussian.(exp.(eta),2.)) : error("unknown family")
ys=family in (:gamma,:beta,:inversegaussian) ? [.2,.3] : [.2,.8]
hi=family===:beta ? .9 : 2.8
expr=quote a ~ Normal(0,1);eta=a .+ 0*x;y .~ interval_censored.($ctor,$hi) end
cols=Dict(:y=>ys,:x=>zeros(length(ys)))
b=bind_data(lower_rkppl(expr,cols;conditioned=(:y,)),cols);built=build_kernel(b)
k=prepare_query(built,b,:sampler);u=unconstrain(built.layout,(a=.2,))
function attempt(built,bound,k,u,mode,optimize)
    ru=Reactant.to_rarray(u)
    native=k(u)
    println("native=",native);flush(stdout)
    if mode===:primal
        hlo=repr(Reactant.@code_hlo optimize=false k(ru))
        println("while_count=",count("stablehlo.while",hlo));flush(stdout)
        c=Reactant.@compile k(ru)
        compiled=Reactant.to_number(c(ru))
        println("compiled=",compiled)
        @assert isapprox(compiled,native;rtol=1e-10,atol=1e-10)
    else
        g(x)=only(Enzyme.gradient(Enzyme.Reverse,k,x))
        sampler=prepare_sampler(built,bound,u;backend=AutoEnzyme(;mode=Enzyme.Reverse))
        _,native_gradient=sampler_value_and_gradient!(sampler,similar(u),u)
        c=optimize===nothing ? Reactant.compile(g,(ru,)) :
            Reactant.compile(g,(ru,);optimize)
        compiled_gradient=Array(c(ru))
        println("native_gradient=",native_gradient)
        println("compiled_gradient=",compiled_gradient)
        @assert isapprox(compiled_gradient,native_gradient;rtol=1e-7,atol=1e-8)
    end
end
Base.invokelatest(attempt,built,b,k,u,mode,optimize)
