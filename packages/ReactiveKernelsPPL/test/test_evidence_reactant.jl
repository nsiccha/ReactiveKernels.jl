using Test, ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Reactant

function _evidence_backend_measure(built, bound, kernel, u, optimize, reverse_supported)
    native = kernel(u)
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ native atol=1e-12 rtol=1e-12
    finite = similar(u)
    for i in eachindex(u)
        up, down = copy(u), copy(u)
        up[i] += 1e-5; down[i] -= 1e-5
        finite[i] = (kernel(up) - kernel(down)) / 2e-5
    end
    @test grad ≈ finite atol=1e-7 rtol=1e-6
    println("EVIDENCE_BACKEND_STAGE primal");flush(stdout)
    ru = Reactant.to_rarray(u)
    hlo = repr(Reactant.@code_hlo optimize=false kernel(ru))
    compiled = Reactant.@compile kernel(ru)
    @test Float64(compiled(ru)) ≈ native atol=1e-10 rtol=1e-10
    if reverse_supported
        println("EVIDENCE_BACKEND_STAGE reverse");flush(stdout)
        cad = compile_ad_value_and_gradient(sampler.ad, ru; optimize)
        rvalue, rgrad = cad(ru)
        @test Float64(rvalue) ≈ value atol=1e-10 rtol=1e-10
        @test Array(rgrad) ≈ grad atol=1e-9 rtol=1e-9
    else
        # Mixed beta-binomial clamp arms: default reverse aborts on mismatched
        # slices; only_enzyme cannot remove the retained count loop's cache.
        # Backend-only reproducers and exact boundaries are in constraints.md.
        @test_skip false
    end
    ops = Dict{String,Int}()
    for match in eachmatch(r"stablehlo\.[a-z_]+", hlo)
        ops[match.match] = get(ops, match.match, 0) + 1
    end
    return ops
end

function _evidence_backend_case(expr,data,q;optimize=nothing,reverse_supported=true)
    bound=bind_data(lower_rkppl(expr,data;conditioned=(:y,)),data)
    built=build_kernel(bound)
    u=unconstrain(built.layout,q)
    k=prepare_query(built,bound,:sampler)
    return Base.invokelatest(_evidence_backend_measure,built,bound,k,u,optimize,reverse_supported)
end

@testset "response evidence: native and compiled primal, reverse and structure" begin
    for family in (:normal,:student,:lognormal,:weibull,:exponential,:bernoulli,:mixture,:inversegaussian,:betabinomial), kind in (:truncated,:censored,:interval_censored)
        selected=get(ENV,"RK_EVIDENCE_BACKEND_FAMILIES","")
        isempty(selected) || string(family) in split(selected,',') || continue
        selected_kind=get(ENV,"RK_EVIDENCE_BACKEND_KIND","")
        isempty(selected_kind) || string(kind)==selected_kind || continue
        structures=Dict{String,Int}[]
        for n in (6,12,24)
            println("EVIDENCE_BACKEND_BEGIN ",family," ",kind," n=",n);flush(stdout)
            ctor=family===:normal ? :(Normal.(eta,s)) : family===:student ? :(StudentT.(5.,eta,s)) :
                 family===:lognormal ? :(LogNormal.(eta,s)) : family===:weibull ? :(Weibull.(s,exp.(eta))) :
                 family===:exponential ? :(Exponential.(exp.(eta))) : family===:bernoulli ? :(Bernoulli.(logistic.(eta))) :
                 family===:inversegaussian ? :(InverseGaussian.(exp.(eta),s)) :
                 family===:betabinomial ? :(BetaBinomial2.(5,logistic.(eta),s)) :
                 :(MixtureModel.(vcat.(Normal.(eta,s),Normal.(-1.,.7)),Ref([.3,.7])))
            lo,hi=family===:bernoulli ? (.2,1.) : (.2,2.8)
            wrap=kind===:interval_censored ? Expr(:.,kind,Expr(:tuple,ctor,:hi)) : Expr(:.,kind,Expr(:tuple,ctor,:lo,:hi))
            # Live bounds depend on a separate parameter; their reverse path is
            # exercised along with the response parameters and scale.
            expr=quote a ~ Normal(0,1); b ~ Normal(0,1); s ~ Exponential(1);
                eta=a .+ 0*x; lo=$lo + b; hi=$hi + b; y .~ $wrap end
            ys=kind===:censored ? [i%3==1 ? lo : i%3==2 ? 1. : hi for i in 1:n] :
               kind===:interval_censored ? fill(lo,n) : fill(1.,n)
            # Bernoulli moving endpoints are a discrete step law. Avoid placing
            # an observation on a moving atom for finite-difference checks.
            if family===:bernoulli || kind===:censored
                expr=quote a ~ Normal(0,1); b ~ Normal(0,1); s ~ Exponential(1);
                    eta=a .+ 0*x; lo=$lo; hi=$hi; y .~ $wrap end
            end
            family===:bernoulli && kind===:truncated && (ys=Int.(ys))
            data=Dict(:y=>ys,:x=>zeros(n))
            # Mixed clamp arms produce chained gathers of the live location.
            # Reactant's default slice optimizer aborts this reverse shape;
            # the backend-only partition-gather reproducer records the gap.
            optimize=kind===:censored ?
                (family===:betabinomial ? :no_slice_slice :
                 family in (:lognormal,:weibull) ? :only_enzyme : nothing) : nothing
            reverse_supported=true
            push!(structures,_evidence_backend_case(expr,data,(a=.2,b=0.,s=1.3);optimize,reverse_supported))
        end
        # Small arm sizes can switch slice/gather and scalar-fold choices.
        # Control-flow regions do not replicate at any size; the complete
        # operation multiset is constant across the larger observation sizes.
        for op in ("stablehlo.if","stablehlo.while")
            @test length(unique(get(d,op,0) for d in structures))==1
        end
        @test structures[2]==structures[3]
    end
end
