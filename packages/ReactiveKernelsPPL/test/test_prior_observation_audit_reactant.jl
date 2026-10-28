using Reactant

function _audit_backend_structure(sampler,ru)
    primal=sampler.kernel
    ad=sampler.ad
    both(v)=ad_value_and_gradient(ad,v)
    # Compare complete operation counts in the ordinary executable pipeline.
    # Transient unoptimized broadcasts can change when a larger shape folds
    # a scalar; their ordered listing does not measure body replication.
    counts=Dict{String,Int}()
    for (prefix,hlo) in (("primal.",repr(Reactant.@code_hlo primal(ru))),
            ("reverse.",repr(Reactant.@code_hlo both(ru))))
        for m in eachmatch(r"\b(?:stablehlo|chlo|enzyme|func|arith|scf|tensor)\.\w+",hlo)
            key=prefix*m.match
            counts[key]=get(counts,key,0)+1
        end
    end
    return counts
end

function _audit_backend_check(bound,built,q,oracle)
    u=unconstrain(built.layout,q)
    sampler=prepare_sampler(built,bound,u;backend=_GEN_BACKEND)
    native,gradient=sampler_value_and_gradient!(sampler,similar(u),u)
    @test native ≈ oracle(u) atol=1e-11
    @test gradient ≈ _findiff_grad(oracle,u) atol=1e-7 rtol=1e-5
    ru=Reactant.to_rarray(u)
    compiled=Reactant.@compile sampler.kernel(ru)
    reverse=compile_ad_value_and_gradient(sampler.ad,ru)
    for shift in (0.0,0.17)
        v=u .+ shift
        rv=Reactant.to_rarray(v)
        @test Float64(compiled(rv)) ≈ oracle(v) atol=1e-10
        value,grad=reverse(rv)
        @test Float64(value) ≈ oracle(v) atol=1e-10
        @test Array(grad) ≈ _findiff_grad(oracle,v) atol=1e-7 rtol=1e-5
    end
    return sampler,u,Base.invokelatest(_audit_backend_structure,sampler,ru)
end

@testset "later prior and ordinal audit: compiled primal, reverse and retained structure" begin
    structures=Dict{String,Dict{String,Int}}()
    for n in (12,24),case in _audit_cases(n)
        @testset "$(case.label) n=$n" begin
            println("AUDIT_COMPILED ",case.label," n=",n);flush(stdout)
            bound=bind_data(lower_rkppl(case.expr,case.data;conditioned=keys(case.data)),case.data)
            built=build_kernel(bound)
            oracle=w->begin q=constrain(built.layout,w);case.oracle(q)+case.jac(q) end
            _,_,ops=_audit_backend_check(bound,built,case.q,oracle)
            @test !isempty(ops)
            if n == 12
                structures[case.label]=ops
            else
                @test ops == structures[case.label]
            end
        end
    end
end

@testset "ordinary cumulative support: compiled inactive logarithms" begin
    case=first(_audit_cases(12))
    bound=bind_data(lower_rkppl(case.expr,case.data;conditioned=keys(case.data)),case.data)
    built=build_kernel(bound)
    u=unconstrain(built.layout,case.q)
    sampler=prepare_sampler(built,bound,u;backend=_GEN_BACKEND)
    ru=Reactant.to_rarray(u)
    compiled=Reactant.@compile sampler.kernel(ru)
    reverse=compile_ad_value_and_gradient(sampler.ad,ru)
    for cuts in ([0.8,-0.7],[0.8,0.8])
        v=unconstrain(built.layout,(b=0.25,c=cuts))
        rv=Reactant.to_rarray(v)
        @test Float64(compiled(rv)) == -Inf
        value,gradient=reverse(rv)
        @test Float64(value) == -Inf
        @test Array(gradient) ≈ -v atol=1e-12
    end
end

@testset "scalar observed Binomial: compiled doors and structure" begin
    for door in (:names,:values,:public)
        structures=Dict{String,Int}[]
        for (n,k) in ((5,2),(9,4),(0,0))
            println("AUDIT_COMPILED scalar ",door," n=",n);flush(stdout)
            bound=_audit_scalar_binomial(door,n,k)
            built=build_kernel(bound)
            oracle=w->begin
                p=1/(1+exp(-w[1]))
                D.logpdf(D.Beta(1,1),p)+D.logpdf(D.Binomial(n,p),k)+log(p)+log1p(-p)
            end
            _,_,ops=_audit_backend_check(bound,built,(theta=0.37,),oracle)
            n == 0 || push!(structures,ops)
        end
        @test structures[1] == structures[2]
    end
end

@testset "scalar observed Binomial: compiled out-of-support observations" begin
    for k in (-1,6,2.5)
        bound=_audit_scalar_binomial(:public,5,k)
        built=build_kernel(bound)
        u=unconstrain(built.layout,(theta=0.37,))
        sampler=prepare_sampler(built,bound,u;backend=_GEN_BACKEND)
        ru=Reactant.to_rarray(u)
        compiled=Reactant.@compile sampler.kernel(ru)
        reverse=compile_ad_value_and_gradient(sampler.ad,ru)
        @test Float64(compiled(ru)) == -Inf
        value,gradient=reverse(ru)
        @test Float64(value) == -Inf
        @test Array(gradient) ≈ [1-2*0.37] atol=1e-12
    end
end
