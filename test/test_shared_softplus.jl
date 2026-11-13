module SharedSoftplusTests
using Test, ReactiveKernels, MutatingFunctions
using DifferentiationInterface: AutoEnzyme
using LogExpFunctions: log1pexp, logistic
import Enzyme

@kernel softplus_total(x::Vector{Float64}, seed::Float64) = begin
    cells = plate(x) do xi
        rk_log1pexp(xi)
    end
    total::Float64 = seed * sum(cells)
    return total
end

@testset "shared stable softplus value and partial" begin
    f = rk_log1pexp
    setprecision(BigFloat, 512) do
        for x in (-1000.0,-745.0,-700.0,-40.0,-20.0,-1.0,-eps(),-0.0,
                0.0,eps(),1.0,20.0,40.0,700.0,1000.0)
            ref = Float64(log1p(exp(BigFloat(x))))
            dref = Float64(inv(1+exp(-BigFloat(x))))
            v,d = derivative_cut(f,(true,),x)
            @test v ≈ ref rtol=2eps(Float64)
            @test d ≈ dref rtol=2eps(Float64)
            @test f(x) == v
            for seed in (-1.3,0.0,1.0)
                wrapped = z -> seed*f(z)
                dr = Enzyme.autodiff(Enzyme.Reverse,wrapped,Enzyme.Active(x))[1][1]
                df = Enzyme.autodiff(Enzyme.Forward,f,Enzyme.Duplicated(x,seed))[1]
                @test dr ≈ seed*dref rtol=4eps(Float64)
                @test df ≈ seed*dref rtol=4eps(Float64)
            end
        end
    end
    @test derivative_cut(f,(true,),-Inf)==(0.0,0.0)
    @test derivative_cut(f,(true,),Inf)==(Inf,1.0)
    @test all(isnan,derivative_cut(f,(true,),NaN))

    backend=AutoEnzyme(;mode=Enzyme.Reverse)
    for nonallocating in (false,true), enabled in (false,true)
        x0=[-.2,.3,40.0]
        op=with_plate_reverse(enabled) do
            k=nonallocating ? prepare_nonallocating(softplus_total,x0,1.0) : prepare(softplus_total)
            prepare_ad(k,backend,x0,1.0;active=:x)
        end
        for n in (0,3,257), seed in (-1.3,0.0,1.0)
            x=[sin(i)*100 for i in 1:n]
            saved=copy(x)
            g=fill(99.0,n)
            value,gradient=ad_value_and_gradient!(op,g,x,seed)
            @test value ≈ seed*sum(log1pexp,x;init=0.0)
            @test gradient ≈ seed.*logistic.(x)
            @test gradient === g
            @test x == saved
        end
    end
end
end
