using Test, Reactant, SpecialFunctions
isdefined(@__MODULE__, :_sc_build) || include("test_smooth_capabilities_helpers.jl")

function _scr_fixture(kind, K, n, G)
    data = _sc_data(n)
    data[:g] = [mod1(i,G) for i in 1:n]
    effect = if kind === :cr
        quote kk = min(length(x),$K); (X,Z) = cr_basis(x;k=kk); f ~ penalized_smooth(X,Z) end
    elseif kind === :tps2
        quote (X,Z) = tps_basis(x,z;k=$(K+2)); f ~ penalized_smooth(X,Z) end
    elseif kind === :t2one
        quote (X,Z) = t2_basis(x;k=$K); f ~ penalized_smooth(X,Z) end
    elseif kind === :t2three
        _sc_tensor_body(K)
    elseif kind === :shared
        quote
            (X,Z) = cr_basis(x;k=$K)
            f ~ penalized_smooth(X,Z)
            mu = f .+ x
            nu = f .- x
            y .~ Normal.(mu,1.0)
            z .~ Normal.(nu,1.0)
        end
    elseif kind === :matern_aniso
        quote
            (P,lam) = hsgp_basis(x,z;k=$K)
            rho[1:2] .~ LogNormal.(0,1)
            sigma ~ LogNormal(0,1)
            raw[axes(P,2)] .~ Normal.(0,1)
            f = P * (hsgp_matern_sqrt_spd(lam,sigma,rho,1.5) .* raw)
        end
    elseif kind in (:matern32,:matern52)
        nu = kind === :matern32 ? 1.5 : 2.5
        quote (P,lam) = hsgp_basis(x,z;k=$K); f ~ _sc_matern(P,lam,$nu) end
    elseif kind === :grouped2
        quote (P,lam) = hsgp_basis(x,z;k=$K,by=g .+ 1); f ~ hsgp_grouped_effect(P,lam,g) end
    elseif kind === :periodic2
        quote
            (P,h) = hsgp_periodic_basis(x,z;k=$K,period=(2.0,3.0))
            f ~ hsgp_periodic_effect(P,h)
        end
    elseif kind === :periodic_aniso
        quote
            (P,h) = hsgp_periodic_basis(x,z;k=$K,period=(2.0,3.0))
            rho_1 ~ LogNormal(0,1); rho_2 ~ LogNormal(0,1); sigma ~ LogNormal(0,1)
            raw[axes(P,2)] .~ Normal.(0,1)
            f = P * (hsgp_periodic_sqrt_spd(h,sigma,[rho_1,rho_2]) .* raw)
        end
    elseif kind === :grouped_periodic
        quote
            (P,h) = hsgp_periodic_basis(x,z;k=$K,period=(2.0,3.0),by=g)
            f ~ _sc_grouped_periodic(P,h,g)
        end
    else
        error("unknown compiled smooth fixture $kind")
    end
    kind === :shared || push!(effect.args, :(y .~ Normal.(f .+ x,1.0)))
    return _sc_build(effect,data)
end

@testset "Reactant: extended smooth capabilities" begin
    for kind in (:cr,:tps2,:t2one,:t2three,:shared,:matern32,:matern52,:matern_aniso,:grouped2,
            :periodic2,:periodic_aniso,:grouped_periodic)
        @testset "$kind" begin
            structures = Dict{String,Int}[]
            for (K,n,G) in ((3,9,2),(4,18,3))
                println("CAPABILITY_COMPILED_BEGIN ",kind," K=",K," n=",n," G=",G)
                flush(stdout)
                fx = _scr_fixture(kind,K,n,G)
                # Check the new per-axis parameters against finite
                # differences even when periodic tracing hits the Bessel gap.
                kind in (:periodic_aniso,:matern_aniso) && _sc_gradient(fx)
                kern = prepare_query(fx.built,fx.bound,:sampler)
                ru = Reactant.to_rarray(fx.u)
                hlo = try
                    repr(Reactant.@code_hlo optimize=false kern(ru))
                catch e
                    # Same measured backend gap as test_lib_smooths_reactant:
                    # arbitrary integer-order scaled Bessel on a traced scalar.
                    if kind in (:periodic2,:periodic_aniso,:grouped_periodic) &&
                            e isa MethodError && e.f === SpecialFunctions.besselix &&
                            length(e.args) == 2 && e.args[1] isa Integer &&
                            e.args[2] isa Reactant.TracedRNumber
                        @test_broken false
                        continue
                    end
                    rethrow()
                end
                ops = Dict{String,Int}()
                for m in eachmatch(r"stablehlo\.[a-z_]+",hlo)
                    ops[m.match] = get(ops,m.match,0)+1
                end
                push!(structures,ops)
                compiled = Reactant.@compile kern(ru)
                @test Float64(compiled(ru)) ≈ Base.invokelatest(kern,fx.u) rtol=1e-9
                q = prepare_sampler(fx.built,fx.bound,fx.u;
                    backend=AutoEnzyme(;mode=Enzyme.Reverse))
                value,grad = sampler_value_and_gradient!(q,similar(fx.u),fx.u)
                cad = compile_ad_value_and_gradient(q.ad,ru)
                rvalue,rgrad = cad(ru)
                @test Float64(rvalue) ≈ value rtol=1e-9
                @test Array(rgrad) ≈ grad rtol=1e-9 atol=1e-10
                println("CAPABILITY_COMPILED_PASS ",kind," K=",K," n=",n," G=",G," ops=",ops)
                flush(stdout)
            end
            isempty(structures) || @test structures[1] == structures[2]
        end
    end
end
