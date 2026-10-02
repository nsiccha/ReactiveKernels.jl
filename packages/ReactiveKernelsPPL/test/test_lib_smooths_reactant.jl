using Test, ReactiveKernels, ReactiveKernelsPPL, ReactiveKernelsDistributionKernels

# Grow observation count, basis width and group count together. The
# backend operation multiset must stay fixed, and compiled primal/reverse
# values must agree with native execution. Data bases run before tracing.
using Distributions, DifferentiationInterface, Enzyme, Reactant, SpecialFunctions
function _lsr_fixture(kind, K, n, G)
    data = Dict(:x => [sin(0.7i) + i/n for i in 1:n],
        :z => [cos(0.5i) + i/n for i in 1:n],
        :y => [0.2sin(i) for i in 1:n], :g => [mod1(i, G) for i in 1:n])
    effect = if kind === :spline
        quote (X, Z) = tps_basis(x; k=$K); f ~ penalized_smooth(X, Z) end
    elseif kind === :t2
        quote (X, rr, rn, nr) = t2_basis(x, z; k=($K, $K)); f ~ t2_smooth(X, rr, rn, nr) end
    elseif kind === :grouped
        quote (P, lam) = hsgp_basis(x; k=$K, by=g); f ~ hsgp_grouped_effect(P, lam, g) end
    elseif kind === :periodic
        quote (P, h) = hsgp_periodic_basis(x; k=$K, period=2.0); f ~ hsgp_periodic_effect(P, h) end
    else
        quote (P, lam) = hsgp_basis(x; k=$K); f ~ hsgp_effect(P, lam) end
    end
    ast = quote a ~ Normal(0, 1) end
    append!(ast.args, effect.args)
    push!(ast.args, :(mu = a .+ f), :(y .~ Normal.(mu, 1.5)))
    bound = bind_data(lower_rkppl(ast, keys(data); conditioned = keys(data)), data)
    built = build_kernel(bound)
    return (; bound, built)
end
@testset "Reactant: library smooths" begin
    for kind in (:spline, :t2, :hsgp, :grouped, :periodic)
        @testset "$kind" begin
            structures = Dict{String,Int}[]
            for (K, n, G) in ((3, 9, 2), (5, 18, 3))
                println("COMPILED_BEGIN ", kind, " K=", K, " n=", n, " G=", G)
                flush(stdout)
                fx = _lsr_fixture(kind, K, n, G)
                u = [0.1cos(i) for i in 1:fx.built.layout.total]
                kern = prepare_query(fx.built, fx.bound, :sampler)
                ru = Reactant.to_rarray(u)
                hlo = try
                    repr(Reactant.@code_hlo optimize=false kern(ru))
                catch e
                    # Reactant 0.2.289 has no arbitrary-order besselix on
                    # traced scalars. Pin only that backend signature;
                    # every other error remains an acceptance failure.
                    if kind === :periodic && e isa MethodError &&
                            e.f === SpecialFunctions.besselix &&
                            length(e.args) == 2 && e.args[1] isa Integer &&
                            e.args[2] isa Reactant.TracedRNumber
                        @test_broken false
                        continue
                    end
                    rethrow()
                end
                ops = Dict{String,Int}()
                for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
                    ops[m.match] = get(ops, m.match, 0) + 1
                end
                push!(structures, ops)
                compiled = Reactant.@compile kern(ru)
                @test Float64(compiled(ru)) ≈ Base.invokelatest(kern, u) rtol=1e-9
                q = prepare_sampler(fx.built, fx.bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
                value, grad = sampler_value_and_gradient!(q, similar(u), u)
                cad = compile_ad_value_and_gradient(q.ad, ru)
                rvalue, rgrad = cad(ru)
                @test Float64(rvalue) ≈ value rtol=1e-9
                @test Array(rgrad) ≈ grad rtol=1e-9 atol=1e-10
                println("COMPILED_PASS ", kind, " K=", K, " n=", n, " G=", G, " ops=", ops)
                flush(stdout)
            end
            # Both periodic shapes may hit the pinned backend limitation.
            # A partially supported pair still fails rather than hiding it.
            isempty(structures) || @test structures[1] == structures[2]
        end
    end
end
