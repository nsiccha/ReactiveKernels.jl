using DifferentiationInterface, Enzyme, Reactant, ReactiveKernels, ReactiveKernelsPPL, Test

function _max_fixture(kind, n, m)
    cols = Dict{Symbol,Any}(:y => collect(range(-0.3, 0.5; length=n)),
        :x => collect(range(-1.0, 2.0; length=n)),
        :other => collect(range(0.2, 0.6; length=m)))
    dims = Dict{Symbol,Int}()
    ast = if kind === :plate
        delete!(cols, :x)
        quote
            @plate for i in eachindex(y)
                z[i] ~ Normal(0, 1)
                y[i] ~ Normal(z[i], 1)
            end
        end
    elseif kind === :scan
        delete!(cols, :x)
        quote
            phi ~ Normal(0, 0.5)
            @scan begin
                h[1] ~ Normal(0, 1)
                for t in 2:T
                    h[t] ~ Normal(phi * h[t-1], 1)
                end
            end
            y .~ Normal.(h, 1)
        end
    elseif kind === :array
        cols[:rows] = collect(1:n)
        quote
            X = hcat(x)
            center = x .+ 0.2
            z[axes(X, 1)] .~ Normal.(center, 1)
            mu = z[rows]
            y .~ Normal.(mu, 1)
        end
    elseif kind === :matrix
        quote
            X = hcat(x)
            a ~ Normal(0, 1)
            b[axes(X, 2)] .~ Normal.(0, 1)
            mu = a .+ X * b
            y .~ Normal.(mu, 1)
        end
    elseif kind === :kernel
        cols[:kx] = collect(range(-0.5, 1.0; length=n+2))
        cols[:ky] = collect(range(-0.2, 0.4; length=n+2))
        dims[:n_kernel] = n + 2
        quote
            a ~ Normal(0, 1)
            mu = a .* x
            y .~ Normal.(mu, 1)
            pk ~ plate(kx, ky; subjects=n_kernel) do xx, yy
                kmu = a .* xx
                yy .~ Normal.(kmu, 1)
                kmu
            end
        end
    else
        error("unknown fixture $kind")
    end
    push!(ast.args, :(other .~ Normal.(0, 0.7)))
    bound = bind_data(lower_rkppl(ast, cols; conditioned=keys(cols)), cols; dims)
    built = build_kernel(bound)
    u = [-0.3 + 0.7 * (i-1) / max(built.layout.total-1, 1) for i in 1:built.layout.total]
    return (; bound, built, u)
end

function _max_compiled(fx)
    q = prepare_query(fx.built, fx.bound, :sampler)
    ru = Reactant.to_rarray(fx.u)
    hlo = repr(Reactant.@code_hlo optimize=false q(ru))
    ops = Dict{String,Int}()
    for m in eachmatch(r"(?:stablehlo|enzyme|chlo|func|arith)\.[a-z_]+", hlo)
        ops[m.match] = get(ops, m.match, 0) + 1
    end
    cq = Reactant.@compile q(ru)
    value = Base.invokelatest(q, fx.u)
    @test Float64(cq(ru)) ≈ value rtol=1e-9
    ad = prepare_ad(q, AutoEnzyme(; mode=Enzyme.Reverse), fx.u; active=:unconstrained)
    _, gradient = Base.invokelatest(ad_value_and_gradient!, ad, similar(fx.u), fx.u)
    cad = compile_ad_value_and_gradient(ad, ru)
    cv, cg = cad(ru)
    @test Float64(cv) ≈ value rtol=1e-9
    @test Array(cg) ≈ gradient rtol=1e-8 atol=1e-9
    h = cbrt(eps(Float64))
    ref = map(eachindex(fx.u)) do i
        hi, lo = copy(fx.u), copy(fx.u)
        hi[i] += h
        lo[i] -= h
        (Base.invokelatest(q, hi) - Base.invokelatest(q, lo)) / (2h)
    end
    @test Array(cg) ≈ ref rtol=2e-5 atol=2e-7
    return ops
end

@testset "Reactant: multiple axes preserve array and recurrence structure" begin
    @testset "$kind" for kind in (:plate, :scan, :array, :matrix, :kernel)
        structures = Dict{String,Int}[]
        for (n, m) in ((8, 3), (17, 6))
            fx = _max_fixture(kind, n, m)
            @test fx.bound.n_obs == n + m + (kind === :kernel ? n + 2 : 0)
            push!(structures, Base.invokelatest(_max_compiled, fx))
        end
        # Compare every operation, including control-flow regions. Growing
        # either observation axis cannot duplicate the data loop's body.
        @test structures[1] == structures[2]
    end
end
