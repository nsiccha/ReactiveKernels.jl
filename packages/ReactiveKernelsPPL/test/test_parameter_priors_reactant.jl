using Test, ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme, Reactant

function _pp_backend_check(expr, data, q)
    bound = bind_data(lower_rkppl(expr, data; conditioned = data), data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, q)
    kernel = prepare_query(built, bound, :sampler)
    return Base.invokelatest(_pp_backend_measure, built, bound, kernel, u)
end

function _pp_backend_measure(built, bound, kernel, u)
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
    ru = Reactant.to_rarray(u)
    hlo = repr(Reactant.@code_hlo optimize=false kernel(ru))
    compiled = Reactant.@compile kernel(ru)
    @test Float64(compiled(ru)) ≈ native atol=1e-10 rtol=1e-10
    cad = compile_ad_value_and_gradient(sampler.ad, ru)
    rvalue, rgrad = cad(ru)
    @test Float64(rvalue) ≈ value atol=1e-10 rtol=1e-10
    @test Array(rgrad) ≈ grad atol=1e-9 rtol=1e-9
    ops = Dict{String,Int}()
    for match in eachmatch(r"stablehlo\.[a-z_]+", hlo)
        ops[match.match] = get(ops, match.match, 0) + 1
    end
    return ops
end

@testset "dynamic priors: native and compiled primal, reverse and structure" begin
    for family in (:normal, :cauchy, :weibull)
        structs = Dict{String,Int}[]
        for n in (3, 7)
            distribution = family === :normal ? :(Normal(a, 1 + exp(a))) :
                family === :cauchy ? :(Cauchy(a, 1 + exp(a))) : :(Weibull(2 + exp(a), 1))
            expr = quote
                a ~ Normal(0, 1)
                x ~ truncated($distribution, exp(a), 3 + exp(a))
                y .~ Normal.(x, 1)
            end
            println("BACKEND_BEGIN ", family, " n=", n); flush(stdout)
            push!(structs, _pp_backend_check(expr, Dict(:y => fill(0.2, n)), (a=0.1, x=1.7)))
        end
        @test structs[1] == structs[2]
    end
end

@testset "hierarchical Dirichlet: native and compiled reverse and structure" begin
    structs = Dict{String,Int}[]
    for k in (3, 7)
        expr = quote
            a ~ Exponential(1)
            p ~ Dirichlet($k, a)
            mu = p[1] .* z
            y .~ Normal.(mu, 1)
        end
        println("BACKEND_BEGIN dirichlet k=", k); flush(stdout)
        push!(structs, _pp_backend_check(expr, Dict(:y => fill(0.2, 5), :z => ones(5)),
            (a=1.2, p=fill(1/k,k))))
    end
    @test length(structs) == 2 && structs[1] == structs[2]
end

_numeric_backend_scale(x) = 2x

@testset "integer prior arguments: native and compiled reverse and structure" begin
    for truncated in (false, true)
        structures = Dict{String,Int}[]
        for n in (3, 7)
            prior = truncated ? :(truncated(Normal(0, sd), lo, hi)) : :(Normal(0, sd))
            expr = quote
                sd = _numeric_backend_scale(seed)
                x ~ $prior
                y .~ Normal.(x, 1)
            end
            data = Dict{Symbol,Any}(:seed=>2, :lo=>-1, :hi=>3, :y=>fill(0.2,n))
            names = ReactiveKernelsPPL._value_symbols(expr)
            filter!(p -> first(p) in names, data)
            before = deepcopy(data)
            push!(structures, _pp_backend_check(expr, data, (x=0.3,)))
            @test data == before
        end
        @test structures[1] == structures[2]
    end
end
