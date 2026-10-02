using CategoricalArrays
using DifferentiationInterface: AutoEnzyme
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# More observations change bound shapes, never the number of mathematical
# operations or control-flow regions in either the primal or derivative.
function _semantics_compiled_fixture(kind, n)
    x = [sin(0.7i) for i in 1:n]
    y = [cos(0.3i) for i in 1:n]
    data = Dict{Symbol,Any}(:x => x, :y => y)
    if kind === :factor
        # Grow the unobserved categorical pool too: its prior dimension is
        # data-dependent and must not replicate the generated prior body.
        pool = vcat(["low", "mid", "high"],
            ["extra$i" for i in 1:(n ÷ 4 - 2)])
        g = categorical([pool[mod1(i, 3)] for i in 1:n]; levels = pool)
        data[:g] = g
        ast = quote
            a ~ Normal(0, 1)
            c[levels(g)[2:end]] .~ Normal.(0, 1)
            mu = a .+ c[g]
            y .~ Normal.(mu, 1)
        end
    elseif kind === :matrix
        ast = quote
            b[axes(X, 2)] .~ Normal.(0, 1)
            X = hcat(ones(length(x)), x)
            mu = X * b
            y .~ Normal.(mu, 1)
        end
    elseif kind === :mixture
        ast = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            y .~ MixtureModel.(vcat.(Normal.(mu, 1), Normal.(1, 1)), Ref([0.4, 0.6]))
        end
    elseif kind === :categorical
        data[:y] = [mod1(i, 3) for i in 1:n]
        ast = quote
            s ~ Dirichlet([1.0, 2.0, 3.0])
            y .~ Categorical(s)
        end
    elseif kind === :multinomial
        data = Dict{Symbol,Any}(:c1 => [mod1(i, 3) for i in 1:n],
            :c2 => ones(Int, n), :c3 => zeros(Int, n))
        data[:N] = data[:c1] .+ data[:c2]
        ast = quote
            s ~ Dirichlet([1.0, 2.0, 3.0])
            eachrow(hcat(c1, c2, c3)) .~ Multinomial.(N, Ref(s))
        end
    elseif kind in (:normcdf, :cexpexp)
        data[:y] = [isodd(i) for i in 1:n]
        prob = Expr(:., kind, Expr(:tuple, :eta))
        ast = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ Bernoulli.($prob)
        end
    elseif kind in (:censored, :truncated)
        data[:y] = [Float64(mod1(i, 3) - 1) for i in 1:n]
        dist = Expr(:., kind, Expr(:tuple, :(Normal.(mu, 1)), 0, 2))
        ast = quote
            a ~ Normal(0, 1)
            mu = a .+ 0 .* x
            y .~ $dist
        end
    else
        ast = quote
            a ~ Normal(0, 1)
            y .~ Normal.(a, 1)
        end
    end
    bound = bind_data(lower_rkppl(ast, data), data)
    built = build_kernel(bound)
    u = collect(range(-0.2; step = 0.15, length = built.layout.total))
    sampler = prepare_sampler(built, bound, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    return Base.invokelatest(_semantics_compiled_measure, sampler, u)
end

function _semantics_compiled_measure(sampler, u)
    native, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
    ru = Reactant.to_rarray(u)
    post, ad = sampler.kernel, sampler.ad
    primal_hlo = repr(Reactant.@code_hlo optimize = false post(ru))
    both(v) = ad_value_and_gradient(ad, v)
    derivative_hlo = repr(Reactant.@code_hlo optimize = false both(ru))
    operations = Dict{String,Int}()
    for (prefix, hlo) in (("primal.", primal_hlo), ("ad.", derivative_hlo))
        for m in eachmatch(r"\b(?:stablehlo|chlo|enzyme|func|arith)\.\w+", hlo)
            key = prefix * m.match
            operations[key] = get(operations, key, 0) + 1
        end
    end
    @test !isempty(operations)
    compiled = Reactant.@compile post(ru)
    @test Float64(compiled(ru)) ≈ native rtol = 1e-9
    compiled_ad = compile_ad_value_and_gradient(ad, ru)
    value, grad = compiled_ad(ru)
    @test Float64(value) ≈ native rtol = 1e-9
    @test Array(grad) ≈ gradient rtol = 1e-8 atol = 1e-9
    return operations
end

# Existing simplex traced-view boundary (reactivekernels-use §7r), also
# pinned by test_leveled_reactant.jl and test_mixture.jl. No substitute
# math or AD rule is introduced to get around this backend limitation.
_semantics_simplex_gap(e) = e isa MethodError && (
    (e.f === Base.reindex && length(e.args) == 2 && !(e.args[2] isa Tuple)) ||
    (e.f === Float64 && length(e.args) == 1 && e.args[1] isa Reactant.TracedRNumber))

@testset "semantics: compiled parity and observation-count invariance" begin
    for kind in (:factor, :matrix, :mixture, :normcdf, :cexpexp,
            :censored, :truncated, :iid, :categorical, :multinomial)
        @testset "$kind" begin
            try
                small = _semantics_compiled_fixture(kind, 12)
                large = _semantics_compiled_fixture(kind, 24)
                @test small == large
                if kind in (:categorical, :multinomial)
                    @test_broken true # self-firing when the pinned gap lifts
                end
            catch e
                kind in (:categorical, :multinomial) && _semantics_simplex_gap(e) || rethrow()
                @test_broken false
            end
        end
    end
end
