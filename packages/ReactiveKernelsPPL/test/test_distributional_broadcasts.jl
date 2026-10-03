using Reactant

@testset "primary parameter links preserve matrix operand shapes" begin
    for family in _DISTRIBUTIONAL_PRIMARY_FAMILIES, link in _DISTRIBUTIONAL_LINKS[1:2]
        f = _distributional_primary_fixture(3, family, link)
        data = (; x = reshape(f.data.x, 3, 1),
            y = repeat(reshape(f.data.y, 3, 1), 1, 2))
        model = _distributional_model(f.ast, data)
        oracle = u -> 2f.oracle(u) - sum(logpdf.(Normal(), u))
        @test model.plan.n_obs == 6
        _distributional_check(model, oracle, f.u)
    end
end

function _distributional_broadcast_fixture(n, family)
    x = reshape([-0.4, 0.6], 2, 1)
    y = family in (:ZeroInflatedBinomial, :Poisson) ?
        reshape(repeat([0, 1], n), 2, n) : family === :Bernoulli ?
        reshape(repeat([false, true], n), 2, n) : fill(0.7, 2, n)
    response = if family === :StudentT
        :(y .~ StudentT.(exp.(eta), eta, logistic.(eta)))
    elseif family === :ZeroInflatedBinomial
        :(y .~ ZeroInflatedBinomial.(3, normcdf.(eta), logistic.(eta)))
    elseif family === :Poisson
        :(y .~ Poisson.(exp.(eta)))
    elseif family === :Bernoulli
        :(y .~ Bernoulli.(logistic.(eta)))
    else
        :(y .~ Normal.(eta, exp.(s)))
    end
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
    end
    family === :Normal && push!(ast.args, :(s ~ Normal(0, 1)))
    push!(ast.args, response)
    u = family === :Normal ? [0.4, 0.1, 0.2] : [0.4, 0.1]
    oracle = u -> begin
        eta = u[1] .+ u[2] .* x
        likelihood = if family === :StudentT
            sum(logpdf.(LocationScale.(eta, logistic.(eta), TDist.(exp.(eta))), y))
        elseif family === :ZeroInflatedBinomial
            p, zi = normcdf.(eta), logistic.(eta)
            sum(broadcast(y, p, zi) do observed, prob, zero
                mass = logpdf(Binomial(3, prob), observed)
                observed == 0 ? logaddexp(log(zero), log1p(-zero) + mass) :
                    log1p(-zero) + mass
            end)
        elseif family === :Poisson
            sum(logpdf.(Poisson.(exp.(eta)), y))
        elseif family === :Bernoulli
            sum(logpdf.(Bernoulli.(logistic.(eta)), y))
        else
            sum(logpdf.(Normal.(eta, exp(u[3])), y))
        end
        likelihood + sum(logpdf.(Normal(), u))
    end
    return (; ast, data = (; x, y), u, oracle)
end

@testset "distributional links retain Julia broadcast axes" begin
    operations = Dict{Symbol,Tuple{Vector{String},Vector{String}}}()
    for n in (3, 7), family in (:StudentT, :ZeroInflatedBinomial, :Normal,
            :Poisson, :Bernoulli)
        f = _distributional_broadcast_fixture(n, family)
        model = _distributional_model(f.ast, f.data)
        @test model.plan.n_obs == 2n
        @test coordinate_names(model.built.layout) ==
            (family === :Normal ? [:a, :b, :s] : [:a, :b])
        _distributional_check(model, f.oracle, f.u)
        kernel = model.kernel
        ru = Reactant.to_rarray(f.u)
        ad = Base.invokelatest(prepare_ad, kernel,
            AutoEnzyme(; mode = Enzyme.Reverse), f.u; active = :unconstrained)
        compiled = Reactant.@compile kernel(ru)
        cad = compile_ad_value_and_gradient(ad, ru)
        value, gradient = cad(ru)
        @test Float64(compiled(ru)) ≈ f.oracle(f.u)
        @test Float64(value) ≈ f.oracle(f.u)
        @test Array(gradient) ≈ _distributional_findiff(f.oracle, f.u) rtol=2e-5 atol=2e-7
        adcall = cad.f
        ops = [m.match for m in eachmatch(r"stablehlo\.[a-z_]+", string(Reactant.@code_hlo optimize=false kernel(ru)))]
        adops = [m.match for m in eachmatch(r"stablehlo\.[a-z_]+", string(Reactant.@code_hlo optimize=false adcall(ru)))]
        n == 3 ? (operations[family] = (ops, adops)) : (@test (ops, adops) == operations[family])
    end
end
