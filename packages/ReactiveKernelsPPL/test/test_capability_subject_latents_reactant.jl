using DifferentiationInterface, Enzyme, Reactant

function _cap_subject_operations(hlo)
    out = Dict{String, Int}()
    for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
        out[m.match] = get(out, m.match, 0)+1
    end
    out
end

# A latent axis and an observation axis own independent authored extents.
@testset "Reactant: independent latent priors retain their own range" begin
    primal, reverse = Dict{String, Int}[], Dict{String, Int}[]
    for copies in (1, 3)
        cols = Dict{Symbol, AbstractVector}(:domain => zeros(2*copies),
            :y => zeros(4*copies))
        ast = quote
            sigma ~ Exponential(1)
            a ~ Normal(0, 1)
            @plate for s in eachindex(domain)
                eta[s] ~ Normal(0, 1)
            end
            y .~ Normal.(a, sigma)
        end
        bound, built = _pl_bind(ast, (:domain, :y), cols)
        u = [0.1cos(i) for i in 1:built.layout.total]
        q = constrain(built.layout, u)
        expected = logpdf(Exponential(), q.sigma) +
            logpdf(Normal(), q.a) +
            sum(logpdf.(Normal(), q.eta))
        @test length(q.eta) == 2*copies
        @test bound.n_obs == 4*copies
        kernel = prepare_query(built, bound, :prior)
        ad = Base.invokelatest(prepare_ad, kernel, AutoEnzyme(; mode=Enzyme.Reverse),
            u; active=:unconstrained)
        gradient = -copy(u)
        sigma = only(e for e in built.layout.entries if e.name === :sigma)
        gradient[sigma.offset] = -q.sigma
        @test Base.invokelatest(kernel, u) ≈ expected
        @test Base.invokelatest(ad_gradient, ad, u) ≈ gradient
        ru = Reactant.to_rarray(u)
        push!(primal, _cap_subject_operations(repr(Reactant.@code_hlo optimize=false kernel(ru))))
        cp = Reactant.@compile kernel(ru)
        @test Float64(cp(ru)) ≈ expected
        cad = compile_ad_value_and_gradient(ad, ru)
        rv, rg = cad(ru)
        @test Float64(rv) ≈ expected
        @test Array(rg) ≈ gradient
        grad = v -> only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(kernel), v))
        push!(reverse, _cap_subject_operations(repr(Reactant.@code_hlo optimize=:only_enzyme grad(ru))))
    end
    @test primal[1] == primal[2]
    # A larger packed port eliminates one shape broadcast. Every other
    # reverse operation is identical; no cell or derivative body grows.
    shapeop = "stablehlo.broadcast_in_dim"
    @test filter(p -> first(p) != shapeop, reverse[1]) ==
        filter(p -> first(p) != shapeop, reverse[2])
    @test get(reverse[2], shapeop, 0) <= get(reverse[1], shapeop, 0)
end
