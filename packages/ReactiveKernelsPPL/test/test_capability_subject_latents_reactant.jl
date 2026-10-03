using DifferentiationInterface, Enzyme, Reactant

function _cap_subject_operations(hlo)
    out = Dict{String, Int}()
    for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
        out[m.match] = get(out, m.match, 0)+1
    end
    out
end

# The original grouped-PK fixture is accepted natively in test_plates.jl.
# This compiled prior cut prunes the known system-matrix batching boundary.
@testset "Reactant: subject-axis latent priors retain their own range" begin
    primal, reverse = Dict{String, Int}[], Dict{String, Int}[]
    for copies in (1, 3)
        original = _pl_pk_cols()
        cols = Dict{Symbol, AbstractVector}()
        for (name, values) in original
            cols[name] = name in (:subj, :dsubj) ?
                vcat((values .+ 2*i for i in 0:copies-1)...) : repeat(values, copies)
        end
        ast = _pl_pk_chain(:(dv .~ Normal.(conc, sigma)),
            Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
                Expr(:for, :(s = eachindex(age_s)), Expr(:block,
                    :(eta[s] ~ Normal(0, 1))))))
        bound, built = _pl_bind(ast, _PL_PK_DATA, cols)
        u = [0.1cos(i) for i in 1:built.layout.total]
        q = constrain(built.layout, u)
        expected = logpdf(Exponential(), q.sigma) +
            sum(logpdf(Normal(), getproperty(q, name)) for name in
                (:b0_vc, :b1_vc, :b0_k10, :b0_k12, :b0_k21, :b0_ka)) +
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
