# Reactant half of the model-only startup acceptance driver. Included by
# acceptance_startup_ad_reactant.jl ONLY when Reactant is loaded (it uses the
# `@compile` macro + Reactant types, resolved at lowering time). Asserts that a
# glm_poisson graph built from the MODEL-ONLY template compiles + evaluates
# through Reactant identically to one built from the full source (== , same
# backend), and matches native (≈, cross-backend).

using Test
using Reactant: @compile

let RX = Reactant, MR = ReactiveKernelsPPLExamples
    G = MR.GLMPoissonExample
    have = (:unconstrained, :year, :counts)
    bnd  = (; year = G.GLM_POISSON_YEAR, counts = G.GLM_POISSON_C)
    q    = [0.2, 0.1, -0.05, 0.03]

    host(v) = v isa RX.AbstractConcreteArray  ? Array(v) :
              v isa RX.AbstractConcreteNumber ? RX.to_number(v) :
              v isa Tuple                     ? map(host, v) : v
    tracev(v) = v isa AbstractArray ? RX.to_rarray(v) : RX.to_rarray(v; track_numbers = true)
    crun(k, inputs) = (tr = map(tracev, inputs); c = @compile sync = true k(tr...); host(c(tr...)))

    gmo = G.build_glm_poisson_graph()                          # model-only template
    gfu = compose(G.evaluate_glm_poisson_source().model)       # full-source model
    kk(g) = prepare(g; have = have, want = :posterior, bound = bnd)
    kmo, kfu = kk(gmo), kk(gfu)

    @testset "Reactant — model-only ≡ full (==) and matches native (≈)" begin
        nat_mo = kmo(q)
        rea_mo = crun(kmo, (q,))
        rea_fu = crun(kfu, (q,))
        @test isfinite(rea_mo)
        @test rea_mo == rea_fu                                   # same backend + graph -> exact
        @test isapprox(nat_mo, rea_mo; rtol = 1e-6, atol = 1e-7) # cross-backend -> approximate
    end
end
