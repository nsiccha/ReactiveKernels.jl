# Durable, HARD-ASSERTING acceptance driver for the model-only startup contract's
# AD and Reactant paths. Not part of runtests.jl (Enzyme is a test-target dep but
# Reactant is a weakdep of ReactiveKernels, not a package test dep), so run it
# explicitly in the relevant environment. It asserts that a graph built from the
# MODEL-ONLY template (`build_*_graph()`) is identical to one built from the FULL
# authored source (`evaluate_*_source().model`) under each backend.
#
# Preserve backend contexts by running it in two environments:
#   * native / plain-Enzyme (Reactant NOT loaded):
#       using ReactiveKernels, ReactiveKernelsPPLExamples, DifferentiationInterface, Enzyme, Test
#       include("acceptance_startup_ad_reactant.jl")
#   * Reactant (loaded):
#       using ReactiveKernels, ReactiveKernelsPPLExamples, Reactant, Test
#       include("acceptance_startup_ad_reactant.jl")
# Native asserts always run; Enzyme asserts run iff Enzyme is loaded; Reactant
# asserts run iff Reactant is loaded. Same-backend model-only-vs-full comparisons
# assert EXACT NUMERIC equality (`==`; note `==` is exact numeric equality, e.g.
# +0.0 == -0.0, not bit identity); the only approximate assert is
# native-vs-Reactant (different backends). The finite-difference check is manual
# central differences, NOT ForwardDiff.

using Test
const _AMR = ReactiveKernelsPPLExamples

# (model_only template graph, full-source graph) for a module.
_mo_full(mod, buildfn, evalfn) =
    (getproperty(mod, buildfn)(), compose(getproperty(mod, evalfn)().model))

# Package-identity check by loaded PkgId name — stronger than `isdefined(Main, …)`
# because an INDIRECT import (a dependency loading Reactant) need not bind the
# name in Main, yet WOULD change the numerical backend context. Used to assert
# the native/Enzyme path runs with Reactant genuinely not loaded.
_pkg_loaded(name) = any(id -> id.name == name, keys(Base.loaded_modules))

@testset "startup model-only ≡ full-source graph: AD / Reactant acceptance" begin
    E = _AMR.EightSchoolsExample
    G = _AMR.GLMPoissonExample
    es_have = (:unconstrained, :observations, :observation_scales)
    es_bnd  = (; observations = E.EIGHT_SCHOOLS_Y, observation_scales = E.EIGHT_SCHOOLS_SIGMA)
    es_q    = [0.0, log(5.0), zeros(E.NSCHOOLS)...]
    gp_have = (:unconstrained, :year, :counts)
    gp_bnd  = (; year = G.GLM_POISSON_YEAR, counts = G.GLM_POISSON_C)
    gp_q    = [0.2, 0.1, -0.05, 0.03]

    @testset "native primal — exact numeric equality (==)" begin
        for (mod, bf, ef, have, bnd, q) in (
            (E, :build_eight_schools_graph, :evaluate_eight_schools_source, es_have, es_bnd, es_q),
            (G, :build_glm_poisson_graph,   :evaluate_glm_poisson_source,   gp_have, gp_bnd, gp_q),
        )
            gmo, gfu = _mo_full(mod, bf, ef)
            k(g) = prepare(g; have = have, want = :posterior, bound = bnd)
            vmo, vfu = k(gmo)(q), k(gfu)(q)
            @test isfinite(vmo)
            @test vmo == vfu
        end
    end

    if isdefined(Main, :Enzyme) && isdefined(Main, :DifferentiationInterface)
        @testset "plain-Enzyme reverse gradient — exact numeric equality (Reactant-free context)" begin
            # Context assertion by loaded-package IDENTITY, before AND after the
            # native/Enzyme path — an indirect import would not bind Main.Reactant
            # yet would change the backend, so `isdefined(Main,:Reactant)` is too
            # weak. Reactant must genuinely not be loaded around this path.
            @test !_pkg_loaded("Reactant")             # BEFORE
            BE = Main.AutoEnzyme(; mode = Main.Enzyme.Reverse)
            gmo, gfu = _mo_full(E, :build_eight_schools_graph, :evaluate_eight_schools_source)
            grad(g) = begin
                k = prepare(g; have = es_have, want = :posterior, bound = es_bnd)
                prep = prepare_ad(k, BE, es_q; active = :unconstrained)
                ReactiveKernels.ad_value_and_gradient(prep, es_q)
            end
            (vmo, gmo_grad) = grad(gmo)
            (vfu, gfu_grad) = grad(gfu)
            @test all(isfinite, gmo_grad)
            @test vmo == vfu                            # same backend + graph -> exact numeric
            @test gmo_grad == gfu_grad                  # exact numeric gradient equality
            # finite-difference sanity (manual central differences, NOT ForwardDiff)
            h = 1e-6
            kk = prepare(gmo; have = es_have, want = :posterior, bound = es_bnd)
            for i in (2, 3)                             # μ-gradient is 0 at this symmetric point
                qp = copy(es_q); qp[i] += h
                qm = copy(es_q); qm[i] -= h
                fd = (kk(qp) - kk(qm)) / (2h)
                @test isapprox(gmo_grad[i], fd; rtol = 1e-4, atol = 1e-6)
            end
            @test !_pkg_loaded("Reactant")             # AFTER
        end
    end

    # The Reactant block uses the `@compile` MACRO and Reactant types, which are
    # resolved at lowering time — so it lives in a separate file included ONLY
    # when Reactant is actually loaded (a runtime `if` around it would still be
    # lowered and fail without Reactant present). Run it in a Reactant env.
    if isdefined(Main, :Reactant)
        include(joinpath(@__DIR__, "acceptance_startup_reactant.jl"))
    end
end
