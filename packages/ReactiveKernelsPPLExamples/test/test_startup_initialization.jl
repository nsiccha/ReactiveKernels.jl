# Regression controls for the model-only template initialization contract.
#
# Each example module builds its `_<MODEL>_GRAPH_TEMPLATE` eagerly in `__init__`
# (a world-age boundary at package load), but MODEL-ONLY: `_evaluate_ppl_source`
# stops as soon as the authored source defines `model`, skipping the source's
# trailing demo/self-check (`prepare(...)` + kernel execution + `@assert`). These
# tests pin that contract:
#   * import builds a GRAPH template, never a prepared/executed demo;
#   * the default (full) `evaluate_*_source()` still runs the demo/self-check;
#   * `build_*_graph()` + `prepare` + evaluate is safe INSIDE ONE ordinary
#     function (the benchmark run_one shape) — no world-age hazard, because the
#     source eval already happened at load, not in the caller;
#   * repeated `build_*_graph()` yields independent composed graphs;
#   * the model-only graph is behaviorally identical to the full-source graph.

const _RK = ReactiveKernels
const _M = ReactiveKernelsPPLExamples

@testset "startup initialization contract (model-only templates)" begin
    E = _M.EightSchoolsExample

    @testset "tail sentinel: package load ran NO demo/self-check tail" begin
        # THE init-path regression guard. `_DEMO_TAIL_EXECUTIONS` counts every
        # time a source's demo tail (any expression after `model` — the
        # prepare/execute/@assert that build `docs_example`) actually executes.
        # All ~89 `__init__` template builds are `model_only=true`, so package
        # load must leave this at 0. This is STRICTLY stronger than checking the
        # returned artifact's fields: the OLD full-demo `__init__` ALSO cached a
        # KernelSpec and discarded the PreparedKernel/output, so a field check
        # cannot tell it apart — but it ran the tail, which this counter records.
        # If `__init__` regressed to the full evaluator, this would be ~89 (fail).
        # (Runs first, before any full `evaluate_*_source()` below trips it.)
        @test _M._DEMO_TAIL_EXECUTIONS[] == 0

        # The sentinel is a REAL observable of tail execution: a FULL evaluate
        # trips it by exactly one, a model_only evaluate does NOT — proving the
        # discarded prepare/execute/@assert genuinely did not run in the template
        # path (not merely that `.kernel`/`.output` fields are absent).
        before = _M._DEMO_TAIL_EXECUTIONS[]
        E.evaluate_eight_schools_source()                      # full: runs the tail
        @test _M._DEMO_TAIL_EXECUTIONS[] == before + 1
        E.evaluate_eight_schools_source(; model_only = true)   # model_only: no tail
        @test _M._DEMO_TAIL_EXECUTIONS[] == before + 1         # unchanged
    end

    @testset "import builds a graph template, not a prepared/executed demo" begin
        # The model-only artifact carries the graph but NOT a prepared kernel or
        # a demo output — i.e. no `prepare`/execute ran to build the template.
        mo = E.evaluate_eight_schools_source(; model_only = true)
        @test mo.model isa _RK.KernelSpec
        @test !hasproperty(mo, :kernel)
        @test !hasproperty(mo, :output)
        # The cached template `__init__` built is a KernelSpec (a graph), never a
        # PreparedKernel — so package load did no eager preparation.
        @test isassigned(E._EIGHT_SCHOOLS_GRAPH_TEMPLATE)
        @test E._EIGHT_SCHOOLS_GRAPH_TEMPLATE[] isa _RK.KernelSpec
        @test !(E._EIGHT_SCHOOLS_GRAPH_TEMPLATE[] isa _RK.PreparedKernel)
    end

    @testset "the default full source still runs its demo/self-check" begin
        full = E.evaluate_eight_schools_source()
        @test full.model isa _RK.KernelSpec
        @test full.kernel isa _RK.PreparedKernel          # prepare ran
        @test hasproperty(full, :output)                  # kernel executed
    end

    @testset "no eager preparation across representative modules" begin
        for (mod, fn) in (
            (_M.EightSchoolsExample, :evaluate_eight_schools_source),
            (_M.LinearRegressionExample, :evaluate_linear_regression_source),
            (_M.BetaBinomialExample, :evaluate_beta_binomial_source),
            (_M.PoissonGammaExample, :evaluate_poisson_gamma_source),
            (_M.GLMPoissonExample, :evaluate_glm_poisson_source),
            (_M.Rate2Example, :evaluate_rate_2_source),
            (_M.ARMA11Example, :evaluate_arma11_source),
            (_M.MNISTLogisticExample, :evaluate_mnist_logistic_source),
            (_M.DiamondsExample, :evaluate_diamonds_source),
            (_M.NormalMixtureKExample, :evaluate_normal_mixture_k_source),
            (_M.DogsNonhierarchicalExample, :evaluate_dogs_nonhierarchical_source),
            (_M.LogisticRegressionRHSExample, :evaluate_logistic_regression_rhs_source),
        )
            mo = getproperty(mod, fn)(; model_only = true)
            @test mo.model isa _RK.KernelSpec
            @test !hasproperty(mo, :kernel)
        end
        # mnist carries a SECOND template (the optimized variant); both are
        # model-only and both are built eagerly at load.
        @test _M.MNISTLogisticExample.evaluate_mnist_logistic_optimized_source(;
            model_only = true).model isa _RK.KernelSpec
        @test isassigned(_M.MNISTLogisticExample._MNIST_LOGISTIC_GRAPH_TEMPLATE)
        @test isassigned(_M.MNISTLogisticExample._MNIST_LOGISTIC_OPTIMIZED_GRAPH_TEMPLATE)
    end

    @testset "first use is world-age-safe inside one ordinary function" begin
        # build + prepare + evaluate all in a SINGLE compiled function call — the
        # exact shape of the posteriordb benchmark's per-model run and the other
        # example tests. Eager `build_*_graph()` (a pure `compose`) makes this
        # safe; a lazy build_*_graph() that Core.eval'd here would throw
        # `method too new`.
        function first_use_one_call()
            g = E.build_eight_schools_graph()
            k = prepare(g; have = (:unconstrained, :observations, :observation_scales),
                           want = :posterior)
            q = [0.0, log(5.0), zeros(E.NSCHOOLS)...]
            k(q, E.EIGHT_SCHOOLS_Y, E.EIGHT_SCHOOLS_SIGMA)
        end
        v = first_use_one_call()
        @test isfinite(v)

        # Repeated use: each build yields an INDEPENDENT composed graph object,
        # and every one evaluates to the same value.
        g1 = E.build_eight_schools_graph()
        g2 = E.build_eight_schools_graph()
        @test g1 !== g2
        pv(g) = prepare(g; have = (:unconstrained, :observations, :observation_scales),
                           want = :posterior)(
            [0.0, log(5.0), zeros(E.NSCHOOLS)...], E.EIGHT_SCHOOLS_Y, E.EIGHT_SCHOOLS_SIGMA)
        @test pv(g1) == v
        @test pv(g2) == v
    end

    @testset "model-only graph matches the full-source graph exactly" begin
        # Same authored `@kernel model` -> same KernelSpec -> same prepared
        # kernel -> EXACT NUMERIC equality (`==`, not `≈`). (`==` is exact
        # numeric equality, e.g. +0.0 == -0.0 — not a byte comparison.)
        q = [0.0, log(5.0), zeros(E.NSCHOOLS)...]
        pv(g) = prepare(g; have = (:unconstrained, :observations, :observation_scales),
                           want = :posterior)(q, E.EIGHT_SCHOOLS_Y, E.EIGHT_SCHOOLS_SIGMA)
        @test pv(E.build_eight_schools_graph()) ==
              pv(compose(E.evaluate_eight_schools_source().model))

        R = _M.Rate2Example
        pr(g) = prepare(g; have = (:unconstrained, :n1, :n2, :k1, :k2), want = :posterior,
                           bound = (; n1 = R.RATE2_N1, n2 = R.RATE2_N2,
                                      k1 = R.RATE2_K1, k2 = R.RATE2_K2))([0.1, -0.1])
        @test pr(R.build_rate_2_graph()) ==
              pr(compose(R.evaluate_rate_2_source().model))
    end

    @testset "batch-1 posteriordb modules: first use world-age-safe in one function" begin
        # The exact public first-use path for the four batch-1 translations
        # (diamonds, normal_mixture_k, dogs_nonhierarchical, logistic_regression_rhs):
        # build_X_graph() -> prepare -> evaluate ALL INSIDE ONE ordinary function.
        # Eager model-only `__init__` (a pure `compose`, no `Core.eval` in the
        # caller) is what makes this safe; a lazy build_X_graph would throw
        # `method too new`. Each also checks the model-only template graph is
        # NUMERICALLY EXACT (`==`) against a graph composed from the full source.
        D = _M.DiamondsExample
        function diamonds_first_use()
            g = D.build_diamonds_graph()
            k = prepare(g; have = (:unconstrained, :X, :Y, :prior_only), want = :posterior,
                bound = (; X = D.DIAMONDS_X, Y = D.DIAMONDS_Y, prior_only = D.DIAMONDS_PRIOR_ONLY))
            k(vcat(fill(0.05, size(D.DIAMONDS_X, 2) - 1), 8.0, log(0.6)))
        end
        @test isfinite(diamonds_first_use())
        dpv(g) = prepare(g; have = (:unconstrained, :X, :Y, :prior_only), want = :posterior,
            bound = (; X = D.DIAMONDS_X, Y = D.DIAMONDS_Y, prior_only = D.DIAMONDS_PRIOR_ONLY))(
            vcat(fill(0.05, size(D.DIAMONDS_X, 2) - 1), 8.0, log(0.6)))
        @test dpv(D.build_diamonds_graph()) == dpv(compose(D.evaluate_diamonds_source().model))

        M = _M.NormalMixtureKExample
        _mq() = [0.1, -0.1, 0.2, 0.0, -3.0, 3.0, 2.0, -9.0, 5.0,
                 log(1.9 / 8.1), log(0.6 / 9.4), log(2.8 / 7.2), log(2.2 / 7.8), log(2.1 / 7.9)]
        function mixture_first_use()
            g = M.build_normal_mixture_k_graph()
            k = prepare(g; have = (:unconstrained, :y, :K), want = :posterior,
                bound = (; y = M.NORMAL_MIXTURE_K_Y, K = M.NORMAL_MIXTURE_K_K))
            k(_mq())
        end
        @test isfinite(mixture_first_use())
        mpv(g) = prepare(g; have = (:unconstrained, :y, :K), want = :posterior,
            bound = (; y = M.NORMAL_MIXTURE_K_Y, K = M.NORMAL_MIXTURE_K_K))(_mq())
        @test mpv(M.build_normal_mixture_k_graph()) ==
              mpv(compose(M.evaluate_normal_mixture_k_source().model))

        G = _M.DogsNonhierarchicalExample
        _gq() = vcat(-1.0, 0.5, log(0.5), log(0.4), 0.2,
                     0.1 .* range(-1.0, 1.0; length = 2 * G.DOGS_NH_J))
        function dogs_first_use()
            g = G.build_dogs_nonhierarchical_graph()
            k = prepare(g; have = (:unconstrained, :y), want = :posterior,
                bound = (; y = G.DOGS_NH_Y))
            k(_gq())
        end
        @test isfinite(dogs_first_use())
        gpv(g) = prepare(g; have = (:unconstrained, :y), want = :posterior,
            bound = (; y = G.DOGS_NH_Y))(_gq())
        @test gpv(G.build_dogs_nonhierarchical_graph()) ==
              gpv(compose(G.evaluate_dogs_nonhierarchical_source().model))

        L = _M.LogisticRegressionRHSExample
        _lhave() = (:unconstrained, :x, :y, :scale_icept, :scale_global, :nu_global,
                    :nu_local, :slab_scale, :slab_df)
        _lbound() = (h = L.LOGISTIC_RHS_HYPER; (; x = L.LOGISTIC_RHS_X, y = L.LOGISTIC_RHS_Y,
            scale_icept = h.scale_icept, scale_global = h.scale_global, nu_global = h.nu_global,
            nu_local = h.nu_local, slab_scale = h.slab_scale, slab_df = h.slab_df))
        _lq() = (d = size(L.LOGISTIC_RHS_X, 2);
            vcat(0.0, fill(0.05, d), log(0.1), fill(log(0.5), d), log(2.0)))
        function logistic_first_use()
            g = L.build_logistic_regression_rhs_graph()
            k = prepare(g; have = _lhave(), want = :posterior, bound = _lbound())
            k(_lq())
        end
        @test isfinite(logistic_first_use())
        lpv(g) = prepare(g; have = _lhave(), want = :posterior, bound = _lbound())(_lq())
        @test lpv(L.build_logistic_regression_rhs_graph()) ==
              lpv(compose(L.evaluate_logistic_regression_rhs_source().model))
    end
end
