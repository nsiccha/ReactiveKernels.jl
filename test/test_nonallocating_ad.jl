# `prepare_ad` (and pullbacks) over `NonAllocatingKernel`. Runs only through
# `test/run_nonallocating_integration.jl`, not the default suite, because it
# needs the unregistered MutatingFunctions weak dependency.
using ReactiveKernels
using MutatingFunctions
using DifferentiationInterface
using DifferentiationInterface: Cache
import Enzyme
using Test

const NA_AD_BACKEND = AutoEnzyme(mode = Enzyme.Reverse,
                                 function_annotation = Enzyme.Const)

nlp_naad(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
g_prior_naad(q) = nlp_naad(q[1], 0.0, 5.0) + nlp_naad(q[2], 0.0, 5.0) +
                  nlp_naad(q[3], 0.0, 5.0) + nlp_naad(q[4], 0.0, 5.0)
g_eta_naad(a, b, X) = a .+ X * b
pois_ll_naad(Cf, eta, cterm) = sum(Cf .* eta) - sum(exp, eta) - cterm

function naad_poisson_data(n)
    yr = collect(range(-1.5, 1.5; length = n))
    X = hcat([yr .^ d for d in (1, 2, 3)]...)
    Cf = Float64.(1:n)
    cterm = sum(log, Cf .+ 1.0)
    X, Cf, cterm
end

naad_spec = @kernel naad_model(unconstrained, X, Cf, cterm) = begin
    a = unconstrained[1]
    b = unconstrained[2:4]
    eta = g_eta_naad(a, b, X)
    prior = g_prior_naad(unconstrained)
    likelihood = pois_ll_naad(Cf, eta, cterm)
    posterior = prior + likelihood
    return posterior
end

function naad_findiff(kb, qq)
    h = 1e-6
    g = similar(qq)
    for i in eachindex(qq)
        qp = copy(qq)
        qm = copy(qq)
        qp[i] += h
        qm[i] -= h
        g[i] = (kb(qp) - kb(qm)) / (2h)
    end
    g
end

@testset "prepare_ad over NonAllocatingKernel" begin
    @test Base.get_extension(ReactiveKernels,
                             :ReactiveKernelsMutatingFunctionsExt) !== nothing

    n = 2000
    X, Cf, cterm = naad_poisson_data(n)
    q = [0.2, 0.3, -0.1, 0.0]
    qalt = [0.5, -0.2, 0.1, 0.3]
    kb = prepare(naad_spec; have = (:unconstrained, :X, :Cf, :cterm),
                 want = :posterior, bound = (; X = X, Cf = Cf, cterm = cterm))
    kbna = prepare_nonallocating(kb)
    prep = prepare_ad(kb, NA_AD_BACKEND, q; active = :unconstrained)
    prepna = prepare_ad(kbna, NA_AD_BACKEND, q; active = :unconstrained)

    @testset "owned Cache threading" begin
        @test length(prepna.external_values) == 1
        @test only(prepna.external_values) isa Cache
    end

    @testset "repeat calls and fresh points match dataflow" begin
        for qq in (q, q, qalt, q)
            gna = similar(qq)
            g = similar(qq)
            vna, _ = ad_value_and_gradient!(prepna, gna, qq)
            v, _ = ad_value_and_gradient!(prep, g, qq)
            @test vna == v
            @test gna ≈ g
            # The AD object owns separate caches: interleaved primal calls
            # observe the kernel's own borrowed caches undisturbed.
            @test kbna(qq) ≈ v
        end
        @test ad_value_and_gradient(prepna, q)[2] ≈
              ad_value_and_gradient(prep, q)[2]
        @test naad_findiff(kb, qalt) ≈
              ad_value_and_gradient(prepna, qalt)[2] rtol = 1e-4
    end

    @testset "one-shot and pullback surfaces agree" begin
        @test ad_gradient(kbna, NA_AD_BACKEND, q;
                          active = :unconstrained) ≈
              ad_gradient(kb, NA_AD_BACKEND, q; active = :unconstrained)
        ppbna = prepare_ad_pullback(kbna, NA_AD_BACKEND, 1.0, q;
                                    active = :unconstrained)
        ppb = prepare_ad_pullback(kb, NA_AD_BACKEND, 1.0, q;
                                  active = :unconstrained)
        @test ad_pullback(ppbna, 2.5, q) ≈ ad_pullback(ppb, 2.5, q)
        @test ad_pullback(kbna, NA_AD_BACKEND, 2.5, q;
                          active = :unconstrained) ≈
              ad_pullback(kb, NA_AD_BACKEND, 2.5, q; active = :unconstrained)
    end

    @testset "bound views cross as owning copies" begin
        Xfull = copy(X)
        Xview = @view Xfull[:, :]
        kbv = prepare(naad_spec; have = (:unconstrained, :X, :Cf, :cterm),
                      want = :posterior,
                      bound = (; X = Xview, Cf = Cf, cterm = cterm))
        kbnav = prepare_nonallocating(kbv)
        prepv = prepare_ad(kbv, NA_AD_BACKEND, q; active = :unconstrained)
        prepnav = prepare_ad(kbnav, NA_AD_BACKEND, q;
                             active = :unconstrained)
        gv = similar(q)
        gvna = similar(q)
        vv, _ = ad_value_and_gradient!(prepv, gv, q)
        vvna, _ = ad_value_and_gradient!(prepnav, gvna, q)
        @test vvna == vv
        @test gvna ≈ gv
    end

    @testset "fail closed" begin
        @test_throws ArgumentError prepare_ad(kbna, NA_AD_BACKEND, q;
                                              active = :nope)
        @test_throws ArgumentError prepare_ad(kbna, NA_AD_BACKEND, q;
                                              active = :unconstrained, foo = 1)
        @test_throws ArgumentError compile_ad_gradient(prepna, q)
        @test_throws ArgumentError compile_ad_value_and_gradient(prepna, q)
    end
end

fscale_naad(a, b) = a * b

@testset "decomposed non-allocating gradient allocates ~nothing" begin
    spec = @kernel scaled_sum_naad(f::typeof(fscale_naad),
                                   x::Vector{Float64}, c::Float64) = begin
        y::Vector{Float64} = broadcast(f, x, c)
        total::Float64 = sum(y)
    end
    x = collect(1.0:5000.0)
    c = 2.5
    kb = prepare(spec; have = (:f, :x, :c), want = :total)
    kbna = prepare_nonallocating(spec; have = (:f, :x, :c), want = :total)
    prep = prepare_ad(kb, NA_AD_BACKEND, fscale_naad, x, c; active = :x)
    prepna = prepare_ad(kbna, NA_AD_BACKEND, fscale_naad, x, c; active = :x)
    g = similar(x)
    gna = similar(x)
    v, _ = ad_value_and_gradient!(prep, g, fscale_naad, x, c)
    vna, _ = ad_value_and_gradient!(prepna, gna, fscale_naad, x, c)
    @test vna == v
    @test gna == g
    @test all(==(c), gna)
    dataflow_bytes = @allocated ad_value_and_gradient!(
        prep, g, fscale_naad, x, c)
    na_bytes = @allocated ad_value_and_gradient!(
        prepna, gna, fscale_naad, x, c)
    println("NONALLOCATING_AD_ALLOC\tdataflow\t", dataflow_bytes)
    println("NONALLOCATING_AD_ALLOC\tnonalloc\t", na_bytes)
    @test dataflow_bytes > 0
    @test na_bytes <= 64
    @test na_bytes < dataflow_bytes
end
