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

@testset "multiple active non-allocating ports share one reverse pass" begin
    spec = @kernel multi_active_naad(
            alpha::Vector{Float64}, beta::Vector{Float64},
            data::Vector{Float64}) = begin
        alpha_term::Float64 = sum(abs2, alpha)
        beta_term::Float64 = sum(beta .* data)
        objective::Float64 = alpha_term + beta_term
    end
    alpha = [0.3, -0.4, 0.2]
    beta = [-0.1, 0.7, 0.5]
    data = [2.0, -1.0, 0.5]
    kernel = prepare(spec; have = (:alpha, :beta, :data), want = :objective)
    nonallocating = prepare_nonallocating(kernel)

    prepared = prepare_ad(
        kernel, NA_AD_BACKEND, alpha, beta, data;
        active = (:beta, :alpha))
    prepared_nonallocating = prepare_ad(
        nonallocating, NA_AD_BACKEND, alpha, beta, data;
        active = (:beta, :alpha))

    reference_value, reference_gradient =
        ad_value_and_gradient(prepared, alpha, beta, data)
    value, gradient = ad_value_and_gradient(
        prepared_nonallocating, alpha, beta, data)
    @test value == reference_value
    @test gradient[1] ≈ data
    @test gradient[2] ≈ 2 .* alpha
    @test gradient[1] ≈ reference_gradient[1]
    @test gradient[2] ≈ reference_gradient[2]

    destination = (similar(beta), similar(alpha))
    inplace_value, returned = ad_value_and_gradient!(
        prepared_nonallocating, destination, alpha, beta, data)
    @test inplace_value == value
    @test returned === destination
    @test destination[1] ≈ gradient[1]
    @test destination[2] ≈ gradient[2]

    mixed_spec = @kernel multi_active_mixed_naad(
            beta::Vector{Float64}, phi::Float64) = begin
        beta_term::Float64 = sum(abs2, beta)
        phi_term::Float64 = phi^2
        objective::Float64 = beta_term + phi_term
    end
    phi = 1.7
    mixed_nonallocating = prepare_nonallocating(
        mixed_spec; have = (:beta, :phi), want = :objective)
    mixed = prepare_ad(
        mixed_nonallocating, NA_AD_BACKEND, beta, phi;
        active = (:beta, :phi))
    mixed_value, mixed_gradient = ad_value_and_gradient(mixed, beta, phi)
    @test mixed_value ≈ sum(abs2, beta) + phi^2
    @test mixed_gradient[1] ≈ 2 .* beta
    @test mixed_gradient[2] ≈ 2phi
    mixed_destination = (similar(beta), Ref(NaN))
    _, mixed_returned = ad_value_and_gradient!(
        mixed, mixed_destination, beta, phi)
    @test mixed_returned === mixed_destination
    @test mixed_destination[1] ≈ mixed_gradient[1]
    @test mixed_destination[2][] ≈ mixed_gradient[2]
end

isdefined(@__MODULE__, :AuthoredPlateChains) ||
    include(joinpath(@__DIR__, "fixtures", "authored_plate_chains.jl"))

# Authored plates under NA reverse AD take the AD-only scalar loop with the
# cell kernel hoisted to a direct literal (`_ad_na_plate_loop!`): calling the
# closure-held nested kernel trips Enzyme static activity analysis. Single
# (`flat`) and chained (`chain`) plates agree with dataflow, the analytic
# gradient, and central differences, with size-invariant allocations.
@testset "prepare_ad over NonAllocatingKernel with authored plates" begin
    C = AuthoredPlateChains
    q = [0.7]
    for (tag, spec) in (("chain", C.chain), ("flat", C.flat))
        n = 32
        x = collect(range(-1.0, 1.0; length = n))
        y = fill(0.3, n)
        kb = prepare(spec; bound = (; x, y))
        kbna = prepare_nonallocating(kb)
        prep = prepare_ad(kb, NA_AD_BACKEND, q; active = :q)
        prepna = prepare_ad(kbna, NA_AD_BACKEND, q; active = :q)
        g, gna = zeros(1), zeros(1)
        v, _ = ad_value_and_gradient!(prep, g, q)
        vna, _ = ad_value_and_gradient!(prepna, gna, q)
        @test vna ≈ v
        @test kbna(q) ≈ v
        @test gna ≈ g
        @test only(gna) ≈ -sum((only(q) .* x .- y) .* x)
        @test naad_findiff(kb, q) ≈ gna rtol = 1e-4
    end
    small, big = 8, 64
    alloc_bytes = map((small, big)) do n
        x = collect(range(-1.0, 1.0; length = n))
        y = fill(0.3, n)
        kbna = prepare_nonallocating(prepare(C.flat; bound = (; x, y)))
        prepna = prepare_ad(kbna, NA_AD_BACKEND, q; active = :q)
        gna = zeros(1)
        ad_value_and_gradient!(prepna, gna, q)
        @allocated ad_value_and_gradient!(prepna, gna, q)
    end
    @test alloc_bytes[1] == alloc_bytes[2]
end

# Partitioned plates gather live lane arguments at run time (`_LaneGather`).
# Arithmetic cells keep the plate bodies free of transcendental calls, so the
#gradient's only per-lane work is gathers plus arithmetic: on Julia 1.12 the
#gather loop's per-element bounds checks allocated ~17.6 bytes/lane under
#reverse-mode Enzyme (the same class as the plate-loop `checkbounds`
#regression above), and the NA gradient must stay size-invariant there too.
@kernel arith_partition_naad(x::Vector{Float64}, y::Vector{Float64}) = begin
    pointwise = plate(x, y) do xi, yi
        cell::Float64 = yi > 0 ? xi * yi : 2.0 * xi + 1.0
        cell
    end
    total::Float64 = sum(pointwise)
    return total
end

@testset "partitioned NA gradients over arithmetic cells stay size-invariant" begin
    # Warmup prep: the first partitioned-AD preparation in a process carries
    # a fixed ~600B per-call cost on every Julia version (process-order
    # effect, unrelated to sizes); measure only after it is settled.
    let xw = [0.5, -0.5], yw = [1.0, -1.0]
        kbw = prepare_nonallocating(arith_partition_naad;
                                    have = (:x, :y), want = :total,
                                    bound = (; y = yw))
        prepw = prepare_ad(kbw, NA_AD_BACKEND, xw; active = :x)
        gw = similar(xw)
        ad_value_and_gradient!(prepw, gw, xw)
        @allocated ad_value_and_gradient!(prepw, gw, xw)
    end
    alloc_bytes = map((16, 128)) do n
        x = collect(range(-2.0, 2.0; length = n))
        y = Float64[isodd(i) ? 1.0 : -1.0 for i in 1:n]
        kbna = prepare_nonallocating(arith_partition_naad;
                                     have = (:x, :y), want = :total,
                                     bound = (; y))
        # Vacuity guard: without partitioning there are no lane gathers and
        # this test would pass while exercising nothing.
        @test any(op -> op isa ReactiveKernels._LaneGather, kbna.ops)
        kb = prepare(arith_partition_naad; have = (:x, :y), want = :total,
                     bound = (; y))
        prepna = prepare_ad(kbna, NA_AD_BACKEND, x; active = :x)
        prep = prepare_ad(kb, NA_AD_BACKEND, x; active = :x)
        gna, g = similar(x), similar(x)
        vna, _ = ad_value_and_gradient!(prepna, gna, x)
        v, _ = ad_value_and_gradient!(prep, g, x)
        @test vna ≈ v
        @test gna ≈ g
        dataflow_bytes = @allocated ad_value_and_gradient!(prep, g, x)
        na_bytes = @allocated ad_value_and_gradient!(prepna, gna, x)
        @test na_bytes < dataflow_bytes
        na_bytes
    end
    println("NONALLOCATING_AD_ALLOC\tpartition_arith\t", alloc_bytes)
    @test alloc_bytes[1] == alloc_bytes[2]
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

@testset "bound-only steps fold once per preparation" begin
    spec = @kernel fold_mixed_naad(m, v, a) = begin
        t = m * v .+ a
        s = sum(t)
        return s
    end
    n = 2000
    m = hcat(ones(n), collect(1.0:n))
    v = [0.5, -0.25]
    a0 = fill(0.1, n)
    kb = prepare(spec; have = (:m, :v, :a), want = :s, bound = (; m, v))
    kbna = prepare_nonallocating(spec; have = (:m, :v, :a), want = :s,
                                 bound = (; m, v))
    matmul_step = findfirst(
        op -> op isa ReactiveKernels._MatMulStep, kbna.ops)
    @test matmul_step !== nothing
    prep = prepare_ad(kb, NA_AD_BACKEND, a0; active = :a)
    prepna = prepare_ad(kbna, NA_AD_BACKEND, a0; active = :a)
    g, gna = similar(a0), similar(a0)
    v_, _ = ad_value_and_gradient!(prep, g, a0)
    vna, _ = ad_value_and_gradient!(prepna, gna, a0)
    @test vna == v_
    @test gna == g == ones(n)
    # The bound-only product folded to exactly one appended constant.
    primal_bounds = count(
        op -> op isa ReactiveKernels._BoundConstant, kbna.ops)
    @test length(prepna.call.ops) == length(kbna.ops) + 1
    @test count(op -> op isa ReactiveKernels._BoundConstant,
                prepna.call.ops) == primal_bounds + 1
    # The folded value is an owned copy, never the primal slot's buffer.
    folded = last(prepna.call.ops).value
    @test folded == kbna.caches[matmul_step][] == m * v
    @test folded !== kbna.caches[matmul_step][]
    # The gradient never recomputes the bound-only product.
    dataflow_bytes = @allocated ad_value_and_gradient!(prep, g, a0)
    na_bytes = @allocated ad_value_and_gradient!(prepna, gna, a0)
    println("NONALLOCATING_AD_ALLOC\tfold_dataflow\t", dataflow_bytes)
    println("NONALLOCATING_AD_ALLOC\tfold_nonalloc\t", na_bytes)
    @test dataflow_bytes > 0
    @test na_bytes < sizeof(a0)
    @test na_bytes < dataflow_bytes
end

@testset "inactive unbound steps still recompute per call" begin
    spec = @kernel fold_live_naad(m, v, a) = begin
        t = m * v .+ a
        s = sum(t)
        return s
    end
    n = 2000
    m = hcat(ones(n), collect(1.0:n))
    v = [0.5, -0.25]
    a0 = fill(0.1, n)
    kbna = prepare_nonallocating(spec; have = (:m, :v, :a), want = :s,
                                 bound = (; m))
    prepna = prepare_ad(kbna, NA_AD_BACKEND, v, a0; active = :a)
    # The product sees the unbound port, so nothing folds.
    @test length(prepna.call.ops) == length(kbna.ops)
    gna = similar(a0)
    va, _ = ad_value_and_gradient!(prepna, gna, v, a0)
    @test va ≈ sum(m * v) + sum(a0)
    @test gna == ones(n)
    vb, _ = ad_value_and_gradient!(prepna, gna, 2v, a0)
    @test vb ≈ sum(m * (2v)) + sum(a0)
    @test gna == ones(n)
    @test vb != va
end

# Tainted NA steps over in-place broadcast with a `Const` array operand (a
# bound, folded, or inactive port) take the AD-only scalar loops
# (`_ad_na_materialize_loop!`, `_ad_na_arith_loop!`): on Julia 1.12 Enzyme's
# `copyto!` override unaliases the shadowed destination against every source,
# and the fresh-or-`Const` array phi trips static activity analysis
# (`EnzymeRuntimeActivityError`), while the identical program differentiates
# cleanly on Julia 1.10 (snag `na-broadcast-act-38b42481`).
@testset "tainted NA arithmetic steps with bound arrays" begin
    spec = @kernel arith_const_naad(a, b) = begin
        c = a + b
        d = a - b
        e = +(a, b, a)
        s = sum(c) + sum(d) + sum(e)
        return s
    end
    a0 = [0.3, -0.4, 0.2]
    b0 = [2.0, -1.0, 0.5]
    kb = prepare(spec; have = (:a, :b), want = :s, bound = (; b = b0))
    kbna = prepare_nonallocating(spec; have = (:a, :b), want = :s,
                                 bound = (; b = b0))
    # Vacuity: the arithmetic must decompose to bare `+`/`-` steps (not the
    # fused fallback), or this test exercises nothing.
    @test any(op -> op === +, kbna.ops)
    @test any(op -> op === -, kbna.ops)
    prep = prepare_ad(kb, NA_AD_BACKEND, a0; active = :a)
    prepna = prepare_ad(kbna, NA_AD_BACKEND, a0; active = :a)
    @test any(op -> op === +, prepna.call.ops)
    @test any(op -> op === -, prepna.call.ops)
    g, gna = similar(a0), similar(a0)
    v, _ = ad_value_and_gradient!(prep, g, a0)
    vna, _ = ad_value_and_gradient!(prepna, gna, a0)
    @test vna == v
    @test gna == g == fill(4.0, length(a0))
end

@testset "tainted NA dotted steps with bound arrays" begin
    spec = @kernel dotted_const_naad(a, d) = begin
        t = d .+ a
        s = sum(t)
        return s
    end
    a0 = [0.3, -0.4, 0.2]
    d0 = [2.0, -1.0, 0.5]
    kb = prepare(spec; have = (:a, :d), want = :s, bound = (; d = d0))
    kbna = prepare_nonallocating(spec; have = (:a, :d), want = :s,
                                 bound = (; d = d0))
    @test any(op -> op isa ReactiveKernels._MaterializeStep, kbna.ops)
    prep = prepare_ad(kb, NA_AD_BACKEND, a0; active = :a)
    prepna = prepare_ad(kbna, NA_AD_BACKEND, a0; active = :a)
    @test any(op -> op isa ReactiveKernels._MaterializeStep, prepna.call.ops)
    g, gna = similar(a0), similar(a0)
    v, _ = ad_value_and_gradient!(prep, g, a0)
    vna, _ = ad_value_and_gradient!(prepna, gna, a0)
    @test vna == v
    @test gna == g == ones(length(a0))

    # N-D: the scalar loop indexes every dimension, not just vectors.
    matspec = @kernel mat_const_naad(A, B) = begin
        C = A + B
        S = sum(C)
        return S
    end
    A0 = [0.3 -0.4 0.2; 1.1 0.7 -0.9]
    B0 = [2.0 -1.0 0.5; 0.25 1.5 -2.25]
    matkb = prepare(matspec; have = (:A, :B), want = :S, bound = (; B = B0))
    matkbna = prepare_nonallocating(matspec; have = (:A, :B), want = :S,
                                    bound = (; B = B0))
    @test any(op -> op === +, matkbna.ops)
    matprep = prepare_ad(matkb, NA_AD_BACKEND, A0; active = :A)
    matprepna = prepare_ad(matkbna, NA_AD_BACKEND, A0; active = :A)
    gmat, gmatna = similar(A0), similar(A0)
    vmat, _ = ad_value_and_gradient!(matprep, gmat, A0)
    vmatna, _ = ad_value_and_gradient!(matprepna, gmatna, A0)
    @test vmatna == vmat
    @test gmatna == gmat == ones(size(A0))

    # N-D dotted: the materialize loop indexes every dimension too.
    dotmatspec = @kernel dotmat_const_naad(A, D) = begin
        T = D .+ A
        S = sum(T)
        return S
    end
    dotmatkb = prepare(dotmatspec; have = (:A, :D), want = :S,
                       bound = (; D = B0))
    dotmatkbna = prepare_nonallocating(dotmatspec; have = (:A, :D),
                                       want = :S, bound = (; D = B0))
    @test any(op -> op isa ReactiveKernels._MaterializeStep, dotmatkbna.ops)
    dotmatprep = prepare_ad(dotmatkb, NA_AD_BACKEND, A0; active = :A)
    dotmatprepna = prepare_ad(dotmatkbna, NA_AD_BACKEND, A0; active = :A)
    gdot, gdotna = similar(A0), similar(A0)
    vdot, _ = ad_value_and_gradient!(dotmatprep, gdot, A0)
    vdotna, _ = ad_value_and_gradient!(dotmatprepna, gdotna, A0)
    @test vdotna == vdot
    @test gdotna == gdot == ones(size(A0))
end
