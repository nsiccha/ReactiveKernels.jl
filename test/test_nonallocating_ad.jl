# `prepare_ad` (and pullbacks) over `NonAllocatingKernel`. Runs only through
# `test/run_nonallocating_integration.jl`, not the default suite, because it
# needs the unregistered MutatingFunctions weak dependency.
using ReactiveKernels
using MutatingFunctions
using DifferentiationInterface
using DifferentiationInterface: Cache
using LinearAlgebra
import Enzyme
using Test

# Ordinary reverse mode, with no function annotation: bound and folded values
# reach the differentiated program as hidden `Constant` operands, so the
# callable holds no data a plain backend must prove readonly.
const NA_AD_BACKEND = AutoEnzyme(mode = Enzyme.Reverse)

# NA gradients stay size-invariant on every Julia version, bounds checks on.
# Registered Enzyme on Julia 1.12+ with `--check-bounds=yes` taped the shadow
# of a cached buffer's `Memory` pointer once per element when a loop stored
# into the buffer and the buffer was then stored back into its `Ref` slot
# (`repro_enzyme_reuse_or_similar_shadow_tape.jl`, Enzyme snag
# `julia-1-12-rever-30566488`). These checks pass on Julia 1.13.1 with the
# Enzyme dogfood line (b2bb4cb6), and the differentiated program no longer
# stores into its slots (`ReactiveKernels._ad_na_cache_driver`).

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

# The differentiated program's constants table and hidden operands. Bound and
# folded values leave the table as placeholders and cross as the pack.
naad_ad_constants(prepared) = prepared.call.constants === nothing ?
    prepared.call.f.call.constants : prepared.call.constants
naad_ad_callables(prepared) = prepared.call.callables === nothing ?
    prepared.call.f.call.callables : prepared.call.callables
naad_hidden(prepared) =
    ReactiveKernels._ad_hidden_operands(prepared.call, prepared.external_values)

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
    # The HAVE ports are untyped, so only an exemplar-typed kernel keeps owned
    # cache slots (a step without a destination-passing method for concrete
    # types is a plain call); those slots are what the AD object threads.
    kbna = prepare_nonallocating(kb, q)
    prep = prepare_ad(kb, NA_AD_BACKEND, q; active = :unconstrained)
    prepna = prepare_ad(kbna, NA_AD_BACKEND, q; active = :unconstrained)

    @testset "owned Cache threading" begin
        @test length(prepna.external_values) == 2
        @test last(prepna.external_values) isa Cache
        untyped = prepare_nonallocating(kb)
        @test all(isnothing, untyped.caches)
        prepu = prepare_ad(untyped, NA_AD_BACKEND, q; active = :unconstrained)
        @test last(prepu.external_values) isa DifferentiationInterface.Constant
        gu = similar(q)
        g = similar(q)
        vu, _ = ad_value_and_gradient!(prepu, gu, q)
        v, _ = ad_value_and_gradient!(prep, g, q)
        @test vu == v
        @test gu ≈ g
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

    @testset "bound values cross as hidden operands" begin
        # `X * b` reaches BLAS `gemv`, an opaque call a plain reverse backend
        # cannot prove readonly when `X` is a field of the differentiated
        # callable (EnzymeMutabilityException). The bound values instead form
        # one named `Constant` pack before the owned caches.
        @test prepna.call.constants === nothing
        @test keys(first(prepna.external_values)) == (:X, :Cf, :cterm)
        @test naad_hidden(prepna) == (X, Cf, cterm)
        annotated = prepare_ad(kbna, AutoEnzyme(mode = Enzyme.Reverse,
            function_annotation = Enzyme.Const), q; active = :unconstrained)
        @test ad_gradient(annotated, qalt) == ad_gradient(prepna, qalt)
        @test ad_gradient(prepna, qalt) ≈ ad_gradient(prep, qalt)
    end

    @testset "fail closed" begin
        @test_throws ArgumentError prepare_ad(kbna, NA_AD_BACKEND, q;
                                              active = :nope)
        @test_throws ArgumentError prepare_ad(kbna, NA_AD_BACKEND, q;
                                              active = :unconstrained, foo = 1)
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
# gradient's only per-lane work is gathers plus arithmetic: on Julia 1.12 the
# gather loop's per-element bounds checks allocated ~17.6 bytes/lane under
# reverse-mode Enzyme (the same class as the plate-loop `checkbounds`
# regression), and the NA gradient must stay size-invariant there too.
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
        @test any(op -> op isa ReactiveKernels._LaneGather, kbna.callables)
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
        # The ordinary kernel reads each arm's lanes through a view, so on
        # this shape it may allocate as little as the non-allocating one.
        @test na_bytes <= dataflow_bytes
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

# `vcat` of operands of different array types: a fresh row (a `Matrix` or an
# adjoint) or a view's row over a parameter block read in place as a reshaped
# view, as an RKPPL model's `vcat(permutedims(c), D)` stacks them (snag
# `rkppl-vcat-of-pa-decf6725`). The step filled its cache through a loop over
# the operand tuple, which the type-stability check flagged and native Enzyme
# reverse refused (`IllegalTypeAnalysisException`, `EnzymeNoTypeError`).
#
# Registered Enzyme (0.13.219) also refuses Base's own `vcat` of a fresh
# `Matrix` row beside a reshaped view, the step's result for a buffer of
# another shape, even where that call never runs (`IllegalTypeAnalysisException`
# in `_typed_vcat`). The Enzyme dogfood line's concatenation overrides take
# views, reshapes and transposes of arrays (Enzyme decision `0prqqvj`), and
# there every kernel below differentiates. The repair is Enzyme's:
# `rejectable` names the kernels with a fresh `Matrix` row, whose rejection is
# pinned broken; any other error, or a rejection of the view or adjoint row,
# still fails.
function _naad_vcat_gradient_or_rejected(prepared, g, u; rejectable::Bool)
    try
        ad_value_and_gradient!(prepared, g, u)
    catch err
        rejectable && err isa Enzyme.Compiler.IllegalTypeAnalysisException || rethrow()
        nothing
    end
end
@kernel naad_vcat_fresh_row(u::Vector{Float64}, y::Matrix{Float64}) = begin
    c::Vector{Float64} = exp.(u[1:2])
    D = reshape(view(u, 3:4), 1, 2)
    C = vcat(permutedims(c), D)
    total::Float64 = -0.5 * sum((y .- C) .^ 2)
end
@kernel naad_vcat_view_row(u::Vector{Float64}, y::Matrix{Float64}) = begin
    c = view(u, 1:2)
    D = reshape(view(u, 3:4), 1, 2)
    C = vcat(permutedims(c), D)
    total::Float64 = -0.5 * sum((y .- C) .^ 2)
end
@kernel naad_vcat_adjoint_row(u::Vector{Float64}, y::Matrix{Float64}) = begin
    c::Vector{Float64} = exp.(u[1:2])
    D = reshape(view(u, 3:4), 1, 2)
    C = vcat(c', D)
    total::Float64 = -0.5 * sum((y .- C) .^ 2)
end
@kernel naad_vcat_three(u::Vector{Float64}, y::Matrix{Float64}) = begin
    c::Vector{Float64} = exp.(u[1:2])
    D = reshape(view(u, 3:6), 2, 2)
    E::Matrix{Float64} = reshape(u[7:8], 1, 2) .* 2.0
    C = vcat(permutedims(c), D, E)
    total::Float64 = -0.5 * sum((y .- C) .^ 2)
end
@testset "native Reverse through vcat of operands of different array types" begin
    include(joinpath(@__DIR__, "fixtures", "nonallocating_called_ops.jl"))
    u = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8]
    for (spec, rows, rejectable) in ((naad_vcat_fresh_row, 2, true),
                                     (naad_vcat_view_row, 2, false),
                                     (naad_vcat_adjoint_row, 2, false),
                                     (naad_vcat_three, 4, true))
        y = reshape(collect(range(-1.0, 1.0; length = 2rows)), rows, 2)
        kb = prepare(spec; have = (:u, :y), want = :total, bound = (; y))
        kbna = prepare_nonallocating(kb, u)
        @test any(op -> op isa ReactiveKernels._ConcatenateStep{typeof(vcat)},
                  called_ops(kbna))
        prep = prepare_ad(kb, NA_AD_BACKEND, u; active = :u)
        prepna = prepare_ad(kbna, NA_AD_BACKEND, u; active = :u,
                            type_stability = :error)
        g, gna = similar(u), similar(u)
        v, _ = ad_value_and_gradient!(prep, g, u)
        result = _naad_vcat_gradient_or_rejected(prepna, gna, u; rejectable)
        if result === nothing
            @test_broken result !== nothing
        else
            vna, _ = result
            @test vna ≈ v rtol = 1e-12
            @test gna ≈ g rtol = 1e-12
            @test gna ≈ naad_findiff(kb, u) rtol = 1e-6 atol = 1e-8
        end
    end
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
        op -> op isa ReactiveKernels._MatMulStep, kbna.callables)
    @test matmul_step !== nothing
    prep = prepare_ad(kb, NA_AD_BACKEND, a0; active = :a)
    prepna = prepare_ad(kbna, NA_AD_BACKEND, a0; active = :a)
    g, gna = similar(a0), similar(a0)
    v_, _ = ad_value_and_gradient!(prep, g, a0)
    vna, _ = ad_value_and_gradient!(prepna, gna, a0)
    @test vna == v_
    @test gna == g == ones(n)
    # Locate the product by its value, independently of operation-table layout.
    product = m * v
    folded_values = filter(value -> value isa AbstractVector &&
        value == product, naad_hidden(prepna))
    @test length(folded_values) == 1
    # The folded value is an owned copy, never the primal slot's buffer.
    folded = only(folded_values)
    @test folded == kbna.caches[matmul_step][] == product
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

# Test-only observation of product execution; the numerical inputs stay intact.
struct NAADCountedMatrix <: AbstractMatrix{Float64}
    data::Matrix{Float64}
    products::Base.RefValue{Int}
end
Base.size(m::NAADCountedMatrix) = size(m.data)
Base.getindex(m::NAADCountedMatrix, i::Int, j::Int) = m.data[i, j]
function Base.:*(m::NAADCountedMatrix, v::Vector{Float64})
    m.products[] += 1
    m.data * v
end
function LinearAlgebra.mul!(out::Vector{Float64}, m::NAADCountedMatrix,
                            v::Vector{Float64})
    m.products[] += 1
    LinearAlgebra.mul!(out, m.data, v)
end

@testset "bound-only folding execution, ownership and growth" begin
    spec = @kernel counted_fold_naad(m, v::Vector{Float64},
                                     a::Vector{Float64}) = begin
        t = m * v .+ a
        # A declared scalar avoids the separate exemplar value/type probe.
        s::Float64 = sum(t)
        return s
    end
    allocations = Int[]
    for n in (0, 64, 4096)
        data = hcat(ones(n), collect(1.0:n))
        m = NAADCountedMatrix(data, Ref(0))
        v = [0.5, -0.25]
        # Binary-exact inputs keep the independent reduction oracle exact.
        a = fill(0.125, n)
        originals = (copy(data), copy(v), copy(a))
        product = data * v
        kernel = prepare_nonallocating(spec; have = (:m, :v, :a),
            want = :s, bound = (; m, v))
        @test m.products[] == 0
        prepared = prepare_ad(kernel, NA_AD_BACKEND, a; active = :a)
        @test m.products[] == 1
        folded = only(value for value in naad_hidden(prepared)
            if value isa AbstractVector && value == product)
        matmul = only(i for i in eachindex(kernel.callables)
            if kernel.callables[i] isa ReactiveKernels._MatMulStep)
        @test folded == product
        @test folded !== kernel.caches[matmul][]
        preparation = prepared.preparation
        gradient = similar(a)
        for point in (a, fill(-0.25, n), a)
            value, returned = ad_value_and_gradient!(prepared, gradient, point)
            @test value == sum(product .+ point)
            @test returned === gradient
            @test gradient == ones(n)
            @test m.products[] == 1
        end
        @test prepared.preparation === preparation
        push!(allocations, @allocated ad_value_and_gradient!(
            prepared, gradient, a))
        @test m.products[] == 1
        second = prepare_ad(kernel, NA_AD_BACKEND, a; active = :a)
        @test m.products[] == 2
        second_folded = only(value for value in naad_hidden(second)
            if value isa AbstractVector && value == product)
        @test second_folded == folded
        @test second_folded !== folded
        @test kernel(a) == sum(product .+ a)
        @test m.products[] == 3
        @test ad_value_and_gradient!(prepared, gradient, a)[1] ==
              sum(product .+ a)
        @test m.products[] == 3
        @test (data, v, a) == originals

        # An inactive but unbound operand still requires a product per call.
        live = prepare_nonallocating(spec; have = (:m, :v, :a),
            want = :s, bound = (; m))
        live_prepared = prepare_ad(live, NA_AD_BACKEND, v, a; active = :a)
        for live_v in (v, 2v)
            before = m.products[]
            value, _ = ad_value_and_gradient!(live_prepared, gradient, live_v, a)
            @test m.products[] == before + 1
            @test value == sum(data * live_v .+ a)
            @test gradient == ones(n)
        end
        @test (data, v, a) == originals
    end
    println("NONALLOCATING_AD_ALLOC\tfold_growth\t", allocations)
    @test maximum(allocations) <= 64
    @test allocations[end] <= allocations[2] + 64
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
    # The product sees the unbound port, so nothing folds: the bound matrix
    # is the only hidden operand.
    @test length(naad_ad_constants(prepna)) == length(kbna.constants)
    @test naad_hidden(prepna) == (m,)
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
    @test any(op -> op === +, kbna.callables)
    @test any(op -> op === -, kbna.callables)
    prep = prepare_ad(kb, NA_AD_BACKEND, a0; active = :a)
    prepna = prepare_ad(kbna, NA_AD_BACKEND, a0; active = :a)
    @test any(op -> op === +, naad_ad_callables(prepna))
    @test any(op -> op === -, naad_ad_callables(prepna))
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
    @test any(op -> op isa ReactiveKernels._MaterializeStep, kbna.callables)
    prep = prepare_ad(kb, NA_AD_BACKEND, a0; active = :a)
    prepna = prepare_ad(kbna, NA_AD_BACKEND, a0; active = :a)
    @test any(op -> op isa ReactiveKernels._MaterializeStep, naad_ad_callables(prepna))
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
    @test any(op -> op === +, matkbna.callables)
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
    @test any(op -> op isa ReactiveKernels._MaterializeStep, dotmatkbna.callables)
    dotmatprep = prepare_ad(dotmatkb, NA_AD_BACKEND, A0; active = :A)
    dotmatprepna = prepare_ad(dotmatkbna, NA_AD_BACKEND, A0; active = :A)
    gdot, gdotna = similar(A0), similar(A0)
    vdot, _ = ad_value_and_gradient!(dotmatprep, gdot, A0)
    vdotna, _ = ad_value_and_gradient!(dotmatprepna, gdotna, A0)
    @test vdotna == vdot
    @test gdotna == gdot == ones(size(A0))
end

# Three or more axes: the AD-only fills run one loop per axis
# (`ReactiveKernels._ad_na_fill!`), because Enzyme differentiates a
# `CartesianIndices` iteration over three axes many times slower than its
# primal (snag `hmm-per-step-tra-0e27954b`: an HMM transition with a step
# axis). The loops visit the indices of `eachindex` in its order, so an
# extruded build, a same-shape `log.` and a three-axis `+` give the ordinary
# gradient.
@kernel naad_three_axes(theta, keep, w) = begin
    P = theta .* keep
    L = log.(P)
    Q = L + w
    s = sum(Q .* w)
    return s
end
@testset "tainted NA steps over three axes" begin
    n = 7
    theta0 = [0.7 0.3; 0.2 0.8]
    keep0 = reshape(collect(range(0.5, 1.5; length = n)), 1, 1, n)
    w0 = reshape(collect(range(-1.0, 1.0; length = 4n)), 2, 2, n)
    bound = (; keep = keep0, w = w0)
    kb = prepare(naad_three_axes; have = (:theta, :keep, :w), want = :s, bound)
    kbna = prepare_nonallocating(naad_three_axes; have = (:theta, :keep, :w),
                                 want = :s, bound)
    prep = prepare_ad(kb, NA_AD_BACKEND, theta0; active = :theta)
    prepna = prepare_ad(kbna, NA_AD_BACKEND, theta0; active = :theta)
    # Vacuity: the three-axis broadcasts and the `+` stay tainted steps.
    @test count(op -> op isa ReactiveKernels._MaterializeStep,
                naad_ad_callables(prepna)) >= 2
    @test any(op -> op === +, naad_ad_callables(prepna))
    g, gna = zero(theta0), zero(theta0)
    v, _ = ad_value_and_gradient!(prep, g, theta0)
    vna, _ = ad_value_and_gradient!(prepna, gna, theta0)
    # s = Σ (log(θ_ij keep_t) + w_ijt) w_ijt, so ∂s/∂θ_ij = Σ_t w_ijt / θ_ij.
    @test vna ≈ v
    @test gna ≈ g ≈ dropdims(sum(w0; dims = 3); dims = 3) ./ theta0
end

@testset "AD-only fills visit eachindex's indices in order" begin
    for dims in ((), (3,), (2, 3), (2, 3, 4), (2, 1, 3, 2))
        A = reshape(collect(1.0:prod(dims)), dims)
        bc = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(identity, A))
        f, args, hoisted = ReactiveKernels._ad_na_broadcast_hoisted(bc)
        seen = Float64[]
        record = x -> (push!(seen, x); x)
        dest = zero(A)
        @test ReactiveKernels._ad_na_fill!(dest, record, args, hoisted, bc,
                                           axes(bc)) === dest
        @test dest == A
        @test seen == vec(A)
    end
    # An extruded leaf and a nested broadcast read the same elements.
    A = reshape(collect(1.0:24.0), 2, 3, 4)
    row = reshape([10.0, 20.0, 30.0, 40.0], 1, 1, 4)
    bc = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(+, A,
        Base.Broadcast.broadcasted(*, row, 2.0)))
    f, args, hoisted = ReactiveKernels._ad_na_broadcast_hoisted(bc)
    dest = zero(A)
    ReactiveKernels._ad_na_fill!(dest, f, args, hoisted, bc, axes(bc))
    @test dest == A .+ row .* 2.0
end

# A non-allocating program takes its operation table, cache tuple and cache
# driver ahead of the HAVE ports. Called through the RGF vararg wrapper, whose
# one-tuple splat of every operand stays a dynamic `Core._apply_iterate` past
# 32 operands, the primal allocated per call from 33 HAVE ports, and native
# Reverse rejected the constant HAVE arrays stored in that tuple beside the
# shadowed caches from 30 HAVE ports, plate or no plate. Both calls now enter
# the generated body directly (`_native_body_call_expr`). The plate here has
# 34 batched operands, one per HAVE port.
const WIDE_NA_PORTS = 33
let ports = [Symbol(:xs, k) for k in 1:WIDE_NA_PORTS],
    cells = [Symbol(:c, k) for k in 1:WIDE_NA_PORTS]
    declared = (:($p::Vector{Vector{Float64}}) for p in ports)
    terms = (:(l * sum($c; init = 0.0)) for c in cells)
    @eval @kernel wide_plate_naad(live::Vector{Float64}, $(declared...)) = begin
        out = plate(live, $(ports...)) do l, $(cells...)
            $(foldl((a, t) -> :($a + $t), terms; init = :(l * l)))
        end
        total::Float64 = sum(out)
    end
    @eval @kernel wide_noplate_naad(live::Vector{Float64}, $(declared...)) = begin
        total::Float64 = sum(abs2, live)
    end
end

# Fixed-arity measurements: `kernel` is called with explicit arguments, so the
# measured call has no splat of its own. The call sits in a `begin … end`
# block: from Julia 1.12, `@allocated f(args...)` runs
# `Base.allocated(f, args...)`, whose own `f(args...)` splat dispatches
# dynamically past 32 arguments and would be measured too.
for count in (1, WIDE_NA_PORTS + 1)
    arguments = [Symbol(:argument_, index) for index in 1:count]
    @eval function wide_naad_call_bytes(kernel, $(arguments...))
        kernel($(arguments...))
        @allocated begin
            kernel($(arguments...))
        end
    end
end

@testset "non-allocating programs with more than 32 operands" begin
    live = [0.1, 0.2, 0.3]
    data = Tuple([[Float64(k + i + j) for j in 1:mod(i, 3)] for i in 1:3]
                 for k in 1:WIDE_NA_PORTS)
    saved = deepcopy(data)
    have = (:live, (Symbol(:xs, k) for k in 1:WIDE_NA_PORTS)...)
    shifted(i) = sum(sum(d[i]; init = 0.0) for d in data)
    expected = sum(l^2 + l * shifted(i) for (i, l) in enumerate(live))
    gradient = [2l + shifted(i) for (i, l) in enumerate(live)]

    kbna = prepare_nonallocating(wide_plate_naad; have, want = :total)
    @test kbna(live, data...) ≈ expected
    prepared = prepare_ad(kbna, NA_AD_BACKEND, live, data...; active = :live)
    value, g = ad_value_and_gradient(prepared, live, data...)
    @test value ≈ expected
    @test g ≈ gradient
    destination = similar(live)
    inplace, returned = ad_value_and_gradient!(prepared, destination,
                                               live, data...)
    @test returned === destination
    @test inplace ≈ expected
    @test destination ≈ gradient

    noplate = prepare_nonallocating(wide_noplate_naad; have, want = :total)
    narrow = prepare_nonallocating(wide_noplate_naad; have = (:live,),
                                   want = :total)
    @test wide_naad_call_bytes(noplate, live, data...) ==
          wide_naad_call_bytes(narrow, live)
    noplate_ad = prepare_ad(noplate, NA_AD_BACKEND, live, data...;
                            active = :live)
    value, g = ad_value_and_gradient(noplate_ad, live, data...)
    @test value ≈ sum(abs2, live)
    @test g ≈ 2 .* live

    @test data == saved
    @test live == [0.1, 0.2, 0.3]
end

@kernel tangent_naad(q::Vector{Float64}, s::Float64, data::Vector{Float64}) = begin
    density::Float64 = sum(data .* exp.(s .* q)) - 0.5 * sum(abs2, q) - s^2
end

@testset "NonAllocatingKernel pushforwards and Hessian-vector products" begin
    data, q, s, v = [2.0, -1.0, 0.5], [0.3, -0.4, 0.2], 0.7, [1.0, 0.5, -2.0]
    kernel = prepare_nonallocating(plan(tangent_naad; want = :density))
    pushforward = prepare_ad_pushforward(
        kernel, AutoEnzyme(; mode = Enzyme.Forward), (v,), q, s, data;
        active = :q)
    @test only(ad_pushforward(pushforward, (v,), q, s, data)) ≈
          dot(data .* s .* exp.(s .* q) .- q, v)
    # Second-order passes through the AD-owned `Cache` contexts, in both
    # orders, and the third order. Julia 1.10 used to fail Enzyme's static
    # activity analysis here (EnzymeRuntimeActivityError, Enzyme 0.13.210
    # through 0.13.216); it passes with the Enzyme dogfood line (b2bb4cb6).
    hessian = Diagonal(data .* s^2 .* exp.(s .* q)) - I
    for backend in (SecondOrder(AutoEnzyme(; mode = Enzyme.Reverse),
                                AutoEnzyme(; mode = Enzyme.Forward)),
                    SecondOrder(AutoEnzyme(; mode = Enzyme.Forward),
                                AutoEnzyme(; mode = Enzyme.Reverse)))
        prepared = prepare_ad_hvp(kernel, backend, (v,), q, s, data; active = :q)
        @test only(ad_hvp(prepared, (v,), q, s, data)) ≈ hessian * v
    end
    third_order = SecondOrder(SecondOrder(AutoEnzyme(; mode = Enzyme.Reverse),
                                          AutoEnzyme(; mode = Enzyme.Forward)),
                              AutoEnzyme(; mode = Enzyme.Forward))
    prepared = prepare_ad_third(kernel, third_order, (v,), q, s, data; active = :q)
    @test only(ad_third(prepared, (v,), q, s, data)) ≈
        data .* s^3 .* exp.(s .* q) .* v .^ 2
end

# The differentiated program's cache driver (`_ad_na_cache_driver`) and its AD
# loops fill a slot's seeded buffer in place and never store into the slot: a
# result of another shape is returned and not kept. Storing the filled buffer
# back cost reverse mode a second pass over a matrix-vector product's operands
# (Enzyme snag `enzyme-reverse-g-294ad104`) and the per-element shadow tape the
# size-invariance tests above pin.
@testset "the differentiated program's cache slots are read-only" begin
    ext = Base.get_extension(ReactiveKernels, :ReactiveKernelsMutatingFunctionsExt)
    driver = ReactiveKernels._ad_na_cache_driver(ext.cache_apply!)
    @test driver === ext.cache_fill!
    A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    x = [1.0, -1.0]
    buffer = zeros(3)
    slot = Ref(buffer)
    y = driver(slot, ReactiveKernels._MatMulStep(), A, x)
    @test y === buffer
    @test y == A * x
    z = driver(slot, ReactiveKernels._MatMulStep(), A[1:2, :], x)
    @test z == A[1:2, :] * x
    @test slot[] === buffer

    bc = Base.Broadcast.broadcasted(*, 2.0, [1.0, 2.0, 3.0])
    @test ReactiveKernels._ad_na_materialize_loop!(slot, bc) === buffer
    @test buffer == [2.0, 4.0, 6.0]
    wide = Base.Broadcast.broadcasted(*, 2.0, [1.0, 2.0, 3.0, 4.0])
    @test ReactiveKernels._ad_na_materialize_loop!(slot, wide) == [2.0, 4.0, 6.0, 8.0]
    @test slot[] === buffer
    @test ReactiveKernels._ad_na_arith_loop!(slot, +, [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) === buffer
    @test buffer == [3.0, 3.0, 3.0]
    @test ReactiveKernels._ad_na_arith_loop!(slot, -, [1.0, 1.0], [2.0, 2.0]) == [-1.0, -1.0]
    @test slot[] === buffer
end

# A dotted composition's stable form is what the differentiated program runs:
# the same gradient as `prepare` where `exp` overflows, and a step the rewrite
# removed (`a` below) needs no seeding.
@kernel naad_stable_fused(x::Vector{Float64}) = begin
    out = log.(exp.(x))
    return sum(out)
end
@kernel naad_stable_dotted(x::Vector{Float64}) = begin
    a = exp.(x)
    b = log.(a)
    return sum(b)
end
@testset "prepare_ad over stable rewrites of a NonAllocatingKernel" begin
    x = [800.0, -800.0, 1.5]
    for spec in (naad_stable_fused, naad_stable_dotted)
        prep = prepare_ad(prepare(spec), NA_AD_BACKEND, x; active = :x)
        prepna = prepare_ad(prepare_nonallocating(spec, x), NA_AD_BACKEND, x; active = :x)
        literal = prepare_ad(prepare_nonallocating(spec, x; simplify = false),
                             NA_AD_BACKEND, x; active = :x)
        g, gna, gliteral = zero(x), zero(x), zero(x)
        v, _ = ad_value_and_gradient!(prep, g, x)
        vna, _ = ad_value_and_gradient!(prepna, gna, x)
        vliteral, _ = ad_value_and_gradient!(literal, gliteral, x)
        @test isequal(vna, v) && vna == 1.5
        @test isequal(gna, g) && gna == [1.0, 1.0, 1.0]
        @test isnan(vliteral) && any(isnan, gliteral)
    end
end
