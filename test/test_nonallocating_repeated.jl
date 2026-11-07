# Focused optional-extension regression; also run by the NA integration runner.
using ReactiveKernels, MutatingFunctions, DifferentiationInterface, Test
import Enzyme

repeated_na_spec = @kernel repeated_nonallocating(q, d) = begin
    z = q .* d
    y = sum(z .* z)
    return y
end

function repeated_na_source_snapshot(spec)
    g = spec.graph
    (; values = copy(g.values), recipes = copy(g.recipes),
       producers = deepcopy(g.producers), aliases = copy(g.aliases),
       version = g.version, preparations = g.preparations,
       source = map(r -> repr(r.source), g.recipes),
       ports = copy(spec.ports), port_order = copy(spec.port_order),
       have = copy(spec.have_names), want = copy(spec.want_names))
end

@testset "independent non-allocating preparations" begin
    @test Base.get_extension(ReactiveKernels,
                             :ReactiveKernelsMutatingFunctionsExt) !== nothing
    @test Base.get_extension(ReactiveKernels,
                             :ReactiveKernelsEnzymeExt) !== nothing
    spec = repeated_na_spec
    source_before = repeated_na_source_snapshot(spec)
    d = collect(range(0.2, 1.1; length = 16))
    q0 = collect(range(-0.5, 0.4; length = 16))
    q1 = q0 .+ 0.13
    inputs_before = map(copy, (d, q0, q1))

    @testset "primal buffers belong to each callable" begin
        # Both kernels run one compiled body while every cache slot still holds
        # nothing, so the slots can only differ by the kernel that holds them.
        k1 = prepare_nonallocating(spec; have = (:q, :d), want = :z)
        k2 = prepare_nonallocating(spec; have = (:q, :d), want = :z)
        asts = map(k -> deepcopy(code_expr(k)), (k1, k2))
        z1 = k1(q0, d)
        saved_z1 = copy(z1)
        z2 = k2(q1, d)
        @test z1 ≈ q0 .* d
        @test z2 ≈ q1 .* d
        @test z1 !== z2
        @test z1 == saved_z1
        @test any(slot -> slot isa Base.RefValue && slot[] === z1, k1.caches)
        @test any(slot -> slot isa Base.RefValue && slot[] === z2, k2.caches)
        @test k1(q1, d) === z1
        saved_z2 = copy(z2)
        @test k1(q0, d) ≈ q0 .* d
        @test z2 == saved_z2
        @test map(code_expr, (k1, k2)) == asts
    end

    for (label, backend) in (
            ("ordinary Reverse", AutoEnzyme(mode = Enzyme.Reverse)),
            ("explicit Const", AutoEnzyme(mode = Enzyme.Reverse,
                                          function_annotation = Enzyme.Const)))
        @testset "$label" begin
            # Keep the established Const recipe separate from default Reverse.
            # Compile two unseeded kernels together, then a third after warming.
            kernels = [prepare_nonallocating(spec; have = (:q, :d),
                         want = :y, bound = (; d)) for _ in 1:2]
            asts = map(k -> deepcopy(code_expr(k)), kernels)
            prepared = [prepare_ad(k, backend, q0; active = :q) for k in kernels]
            push!(kernels, prepare_nonallocating(spec; have = (:q, :d),
                                                want = :y, bound = (; d)))
            push!(asts, deepcopy(code_expr(last(kernels))))
            push!(prepared, prepare_ad(last(kernels), backend, q1; active = :q))
            for (i, k) in enumerate(kernels), other in kernels[1:i-1]
                for (slot, other_slot) in zip(k.caches, other.caches)
                    slot isa Base.RefValue || continue
                    @test slot !== other_slot
                    slot[] isa AbstractArray && @test slot[] !== other_slot[]
                end
            end
            for (round, q) in enumerate((q0, q1, q0)), i in eachindex(kernels)
                k, ad = kernels[i], prepared[i]
                # Interleave changed points in the other primal and AD objects.
                other_q = isodd(round) ? q1 : q0
                other = mod1(i + 1, length(kernels))
                @test kernels[other](other_q) ≈ sum((other_q .* d).^2)
                @test ad_value_and_gradient(prepared[other], other_q)[2] ≈
                      2 .* other_q .* d.^2
                value, gradient = ad_value_and_gradient(ad, q)
                @test value ≈ sum((q .* d).^2)
                @test gradient ≈ 2 .* q .* d.^2
                @test k(q) ≈ value
                @test map(code_expr, kernels) == asts
                @test repeated_na_source_snapshot(spec) == source_before
                @test (d, q0, q1) == inputs_before
            end
        end
    end
    @test repeated_na_source_snapshot(spec) == source_before
    @test (d, q0, q1) == inputs_before
end

bound_number_na_spec = @kernel bound_number_nonallocating(
        q::Vector{Float64}, d::Vector{Float64}, s::Float64) = begin
    z = q .* d .* s
    y::Float64 = sum(z .* z)
    return y
end

function bound_number_na_primal_bytes(k, q)
    k(q)
    k(q)
    @allocated k(q)
end

# The user's direction on numeric values (`ODLPlaus:precompile` decision
# `2026-11-06T01-09-46-212-0ikeidi`): a bound number is a runtime input, so its
# value is not part of a kernel's type and costs no compile.
@testset "non-allocating preparations share a body, not their state" begin
    d = collect(range(0.2, 1.1; length = 16))
    q0 = collect(range(-0.5, 0.4; length = 16))
    q1 = q0 .+ 0.13
    backend = AutoEnzyme(mode = Enzyme.Reverse)
    prepare_at(s; want = :y) = prepare_nonallocating(bound_number_na_spec;
        have = (:q, :d, :s), want, bound = (; d, s))
    expected(q, s) = sum((q .* d .* s) .^ 2)

    @testset "one type across values of a bound number" begin
        k, same, other = prepare_at(2.0), prepare_at(2.0), prepare_at(5.0)
        @test typeof(k) === typeof(same) === typeof(other)
        ads = map(x -> prepare_ad(x, backend, q0; active = :q), (k, same, other))
        @test typeof(ads[1]) === typeof(ads[2]) === typeof(ads[3])
        # Each kernel and gradient reads its own values from one compiled body.
        for (x, ad, s) in zip((k, same, other), ads, (2.0, 2.0, 5.0)), q in (q0, q1)
            @test x(q) ≈ expected(q, s)
            value, gradient = ad_value_and_gradient(ad, q)
            @test value ≈ expected(q, s)
            @test gradient ≈ 2 .* q .* (d .* s) .^ 2
        end
    end

    @testset "kernels of one type share no slot" begin
        k1, k2 = prepare_at(2.0; want = :z), prepare_at(5.0; want = :z)
        @test typeof(k1) === typeof(k2)
        z1 = k1(q0)
        saved = copy(z1)
        z2 = k2(q1)
        @test z1 !== z2
        @test z1 == saved
        @test z2 ≈ q1 .* d .* 5.0
        slots(k) = filter(slot -> slot isa Base.RefValue, collect(k.caches))
        @test !isempty(slots(k1))
        @test all(pair -> first(pair) !== last(pair), zip(slots(k1), slots(k2)))
        @test k1(q1) === z1
        @test z2 ≈ q1 .* d .* 5.0
    end

    @testset "a deepcopy owns its slots" begin
        k = prepare_at(2.0; want = :z)
        z = k(q0)
        saved = copy(z)
        copied = deepcopy(k)
        @test typeof(copied) === typeof(k)
        result = copied(q1)
        @test result !== z
        @test result ≈ q1 .* d .* 2.0
        @test z == saved
    end

    @testset "a warm call allocates nothing" begin
        k = prepare_at(2.0)
        @test bound_number_na_primal_bytes(k, q0) == 0
    end
end
